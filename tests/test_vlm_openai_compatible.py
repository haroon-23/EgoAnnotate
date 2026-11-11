"""OpenAI-compatible backend against a local fake HTTP server (no real network)."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vlm_backend import (
    BudgetExceeded,
    ManagedBackend,
    OpenAICompatibleBackend,
    VLMBackendConfig,
    VLMBudget,
    VLMRequest,
    strip_reasoning,
)


class _Server:
    """Scripted responses: list of (status, body, headers) consumed in order."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                outer.seen.append({
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "body": json.loads(self.rfile.read(n)),
                })
                status, body, headers = outer.script.pop(0) if outer.script else outer.last
                outer.last = (status, body, headers)
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _ok(text):
    return (200, {"choices": [{"message": {"role": "assistant", "content": text}}]}, None)


@pytest.fixture
def server():
    made = []

    def make(script):
        s = _Server(script)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


def _backend(url, **kw):
    cfg = VLMBackendConfig(
        backend="openai_compatible", base_url=url, model="nvidia/cosmos-reason1-7b",
        api_key_env="TEST_NV_KEY", request_timeout_s=5.0, max_retries=2, **kw,
    )
    sleeps = []
    return OpenAICompatibleBackend(cfg, sleep=sleeps.append), sleeps


def test_sends_chat_completion_with_image_parts_and_bearer(server, monkeypatch):
    monkeypatch.setenv("TEST_NV_KEY", "secret-token")
    srv = server([_ok('{"a": 1}')])
    b, _ = _backend(srv.url)
    assert b.is_available()
    img = Image.new("RGB", (32, 24), (200, 10, 10))
    resp = b.generate(VLMRequest(images=[img, img], prompt="describe", max_new_tokens=77, temperature=0.2))
    assert resp.text == '{"a": 1}' and resp.backend == "openai_compatible"
    seen = srv.seen[0]
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer secret-token"
    body = seen["body"]
    assert body["model"] == "nvidia/cosmos-reason1-7b"
    assert body["max_tokens"] == 77 and body["temperature"] == pytest.approx(0.2)
    parts = body["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "describe"}
    assert [p["type"] for p in parts[1:]] == ["image_url", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_retries_on_429_honouring_retry_after_and_counts_each_attempt(server, monkeypatch):
    monkeypatch.setenv("TEST_NV_KEY", "k")
    srv = server([(429, {"error": "slow down"}, {"Retry-After": "2"}), _ok("done")])
    b, sleeps = _backend(srv.url)
    budget = VLMBudget(3)
    resp = b.generate(VLMRequest(prompt="p", on_attempt=budget.consume))
    assert resp.text == "done"
    assert len(srv.seen) == 2 and budget.used == 2
    assert sleeps == [2.0]


def test_retries_are_bounded_and_stop_at_budget(server, monkeypatch):
    monkeypatch.setenv("TEST_NV_KEY", "k")
    srv = server([(503, {"error": "down"}, None)] * 10)
    b, _ = _backend(srv.url)
    with pytest.raises(RuntimeError, match="after 3 attempt"):
        b.generate(VLMRequest(prompt="p"))
    assert len(srv.seen) == 3  # 1 + max_retries

    srv2 = server([(503, {"error": "down"}, None)] * 10)
    b2, _ = _backend(srv2.url)
    managed = ManagedBackend(b2, VLMBudget(2), None)
    with pytest.raises(BudgetExceeded):
        managed.generate(VLMRequest(prompt="p"))
    assert len(srv2.seen) == 2  # the budget vetoes attempt 3


def test_permanent_error_is_not_retried_and_never_leaks_the_key(server, monkeypatch):
    monkeypatch.setenv("TEST_NV_KEY", "super-secret")
    srv = server([(401, {"error": "bad key"}, None)])
    b, sleeps = _backend(srv.url)
    with pytest.raises(RuntimeError) as exc:
        b.generate(VLMRequest(prompt="p"))
    assert "HTTP 401" in str(exc.value) and "super-secret" not in str(exc.value)
    assert len(srv.seen) == 1 and sleeps == []


def test_unreachable_server_raises_after_bounded_retries():
    cfg = VLMBackendConfig(base_url="http://127.0.0.1:9/v1", model="m", api_key_env="",
                           request_timeout_s=1.0, max_retries=1)
    b = OpenAICompatibleBackend(cfg, sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="after 2 attempt"):
        b.generate(VLMRequest(prompt="p"))


def test_availability_rules(monkeypatch):
    monkeypatch.delenv("TEST_NV_KEY", raising=False)
    remote = VLMBackendConfig(base_url="https://integrate.api.nvidia.com/v1", model="m", api_key_env="TEST_NV_KEY")
    assert not OpenAICompatibleBackend(remote).is_available()
    monkeypatch.setenv("TEST_NV_KEY", "x")
    assert OpenAICompatibleBackend(remote).is_available()
    local = VLMBackendConfig(base_url="http://localhost:8000/v1", model="m", api_key_env="NOPE_NOT_SET")
    assert OpenAICompatibleBackend(local).is_available()  # loopback needs no key
    assert not OpenAICompatibleBackend(VLMBackendConfig(base_url="", model="m")).is_available()
    assert not OpenAICompatibleBackend(VLMBackendConfig(base_url="http://x/v1", model="", api_key_env="")).is_available()


def test_json_mode_and_content_list_response(server, monkeypatch):
    monkeypatch.setenv("TEST_NV_KEY", "k")
    srv = server([(200, {"choices": [{"message": {"content": [{"type": "text", "text": "hi"}]}}]}, None)])
    b, _ = _backend(srv.url, json_mode=True)
    assert b.generate(VLMRequest(prompt="p")).text == "hi"
    assert srv.seen[0]["body"]["response_format"] == {"type": "json_object"}


def test_reasoning_blocks_are_stripped():
    assert strip_reasoning("<think>hmm {x}</think>\n{\"a\": 1}") == '{"a": 1}'
    assert strip_reasoning("<think>a</think><answer>fold shirt</answer>") == "fold shirt"
    assert strip_reasoning("thinking...</think>final") == "final"
    assert strip_reasoning("plain") == "plain"
