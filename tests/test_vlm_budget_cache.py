"""Request budget, response cache, backend factory, template fallback and
batched one-request-per-episode annotation (all with fake backends)."""
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.vlm_backend as vlm_mod
from src.cache_manager import StageCache, VLMDiskCache, make_stage_key
from src.datatypes import CandidateSegment
from src.gemini_client import generate_text
from src.language_generator import GeminiLanguageGenerator, LanguageGeneratorConfig
from src.segment_labeler import (
    EpisodeAnnotator,
    SegmentLabeler,
    SegmentLabelerConfig,
    template_instruction,
    template_label,
)
from src.vlm_backend import (
    BudgetExceeded,
    ManagedBackend,
    NullVLMBackend,
    VLMBackendConfig,
    VLMBudget,
    VLMRequest,
    VLMResponse,
    create_managed_backend,
    select_backend,
)


class CountingBackend:
    """Fake backend: counts real calls, returns canned text."""

    name = "fake"
    model_id = "fake-model-1"

    def __init__(self, text="{}"):
        self.text = text
        self.calls = 0
        self.requests = []

    def is_available(self):
        return True

    def generate(self, request):
        self.requests.append(request)
        if request.on_attempt:
            request.on_attempt()
        self.calls += 1
        return VLMResponse(text=self.text, backend=self.name, latency_s=0.0, model=self.model_id)


def _cand(i, start, end, contact="contact", grasp="power_wrap", obj="shirt",
          hands=("right",), tr="contact_on"):
    return CandidateSegment(
        start_frame=int(start * 10), end_frame=int(end * 10), start_time=start, end_time=end,
        transition_type=tr, contact_state=contact, grasp_type=grasp, object_name=obj,
        hands=list(hands), hand_used=hands[0] if len(hands) == 1 else "both",
    )


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "clip.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64))
    for i in range(30):
        frame = np.full((64, 96, 3), (i * 7) % 255, dtype=np.uint8)
        w.write(frame)
    w.release()
    return path


CANDS = [
    _cand(1, 0.0, 1.0, contact="no_contact", grasp="none", obj=None, tr="start"),
    _cand(2, 1.0, 2.0, tr="contact_on"),
    _cand(3, 2.0, 3.0, tr="grasp_change"),
]
GOOD_JSON = json.dumps({
    "task_instruction": "Fold the shirt on the bed",
    "segments": [
        {"index": 1, "action": "idle", "description": "hands resting"},
        {"index": 2, "action": "grasp", "description": "grabs the shirt"},
        {"index": 3, "action": "manipulate", "description": "folds the shirt"},
    ],
})


def _managed(inner, tmp_path, limit=3):
    return ManagedBackend(inner, VLMBudget(limit), VLMDiskCache(tmp_path / "vlm.json"))


# ---------------------------------------------------------------- budget


def test_budget_hard_cap_and_denied_counter():
    b = VLMBudget(2)
    b.consume()
    b.consume()
    with pytest.raises(BudgetExceeded):
        b.consume()
    assert b.used == 2 and b.denied == 1 and b.exhausted


def test_managed_backend_stops_calling_when_budget_spent(tmp_path):
    inner = CountingBackend("x")
    m = _managed(inner, tmp_path, limit=2)
    for i in range(2):
        m.generate(VLMRequest(prompt=f"p{i}"))
    with pytest.raises(BudgetExceeded):
        m.generate(VLMRequest(prompt="p-new"))
    assert inner.calls == 2
    assert m.stats()["requests_used"] == 2 and m.stats()["budget_exhausted"]


def test_zero_budget_means_cache_only(tmp_path):
    inner = CountingBackend("x")
    m = _managed(inner, tmp_path, limit=0)
    with pytest.raises(BudgetExceeded):
        m.generate(VLMRequest(prompt="p"))
    assert inner.calls == 0


def test_gemini_retries_count_against_budget():
    class Boom(Exception):
        code = 429

    class Models:
        def __init__(self):
            self.n = 0

        def generate_content(self, **kw):
            self.n += 1
            raise Boom("429 RESOURCE_EXHAUSTED")

    class Client:
        models = Models()

    budget = VLMBudget(2)
    import src.gemini_client as gc

    orig_sleep, orig_wait = gc.time.sleep, gc._PACER.wait
    gc.time.sleep = lambda s: None
    gc._PACER.wait = lambda: None
    try:
        with pytest.raises(BudgetExceeded):
            generate_text(Client, "m", ["p"], on_attempt=budget.consume, max_attempts=5)
    finally:
        gc.time.sleep, gc._PACER.wait = orig_sleep, orig_wait
    assert Client.models.n == 2  # 3rd attempt vetoed before hitting the network


# ---------------------------------------------------------------- cache


def test_response_cache_second_identical_request_makes_zero_calls(tmp_path):
    inner = CountingBackend("answer")
    m = _managed(inner, tmp_path)
    r1 = m.generate(VLMRequest(prompt="same"))
    # New process simulation: fresh wrapper, fresh budget, same cache file.
    m2 = ManagedBackend(inner, VLMBudget(3), VLMDiskCache(tmp_path / "vlm.json"))
    r2 = m2.generate(VLMRequest(prompt="same"))
    assert (r1.cached, r2.cached) == (False, True)
    assert inner.calls == 1 and m2.budget.used == 0 and m2.budget.cache_hits == 1


def test_cache_key_changes_with_prompt_model_and_image(tmp_path):
    from PIL import Image

    inner = CountingBackend("a")
    m = _managed(inner, tmp_path, limit=10)
    m.generate(VLMRequest(prompt="p"))
    m.generate(VLMRequest(prompt="p2"))
    m.generate(VLMRequest(prompt="p", images=[Image.new("RGB", (8, 8), (1, 2, 3))]))
    inner.model_id = "other-model"
    m.generate(VLMRequest(prompt="p"))
    assert inner.calls == 4


def test_episode_rerun_makes_zero_requests(tmp_path, video):
    inner = CountingBackend(GOOD_JSON)
    mk = lambda: ManagedBackend(inner, VLMBudget(3), VLMDiskCache(tmp_path / "vlm.json"))

    ann1 = EpisodeAnnotator(mk()).annotate(CANDS, video)
    assert inner.calls == 1 and ann1.requests_used == 1 and not ann1.cache_hit
    ann2 = EpisodeAnnotator(mk()).annotate(CANDS, video)
    assert inner.calls == 1  # zero new requests on re-run
    assert ann2.requests_used == 0 and ann2.cache_hit
    assert ann2.task_instruction == "Fold the shirt on the bed"


def test_changed_segment_table_is_a_cache_miss(tmp_path, video):
    inner = CountingBackend(GOOD_JSON)
    mk = lambda: ManagedBackend(inner, VLMBudget(3), VLMDiskCache(tmp_path / "vlm.json"))
    EpisodeAnnotator(mk()).annotate(CANDS, video)
    EpisodeAnnotator(mk()).annotate(CANDS[:2], video)
    assert inner.calls == 2


# ---------------------------------------------------------------- stage cache key


def test_stage_cache_key_covers_config_model_and_prompt(tmp_path):
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"video-bytes")
    cache = StageCache(vid, cache_root=tmp_path / "c")
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return calls["n"]

    k = lambda **kw: make_stage_key(**{"config": {"fps": 5}, "model_id": "m1", "prompt": "p1", **kw})
    assert cache.load_or_compute("s", compute, key=k()) == (1, False)
    assert cache.load_or_compute("s", compute, key=k()) == (1, True)
    assert cache.load_or_compute("s", compute, key=k(config={"fps": 10}))[1] is False
    assert cache.load_or_compute("s", compute, key=k(model_id="m2"))[1] is False
    assert cache.load_or_compute("s", compute, key=k(prompt="p2"))[1] is False
    assert calls["n"] == 4
    # Key order in the config dict must not matter.
    a = make_stage_key({"a": 1, "b": 2}, "m", "p")
    assert a == make_stage_key({"b": 2, "a": 1}, "m", "p")


def test_stage_cache_can_skip_persisting_degraded_results(tmp_path):
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"x")
    cache = StageCache(vid, cache_root=tmp_path / "c")
    n = {"i": 0}

    def compute():
        n["i"] += 1
        return {"source": "template"}

    for _ in range(2):
        cache.load_or_compute("lang", compute, cacheable=lambda r: r["source"] != "template")
    assert n["i"] == 2


# ---------------------------------------------------------------- factory


def _no_dotenv(monkeypatch):
    monkeypatch.setattr(vlm_mod, "ensure_api_key_from_dotenv", lambda: None)


def test_auto_picks_gemini_when_key_present(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setattr(vlm_mod, "GEMINI_AVAILABLE", True)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    sel = select_backend(VLMBackendConfig(backend="auto"))
    assert sel.backend.name == "gemini"


def test_auto_falls_back_to_openai_compatible_then_none(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nv")
    cfg = VLMBackendConfig(backend="auto", base_url="https://integrate.api.nvidia.com/v1", model="m")
    assert select_backend(cfg).backend.name == "openai_compatible"
    monkeypatch.delenv("NVIDIA_API_KEY")
    sel = select_backend(cfg)  # remote host, key missing -> unusable
    assert sel.backend.name == "none" and "auto -> none" in sel.reason
    local = VLMBackendConfig(backend="auto", base_url="http://localhost:11434/v1", model="qwen", api_key_env="")
    assert select_backend(local).backend.name == "openai_compatible"


def test_none_backend_and_managed_factory(monkeypatch):
    _no_dotenv(monkeypatch)
    m = create_managed_backend(VLMBackendConfig(backend="none"))
    assert isinstance(m.inner, NullVLMBackend) and m.name == "none" and m.cache is None
    assert m.stats()["max_requests_per_episode"] == 3


def test_managed_factory_budget_from_config(monkeypatch, tmp_path):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(vlm_mod, "GEMINI_AVAILABLE", True)
    cfg = VLMBackendConfig.from_config({
        "vlm": {"max_requests_per_episode": 1, "cache_path": str(tmp_path / "c.json")},
        "gemini": {"model": "gemini-2.5-flash"},
    })
    m = create_managed_backend(cfg)
    assert cfg.backend == "auto" and m.name == "gemini"
    assert m.budget.limit == 1 and m.model_id == "gemini-2.5-flash"


def test_on_unavailable_raise_for_explicit_backend(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="unavailable"):
        create_managed_backend(VLMBackendConfig(backend="gemini", on_unavailable="raise"))
    # default degrades instead of raising
    assert create_managed_backend(VLMBackendConfig(backend="gemini")).name == "none"


# ---------------------------------------------------------------- templates


def test_template_label_and_instruction_use_measured_signals_only():
    assert template_label(CANDS[0])["action"] == "idle"
    assert template_label(CANDS[1])["action"] == "contact"
    assert template_label(CANDS[2])["action"] == "grasp"
    text = template_instruction(CANDS)
    assert text == "Manipulate the shirt with the right hand using a power wrap grasp."
    both = [_cand(1, 0, 2, hands=("left", "right"), grasp="precision_pinch", obj="cup")]
    assert template_instruction(both) == "Manipulate the cup with both hands using a precision pinch grasp."
    assert template_instruction([CANDS[0]]) == "No hand-object interaction detected."
    assert "unidentified object" in template_instruction([_cand(1, 0, 1, obj=None)])


def test_none_backend_labels_everything_as_template_without_a_request(video):
    vlm = ManagedBackend(NullVLMBackend(), VLMBudget(3), None, "vlm.backend=none")
    labeler = SegmentLabeler(SegmentLabelerConfig(), vlm=vlm)
    segs = labeler.label_segments(CANDS, video)
    assert [s.label_source for s in segs] == ["template"] * 3
    assert labeler.last_annotation.language_source == "template"
    assert "vlm.backend=none" in labeler.last_annotation.degraded_reason
    gen = GeminiLanguageGenerator(LanguageGeneratorConfig(), vlm=vlm)
    assert gen.generate_episode_description(video) == labeler.last_annotation.task_instruction
    assert gen.last_annotation.language_source == "template"
    assert vlm.budget.used == 0


# ---------------------------------------------------------------- batched call


def test_one_request_names_segments_and_writes_instruction(tmp_path, video):
    inner = CountingBackend(GOOD_JSON)
    vlm = _managed(inner, tmp_path)
    labeler = SegmentLabeler(SegmentLabelerConfig(), vlm=vlm)
    gen = GeminiLanguageGenerator(LanguageGeneratorConfig(), vlm=vlm)

    segs = labeler.label_segments(CANDS, video)
    task = gen.generate_episode_description(video)

    assert inner.calls == 1  # segment names + instruction share one request
    assert task == "Fold the shirt on the bed"
    assert [s.name for s in segs] == ["idle", "grasp", "manipulate"]
    assert segs[2].description == "folds the shirt"
    assert all(s.label_source == "vlm:fake:fake-model-1" for s in segs)
    assert gen.last_annotation.language_source == "vlm"
    # keyframes + the deterministic table were what the model saw
    req = inner.requests[0]
    assert 1 < len(req.images) <= 5
    assert "contact=contact | grasp=power_wrap" in req.prompt and "strict JSON" in req.prompt
    # Downscaled keyframes
    assert max(req.images[0].size) <= 448


def test_instruction_only_when_labeler_did_not_run(tmp_path, video):
    inner = CountingBackend(json.dumps({"task_instruction": "Iron the shirt", "segments": []}))
    gen = GeminiLanguageGenerator(LanguageGeneratorConfig(), vlm=_managed(inner, tmp_path))
    assert gen.generate_episode_description(video) == "Iron the shirt"
    assert inner.calls == 1


def test_budget_exhausted_degrades_to_template_with_reason(tmp_path, video):
    inner = CountingBackend(GOOD_JSON)
    vlm = _managed(inner, tmp_path, limit=0)
    ann = EpisodeAnnotator(vlm).annotate(CANDS, video)
    assert inner.calls == 0
    assert ann.language_source == "template" and not ann.from_model
    assert "budget exhausted" in ann.degraded_reason
    assert ann.segment_sources == ["template"] * 3


def test_invalid_json_uses_repair_then_template_and_is_not_cached(tmp_path, video):
    inner = CountingBackend("definitely not json")
    vlm = _managed(inner, tmp_path, limit=3)
    ann = EpisodeAnnotator(vlm).annotate(CANDS, video)
    assert inner.calls == 2  # original + one repair re-prompt, both counted
    assert ann.language_source == "template" and "not valid JSON" in ann.degraded_reason
    # A bad reply must not be replayed forever from the disk cache.
    vlm2 = ManagedBackend(inner, VLMBudget(3), VLMDiskCache(tmp_path / "vlm.json"))
    EpisodeAnnotator(vlm2).annotate(CANDS, video)
    assert inner.calls >= 3


def test_partial_reply_marks_only_missing_segments_as_template(tmp_path, video):
    partial = json.dumps({
        "task_instruction": "Fold the shirt",
        "segments": [{"index": 2, "action": "grasp", "description": "grabs shirt"},
                     {"index": 3, "action": "flying", "description": "bad action"}],
    })
    ann = EpisodeAnnotator(_managed(CountingBackend(partial), tmp_path)).annotate(CANDS, video)
    assert ann.instruction_source == "vlm:fake:fake-model-1"
    assert ann.segment_sources == ["template", "vlm:fake:fake-model-1", "template"]
    assert ann.segment_labels[2]["action"] == "grasp"  # template for seg 3, not "flying"
    assert "2 segment(s)" in ann.degraded_reason


def test_backend_error_degrades_to_template(tmp_path, video):
    class Down(CountingBackend):
        def generate(self, request):
            raise RuntimeError("HTTP 503")

    ann = EpisodeAnnotator(_managed(Down(), tmp_path)).annotate(CANDS, video)
    assert ann.language_source == "template" and "HTTP 503" in ann.degraded_reason


# ---------------------------------------------------------------- labeler fix


def test_local_named_backend_is_actually_used_and_source_recorded(video):
    """Regression: a backend named 'local' was short-circuited to the heuristic."""

    class LocalLike(CountingBackend):
        name = "local"
        model_id = "smol"

    fake = LocalLike('{"action": "grasp", "description": "grasping the shirt"}')
    labeler = SegmentLabeler(SegmentLabelerConfig(batch_requests=False), vlm=fake)
    segs = labeler.label_segments(CANDS[1:2], video)
    assert fake.calls == 1
    assert segs[0].name == "grasp" and segs[0].description == "grasping the shirt"
    assert segs[0].label_source == "vlm:local:smol"


def test_per_segment_mode_falls_back_with_template_source(video):
    fake = CountingBackend("garbage")
    labeler = SegmentLabeler(SegmentLabelerConfig(batch_requests=False), vlm=fake)
    segs = labeler.label_segments(CANDS[1:2], video)
    assert segs[0].label_source == "template"
    assert segs[0].name == "contact"


def test_per_segment_mode_stops_at_budget(tmp_path, video):
    inner = CountingBackend('{"action": "grasp", "description": "d"}')
    vlm = _managed(inner, tmp_path, limit=1)
    labeler = SegmentLabeler(SegmentLabelerConfig(batch_requests=False), vlm=vlm)
    segs = labeler.label_segments(CANDS[1:], video)
    assert inner.calls == 1
    assert [s.label_source for s in segs] == ["vlm:fake:fake-model-1", "template"]
