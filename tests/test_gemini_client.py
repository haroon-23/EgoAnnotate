"""Unit tests for the Gemini API resilience layer (Fix I, 2026-10-09).

Background: every verification run sprinkled "VLM call failed" errors through
the logs. Root cause, two halves:
  1. ``generate_text`` made ONE bare generate_content call — a single 429
     (quota) or 503 (overloaded) killed that segment's label instantly.
  2. The free Gemini tier caps gemini-2.5-flash at ~10-15 requests/minute,
     while the pipeline fires several VLM calls per episode, un-paced.

Fix I sits in ``generate_text`` (the single funnel every stage uses):
process-wide pacing (GEMINI_MIN_INTERVAL_S, default 4s), retry with backoff
that honors the server's "Please retry in Xs" hint, and fail-fast when the
hint indicates the DAILY quota is gone (retrying inside the run is useless).

All tests use a fake client: no network, no quota burned.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np  # noqa: F401  (keeps parity with other hermetic tests)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hermetic_stubs import scipy_stub_for_import  # noqa: E402

with scipy_stub_for_import():
    import src.gemini_client as gc  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    """Records calls; raises scripted exceptions, then succeeds."""

    def __init__(self, script):
        # script: list of exceptions to raise in order, then return "ok"
        self._script = list(script)
        self.calls = 0

    def generate_content(self, model=None, contents=None, config=None):
        self.calls += 1
        if self._script:
            raise self._script.pop(0)
        return _FakeResp("ok")


class _FakeClient:
    def __init__(self, script):
        self.models = _FakeModels(script)


class _FakeTime:
    """Drop-in for the `time` module inside gemini_client: records sleeps."""

    def __init__(self):
        self.sleeps = []
        self._now = 0.0

    def monotonic(self):
        return self._now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self._now += seconds


def _api_error(code, message, status):
    """Build a real google-genai error if the SDK is present."""
    try:
        from google.genai import errors
    except ImportError:
        exc = RuntimeError(message)
        exc.code = code  # type: ignore[attr-defined]
        exc.status = status  # type: ignore[attr-defined]
        return exc
    payload = {"error": {"code": code, "message": message, "status": status}}
    if code >= 500:
        return errors.ServerError(code, payload)
    return errors.ClientError(code, payload)


class TestGenerateTextRetry(unittest.TestCase):
    def setUp(self):
        self._orig_time = gc.time
        self.fake_time = _FakeTime()
        gc.time = self.fake_time
        self._orig_types = gc.types
        gc.types = types.SimpleNamespace(
            GenerateContentConfig=lambda **kw: None,
        )
        self._orig_interval = gc._PACER.min_interval_s
        gc._PACER.min_interval_s = 0.0  # pacing tested separately
        self._orig_env_attempts = gc.os.environ.pop("GEMINI_MAX_ATTEMPTS", None)

    def tearDown(self):
        gc.time = self._orig_time
        gc.types = self._orig_types
        gc._PACER.min_interval_s = self._orig_interval
        if self._orig_env_attempts is not None:
            gc.os.environ["GEMINI_MAX_ATTEMPTS"] = self._orig_env_attempts

    def test_success_first_try(self):
        client = _FakeClient([])
        out = gc.generate_text(client, "m", ["hello"])
        self.assertEqual(out, "ok")
        self.assertEqual(client.models.calls, 1)

    def test_429_retried_honoring_server_hint(self):
        err = _api_error(429, "Quota exceeded. Please retry in 2s",
                         "RESOURCE_EXHAUSTED")
        client = _FakeClient([err, err])
        out = gc.generate_text(client, "m", ["hello"])
        self.assertEqual(out, "ok")
        self.assertEqual(client.models.calls, 3)
        # hint 2s -> delay = min(120, 2+1) = 3s each retry
        self.assertEqual(self.fake_time.sleeps, [3.0, 3.0])

    def test_503_retried_with_backoff(self):
        err = _api_error(503, "The model is overloaded", "UNAVAILABLE")
        client = _FakeClient([err])
        out = gc.generate_text(client, "m", ["hello"])
        self.assertEqual(out, "ok")
        self.assertEqual(client.models.calls, 2)
        self.assertEqual(len(self.fake_time.sleeps), 1)
        # backoff 2s * jitter(±25%) -> in [1.5, 2.5]
        self.assertGreaterEqual(self.fake_time.sleeps[0], 1.5)
        self.assertLessEqual(self.fake_time.sleeps[0], 2.5)

    def test_400_raises_immediately(self):
        err = _api_error(400, "Invalid request", "INVALID_ARGUMENT")
        client = _FakeClient([err])
        with self.assertRaises(Exception):
            gc.generate_text(client, "m", ["hello"])
        self.assertEqual(client.models.calls, 1)
        self.assertEqual(self.fake_time.sleeps, [])

    def test_daily_cap_hint_fails_fast(self):
        # "retry in 3600s" = daily quota gone; retrying in-run is useless.
        err = _api_error(429, "Quota exceeded. Please retry in 3600s",
                         "RESOURCE_EXHAUSTED")
        client = _FakeClient([err, err, err, err, err])
        with self.assertRaises(Exception):
            gc.generate_text(client, "m", ["hello"])
        self.assertEqual(client.models.calls, 1)  # no retries burned
        self.assertEqual(self.fake_time.sleeps, [])

    def test_attempts_exhausted_raises(self):
        gc.os.environ["GEMINI_MAX_ATTEMPTS"] = "3"
        err = _api_error(503, "overloaded", "UNAVAILABLE")
        client = _FakeClient([err, err, err, err, err])
        with self.assertRaises(Exception):
            gc.generate_text(client, "m", ["hello"])
        self.assertEqual(client.models.calls, 3)

    def test_timeout_error_retried(self):
        class ReadTimeout(Exception):
            pass
        client = _FakeClient([ReadTimeout("slow")])
        out = gc.generate_text(client, "m", ["hello"])
        self.assertEqual(out, "ok")
        self.assertEqual(client.models.calls, 2)


class TestPacer(unittest.TestCase):
    def setUp(self):
        self._orig_time = gc.time
        self.fake_time = _FakeTime()
        gc.time = self.fake_time
        self._orig_interval = gc._PACER.min_interval_s

    def tearDown(self):
        gc.time = self._orig_time
        gc._PACER.min_interval_s = self._orig_interval

    def test_pacer_spaces_calls(self):
        gc._PACER.min_interval_s = 4.0
        gc._PACER._last_call_s = 0.0
        gc._PACER.wait()
        self.assertEqual(self.fake_time.sleeps, [4.0])
        # Second call immediately after: must wait the full interval again
        gc._PACER.wait()
        self.assertEqual(self.fake_time.sleeps, [4.0, 4.0])

    def test_pacer_zero_interval_no_wait(self):
        gc._PACER.min_interval_s = 0.0
        gc._PACER.wait()
        self.assertEqual(self.fake_time.sleeps, [])


class TestErrorClassification(unittest.TestCase):
    def test_codes(self):
        self.assertTrue(gc._is_retryable_error(
            _api_error(429, "x", "RESOURCE_EXHAUSTED")))
        self.assertTrue(gc._is_retryable_error(
            _api_error(503, "x", "UNAVAILABLE")))
        self.assertTrue(gc._is_retryable_error(
            _api_error(500, "x", "INTERNAL")))
        self.assertFalse(gc._is_retryable_error(
            _api_error(400, "bad", "INVALID_ARGUMENT")))
        self.assertFalse(gc._is_retryable_error(
            _api_error(404, "no model", "NOT_FOUND")))
        self.assertFalse(gc._is_retryable_error(
            _api_error(403, "denied", "PERMISSION_DENIED")))

    def test_delay_honors_hint(self):
        err = _api_error(429, "Please retry in 10s", "RESOURCE_EXHAUSTED")
        self.assertEqual(gc._retry_delay_s(err, 0), 11.0)
        err2 = _api_error(429, "Please retry in 999s", "RESOURCE_EXHAUSTED")
        self.assertEqual(gc._retry_delay_s(err2, 0), 120.0)  # capped


if __name__ == "__main__":
    unittest.main()
