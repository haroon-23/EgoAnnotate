"""Tests for Phase 3B CLI v4 subcommands, license audit, DLQ halved-batch retry & resume."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.egoannotate.batch_orchestrator import MultiGPUBatchOrchestrator
from src.egoannotate.cli import build_parser, main


def test_cli_audit_licenses_lists_all_v4_commercial_backends(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["audit-licenses"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    by_name = {b["name"]: b for b in payload["backends"]}
    for expected in (
        "rtmw_133_apache",
        "moge2_mit",
        "da3_metric_large_apache",
        "qwen3_vl_local",
        "qwen3_vl_vllm",
    ):
        assert expected in by_name
        assert by_name[expected]["commercial_ok"] is True


def test_cli_parser_supports_v4_commands() -> None:
    parser = build_parser()
    a1 = parser.parse_args(["batch-gpu", "in_dir", "--out", "out_dir", "--shard-lerobot"])
    assert a1.command == "batch-gpu"
    assert a1.shard_lerobot is True

    a2 = parser.parse_args(["shard-lerobot", "runs", "--out", "sharded"])
    assert a2.command == "shard-lerobot"

    a3 = parser.parse_args(["validate-dataset", "lr_dir", "--no-smoke"])
    assert a3.command == "validate-dataset"
    assert a3.no_smoke is True

    a4 = parser.parse_args(["ray-batch", "in_dir", "--out", "out_dir", "--concurrency", "4"])
    assert a4.command == "ray-batch"
    assert a4.concurrency == 4


def test_orchestrator_dlq_halved_batch_retry_and_clean_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good_vid = tmp_path / "good.mp4"
    bad_vid = tmp_path / "bad.mp4"
    good_vid.write_bytes(b"good-video-payload-123456")
    bad_vid.write_bytes(b"corrupt-video-payload-999")

    calls: list[tuple[str, float]] = []

    def _fake_run_video(
        self: Any,
        video_path: Path,
        output_dir: Path,
        *,
        episode_id: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        _ = (self, use_cache)
        scale = float(os.environ.get("EGOANNOTATE_BATCH_SCALE", "1.0"))
        calls.append((video_path.name, scale))
        if video_path.name == "bad.mp4":
            raise RuntimeError("Simulated CUDA OOM on bad clip")
        output_dir.mkdir(parents=True, exist_ok=True)
        rep: dict[str, Any] = {
            "episode_id": episode_id or video_path.stem,
            "video": {"processed_frames": 30, "total_frames": 30, "duration_s": 1.0},
            "quality": {"robot_readiness_score": 0.94, "verdict": "PASS", "flags": []},
            "privacy_audit": {
                "privacy_pass": True,
                "frames_blurred": 0,
                "verified_unblurred_faces": 0,
            },
            "retargeting": {},
            "gpu_enrichment": {},
        }
        (output_dir / "run_report.json").write_text(json.dumps(rep), encoding="utf-8")
        return rep

    monkeypatch.setattr(
        "src.egoannotate.pipeline_v3.EgoAnnotateV3Pipeline.run_video",
        _fake_run_video,
    )

    out_root = tmp_path / "batch_out"
    orch = MultiGPUBatchOrchestrator(out_root=out_root, profile="cpu_smoke", max_gpus=1)
    orch.devices = ["cpu"]
    res1 = orch.run_videos([good_vid, bad_vid])

    assert res1["good"]["status"] == "OK"
    assert res1["bad"]["status"] == "FAILED_DLQ"
    bad_calls = [s for name, s in calls if name == "bad.mp4"]
    assert bad_calls == [1.0, 0.5]

    dlq_lines = (out_root / "dlq.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(dlq_lines) == 1
    assert json.loads(dlq_lines[0])["batch_scale"] == 0.5

    good_calls_before = len([1 for name, _ in calls if name == "good.mp4"])
    res2 = orch.run_videos([good_vid])
    good_calls_after = len([1 for name, _ in calls if name == "good.mp4"])
    assert res2["good"]["status"] == "OK"
    assert good_calls_after == good_calls_before


def test_orchestrator_catches_unbound_local_error_without_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vid = tmp_path / "race_clip.mp4"
    vid.write_bytes(b"rtmlib-race-payload")

    def _raise_unbound(
        self: Any,
        video_path: Path,
        output_dir: Path,
        *,
        episode_id: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        _ = (self, video_path, output_dir, episode_id, use_cache)
        raise UnboundLocalError("cannot access local variable 'cached_onnx'")

    monkeypatch.setattr(
        "src.egoannotate.pipeline_v3.EgoAnnotateV3Pipeline.run_video",
        _raise_unbound,
    )
    out_root = tmp_path / "out_unbound"
    orch = MultiGPUBatchOrchestrator(out_root=out_root, profile="cpu_smoke", max_gpus=1)
    orch.devices = ["cpu"]
    res = orch.run_videos([vid])
    assert res["race_clip"]["status"] == "FAILED_DLQ"
    assert "UnboundLocalError" in res["race_clip"]["error"]
