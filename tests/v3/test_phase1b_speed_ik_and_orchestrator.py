"""Tests for Phase 1B vectorized batched IK, reach clamping, ledger, and DLQ."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.egoannotate.batch_orchestrator import (
    BatchStateLedger,
    DeadLetterQueue,
    compute_video_manifest_id,
)
from src.egoannotate.config import load_config
from src.egoannotate.ik_batched import (
    clamp_targets_to_workspace,
    solve_arm_trajectory_ik_batched,
)
from src.egoannotate.retargeting_v3 import load_kinematic_chains


def test_batched_ik_fast_and_accurate_on_unitree_g1() -> None:
    _, chains = load_kinematic_chains(Path("configs/embodiments_v3/urdfs/unitree_g1_dual_7dof.urdf"))
    chain = next(iter(chains.values()))
    t_len = 300
    t = np.linspace(0.0, 2.0 * np.pi, t_len, dtype=np.float32)
    far_targets = np.stack(
        [
            0.35 + 0.12 * np.cos(t),
            -0.20 + 0.10 * np.sin(t),
            0.25 + 0.08 * np.sin(2.0 * t),
        ],
        axis=1,
    )
    far_targets[10] = np.array([2.5, -2.5, 2.5], dtype=np.float32)

    clamped = clamp_targets_to_workspace(far_targets, chain)
    assert float(np.linalg.norm(clamped[10])) < float(np.linalg.norm(far_targets[10]))

    t0 = time.perf_counter()
    qpos, ee_m, err_mm, used = solve_arm_trajectory_ik_batched(
        chain,
        far_targets,
        dt_s=1.0 / 30.0,
        clamp_reach=True,
        device="cpu",
    )
    elapsed = time.perf_counter() - t0
    assert qpos.shape[0] == t_len
    assert ee_m.shape == (t_len, 3)
    assert used.shape == (t_len, 3)
    assert np.all(np.isfinite(qpos))
    assert float(np.mean(err_mm)) < 15.0
    assert elapsed < 5.0


def test_batch_ledger_6_states_and_dlq(tmp_path: Path) -> None:
    fake_vid = tmp_path / "clip.mp4"
    fake_vid.write_bytes(b"egoannotate-v4-test-bytes")
    mid = compute_video_manifest_id(fake_vid)
    assert len(mid) == 64

    ledger = BatchStateLedger(tmp_path / "ledger.sqlite3")
    row = ledger.register_video(mid, "clip", fake_vid, tmp_path / "out")
    assert row["state"] == "PENDING"

    for st in ("PERCEPTION", "SEMANTICS", "RETARGETING", "VERIFIED", "SHARDED"):
        ledger.transition(mid, st, gpu_device="cuda:0", batch_scale=1.0)
        cur = ledger.get(mid)
        assert cur is not None
        assert cur["state"] == st

    dlq = DeadLetterQueue(tmp_path / "dlq.jsonl")
    dlq.record(mid, "clip", fake_vid, "CUDA OOM", "Traceback...", batch_scale=0.5)
    lines = (tmp_path / "dlq.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["manifest_id"] == mid
    assert payload["batch_scale"] == 0.5


def test_gpu_prod_yaml_has_industrial_and_garment_prompts() -> None:
    cfg = load_config(profile="gpu_prod")
    prompts = set(cfg.perception.open_vocab_prompts)
    for expected in ("trowel", "welding torch", "sewing machine", "fabric", "plastic bin"):
        assert expected in prompts
