"""Smoke-train test (Phase B): 20 real steps on a synthetic 2-episode merge.

Asserts the three success criteria from scripts/train_act_smoke.py:
  1. loss decreases (last-10 mean < 0.9 * first-10 mean),
  2. no NaN/Inf in loss or params,
  3. no-grad rollout returns (B, k, D).
Torch-guarded via pytest.importorskip.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

__import__("pytest").importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from merge_lerobot import merge_lerobot_datasets  # noqa: E402
from test_lerobot_exporter import _make_episode, _make_mp4, export_to_lerobot  # noqa: E402
from train_act_smoke import run_training  # noqa: E402


def _build_merge(tmp_path: Path) -> Path:
    ep_dirs = []
    for name in ("ep_a", "ep_b"):
        ep_dir = tmp_path / name
        ep_dir.mkdir(parents=True, exist_ok=True)
        mp4 = ep_dir / "capture.mp4"
        _make_mp4(mp4, n_frames=24, fps=10.0)
        _make_episode(ep_dir, name, n_frames=24, with_robot=True,
                      video_path=str(mp4), target_robot="franka")
        ep_dirs.append(export_to_lerobot(name, str(tmp_path), export_mode="robot"))
    return merge_lerobot_datasets(ep_dirs, tmp_path / "merged")


class TestActSmoke(unittest.TestCase):

    def test_smoke_train_20_steps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            merged = _build_merge(tmp_path)
            result = run_training(
                merged, steps=20, batch_size=4, chunk_size=8,
                lr=1e-4, seed=0, out=tmp_path / "runs" / "smoke",
                image_size=48)
            checks = result["checks"]
            self.assertTrue(checks["success"], f"checks failed: {checks}")
            self.assertTrue(checks["loss_decreasing"])
            self.assertTrue(checks["loss_finite"])
            self.assertTrue(checks["params_finite"])
            self.assertTrue(checks["rollout_shape_ok"])
            # artifacts written
            out = Path(result["out_dir"])
            self.assertTrue((out / "policy.pt").exists())
            self.assertTrue((out / "loss_curve.json").exists())
            cfg = json.loads((out / "config.json").read_text())
            self.assertEqual(cfg["steps"], 20)
            self.assertEqual(len(result["losses"]), 20)


if __name__ == "__main__":
    unittest.main()
