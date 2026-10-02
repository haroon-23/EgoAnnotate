"""Hermetic tests for src/retargeting/weld_grasp.py — pure numpy, no heavy deps."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hermetic_stubs import scipy_stub_for_import

with scipy_stub_for_import():
    from src.retargeting.weld_grasp import (  # noqa: E402
        WeldGraspConfig,
        GraspWindow,
        find_grasp_windows,
    )


def _synth(n=20, closed_lo=5, closed_hi=15, obj=None, ee_jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    reach = np.ones(n, dtype=bool)
    grip = np.full(n, 0.08)  # open
    grip[closed_lo:closed_hi] = 0.01  # closed
    obj = np.array([0.5, 0.0, 0.31]) if obj is None else np.asarray(obj, float)
    ee = np.tile(obj, (n, 1)) + rng.normal(0, ee_jitter, (n, 3))
    return reach, grip, ee, obj


def test_basic_window_detected():
    reach, grip, ee, obj = _synth()
    wins = find_grasp_windows(reach, grip, ee, obj)
    assert len(wins) == 1
    w = wins[0]
    assert (w.start_idx, w.end_idx) == (5, 15)
    assert w.grasp_frame_idx == 5
    np.testing.assert_allclose(w.object_spawn_pos, ee[5], rtol=0, atol=1e-9)
    assert w.peak_lift_m == 0.0 and w.weld_held is False  # filled by verifier later


def test_short_window_dropped():
    reach, grip, ee, obj = _synth(closed_lo=5, closed_hi=8)  # 3 < 6 frames
    assert find_grasp_windows(reach, grip, ee, obj) == []


def test_exact_min_length_kept():
    reach, grip, ee, obj = _synth(closed_lo=5, closed_hi=11)  # exactly 6 frames
    wins = find_grasp_windows(reach, grip, ee, obj)
    assert len(wins) == 1 and (wins[0].start_idx, wins[0].end_idx) == (5, 11)


def test_proximity_failure_drops_window():
    reach, grip, ee, obj = _synth()
    far_obj = obj + np.array([0.5, 0.0, 0.0])  # 0.5 m away > 0.09
    assert find_grasp_windows(reach, grip, ee, far_obj) == []


def test_unreachable_frames_break_window():
    reach, grip, ee, obj = _synth()
    reach[8:12] = False
    wins = find_grasp_windows(reach, grip, ee, obj)
    # (5,8) is 3 frames -> dropped; (12,15) is 3 frames -> dropped
    assert wins == []


def test_gap_splits_into_two_windows():
    reach, grip, ee, obj = _synth(n=30, closed_lo=0, closed_hi=30)
    grip[12:16] = 0.08  # open gap in the middle
    wins = find_grasp_windows(reach, grip, ee, obj)
    assert len(wins) == 2
    assert (wins[0].start_idx, wins[0].end_idx) == (0, 12)
    assert (wins[1].start_idx, wins[1].end_idx) == (16, 30)


def test_open_gripper_no_window():
    reach, grip, ee, obj = _synth()
    grip[:] = 0.08
    assert find_grasp_windows(reach, grip, ee, obj) == []


def test_empty_input_returns_empty():
    assert find_grasp_windows(
        np.zeros(0, dtype=bool), np.zeros(0), np.zeros((0, 3)), np.zeros(3)
    ) == []


def test_shape_mismatch_returns_empty_never_raises():
    reach, grip, ee, obj = _synth()
    assert find_grasp_windows(reach, grip[:5], ee, obj) == []
    assert find_grasp_windows(reach, grip, ee, np.zeros(2)) == []


def test_gripper_2d_column_shape_accepted():
    reach, grip, ee, obj = _synth()
    wins = find_grasp_windows(reach, grip[:, None], ee, obj)  # (N,1) like HDF5
    assert len(wins) == 1


def test_custom_config_actually_applies():
    cfg = WeldGraspConfig(close_threshold_m=0.02, min_window_frames=4)
    reach, grip, ee, obj = _synth(closed_lo=5, closed_hi=9)  # grip 0.01 < 0.02
    wins = find_grasp_windows(reach, grip, ee, obj, cfg)
    assert len(wins) == 1 and (wins[0].start_idx, wins[0].end_idx) == (5, 9)


def test_window_to_dict_json_safe():
    import json
    reach, grip, ee, obj = _synth()
    d = find_grasp_windows(reach, grip, ee, obj)[0].to_dict()
    json.dumps(d)  # must not raise
    assert d["start_idx"] == 5 and d["weld_held"] is False
