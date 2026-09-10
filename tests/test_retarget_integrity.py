import numpy as np
import pytest


def test_velocity_infeasible_flagged():
    from src.retargeting.retargeter import enforce_velocity_limits
    q = np.zeros((4, 7))
    q[2, 0] = 0.5  # 0.5 rad in 1/30 s = 15 rad/s
    reach = np.array([True, True, True, True])
    out = enforce_velocity_limits(q, reach, dt=1/30.0, vel_limits=np.array([2.17]*7))
    assert out[2] == False and out[1] == True


def test_interpolated_never_reachable():
    from src.retargeting.retargeter import apply_source_gate
    r = apply_source_gate(reachable=True, interpolated=True, present=True, gripper_prev=0.03)
    assert r["reachable"] is False and r["retarget_source"] == "interpolated_hold"


def test_gripper_continuous_inside_bounds():
    from src.retargeting.gripper_mapper import map_opening
    vals = [map_opening(d, "precision_pinch", 0.9) for d in np.linspace(0.02, 0.06, 20)]
    assert len(set(np.round(vals, 5))) >= 15  # no pinning
