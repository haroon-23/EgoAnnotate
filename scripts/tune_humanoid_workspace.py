"""Tune workspace bounds for Unitree H1 Humanoid Arm."""
import json
import numpy as np
from src.retargeting.retargeter import Retargeter, RetargetingConfig

def test_bounds(wb_dict):
    cfg = RetargetingConfig.from_yaml("configs/retargeting_humanoid.yaml")
    cfg.pose_mapper.robot_workspace_bounds = wb_dict
    retargeter = Retargeter(cfg)
    res = retargeter.run_from_annotations("data/output/new_video/frame_annotations.json", episode_id="humanoid_tune")
    n_reach = res.summary['n_reachable']
    pct = res.summary['pct_reachable']
    res_mm = res.summary['avg_ik_residual_mm']
    print(f"Bounds {wb_dict} -> Reachable: {n_reach}/{res.n_frames} ({pct:.1f}%), Avg Residual: {res_mm:.2f} mm")
    return pct

if __name__ == "__main__":
    # Test bounds relative to H1 torso base: shoulder is at [0.0, -0.18, 0.35]
    # Hand working area in front of chest: X in front [0.1, 0.5], Y right side [-0.4, 0.0], Z in front of torso [0.0, 0.45]
    test_bounds({
        "x_min": 0.10, "x_max": 0.50,
        "y_min": -0.40, "y_max": 0.00,
        "z_min": 0.00, "z_max": 0.40
    })
