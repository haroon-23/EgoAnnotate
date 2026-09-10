"""Empirical verification script for Trajectory Smoothing (Change A) & Metric Scaling (Change B).

Runs retargeting on test_10s and new_video with:
  1. Smoothing OFF vs Smoothing ON: compares IK reachability and joint angle jitter ||q_k - q_{k-1}||.
  2. Metric Calibration: computes estimated scale factor, physical wrist motion range (m), and cumulative travel path.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from src.retargeting.retargeter import Retargeter, RetargetingConfig
from src.retargeting.pose_mapper import PoseMapperConfig
from src.retargeting.metric_calibration import MetricCalibrator, MetricCalibrationConfig


def run_evaluation(episode_name: str, annotations_path: str):
    print(f"\n{'='*70}")
    print(f" EVALUATING RETARGETING FOR: {episode_name}")
    print(f"{'='*70}")

    # Load default config
    base_cfg = RetargetingConfig.from_yaml("configs/retargeting_franka.yaml")

    # --- 1. Run with Smoothing OFF -------------------------------------------
    cfg_off = RetargetingConfig.from_yaml("configs/retargeting_franka.yaml")
    cfg_off.pose_mapper.enable_trajectory_smoothing = False

    retargeter_off = Retargeter(cfg_off)
    res_off = retargeter_off.run_from_annotations(annotations_path, episode_id=f"{episode_name}_off")

    # Joint angle deltas (jitter) OFF
    q_off = res_off.joint_trajectories  # (N, 7)
    deltas_off = np.linalg.norm(np.diff(q_off, axis=0), axis=1)  # (N-1,)
    mean_jitter_off = float(np.mean(deltas_off))
    max_jitter_off = float(np.max(deltas_off))

    # --- 2. Run with Smoothing ON --------------------------------------------
    cfg_on = RetargetingConfig.from_yaml("configs/retargeting_franka.yaml")
    cfg_on.pose_mapper.enable_trajectory_smoothing = True
    cfg_on.pose_mapper.smoothing_window = 9
    cfg_on.pose_mapper.smoothing_polyorder = 2

    retargeter_on = Retargeter(cfg_on)
    res_on = retargeter_on.run_from_annotations(annotations_path, episode_id=f"{episode_name}_on")

    # Joint angle deltas (jitter) ON
    q_on = res_on.joint_trajectories  # (N, 7)
    deltas_on = np.linalg.norm(np.diff(q_on, axis=0), axis=1)  # (N-1,)
    mean_jitter_on = float(np.mean(deltas_on))
    max_jitter_on = float(np.max(deltas_on))

    # Jitter reduction %
    jitter_reduction_pct = (1.0 - mean_jitter_on / max(mean_jitter_off, 1e-9)) * 100.0

    # --- 3. Extract Metric Scaling Data --------------------------------------
    meta_sample = res_on.target_poses[0].scaling_metadata.get("metric_calibration", {})

    print(f"\n--- CHANGE A: TRAJECTORY SMOOTHING (window=9, polyorder=2) ---")
    print(f"  Frames Processed            : {res_on.n_frames}")
    print(f"  IK Reachability (OFF)       : {res_off.summary['n_reachable']}/{res_off.n_frames} ({res_off.summary['pct_reachable']:.2f}%)")
    print(f"  IK Reachability (ON)        : {res_on.summary['n_reachable']}/{res_on.n_frames} ({res_on.summary['pct_reachable']:.2f}%)")
    print(f"  Mean Joint Delta (OFF)      : {mean_jitter_off:.4f} rad/frame")
    print(f"  Mean Joint Delta (ON)       : {mean_jitter_on:.4f} rad/frame")
    print(f"  Max Joint Delta (OFF)       : {max_jitter_off:.4f} rad/frame")
    print(f"  Max Joint Delta (ON)        : {max_jitter_on:.4f} rad/frame")
    print(f"  Joint Jitter Reduction      : {jitter_reduction_pct:+.2f}%")

    print(f"\n--- CHANGE B: APPROXIMATE METRIC SCALING ---")
    print(f"  Calibration Method          : {meta_sample.get('calibration_method')}")
    print(f"  Anthropometric Ref Source   : {meta_sample.get('anthropometric_source')}")
    print(f"  Estimated Scale Factor      : {meta_sample.get('scale_factor_m_per_unit'):.4f} m/unit")
    print(f"  Estimated Wrist Motion Span : {meta_sample.get('estimated_wrist_motion_span_m')}")
    print(f"  Cumulative Path Length      : {meta_sample.get('total_wrist_path_length_m'):.3f} m")
    print(f"  Documented Error Margin     : {meta_sample.get('documented_error_margin')}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_evaluation("test_10s", "data/output/test_10s/frame_annotations.json")
    run_evaluation("new_video", "data/output/new_video/frame_annotations.json")
