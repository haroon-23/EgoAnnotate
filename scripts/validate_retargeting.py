"""Simulation validation and side-by-side proof video generation.

Task 5 of the retargeting pipeline.

Usage::

    python scripts/validate_retargeting.py \\
        --annotations data/output/test_10s/frame_annotations.json \\
        --video data/raw_videos/test_10s.mp4 \\
        --output data/output/test_10s/retargeting/

What this script does:
  1. Loads URDF, runs the full retargeting pipeline on the annotation JSON.
  2. Opens PyBullet in DIRECT mode, loads the robot, replays joint trajectories.
  3. Renders the robot from a fixed camera view for each frame.
  4. Produces a SIDE-BY-SIDE video: left=original human video, right=robot sim.
  5. Reports quantitative validation: % reachable, avg solve time, collisions.

PyBullet rendering on macOS notes:
  - DIRECT mode + getCameraImage() uses Mesa/CPU software rendering.
  - If rendered frames are all-black (known on some macOS+GPU combos),
    the script falls back to a matplotlib skeleton-based 2D rendering.
  - The fallback is explicitly reported as "skeleton_2d_fallback" in output.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Ensure src/ is importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retargeting import (
    Retargeter,
    RetargetingConfig,
    RetargetingResult,
)
from src.retargeting.urdf_loader import URDFLoader
from src.retargeting.ik_solver import IKResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robot rendering helpers
# ---------------------------------------------------------------------------

CAMERA_DISTANCE = 1.5       # metres from target
CAMERA_YAW = 45.0           # degrees
CAMERA_PITCH = -30.0        # degrees
CAMERA_TARGET = [0.4, 0.0, 0.3]  # focus point in robot base frame
RENDER_W = 640
RENDER_H = 480


def render_robot_frame_pybullet(
    pb,
    client_id: int,
    robot_id: int,
    kin,
    joint_angles: np.ndarray,
    gripper_opening_m: float,
) -> Optional[np.ndarray]:
    """Render robot in given joint configuration via PyBullet getCameraImage.

    Returns:
        np.ndarray of shape (H, W, 3) BGR, or None if rendering failed.
    """
    # Set joint states
    for idx, angle in zip(kin.arm_joint_indices, joint_angles.tolist()):
        pb.resetJointState(robot_id, idx, angle, physicsClientId=client_id)
    # Set gripper
    for idx in kin.gripper_joint_indices:
        pb.resetJointState(robot_id, idx, gripper_opening_m, physicsClientId=client_id)

    # Compute view matrix
    view_matrix = pb.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=CAMERA_TARGET,
        distance=CAMERA_DISTANCE,
        yaw=CAMERA_YAW,
        pitch=CAMERA_PITCH,
        roll=0,
        upAxisIndex=2,
        physicsClientId=client_id,
    )
    proj_matrix = pb.computeProjectionMatrixFOV(
        fov=60,
        aspect=RENDER_W / RENDER_H,
        nearVal=0.01,
        farVal=5.0,
        physicsClientId=client_id,
    )

    try:
        _, _, px, _, _ = pb.getCameraImage(
            RENDER_W,
            RENDER_H,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pb.ER_TINY_RENDERER,  # CPU renderer, works in DIRECT mode
            physicsClientId=client_id,
        )
    except Exception as exc:
        logger.debug("getCameraImage failed: %s", exc)
        return None

    rgba = np.array(px, dtype=np.uint8).reshape(RENDER_H, RENDER_W, 4)
    rgb = rgba[:, :, :3]

    # Check for all-black frame (rendering failure indicator)
    if rgb.max() < 5:
        return None

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def render_robot_skeleton_2d(
    joint_angles: np.ndarray,
    kin,
    width: int = RENDER_W,
    height: int = RENDER_H,
) -> np.ndarray:
    """Fallback 2D stick-figure rendering of the robot arm.

    Uses a simplified 2D planar projection of the joint angles onto a canvas.
    This is clearly labeled as 'SKELETON FALLBACK' in the rendered image.

    Returns:
        np.ndarray of shape (height, width, 3) BGR.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)  # dark gray background

    # Simple 2D forward kinematics projection for visualization
    # Treat each joint angle as rotating in the XZ plane (sagittal view)
    n_joints = len(joint_angles)

    # Calculate link lengths dynamically based on arm joint count
    base_unit_len = 0.8 / max(n_joints, 1)
    link_lengths = [base_unit_len] * (n_joints + 1)

    # Project onto 2D: x=horizontal, y=vertical (up)
    cx, cy = width // 2, height - 50
    scale = min(width, height) * 0.35

    cum_angle = -np.pi / 2  # start pointing up
    x, y = float(cx), float(cy)
    points = [(x, y)]

    for i in range(n_joints):
        cum_angle += joint_angles[i] * 0.5  # damped for visual clarity
        length = link_lengths[i + 1] if i + 1 < len(link_lengths) else 0.1
        x += length * np.cos(cum_angle) * scale
        y -= length * np.sin(cum_angle) * scale
        points.append((float(x), float(y)))

    # Draw links
    colors = [(100, 200, 255), (100, 255, 100), (255, 200, 100),
              (200, 100, 255), (100, 255, 200), (255, 100, 200), (200, 200, 255)]
    for i in range(len(points) - 1):
        p1 = (int(points[i][0]), int(points[i][1]))
        p2 = (int(points[i + 1][0]), int(points[i + 1][1]))
        color = colors[i % len(colors)]
        cv2.line(canvas, p1, p2, color, 4, cv2.LINE_AA)
        cv2.circle(canvas, p2, 6, (255, 255, 255), -1)

    cv2.circle(canvas, (int(points[0][0]), int(points[0][1])), 10, (200, 200, 200), -1)

    # Label
    cv2.putText(canvas, f"{kin.robot_name.upper()} SIM (skeleton fallback)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(canvas, "PyBullet renderer unavailable on this system",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return canvas


def compose_side_by_side(
    human_frame: np.ndarray,
    robot_frame: np.ndarray,
    frame_idx: int,
    timestamp: float,
    reachable: bool,
    gripper_m: float,
    gripper_method: str,
    robot_name: str = "ROBOT",
    target_h: int = 480,
) -> np.ndarray:
    """Stack human and robot frames side by side with HUD overlay."""
    # Resize both to same height
    h1, w1 = human_frame.shape[:2]
    h2, w2 = robot_frame.shape[:2]

    scale1 = target_h / h1
    new_w1 = int(w1 * scale1)
    lf = cv2.resize(human_frame, (new_w1, target_h))

    scale2 = target_h / h2
    new_w2 = int(w2 * scale2)
    rf = cv2.resize(robot_frame, (new_w2, target_h))

    # Add column labels
    label_h = 30
    left_label = np.zeros((label_h, new_w1, 3), dtype=np.uint8)
    right_label = np.zeros((label_h, new_w2, 3), dtype=np.uint8)
    cv2.putText(left_label, "HUMAN DEMONSTRATION",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
    cv2.putText(right_label, f"{robot_name.upper()} RETARGETED",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)

    left_col = np.vstack([left_label, lf])
    right_col = np.vstack([right_label, rf])

    combined = np.hstack([left_col, right_col])

    # HUD strip at bottom
    hud_h = 40
    total_w = combined.shape[1]
    hud = np.zeros((hud_h, total_w, 3), dtype=np.uint8)
    hud[:] = (30, 30, 30)

    reach_color = (100, 255, 100) if reachable else (100, 100, 255)
    reach_str = "IK: VALID" if reachable else "IK: FALLBACK"
    cv2.putText(hud, f"Frame {frame_idx:05d}  t={timestamp:.2f}s  {reach_str}  "
                f"Gripper={gripper_m*100:.1f}mm  Method={gripper_method}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, reach_color, 1, cv2.LINE_AA)

    return np.vstack([combined, hud])


# ---------------------------------------------------------------------------
# Main validation script
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate retargeting: replay robot in PyBullet, generate side-by-side video."
    )
    parser.add_argument(
        "--annotations", required=True,
        help="Path to frame_annotations.json (EgoAnnotate pipeline output)."
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the original source video (.mp4)."
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for retargeting results and video."
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to retargeting YAML configuration file."
    )
    parser.add_argument(
        "--urdf", default=None,
        help="Path to robot URDF. Overrides config if specified."
    )
    parser.add_argument(
        "--preferred-hand", default="right",
        choices=["right", "left", "auto"],
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit processing to first N frames (for quick testing)."
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Run full retargeting pipeline
    # ------------------------------------------------------------------
    if args.config:
        config = RetargetingConfig.from_yaml(args.config)
        if args.urdf:
            config.urdf_path = args.urdf
        config.output_dir = str(output_dir)
    else:
        pose_cfg = PoseMapperConfig(preferred_hand=args.preferred_hand)
        config = RetargetingConfig(
            urdf_path=args.urdf,
            pose_mapper=pose_cfg,
            output_dir=str(output_dir),
        )

    print("\n" + "=" * 70)
    print("  RETARGETING VALIDATION")
    print("=" * 70)

    retargeter = Retargeter(config)
    result = retargeter.run_from_annotations(
        args.annotations,
        episode_id=Path(args.annotations).parent.name,
    )

    if args.max_frames:
        n_frames = min(args.max_frames, result.n_frames)
        print(f"[INFO] Limiting to first {n_frames} frames for validation.")
    else:
        n_frames = result.n_frames

    # Save retargeting JSON
    result_path = output_dir / "retargeting_result.json"
    result.save_json(str(result_path))
    print(f"[INFO] Retargeting result saved to: {result_path}")

    # ------------------------------------------------------------------
    # Step 2: Open PyBullet, load robot for rendering
    # ------------------------------------------------------------------
    import pybullet as pb
    import pybullet_data

    kin = retargeter.load_kinematics()

    render_client = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=render_client)
    render_robot = pb.loadURDF(
        kin.urdf_path,
        basePosition=[0, 0, 0],
        useFixedBase=True,
        physicsClientId=render_client,
    )

    # Test if PyBullet rendering works on this machine
    test_frame = render_robot_frame_pybullet(
        pb, render_client, render_robot, kin,
        kin.rest_poses,
        kin.gripper_joints[0].upper_limit / 2.0 if kin.gripper_joints else 0.02,
    )
    using_pybullet_renderer = test_frame is not None
    renderer_mode = "pybullet_tiny_renderer" if using_pybullet_renderer else "skeleton_2d_fallback"
    print(f"[INFO] Renderer mode: {renderer_mode}")
    if not using_pybullet_renderer:
        print("[WARN] PyBullet rendered black frames — using 2D skeleton fallback.")

    # ------------------------------------------------------------------
    # Step 3: Open source video and build side-by-side output
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read all source frames needed
    print(f"[INFO] Reading {n_frames} frames from {args.video}...")
    human_frames = []
    for fi in range(n_frames):
        ret, fr = cap.read()
        if not ret:
            break
        human_frames.append(fr)
    cap.release()

    # Set up output video
    sbs_path = output_dir / "side_by_side.mp4"
    target_h = 480

    # Probe composite frame size
    dummy_robot = (
        render_robot_frame_pybullet(pb, render_client, render_robot, kin,
                                    kin.rest_poses, 0.02)
        if using_pybullet_renderer
        else render_robot_skeleton_2d(kin.rest_poses, kin, RENDER_W, RENDER_H)
    )
    if dummy_robot is None:
        dummy_robot = render_robot_skeleton_2d(kin.rest_poses, kin, RENDER_W, RENDER_H)

    dummy_human = human_frames[0] if human_frames else np.zeros((480, 640, 3), np.uint8)
    dummy_composed = compose_side_by_side(
        dummy_human, dummy_robot, 0, 0.0, True, 0.02, "grasp_type", robot_name=kin.robot_name
    )
    out_h, out_w = dummy_composed.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(sbs_path), fourcc, fps, (out_w, out_h))

    # ------------------------------------------------------------------
    # Step 4: Render each frame and write to video
    # ------------------------------------------------------------------
    print(f"[INFO] Rendering {n_frames} frames → {sbs_path}")
    render_failures = 0

    from tqdm import tqdm
    for fi in tqdm(range(min(n_frames, len(human_frames))), desc="Rendering"):
        ik_r = result.ik_results[fi]
        gc = result.gripper_commands[fi]

        # Render robot
        gripper_m = float(gc.opening_m)
        if using_pybullet_renderer:
            robot_img = render_robot_frame_pybullet(
                pb, render_client, render_robot, kin,
                ik_r.joint_angles, gripper_m
            )
            if robot_img is None:
                render_failures += 1
                robot_img = render_robot_skeleton_2d(ik_r.joint_angles, kin)
        else:
            robot_img = render_robot_skeleton_2d(ik_r.joint_angles, kin)

        composed = compose_side_by_side(
            human_frames[fi], robot_img,
            ik_r.frame_idx, ik_r.timestamp,
            ik_r.reachable, gripper_m, gc.gripper_mapping_method,
            robot_name=kin.robot_name,
            target_h=target_h,
        )
        writer.write(composed)

    writer.release()
    pb.disconnect(render_client)

    # ------------------------------------------------------------------
    # Step 5: Print final quantitative report
    # ------------------------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  VALIDATION REPORT — {kin.robot_name.upper()}")
    print(f"{sep}")
    print(f"  Renderer used       : {renderer_mode}")
    if render_failures > 0:
        print(f"  Render fallbacks    : {render_failures} frames fell back to skeleton")
    print(f"  Side-by-side video  : {sbs_path}")
    print(f"  Retargeting JSON    : {result_path}")
    print()
    summary = result.summary
    print(f"  Frames processed    : {min(n_frames, len(human_frames))}")
    print(f"  Reachable (IK valid): {summary['n_reachable']} / {summary['n_frames']}"
          f" ({summary['pct_reachable']:.1f}%)")
    print(f"  Fallback used       : {summary['n_fallback']} frames")
    print(f"  Self-collision evts : {summary['n_self_collision']}")
    print(f"  Avg IK residual     : {summary['avg_ik_residual_mm']:.2f} mm")
    print(f"  Avg IK solve time   : {summary['avg_solve_time_ms']:.2f} ms/frame")
    print(f"  Total wall-clock    : {summary['wall_clock_seconds']:.1f} s")
    print()
    print("  Gripper method breakdown:")
    for method, cnt in summary["gripper_method_counts"].items():
        pct = 100.0 * cnt / max(summary["n_frames"], 1)
        print(f"    {method:<30} {cnt:>5} frames ({pct:.1f}%)")
    print()
    print("  MODELING CAVEATS (always reported, never hidden):")
    for caveat in summary["modeling_caveats"]:
        print(f"    • {caveat}")
    print(f"{sep}\n")

    # Describe what the video shows
    n_reach = summary["n_reachable"]
    pct_reach = summary["pct_reachable"]
    print("  SIDE-BY-SIDE VIDEO DESCRIPTION:")
    print(f"    Left panel : original human egocentric video.")
    print(f"    Right panel: {kin.robot_name.upper()} arm replaying the retargeted trajectory.")
    if using_pybullet_renderer:
        print(f"    Rendering  : PyBullet 3D (tiny renderer, DIRECT mode).")
    else:
        print(f"    Rendering  : 2D skeleton fallback (PyBullet 3D unavailable on this machine).")
    print(f"    IK status  : {pct_reach:.1f}% of frames show valid robot configurations.")
    if pct_reach < 50.0:
        print(f"    ⚠ Low reachability: many human hand poses fall outside {kin.robot_name}'s")
        print("      joint limits or workspace. This is real, expected information about")
        print("      the robot's constraints for this motion type.")
    print(f"    Gripper HUD: shows per-frame opening in mm and mapping method used.")
    print(f"    Frames with reachable=False show the PREVIOUS valid config (visible")
    print(f"    as brief pauses or holds in the robot motion).")
    print()


if __name__ == "__main__":
    # Fix relative imports when run as script
    from src.retargeting.pose_mapper import PoseMapperConfig
    main()
