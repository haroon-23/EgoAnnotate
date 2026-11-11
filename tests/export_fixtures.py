"""Synthetic episodes with real PNG frames for the export/validation tests."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from src.datatypes import AnnotatedEpisode, AnnotationFrame, MetricHandPose

NQ = 7
JOINT_NAMES = [f"panda_joint{i + 1}" for i in range(NQ)]
JOINT_LOWER = [-2.8, -1.7, -2.8, -3.0, -2.8, -0.0, -2.8]
JOINT_UPPER = [2.8, 1.7, 2.8, -0.07, 2.8, 3.7, 2.8]


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _hand(t: int, side: float) -> MetricHandPose:
    R = _rot_z(0.1 * t + side)
    wrist = np.array([0.05 * side + 0.01 * t, 0.02, 0.4 + 0.005 * t])
    local = np.random.default_rng(3).normal(scale=0.04, size=(21, 3))
    local[0] = 0.0
    local[4] = [0.06, 0.04, 0.0]
    local[8] = [0.09 + 0.002 * t, -0.02, 0.0]
    kp = wrist + local @ R.T
    return MetricHandPose(wrist, R, kp, 0.8, "calibrated", False)


def make_episode(
    root: Path,
    episode_id: str = "ep0",
    *,
    n_frames: int = 12,
    size: tuple = (64, 48),
    fps: float = 10.0,
    task: str = "pick up the red cup",
    world: bool = True,
    robot: bool = True,
    unreachable: Sequence[int] = (),
    gate_passed: Optional[bool] = True,
    left_hand_from: int = 4,
    joint_scale: float = 0.2,
    save: bool = True,
) -> AnnotatedEpisode:
    """Episode with PNG frames under ``root/<episode_id>/frames`` and annotations.json."""
    ep_dir = Path(root) / episode_id
    (ep_dir / "frames").mkdir(parents=True, exist_ok=True)
    w, h = size
    frames = []
    for t in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        img[..., 1] = (t * 17) % 256
        img[..., 2] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
        path = ep_dir / "frames" / f"frame_{t:06d}.png"
        cv2.imwrite(str(path), img)
        T = np.eye(4)
        T[:3, :3] = _rot_z(0.3)
        T[:3, 3] = [0.1, 0.0, 1.0 + 0.001 * t]
        q = joint_scale * np.sin(0.3 * t + np.arange(NQ)) - 0.3
        fr = AnnotationFrame(
            frame_idx=3 * t, timestamp=3 * t / 30.0, image_path=str(path),
            right_hand_metric=_hand(t, 1.0),
            left_hand_metric=_hand(t, -1.0) if t >= left_hand_from else None,
            T_world_cam=T if world else None,
            frame_description="reach and grasp",
        )
        if robot:
            fr.robot_joint_angles = [float(x) for x in q]
            fr.robot_gripper_opening_m = float(0.08 - 0.005 * t)
            fr.robot_reachable = t not in set(unreachable)
            fr.robot_ee_pos_error_m = 0.002
        frames.append(fr)
    retargeting = {}
    if robot:
        retargeting = {
            "report": {"tracked_fraction": 1.0,
                       "gate": {"passed": gate_passed,
                                "reasons": [] if gate_passed else ["valid_fraction 0.2 < 0.5"]}},
            "joint_limits": {"names": JOINT_NAMES, "lower": JOINT_LOWER, "upper": JOINT_UPPER},
        }
    ep = AnnotatedEpisode(
        episode_id=episode_id, video_path=str(ep_dir / "src.mp4"), task_description=task,
        frames=frames, segments=[], num_frames=n_frames,
        duration_seconds=n_frames / fps, target_robot="franka_panda",
        effective_fps=fps, geometry={"world_source": "aruco", "tier": "approximate"},
        robot_joint_names=JOINT_NAMES if robot else [], retargeting=retargeting,
    )
    if save:
        ep.save_json(ep_dir / "annotations.json")
    return ep
