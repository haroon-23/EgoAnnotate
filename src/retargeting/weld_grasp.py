"""Weld-grasp semantics for physics verification (Phase E).

"What weld grasp means" (concrete, given the current representation): today a
grasp is only a 1-DOF gripper opening (``GripperCommand.opening_m``); there is
no object-attachment model, which is why ``scripts/physics_replay.py`` relies
on *friction* to lift (and measured 0.017 m < 0.02 m — friction luck, not a
grasp). A **weld grasp** = during a validated grasp window, the object is
rigidly attached to the end-effector (a MuJoCo weld equality constraint) for
the replay. Verification then tests *trajectory dynamic feasibility +
attachment stability* rather than friction luck.

This module is pure numpy — no heavy dependencies. The MuJoCo weld itself is
implemented by :mod:`src.retargeting.episode_verifier`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WeldGraspConfig:
    """Configuration for weld-grasp window detection.

    Attributes:
        close_threshold_m: Gripper opening (per-finger, metres) below which the
            gripper counts as closed. Matches ``physics_replay.py`` (0.03).
        min_window_frames: Minimum contiguous frames for a grasp window.
            Matches ``physics_replay.py`` (6).
        grasp_proximity_m: End-effector to object distance (metres) required to
            qualify a frame as a grasp candidate. Matches "near" (0.09).
        slip_tolerance_m: During a weld trial, the weld counts as held while
            the object stays within this distance of its welded pose.
    """

    close_threshold_m: float = 0.03
    min_window_frames: int = 6
    grasp_proximity_m: float = 0.09
    slip_tolerance_m: float = 0.005


@dataclass
class GraspWindow:
    """One validated grasp window.

    Attributes:
        start_idx: First frame index of the window (inclusive).
        end_idx: Frame index one past the window (exclusive).
        grasp_frame_idx: Frame where the weld is declared — the first
            closed-and-near frame of the window.
        object_spawn_pos: (3,) world position where the object is spawned for
            the weld trial — the end-effector position at ``grasp_frame_idx``.
        peak_lift_m: Maximum object lift measured during the trial (filled by
            the verifier).
        weld_held: Whether the weld held for the whole trial (filled by the
            verifier).
    """

    start_idx: int
    end_idx: int
    grasp_frame_idx: int
    object_spawn_pos: np.ndarray
    peak_lift_m: float = 0.0
    weld_held: bool = False

    def to_dict(self) -> dict:
        return {
            "start_idx": int(self.start_idx),
            "end_idx": int(self.end_idx),
            "grasp_frame_idx": int(self.grasp_frame_idx),
            "object_spawn_pos": [round(float(v), 4) for v in np.asarray(self.object_spawn_pos).ravel()],
            "peak_lift_m": round(float(self.peak_lift_m), 4),
            "weld_held": bool(self.weld_held),
        }


def find_grasp_windows(
    reachability: np.ndarray,
    gripper_opening_m: np.ndarray,
    ee_positions: np.ndarray,
    object_position: np.ndarray,
    config: Optional[WeldGraspConfig] = None,
) -> List[GraspWindow]:
    """Find validated grasp windows. Pure numpy; never raises on bad input.

    A frame qualifies when it is kinematically reachable AND the gripper is
    closed (``opening_m < close_threshold_m``) AND the end-effector is near the
    object (``||ee - object|| < grasp_proximity_m``). A window is a maximal
    run of qualifying frames with length >= ``min_window_frames``.

    Args:
        reachability: (N,) bool — per-frame kinematic reachability.
        gripper_opening_m: (N,) or (N,1) float — per-finger opening in metres.
        ee_positions: (N,3) float — end-effector world positions in metres.
        object_position: (3,) float — object world position in metres.
        config: WeldGraspConfig (defaults if None).

    Returns:
        List of GraspWindow, in frame order. Empty list when nothing qualifies
        (including empty input) — never raises.
    """
    cfg = config or WeldGraspConfig()
    try:
        reach = np.asarray(reachability, dtype=bool).ravel()
        grip = np.asarray(gripper_opening_m, dtype=float).ravel()
        ee = np.asarray(ee_positions, dtype=float).reshape(-1, 3)
        obj = np.asarray(object_position, dtype=float).ravel()
    except Exception as exc:
        logger.warning("find_grasp_windows: could not parse inputs (%s); returning []", exc)
        return []
    n = len(reach)
    if n == 0 or len(grip) != n or ee.shape[0] != n or obj.shape != (3,):
        if n != 0:
            logger.warning(
                "find_grasp_windows: shape mismatch (reach=%d grip=%d ee=%s obj=%s); returning []",
                n, len(grip), ee.shape, obj.shape,
            )
        return []

    closed = grip < cfg.close_threshold_m
    dist = np.linalg.norm(ee - obj[None, :], axis=1)
    near = dist < cfg.grasp_proximity_m
    ok = reach & closed & near

    windows: List[GraspWindow] = []
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i >= cfg.min_window_frames:
            grasp_frame = i  # first closed-and-near frame: weld declared here
            windows.append(
                GraspWindow(
                    start_idx=int(i),
                    end_idx=int(j),
                    grasp_frame_idx=int(grasp_frame),
                    object_spawn_pos=np.array(ee[grasp_frame], dtype=float).copy(),
                )
            )
        i = j
    return windows
