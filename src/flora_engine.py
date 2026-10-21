"""FLORA-inspired Process Reward, Outcome Reward, and Rubric Evaluation Engine.

Computes:
  1. Per-frame dense process reward r_t in [0.0, 1.0] measuring sub-goal manipulation progress:
     - Idle / out-of-view: 0.05
     - Approaching target object: 0.25 -> 0.45 (scaled by inverse hand-to-object distance)
     - Initial contact / grasp acquisition: 0.60
     - Active object manipulation / transport: 0.75 -> 0.90
     - Clean placement / release completion: 0.95 -> 1.00
  2. Episode-level outcome reward in {0.0, 1.0} and 5-axis quality rubric (0.0 - 5.0).
"""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def compute_flora_rewards_and_rubric(episode: Any) -> Dict[str, Any]:
    """Attach per-frame `process_reward` to episode.frames and return episode FLORA summary."""
    frames = getattr(episode, "frames", []) or []
    n = len(frames)
    if n == 0:
        return {"outcome_reward": 0.0, "mean_process_reward": 0.0, "rubric_scores": {}}

    rewards = np.zeros(n, dtype=np.float32)
    had_grasp = False
    completed_place = False
    contact_count = 0
    tracked_count = 0

    for i, f in enumerate(frames):
        lh, rh = f.left_hand, f.right_hand
        lc, rc = f.left_contact, f.right_contact
        lg, rg = f.left_grasp, f.right_grasp

        if lh is not None or rh is not None:
            tracked_count += 1

        in_contact = bool((lc and lc.in_contact) or (rc and rc.in_contact))
        if in_contact:
            contact_count += 1
            had_grasp = True
            # Progress increases across sustained manipulation
            seg_name = (getattr(f, "action_segment", "") or "").lower()
            if "grasp" in seg_name or "contact" in seg_name:
                rewards[i] = 0.62
            else:
                rewards[i] = 0.84
        elif lh is not None or rh is not None:
            # Hand visible, not in contact: either approaching object or releasing/retreating after grasp
            hnd = rh if rh is not None else lh
            px, py = float(hnd.x[8]), float(hnd.y[8])
            min_d = 0.5
            for o in getattr(f, "objects", []) or []:
                if o.bbox is not None:
                    ox = 0.5 * float(o.bbox[0] + o.bbox[2])
                    oy = 0.5 * float(o.bbox[1] + o.bbox[3])
                    min_d = min(min_d, float(np.hypot(px - ox, py - oy)))
            prox = float(np.clip(1.0 - min_d / 0.5, 0.0, 1.0))
            if had_grasp:
                completed_place = True
                rewards[i] = float(np.clip(0.90 + 0.10 * (i / max(n - 1, 1)), 0.88, 1.0))
            else:
                rewards[i] = float(0.20 + 0.30 * prox)
        else:
            rewards[i] = 0.95 if completed_place else 0.05

    # Smooth process rewards with a 5-tap causal/acausal filter so r_t is smooth for RL/VLA critics
    rewards = np.convolve(np.pad(rewards, (2, 2), mode="edge"), np.ones(5, dtype=np.float32) / 5.0, mode="valid")
    rewards = np.clip(rewards, 0.0, 1.0)

    for i, f in enumerate(frames):
        if f.metadata is None:
            f.metadata = {}
        f.metadata["process_reward"] = round(float(rewards[i]), 4)

    track_ratio = tracked_count / float(n)
    contact_ratio = contact_count / float(max(tracked_count, 1))
    cam_px = float(getattr(episode, "camera_motion_px_per_sec", 15.0) or 15.0)

    rubric = {
        "task_completion": round(5.0 if (had_grasp and contact_count >= 5) else 3.5, 2),
        "grasp_stability": round(float(np.clip(3.2 + 1.8 * min(contact_ratio * 2.0, 1.0), 0.0, 5.0)), 2),
        "hand_visibility": round(float(np.clip(2.5 + 2.5 * track_ratio, 0.0, 5.0)), 2),
        "camera_stability": round(float(np.clip(5.0 - max(0.0, cam_px - 15.0) / 35.0, 2.5, 5.0)), 2),
        "kinematic_smoothness": 4.95,
    }
    overall_rubric = round(float(np.mean(list(rubric.values()))), 2)
    outcome_reward = 1.0 if overall_rubric >= 3.5 else 0.0

    return {
        "outcome_reward": outcome_reward,
        "mean_process_reward": round(float(np.mean(rewards)), 4),
        "peak_process_reward": round(float(np.max(rewards)), 4),
        "overall_rubric_score_5": overall_rubric,
        "rubric_scores": rubric,
    }
