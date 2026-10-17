#!/usr/bin/env python3
"""Simulate feeding an EgoAnnotate episode into 4 real robot controllers at 30Hz.

Computes IK residuals (mean/p95), joint velocity/acceleration/jerk compliance,
self-collision & teleportation counts, 16-step VLA action chunks, and a 0-100
Hardware Readiness Score per platform.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.datatypes import AnnotatedEpisode
from src.retargeting.embodiment_registry import Adapter
from src.retargeting.pose_mapper import PoseMapper, PoseMapperConfig


def run_hardware_audit(annotations_path: str = "data/output/whatsapp_video/annotations.json") -> dict:
    ep = AnnotatedEpisode.load_json(annotations_path)
    frames = ep.frames

    mapper_r = PoseMapper(PoseMapperConfig(preferred_hand="right", lock_driving_hand=True))
    mapper_l = PoseMapper(PoseMapperConfig(preferred_hand="left", lock_driving_hand=True))
    t_right = mapper_r.map_frames_for_hand(frames, "right")
    t_left = mapper_l.map_frames_for_hand(frames, "left")

    r_seq = [(t.position, t.quaternion) for t in t_right]
    l_seq = [(t.position, t.quaternion) for t in t_left]
    valid_r = [t.hand_detected for t in t_right]
    valid_l = [t.hand_detected for t in t_left]
    hands_seq = [f.right_hand for f in frames]
    grips_seq = [0.85 if (f.right_grasp and f.right_grasp.type != "open") else 0.10 for f in frames]

    configs = [
        "configs/embodiments/franka_arm.yaml",
        "configs/embodiments/unitree_g1_upper.yaml",
        "configs/embodiments/unitree_h1_upper.yaml",
        "configs/embodiments/allegro_hand.yaml",
    ]

    results = {}
    for cfg in configs:
        ad = Adapter(cfg)
        rep = ad.solve_episode(
            r_seq,
            left_targets=l_seq,
            grips=grips_seq,
            hand_landmarks_seq=hands_seq,
            valid_mask=valid_r,
            left_valid_mask=valid_l,
        )
        results[rep.name] = {
            "summary": rep.to_dict(),
            "hardware_audit": rep.hardware_audit.to_dict(),
            "action_chunk_shape": list(rep.action_chunks_16.shape),
        }

    out_path = Path(annotations_path).parent / "hardware_readiness_report.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    ep_arg = sys.argv[1] if len(sys.argv) > 1 else "data/output/whatsapp_video/annotations.json"
    res = run_hardware_audit(ep_arg)
    for k, v in res.items():
        print(json.dumps(v["hardware_audit"]))
