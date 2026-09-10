"""Test script to verify path length calculation fix and aspect-ratio scale factor analysis."""
import json
import numpy as np
from src.retargeting.pose_mapper import PoseMapper, PoseMapperConfig
from src.retargeting.retargeter import Retargeter
from src.datatypes import AnnotationFrame

def analyze_episode(name: str, path: str):
    with open(path, "r") as f:
        raw_frames = json.load(f)
    
    retargeter = Retargeter()
    frames = [retargeter._parse_flat_frame(fr) if "left_hand_keypoints" in fr else AnnotationFrame.from_dict(fr) for fr in raw_frames]
    
    pm = PoseMapper(PoseMapperConfig(preferred_hand="right"))
    
    raw_wrists = []
    hands = []
    for f in frames:
        hand, side = pm._select_hand(f)
        hands.append(hand)
        if hand is not None:
            raw_wrists.append(np.array([hand.x[0], hand.y[0], hand.z[0]], dtype=np.float64))
        else:
            raw_wrists.append(None)
            
    # Calculate old path length
    valid_wrists = [w for w in raw_wrists if w is not None]
    wrist_arr = np.stack(valid_wrists, axis=0)
    
    # Hand reference length in normalized space
    valid_hand_lengths = []
    for hand in hands:
        if hand is not None and len(hand.x) >= 10:
            p_wrist = np.array([hand.x[0], hand.y[0], hand.z[0]], dtype=np.float64)
            p_mcp = np.array([hand.x[9], hand.y[9], hand.z[9]], dtype=np.float64)
            dist = float(np.linalg.norm(p_mcp - p_wrist))
            if dist > 1e-4:
                valid_hand_lengths.append(dist)
    
    mean_hand_units = float(np.mean(valid_hand_lengths))
    scale_factor = 0.090 / mean_hand_units
    
    old_diffs = np.diff(wrist_arr, axis=0)
    old_path_length_m = float(np.sum(np.linalg.norm(old_diffs, axis=1) * scale_factor))
    
    # Calculate NEW gap-disciplined path length
    new_path_length_m = 0.0
    step_count = 0
    jump_count = 0
    jump_distance_total = 0.0
    
    for i in range(len(raw_wrists) - 1):
        w1 = raw_wrists[i]
        w2 = raw_wrists[i + 1]
        h1 = hands[i]
        h2 = hands[i + 1]
        
        if w1 is not None and w2 is not None and h1 is not None and h2 is not None:
            if not h1.is_interpolated and not h2.is_interpolated:
                step_m = float(np.linalg.norm(w2 - w1)) * scale_factor
                if step_m < 0.15:  # 0.15m per 1/30s frame = 4.5 m/s max hand speed
                    new_path_length_m += step_m
                    step_count += 1
                else:
                    jump_count += 1
                    jump_distance_total += step_m

    fps = 30.0
    duration_sec = len(frames) / fps
    active_motion_sec = step_count / fps
    avg_speed_m_s = new_path_length_m / max(active_motion_sec, 1e-3)
    
    print(f"=== {name} ===")
    print(f"Total Frames           : {len(frames)} ({duration_sec:.1f}s)")
    print(f"Valid Consecutive Steps: {step_count} ({active_motion_sec:.1f}s active motion)")
    print(f"Omitted Gap Jumps      : {jump_count} jumps totaling {jump_distance_total:.2f}m")
    print(f"Mean Hand Length (units): {mean_hand_units:.4f}")
    print(f"Scale Factor (m/unit)  : {scale_factor:.4f}")
    print(f"OLD Path Length        : {old_path_length_m:.3f} m")
    print(f"NEW Corrected Path Len : {new_path_length_m:.3f} m")
    print(f"Average Hand Speed     : {avg_speed_m_s:.3f} m/s")
    print()

analyze_episode("test_10s.mp4", "data/output/test_10s/frame_annotations.json")
analyze_episode("new_video.mp4", "data/output/new_video/frame_annotations.json")
