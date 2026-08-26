#!/usr/bin/env python3
"""Evaluation harness for EgoAnnotate pipeline.

Measures contact detection, grasp classification, and temporal segmentation accuracy
against hand-labeled ground truth CSV files.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline import EgoAnnotatePipeline
from src.datatypes import (
    AnnotatedEpisode,
    HandLandmarks,
    ContactState,
    GraspType,
    RobotAgnosticAction,
    AnnotationFrame,
    ActionSegment
)


@dataclass
class FrameEvalResult:
    """Per-frame evaluation results."""
    frame_idx: int
    timestamp_sec: float
    object_name: str
    hand: str
    contact_true: int
    contact_pred: Optional[int]
    grasp_true: str
    grasp_pred: Optional[str]
    contact_correct: Optional[bool]
    grasp_correct: Optional[bool]


@dataclass
class SegmentEvalResult:
    """Per-segment evaluation results."""
    segment_name: str
    pred_start: float
    pred_end: float
    gt_start: Optional[float]
    gt_end: Optional[float]
    start_error_sec: Optional[float]
    end_error_sec: Optional[float]


@dataclass
class EvalReport:
    """Complete evaluation report."""
    video_id: str
    video_path: str
    timestamp: str
    num_frames_evaluated: int
    contact_accuracy: float
    contact_precision: float
    contact_recall: float
    grasp_accuracy: float
    frame_results: list[dict]
    segment_results: list[dict]
    notes: str


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    """Load ground truth CSV and validate required columns."""
    required_cols = [
        "video_id", "frame_idx", "timestamp_sec", "object_name",
        "hand", "contact_true", "grasp_type_true", "notes"
    ]
    df = pd.read_csv(csv_path)
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return df


def find_gt_csv(video_path: Path, eval_dir: Path) -> Optional[Path]:
    """Find ground truth CSV for a video."""
    video_id = video_path.stem
    gt_path = eval_dir / f"ground_truth_{video_id}.csv"
    if gt_path.exists():
        return gt_path
    return None


def get_frame_contact_grasp(episode: AnnotatedEpisode, frame_idx: int, hand: str, object_name: str) -> tuple[Optional[int], Optional[str]]:
    """Extract contact and grasp predictions for a specific frame, hand, and object."""
    if frame_idx >= len(episode.frames):
        return None, None
    
    frame = episode.frames[frame_idx]
    
    # Get hand data
    if hand == "left":
        contact_state = frame.left_contact
        grasp_type = frame.left_grasp
    else:
        contact_state = frame.right_contact
        grasp_type = frame.right_grasp
    
    # Contact: check if this specific object is in contact
    contact_pred = None
    if contact_state and contact_state.in_contact:
        # Object name match (allow partial match for flexibility)
        if contact_state.object_name and object_name.lower() in contact_state.object_name.lower():
            contact_pred = 1
        elif contact_state.object_name is None:
            # No specific object name from pipeline, assume contact with target object
            contact_pred = 1
        else:
            contact_pred = 0
    elif contact_state:
        contact_pred = 0
    
    # Grasp type
    grasp_pred = None
    if grasp_type:
        grasp_pred = grasp_type.type
    
    return contact_pred, grasp_pred


def evaluate_segments(episode: AnnotatedEpisode, gt_df: pd.DataFrame) -> list[SegmentEvalResult]:
    """Evaluate temporal segmentation boundaries."""
    results = []
    
    # Get predicted segments from pipeline
    pred_segments = episode.segments
    
    # Get ground truth segments from CSV (unique segments by name + time ranges)
    # We'll derive GT segments from frame-level annotations
    gt_segments = []
    if not gt_df.empty:
        # Group by action type (using object + hand transitions as segment boundaries)
        # For now, use unique (object_name, hand) combinations with time ranges
        for (obj, hand), group in gt_df.groupby(["object_name", "hand"]):
            gt_start = group["timestamp_sec"].min()
            gt_end = group["timestamp_sec"].max()
            gt_segments.append({
                "name": obj,  # Use object as segment label
                "start": gt_start,
                "end": gt_end
            })
    
    # Match predicted to ground truth segments
    for pred_seg in pred_segments:
        pred_start = pred_seg.start_time
        pred_end = pred_seg.end_time
        pred_name = pred_seg.name
        
        # Find nearest GT segment
        best_gt = None
        best_start_err = float('inf')
        best_end_err = float('inf')
        
        for gt_seg in gt_segments:
            start_err = abs(pred_start - gt_seg["start"])
            end_err = abs(pred_end - gt_seg["end"])
            total_err = start_err + end_err
            
            if total_err < best_start_err + best_end_err:
                best_gt = gt_seg
                best_start_err = start_err
                best_end_err = end_err
        
        if best_gt:
            results.append(SegmentEvalResult(
                segment_name=pred_name,
                pred_start=pred_start,
                pred_end=pred_end,
                gt_start=best_gt["start"],
                gt_end=best_gt["end"],
                start_error_sec=best_start_err,
                end_error_sec=best_end_err
            ))
        else:
            results.append(SegmentEvalResult(
                segment_name=pred_name,
                pred_start=pred_start,
                pred_end=pred_end,
                gt_start=None,
                gt_end=None,
                start_error_sec=None,
                end_error_sec=None
            ))
    
    return results


def load_existing_episode(video_id: str, output_dir: Path) -> Optional[AnnotatedEpisode]:
    """Load existing AnnotatedEpisode from pipeline output directory."""
    episode_dir = output_dir / video_id
    frame_annot_path = episode_dir / "frame_annotations.json"
    metadata_path = episode_dir / "metadata.json"
    segments_path = episode_dir / "action_segments.json"
    
    if not all(p.exists() for p in [frame_annot_path, metadata_path, segments_path]):
        return None
    
    with open(frame_annot_path, "r") as f:
        frames_data = json.load(f)
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    with open(segments_path, "r") as f:
        segments_data = json.load(f)
    
    # Reconstruct AnnotatedEpisode from saved data
    frames = []
    for fd in frames_data:
        # Parse hand landmarks
        left_hand = None
        if fd.get("left_hand_present"):
            left_hand = HandLandmarks(
                x=np.array(fd["left_hand_keypoints"][::3], dtype=np.float32),
                y=np.array(fd["left_hand_keypoints"][1::3], dtype=np.float32),
                z=np.array(fd["left_hand_keypoints"][2::3], dtype=np.float32),
                confidence=1.0,  # Not saved in export
                handedness="Left"
            )
        right_hand = None
        if fd.get("right_hand_present"):
            right_hand = HandLandmarks(
                x=np.array(fd["right_hand_keypoints"][::3], dtype=np.float32),
                y=np.array(fd["right_hand_keypoints"][1::3], dtype=np.float32),
                z=np.array(fd["right_hand_keypoints"][2::3], dtype=np.float32),
                confidence=1.0,
                handedness="Right"
            )
        
        # Parse contact states
        left_contact = None
        if fd.get("left_contact") is not None:
            left_contact = ContactState(
                fingers=np.array([fd["left_contact"]] * 5, dtype=bool),  # Simplified
                object_name=fd.get("left_contact_object"),
                in_contact=fd.get("left_contact", False)
            )
        right_contact = None
        if fd.get("right_contact") is not None:
            right_contact = ContactState(
                fingers=np.array([fd["right_contact"]] * 5, dtype=bool),
                object_name=fd.get("right_contact_object"),
                in_contact=fd.get("right_contact", False)
            )
        
        # Parse grasp types
        left_grasp = None
        if fd.get("left_grasp_type"):
            left_grasp = GraspType(
                type=fd["left_grasp_type"],
                confidence=0.7,
                thumb_index_distance=0.05,
                num_curled_fingers=3
            )
        right_grasp = None
        if fd.get("right_grasp_type"):
            right_grasp = GraspType(
                type=fd["right_grasp_type"],
                confidence=0.7,
                thumb_index_distance=0.05,
                num_curled_fingers=3
            )
        
        # Parse action
        action = None
        if fd.get("action_wrist_delta"):
            action = RobotAgnosticAction(
                wrist_delta=np.array(fd["action_wrist_delta"], dtype=np.float32),
                finger_angles=np.array(fd.get("action_finger_angles", [0.0]*15), dtype=np.float32),
                gripper_openness=fd.get("action_gripper_openness", 1.0),
                hand_orientation=None
            )
        
        frame = AnnotationFrame(
            frame_idx=fd["frame_idx"],
            timestamp=fd["timestamp"],
            image_path=fd["image_path"],
            left_hand=left_hand,
            right_hand=right_hand,
            objects=[],  # Not needed for eval
            left_contact=left_contact,
            right_contact=right_contact,
            left_grasp=left_grasp,
            right_grasp=right_grasp,
            action=action,
            frame_description=fd.get("language_instruction", ""),
            action_segment=fd.get("action_segment", "idle")
        )
        frames.append(frame)
    
    segments = [ActionSegment.from_dict(s) for s in segments_data]
    
    return AnnotatedEpisode(
        episode_id=meta["episode_id"],
        video_path=meta["video_path"],
        task_description=meta["task_description"],
        frames=frames,
        segments=segments,
        num_frames=meta["num_frames"],
        duration_seconds=meta["duration_seconds"],
        target_robot=meta.get("target_robot", "humanoid_generic")
    )


def evaluate_video(video_path: Path, config_path: Path, eval_dir: Path, output_dir: Path, use_existing: bool = True) -> EvalReport:
    """Run full evaluation on a single video."""
    video_id = video_path.stem
    
    # Find and load ground truth
    gt_csv_path = find_gt_csv(video_path, eval_dir)
    if gt_csv_path is None:
        warnings.warn(f"No ground truth CSV found for {video_id} at {eval_dir / f'ground_truth_{video_id}.csv'}")
        gt_df = pd.DataFrame()
    else:
        gt_df = load_ground_truth(gt_csv_path)
        gt_df = gt_df[gt_df["video_id"] == video_id]
    
    # Try to load existing episode first
    episode = None
    if use_existing:
        episode = load_existing_episode(video_id, output_dir)
        if episode:
            print(f"Loaded existing episode from {output_dir / video_id}")
    
    # Fall back to running pipeline if no existing episode
    if episode is None:
        print(f"Running EgoAnnotate pipeline on {video_path}...")
        pipeline = EgoAnnotatePipeline(config_path=str(config_path))
        episode = pipeline.process_video(str(video_path), episode_id=video_id)
    
    # Evaluate each frame with ground truth
    frame_results = []
    contact_correct = 0
    contact_total = 0
    contact_tp = 0
    contact_fp = 0
    contact_fn = 0
    grasp_correct = 0
    grasp_total = 0
    
    for _, gt_row in gt_df.iterrows():
        frame_idx = int(gt_row["frame_idx"])
        timestamp = float(gt_row["timestamp_sec"])
        object_name = str(gt_row["object_name"])
        hand = str(gt_row["hand"])
        contact_true = int(gt_row["contact_true"])
        grasp_true = str(gt_row["grasp_type_true"])
        
        contact_pred, grasp_pred = get_frame_contact_grasp(episode, frame_idx, hand, object_name)
        
        # Contact evaluation
        contact_correct_frame = None
        if contact_pred is not None:
            contact_total += 1
            contact_correct_frame = (contact_pred == contact_true)
            if contact_correct_frame:
                contact_correct += 1
            
            # Precision/Recall
            if contact_pred == 1 and contact_true == 1:
                contact_tp += 1
            elif contact_pred == 1 and contact_true == 0:
                contact_fp += 1
            elif contact_pred == 0 and contact_true == 1:
                contact_fn += 1
        
        # Grasp evaluation
        grasp_correct_frame = None
        if grasp_pred is not None:
            grasp_total += 1
            grasp_correct_frame = (grasp_pred == grasp_true)
            if grasp_correct_frame:
                grasp_correct += 1
        
        frame_results.append(FrameEvalResult(
            frame_idx=frame_idx,
            timestamp_sec=timestamp,
            object_name=object_name,
            hand=hand,
            contact_true=contact_true,
            contact_pred=contact_pred,
            grasp_true=grasp_true,
            grasp_pred=grasp_pred,
            contact_correct=contact_correct_frame,
            grasp_correct=grasp_correct_frame
        ))
    
    # Segment evaluation
    segment_results = evaluate_segments(episode, gt_df)
    
    # Compute metrics
    contact_accuracy = contact_correct / contact_total if contact_total > 0 else 0.0
    contact_precision = contact_tp / (contact_tp + contact_fp) if (contact_tp + contact_fp) > 0 else 0.0
    contact_recall = contact_tp / (contact_tp + contact_fn) if (contact_tp + contact_fn) > 0 else 0.0
    grasp_accuracy = grasp_correct / grasp_total if grasp_total > 0 else 0.0
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY: {video_id}")
    print("=" * 70)
    print(f"Frames evaluated:     {len(frame_results)}")
    print(f"Contact Accuracy:     {contact_accuracy:.2%} ({contact_correct}/{contact_total})")
    print(f"Contact Precision:    {contact_precision:.2%} (TP={contact_tp}, FP={contact_fp})")
    print(f"Contact Recall:       {contact_recall:.2%} (TP={contact_tp}, FN={contact_fn})")
    print(f"Grasp Accuracy:       {grasp_accuracy:.2%} ({grasp_correct}/{grasp_total})")
    print("-" * 70)
    
    if segment_results:
        print("Segment Boundary Errors:")
        for seg in segment_results:
            if seg.start_error_sec is not None:
                print(f"  {seg.segment_name}: start_err={seg.start_error_sec:.2f}s, end_err={seg.end_error_sec:.2f}s")
            else:
                print(f"  {seg.segment_name}: no ground truth match")
    print("=" * 70 + "\n")
    
    return EvalReport(
        video_id=video_id,
        video_path=str(video_path),
        timestamp=datetime.now().isoformat(),
        num_frames_evaluated=len(frame_results),
        contact_accuracy=contact_accuracy,
        contact_precision=contact_precision,
        contact_recall=contact_recall,
        grasp_accuracy=grasp_accuracy,
        frame_results=[asdict(fr) for fr in frame_results],
        segment_results=[asdict(sr) for sr in segment_results],
        notes=f"Config: {config_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EgoAnnotate pipeline against ground truth annotations."
    )
    parser.add_argument("video_path", type=str, help="Path to video file to evaluate")
    parser.add_argument(
        "--config", "-c",
        default="configs/default.yaml",
        help="Path to pipeline config YAML (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--eval-dir", "-e",
        default="tests/eval",
        help="Directory containing ground_truth_*.csv files (default: tests/eval)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="eval_results",
        help="Directory to save JSON reports (default: eval_results)"
    )
    parser.add_argument(
        "--use-existing", "-u",
        action="store_true",
        help="Use existing pipeline outputs instead of re-running (default: True)"
    )
    parser.add_argument(
        "--no-existing",
        action="store_true",
        help="Force re-running the pipeline even if outputs exist"
    )
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    config_path = Path(args.config)
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_existing = not args.no_existing
    report = evaluate_video(video_path, config_path, eval_dir, output_dir, use_existing=use_existing)
    
    # Save JSON report
    output_path = output_dir / f"{report.video_id}_eval.json"
    with open(output_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    
    print(f"Detailed report saved to: {output_path}")


if __name__ == "__main__":
    main()