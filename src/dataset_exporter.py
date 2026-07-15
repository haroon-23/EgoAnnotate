"""Dataset export to JSON and Parquet formats.

Implements DatasetExporter to serialize VLA training frames, action segments,
overall episode metadata, grasp distributions, and compile visualization videos.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from .datatypes import AnnotatedEpisode
from .visualizer import EgoVisualizer, VizConfig

logger = logging.getLogger(__name__)


@dataclass
class ExporterConfig:
    """Configuration for the DatasetExporter."""
    output_dir: str
    format: str = "json"                 # "json" or "parquet"
    include_image_bytes: bool = False
    save_viz_video: bool = True


class DatasetExporter:
    """Handles serialization and export of annotated egocentric VLA datasets."""

    def __init__(self, config: ExporterConfig):
        """Initialise the DatasetExporter."""
        self.config = config
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def export_episode(self, episode: AnnotatedEpisode) -> Path:
        """Export all annotations, summaries, and visualization for an episode.

        Returns:
            The Path to the created episode directory.
        """
        episode_dir = self.output_path / episode.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting episode %s to: %s", episode.episode_id, episode_dir)

        # 1. Export frame annotations (JSON or Parquet)
        self._export_frames(episode, episode_dir)

        # 2. Export episode metadata
        self._export_metadata(episode, episode_dir)

        # 3. Export action segments
        self._export_segments(episode, episode_dir)

        # 4. Save visualization video if requested
        if self.config.save_viz_video:
            self._create_viz_video(episode, episode_dir)

        # 5. Create summary metrics file
        self._create_summary(episode, episode_dir)

        return episode_dir

    def _export_frames(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Compile and serialize the detailed frame-by-frame annotations."""
        frame_data = []

        for frame in episode.frames:
            # Flatten Left Hand keypoints to shape (63,)
            left_kp = [0.0] * 63
            if frame.left_hand is not None:
                left_kp = frame.left_hand.to_array().flatten().tolist()

            # Flatten Right Hand keypoints to shape (63,)
            right_kp = [0.0] * 63
            if frame.right_hand is not None:
                right_kp = frame.right_hand.to_array().flatten().tolist()

            # Contact states
            left_contact_val = frame.left_contact.in_contact if frame.left_contact else False
            left_contact_obj = frame.left_contact.object_name if frame.left_contact else None

            right_contact_val = frame.right_contact.in_contact if frame.right_contact else False
            right_contact_obj = frame.right_contact.object_name if frame.right_contact else None

            # Grasp types
            left_grasp_val = frame.left_grasp.type if frame.left_grasp else None
            right_grasp_val = frame.right_grasp.type if frame.right_grasp else None

            # Actions
            action_wrist = [0.0, 0.0, 0.0]
            action_angles = [0.0] * 15
            action_gripper = 1.0

            if frame.action is not None:
                action_wrist = frame.action.wrist_delta.tolist()
                action_angles = frame.action.finger_angles.tolist()
                action_gripper = float(frame.action.gripper_openness)

            # Language instruction (frame_description if filled, fallback to task_description)
            lang_inst = frame.frame_description if frame.frame_description else episode.task_description

            row = {
                "frame_idx": frame.frame_idx,
                "timestamp": frame.timestamp,
                "image_path": frame.image_path,
                "left_hand_present": frame.left_hand is not None,
                "right_hand_present": frame.right_hand is not None,
                "left_hand_keypoints": left_kp,
                "right_hand_keypoints": right_kp,
                "left_contact": left_contact_val,
                "left_contact_object": left_contact_obj,
                "right_contact": right_contact_val,
                "right_contact_object": right_contact_obj,
                "left_grasp_type": left_grasp_val,
                "right_grasp_type": right_grasp_val,
                "action_wrist_delta": action_wrist,
                "action_finger_angles": action_angles,
                "action_gripper_openness": action_gripper,
                "language_instruction": lang_inst,
                "action_segment": frame.action_segment,
            }
            frame_data.append(row)

        # Export in chosen format
        if self.config.format.lower() == "parquet":
            try:
                import pandas as pd
                df = pd.DataFrame(frame_data)
                df.to_parquet(episode_dir / "frame_annotations.parquet", index=False)
                logger.info("Exported frame annotations to Parquet.")
                return
            except ImportError:
                logger.warning("pandas or pyarrow not installed. Falling back to JSON frames export.")

        # Default JSON export
        with open(episode_dir / "frame_annotations.json", "w") as f:
            json.dump(frame_data, f, indent=2)
        logger.info("Exported frame annotations to JSON.")

    def _export_metadata(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Export high-level metadata info for the episode."""
        meta = {
            "episode_id": episode.episode_id,
            "video_path": episode.video_path,
            "task_description": episode.task_description,
            "num_frames": episode.num_frames,
            "duration_seconds": episode.duration_seconds,
            "target_robot": episode.target_robot,
        }
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _export_segments(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Export temporal action segments list."""
        segments_data = [s.to_dict() for s in episode.segments]
        with open(episode_dir / "action_segments.json", "w") as f:
            json.dump(segments_data, f, indent=2)

    def _create_viz_video(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Render and save a visualization HUD video in the output directory."""
        try:
            visualizer = EgoVisualizer(VizConfig())
            visualizer.render_episode(episode, episode_dir / "visualization.mp4")
        except Exception as e:
            logger.error("Failed to render visualization video during export: %s", e)

    def _create_summary(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Compile and serialize general statistics summary for the episode."""
        left_grasps: Dict[str, int] = {}
        right_grasps: Dict[str, int] = {}
        left_contact_count = 0
        right_contact_count = 0

        for frame in episode.frames:
            if frame.left_grasp:
                left_grasps[frame.left_grasp.type] = left_grasps.get(frame.left_grasp.type, 0) + 1
            if frame.right_grasp:
                right_grasps[frame.right_grasp.type] = right_grasps.get(frame.right_grasp.type, 0) + 1

            if frame.left_contact and frame.left_contact.in_contact:
                left_contact_count += 1
            if frame.right_contact and frame.right_contact.in_contact:
                right_contact_count += 1

        total_frames = len(episode.frames)
        summary = {
            "left_grasp_distribution": left_grasps,
            "right_grasp_distribution": right_grasps,
            "left_contact_frames": left_contact_count,
            "right_contact_frames": right_contact_count,
            "total_frames": total_frames,
            "ratio_left_contact": left_contact_count / total_frames if total_frames > 0 else 0.0,
            "ratio_right_contact": right_contact_count / total_frames if total_frames > 0 else 0.0,
        }

        with open(episode_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
