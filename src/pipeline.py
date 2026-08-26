"""Pipeline orchestration — runs all annotation stages.

Main EgoAnnotatePipeline orchestrator that coordinates loading YAML config, sampling videos,
tracking hands, detecting objects, classifying contact/grasp states, computing actions,
segmenting temporal actions, generating descriptions, and exporting the dataset.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2
import yaml

from .action_computer import ActionComputer, ActionComputerConfig
from .contact_detector import ContactDetector, ContactDetectorConfig
from .dataset_exporter import DatasetExporter, ExporterConfig
from .datatypes import AnnotatedEpisode, AnnotationFrame
from .grasp_classifier import GraspClassifier, GraspClassifierConfig
from .grounding_detector import GroundingDINOConfig
from .hand_tracker import HandTracker, HandTrackerConfig
from .language_generator import GeminiLanguageGenerator, LanguageGeneratorConfig
from .object_detector import GeminiObjectDetector, ObjectDetectorConfig
from .segment_labeler import SegmentLabeler, SegmentLabelerConfig, create_segment_labeler
from .signal_segmenter import SignalSegmenter, SignalSegmenterConfig
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class EgoAnnotatePipeline:
    """Orchestrates the entire egocentric video annotation pipeline end-to-end."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        """Load YAML configuration and initialize all annotation stages in order."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        logger.info("Initializing EgoAnnotatePipeline from config: %s", config_path)

        # 1. VideoProcessor
        # Support either "pipeline" key (new) or "video" key (old)
        pipe_cfg = self.config.get("pipeline") or {}
        video_cfg = self.config.get("video") or {}
        target_fps = float(pipe_cfg.get("target_fps", video_cfg.get("sample_fps", 2.0)))
        image_size_list = pipe_cfg.get("image_size", [224, 224])
        image_size = (int(image_size_list[0]), int(image_size_list[1]))
        self.video_processor = VideoProcessor(target_fps=target_fps, image_size=image_size)

        # 2. HandTracker
        # Support "hand_tracking" (new) or "hand_tracker" (old)
        ht_cfg = self.config.get("hand_tracking") or self.config.get("hand_tracker") or {}
        ht_config = HandTrackerConfig(
            model_path=ht_cfg.get("model_path", "models/hand_landmarker.task"),
            running_mode=ht_cfg.get("running_mode", "VIDEO"),
            num_hands=int(ht_cfg.get("num_hands", 2)),
            min_detection_confidence=float(ht_cfg.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(ht_cfg.get("min_tracking_confidence", 0.5)),
            smoothing_window=int(ht_cfg.get("smoothing_window", 5)),
        )
        self.hand_tracker = HandTracker(ht_config)

        # 3. GeminiObjectDetector
        # Support "object_detection" (new) or "object_detector" (old)
        od_cfg = self.config.get("object_detection") or self.config.get("object_detector") or {}
        gemini_cfg = self.config.get("gemini") or {}
        # Grounding DINO config
        gd_cfg = self.config.get("grounding_dino") or {}
        od_prompt = od_cfg.get(
            "prompt",
            "Identify the single primary object being interacted with or manipulated by the hand in this image. Respond in JSON."
        )
        od_config = ObjectDetectorConfig(
            keyframes_per_video=int(od_cfg.get("keyframes_per_video", 3)),
            bbox_keyframe_interval=int(od_cfg.get("bbox_keyframe_interval", 15)),
            gemini_model=gemini_cfg.get("model", od_cfg.get("gemini_model", "gemini-1.5-pro-latest")),
            prompt=od_prompt,
            grounding_dino_model=gd_cfg.get("model_name", "IDEA-Research/grounding-dino-tiny"),
            grounding_dino_confidence=float(gd_cfg.get("confidence_threshold", 0.3)),
            grounding_dino_box_threshold=float(gd_cfg.get("box_threshold", 0.3)),
            grounding_dino_text_threshold=float(gd_cfg.get("text_threshold", 0.25)),
        )
        self.object_detector = GeminiObjectDetector(od_config)

        # 4. ContactDetector
        # Support "contact_detection" (new) or "contact_detector" (old)
        cd_cfg = self.config.get("contact_detection") or self.config.get("contact_detector") or {}
        cd_config = ContactDetectorConfig(
            proximity_threshold_px=int(cd_cfg.get("proximity_threshold_px", cd_cfg.get("distance_threshold_px", 25))),
            fingertip_indices=cd_cfg.get("fingertip_indices", [4, 8, 12, 16, 20]),
        )
        self.contact_detector = ContactDetector(cd_config)

        # 5. GraspClassifier
        # Support "grasp_classification" (new) or "grasp_classifier" (old)
        gc_cfg = self.config.get("grasp_classification") or self.config.get("grasp_classifier") or {}
        gc_config = GraspClassifierConfig(
            pinch_threshold=float(gc_cfg.get("pinch_threshold", 0.05)),
            wrap_threshold=float(gc_cfg.get("wrap_threshold", 0.15)),
        )
        self.grasp_classifier = GraspClassifier(gc_config)

        # 6. ActionComputer
        # Support "action_computation" (new) or "action_computer" (old)
        ac_cfg = self.config.get("action_computation") or self.config.get("action_computer") or {}
        ac_config = ActionComputerConfig(
            compute_wrist_delta=ac_cfg.get("compute_wrist_delta", True),
            compute_finger_angles=ac_cfg.get("compute_finger_angles", True),
            compute_gripper_state=ac_cfg.get("compute_gripper_state", True),
            normalize_actions=ac_cfg.get("normalize_actions", True),
        )
        self.action_computer = ActionComputer(ac_config)

        # 7. SignalSegmenter (deterministic boundaries from contact/grasp signals)
        ss_cfg = self.config.get("signal_segmentation") or {}
        ss_config = SignalSegmenterConfig(
            min_segment_duration_sec=float(ss_cfg.get("min_segment_duration_sec", 0.2)),
            merge_gap_sec=float(ss_cfg.get("merge_gap_sec", 0.3)),
            idle_threshold_sec=float(ss_cfg.get("idle_threshold_sec", 1.0)),
        )
        self.signal_segmenter = SignalSegmenter(ss_config)

        # 8. SegmentLabeler (VLM labels for pre-computed segments)
        sl_cfg = self.config.get("segment_labeling") or {}
        sl_config = SegmentLabelerConfig(
            gemini_model=gemini_cfg.get("model", sl_cfg.get("gemini_model", "gemini-1.5-flash")),
        )
        self.segment_labeler = create_segment_labeler(sl_config)

        # 9. GeminiLanguageGenerator
        # Support "language_annotation" (new) or "language_generator" (old)
        lg_cfg = self.config.get("language_annotation") or self.config.get("language_generator") or {}
        lg_config = LanguageGeneratorConfig(
            gemini_model=gemini_cfg.get("model", lg_cfg.get("gemini_model", "gemini-1.5-pro-latest")),
            episode_prompt=lg_cfg.get("episode_prompt", LanguageGeneratorConfig.episode_prompt),
            segment_prompt=lg_cfg.get("segment_prompt", LanguageGeneratorConfig.segment_prompt),
        )
        self.language_generator = GeminiLanguageGenerator(lg_config)

        # 9. DatasetExporter
        # Support "output" (new) or "exporter" (old)
        output_cfg = self.config.get("output") or {}
        exp_cfg = self.config.get("exporter") or {}
        
        output_dir = pipe_cfg.get("output_dir", self.config.get("output_dir", "data/output"))
        format_val = output_cfg.get("format", exp_cfg.get("format", "json"))
        include_bytes = output_cfg.get("include_image_bytes", exp_cfg.get("include_image_bytes", False))
        save_viz = output_cfg.get("save_viz_video", (self.config.get("visualizer") or {}).get("output_video", True))
        
        exp_config = ExporterConfig(
            output_dir=output_dir,
            format=format_val,
            include_image_bytes=include_bytes,
            save_viz_video=save_viz,
        )
        self.dataset_exporter = DatasetExporter(exp_config)

        print("\n" + "=" * 50)
        print("EGO ANNOTATE initialized with stages:")
        print("  1. VideoProcessor")
        print("  2. HandTracker")
        print("  3. GeminiObjectDetector")
        print("  4. ContactDetector")
        print("  5. GraspClassifier")
        print("  6. ActionComputer")
        print("  7. GeminiActionSegmenter")
        print("  8. GeminiLanguageGenerator")
        print("  9. DatasetExporter")
        print("=" * 50 + "\n")

    def process_video(self, video_path: str, episode_id: Optional[str] = None) -> AnnotatedEpisode:
        """Process a single video through all pipeline stages, exporting and saving telemetry HUD video.

        Args:
            video_path: Path to the input video.
            episode_id: Optional name override for the episode.

        Returns:
            The compiled AnnotatedEpisode object.
        """
        video_name = Path(video_path).stem
        print("\n" + "#" * 60)
        print(f" PROCESSING VIDEO: {video_name}")
        print("#" * 60 + "\n")

        # Determine original video FPS and sample rate
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps <= 0:
            fps = 30.0
        sample_every = max(1, int(round(fps / self.video_processor.target_fps)))

        # Stage 1: Video Uniform Frame Extraction
        frames_dir = self.config.get("pipeline", {}).get("frames_dir", self.config.get("frame_output_dir", "data/frames"))
        frame_output_dir = str(
            Path(frames_dir) / video_name
        )
        image_paths = self.video_processor.process(video_path, frame_output_dir)

        # Stage 2: Hand Landmarks tracking
        hand_results = self.hand_tracker.track_frames(image_paths)

        # Stage 3: Object Detection (Gemini + Grounding DINO with per-frame tracking)
        per_frame_objects = self.object_detector.detect_per_frame_objects_with_bboxes(
            video_path, image_paths
        )
        print(f"[Pipeline] Object bboxes populated via Grounding DINO for {len(per_frame_objects)} frames")

        # Stage 4-5: Contact & Grasp State per Frame
        frames = []
        for i, path in enumerate(image_paths):
            timestamp = (i * sample_every) / fps
            hands = hand_results[i]
            frame_objs = per_frame_objects[i] if i < len(per_frame_objects) else []

            left_hand = hands.get("left")
            right_hand = hands.get("right")

            left_contact = self.contact_detector.detect_contact(left_hand, frame_objs)
            right_contact = self.contact_detector.detect_contact(right_hand, frame_objs)

            left_grasp = self.grasp_classifier.classify(left_hand)
            right_grasp = self.grasp_classifier.classify(right_hand)

            frame = AnnotationFrame(
                frame_idx=i,
                timestamp=timestamp,
                image_path=path,
                left_hand=left_hand,
                right_hand=right_hand,
                objects=frame_objs,
                left_contact=left_contact,
                right_contact=right_contact,
                left_grasp=left_grasp,
                right_grasp=right_grasp,
            )
            frames.append(frame)

        # Build signal timelines for segmentation
        left_contact_timeline = [f.left_contact for f in frames]
        right_contact_timeline = [f.right_contact for f in frames]
        left_grasp_timeline = [f.left_grasp for f in frames]
        right_grasp_timeline = [f.right_grasp for f in frames]
        frame_timestamps = [f.timestamp for f in frames]

        # Stage 6a: Signal-based segment boundary detection (deterministic)
        candidates = self.signal_segmenter.get_candidates(
            left_contact_timeline,
            right_contact_timeline,
            left_grasp_timeline,
            right_grasp_timeline,
            frame_timestamps,
        )
        print(f"[Pipeline] SignalSegmenter found {len(candidates)} candidate segments")

        # Stage 6b: VLM labeling of pre-computed segments
        if self.segment_labeler is not None:
            segments = self.segment_labeler.label_segments(candidates, video_path)
            print(f"[Pipeline] SegmentLabeler labeled {len(segments)} segments")
        else:
            # Fallback: create default segments from candidates
            segments = []
            for cand in candidates:
                segments.append(ActionSegment(
                    name=cand.transition_type if cand.transition_type != "full_video" else "idle",
                    start_time=cand.start_time,
                    end_time=cand.end_time,
                    object_name=cand.object_name or "unknown",
                    hand_used="right",
                    description=f"auto: {cand.contact_state} {cand.grasp_type}",
                ))
            print(f"[Pipeline] SegmentLabeler unavailable, using {len(segments)} auto-labeled segments")

        # Map segment labels to frames
        for frame in frames:
            assigned = False
            for seg in segments:
                if seg.start_time <= frame.timestamp <= seg.end_time:
                    frame.action_segment = seg.name
                    assigned = True
                    break
            if not assigned:
                frame.action_segment = "idle"

        # Stage 7: Language Generation
        task_description = self.language_generator.generate_episode_description(video_path)
        seg_descriptions = self.language_generator.generate_segment_descriptions(
            video_path, segments
        )
        # Assign descriptions to action segments and frame descriptions
        for seg, desc in zip(segments, seg_descriptions):
            seg.description = desc

        for frame in frames:
            for seg in segments:
                if seg.start_time <= frame.timestamp <= seg.end_time:
                    frame.frame_description = seg.description
                    break

        # Stage 8: VLA Action Primitives Computation
        self.action_computer.compute_actions(frames)

        # Build Annotated Episode
        if episode_id is None:
            episode_id = video_name

        duration_sec = total_frames / fps if fps > 0 else 0.0
        episode = AnnotatedEpisode(
            episode_id=episode_id,
            video_path=video_path,
            task_description=task_description,
            frames=frames,
            segments=segments,
            num_frames=len(frames),
            duration_seconds=duration_sec,
        )

        # Stage 9: Export Episode
        self.dataset_exporter.export_episode(episode)

        print("\n" + "=" * 50)
        print("EPISODE ANNOTATION COMPLETE")
        print("=" * 50)
        print(f"Episode ID:       {episode.episode_id}")
        print(f"Task Desc:        {episode.task_description}")
        print(f"Total Frames:     {episode.num_frames}")
        print(f"Duration:         {episode.duration_seconds:.2f} seconds")
        print(f"Action Segments:  {len(episode.segments)}")
        print("=" * 50 + "\n")

        return episode

    def process_videos(self, video_paths: List[str]) -> List[AnnotatedEpisode]:
        """Process a list of videos in a batch, logging errors and continuing on failure.

        Args:
            video_paths: List of absolute paths to the video files.

        Returns:
            List of successfully annotated AnnotatedEpisodes.
        """
        successful_episodes = []
        failures = {}

        print("\n" + "=" * 60)
        print(f"STARTING BATCH PROCESSING OF {len(video_paths)} VIDEOS")
        print("=" * 60 + "\n")

        for video_path in video_paths:
            try:
                episode = self.process_video(video_path)
                successful_episodes.append(episode)
            except Exception as e:
                logger.error("Failed to process video: %s. Error: %s", video_path, e, exc_info=True)
                failures[video_path] = str(e)
                # Fail fast if continue_on_error is disabled in config
                batch_cfg = self.config.get("batch", {})
                if not batch_cfg.get("continue_on_error", True):
                    raise e

        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE SUMMARY")
        print("=" * 60)
        print(f"Total Videos:      {len(video_paths)}")
        print(f"Successful:        {len(successful_episodes)}")
        print(f"Failed:            {len(failures)}")
        if failures:
            print("\nFailed Videos Detail:")
            for path, err in failures.items():
                print(f"  - {Path(path).name}: {err}")
        print("=" * 60 + "\n")

        return successful_episodes
