"""Pipeline orchestration — runs all annotation stages.

Main EgoAnnotatePipeline orchestrator that coordinates loading YAML config, sampling videos,
tracking hands, detecting objects, classifying contact/grasp states, computing actions,
segmenting temporal actions, generating descriptions, and exporting the dataset.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import cv2
import yaml

from .action_computer import ActionComputer, ActionComputerConfig
from .contact_detector import ContactDetector, ContactDetectorConfig
from .dataset_exporter import DatasetExporter, ExporterConfig
from .datatypes import ActionSegment, AnnotatedEpisode, AnnotationFrame
from .grasp_classifier import GraspClassifier, GraspClassifierConfig
from .hand_tracker import HandTracker, HandTrackerConfig
from .language_generator import GeminiLanguageGenerator, LanguageGeneratorConfig
from .object_detector import GeminiObjectDetector, ObjectDetectorConfig
from .perception.depth import (
    DepthConfig,
    create_depth_estimator,
    localize_objects_3d,
)
from .perception.detector import Detector2DConfig, create_detector_2d
from .perception.pnp_pose import CameraIntrinsics, refine_poses_pnp
from .perception.sam2_segmenter import Sam2Config, create_sam2_segmenter, encode_mask_rle
from .retargeting import (
    Retargeter,
    RetargetingConfig,
    PoseMapper,
    GripperMapper,
    create_ik_solver,
    MinkIKConfig,
)
from .segment_labeler import SegmentLabeler, SegmentLabelerConfig, create_segment_labeler, build_instruction
from .signal_segmenter import SignalSegmenter, SignalSegmenterConfig
from .video_processor import VideoProcessor
from .vlm_backend import VLMBackend, VLMBackendConfig, create_vlm_backend

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
        # Phase C: the 2D bbox backend is chosen via the `perception:` section
        # (`detector_backend: "owlvit"` default; `"grounding_dino"` opt-in).
        # The legacy `grounding_dino:` section still works as an alias and
        # fills in settings the `perception:` section does not override.
        perception_raw = self.config.get("perception") or {}
        gd_legacy = self.config.get("grounding_dino") or {}
        det_config = Detector2DConfig(
            backend=str(perception_raw.get("detector_backend", "owlvit")),
            model_name=str(
                perception_raw.get("owlvit_model", gd_legacy.get("model_name", "google/owlvit-base-patch32"))
            ),
            confidence_threshold=float(
                perception_raw.get("confidence_threshold", gd_legacy.get("confidence_threshold", 0.3))
            ),
            box_threshold=float(
                perception_raw.get("box_threshold", gd_legacy.get("box_threshold", 0.30))
            ),
            text_threshold=float(
                perception_raw.get("text_threshold", gd_legacy.get("text_threshold", 0.25))
            ),
            grounding_dino_weights=str(
                perception_raw.get("grounding_dino_weights", "models/groundingdino_swint_ogc.pth")
            ),
            grounding_dino_config=str(
                perception_raw.get("grounding_dino_config", "models/GroundingDINO_SwinT_OGC.py")
            ),
        )
        self.detector_2d = create_detector_2d(det_config)  # None when unavailable

        # Support "object_detection" (new) or "object_detector" (old)
        od_cfg = self.config.get("object_detection") or self.config.get("object_detector") or {}
        gemini_cfg = self.config.get("gemini") or {}
        # Phase D: VLM dual backend. Global default from `vlm:`; each stage may
        # override with its own `vlm_backend` key ("gemini" | "local").
        vlm_raw = self.config.get("vlm") or {}
        self.vlm_config = VLMBackendConfig(
            backend=str(vlm_raw.get("backend", "gemini")),
            gemini_model=str(gemini_cfg.get("model", "gemini-3.8-flash")),
            gemini_api_key=vlm_raw.get("gemini_api_key") or gemini_cfg.get("api_key"),
            local_model=str(vlm_raw.get("local_model", "HuggingFaceTB/SmolVLM-500M-Instruct")),
            local_weights=str(vlm_raw.get("local_weights", "models/smolvlm-500m-instruct")),
            device=str(vlm_raw.get("device", "cpu")),
            max_new_tokens=int(vlm_raw.get("max_new_tokens", 256)),
            temperature=float(vlm_raw.get("temperature", 0.1)),
            video_frames=int(vlm_raw.get("video_frames", 8)),
            timeout_s=float(vlm_raw.get("timeout_s", 300)),
            on_unavailable=str(vlm_raw.get("on_unavailable", "degrade")),
        )
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
            grounding_dino_model=gd_cfg.get("model_name", "google/owlvit-base-patch32"),
            grounding_dino_confidence=float(gd_cfg.get("confidence_threshold", 0.3)),
            grounding_dino_box_threshold=float(gd_cfg.get("box_threshold", 0.3)),
            grounding_dino_text_threshold=float(gd_cfg.get("text_threshold", 0.25)),
        )
        self.object_detector = GeminiObjectDetector(
            od_config,
            detector_2d=self.detector_2d,
            vlm=self._vlm_for_stage(od_cfg, required=True),
        )

        # 3b. Phase C optional perception: SAM 2 masks, metric depth, PnP.
        # All three degrade gracefully (warn-and-continue) when unavailable.
        sam2_raw = self.config.get("sam2") or {}
        sam2_config = Sam2Config(
            enabled=bool(sam2_raw.get("enabled", False)),
            checkpoint=str(sam2_raw.get("checkpoint", "models/sam2.1_hiera_tiny.pt")),
            config=str(sam2_raw.get("config", sam2_raw.get("config_name", "sam2_hiera_t"))),
            device=str(sam2_raw.get("device", "cpu")),
        )
        self.sam2 = create_sam2_segmenter(sam2_config)  # None unless enabled+available

        depth_raw = self.config.get("depth") or {}
        depth_config = DepthConfig(
            backend=str(depth_raw.get("backend", "unidepth")),
            model_path=str(depth_raw.get("model_path", "models/unidepth_v1.onnx")),
            keyframe_only=bool(depth_raw.get("keyframe_only", True)),
        )
        self.depth_estimator = create_depth_estimator(depth_config)

        # Camera intrinsics: null fields -> None -> 3D/PnP stages skip with a warning.
        # Contact/grasp semantics stay 2D (depth-informed contact is future work).
        self.intrinsics = CameraIntrinsics.from_dict(self.config.get("camera"))

        pnp_raw = self.config.get("pnp_refinement") or {}
        self.pnp_enabled = bool(pnp_raw.get("enabled", True))

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
        self.segment_labeler = create_segment_labeler(
            sl_config, vlm=self._vlm_for_stage(sl_cfg, required=False)
        )

        # 9. GeminiLanguageGenerator
        # Support "language_annotation" (new) or "language_generator" (old)
        lg_cfg = self.config.get("language_annotation") or self.config.get("language_generator") or {}
        lg_config = LanguageGeneratorConfig(
            gemini_model=gemini_cfg.get("model", lg_cfg.get("gemini_model", "gemini-1.5-pro-latest")),
            episode_prompt=lg_cfg.get("episode_prompt", LanguageGeneratorConfig.episode_prompt),
            segment_prompt=lg_cfg.get("segment_prompt", LanguageGeneratorConfig.segment_prompt),
        )
        self.language_generator = GeminiLanguageGenerator(
            lg_config, vlm=self._vlm_for_stage(lg_cfg, required=True)
        )

        # 9. DatasetExporter
        # Support "output" (new) or "exporter" (old)
        output_cfg = self.config.get("output") or {}
        exp_cfg = self.config.get("exporter") or {}
        
        output_dir = pipe_cfg.get("output_dir", self.config.get("output_dir", "data/output"))
        format_val = output_cfg.get("format", exp_cfg.get("format", "json"))
        include_bytes = output_cfg.get("include_image_bytes", exp_cfg.get("include_image_bytes", False))
        save_viz = output_cfg.get("save_viz_video", (self.config.get("visualizer") or {}).get("output_video", True))
        save_overlay = output_cfg.get("save_overlay_video", True)
        
        exp_config = ExporterConfig(
            output_dir=output_dir,
            format=format_val,
            include_image_bytes=include_bytes,
            save_viz_video=save_viz,
            save_overlay_video=save_overlay,
            lerobot_export_mode=(self.config.get("export") or {}).get("mode", "human"),
        )
        self.dataset_exporter = DatasetExporter(exp_config)

        # 10. Retargeter (Optional human-to-robot kinematic retargeting)
        ret_cfg = self.config.get("retargeting") or {}
        self.enable_retargeting = bool(ret_cfg.get("enable", False))
        self.save_retargeting_proof_video = bool(ret_cfg.get("save_proof_video", True))
        self.retargeter: Optional[Retargeter] = None

        if self.enable_retargeting:
            try:
                cfg_path = ret_cfg.get("config_path")
                if cfg_path and os.path.exists(cfg_path):
                    r_config = RetargetingConfig.from_yaml(cfg_path)
                else:
                    r_config = RetargetingConfig(
                        urdf_path=ret_cfg.get("target_urdf_path"),
                        urdf_key=ret_cfg.get("urdf_key"),
                        end_effector_link=ret_cfg.get("end_effector_link", "panda_link8"),
                        gripper_joint_names=ret_cfg.get(
                            "gripper_joint_names",
                            ["panda_finger_joint1", "panda_finger_joint2"],
                        ),
                        ik_backend=ret_cfg.get("ik_backend", "pybullet"),
                        mink_ik=MinkIKConfig(**(ret_cfg.get("mink_ik") or {})),
                    )
                self.retargeter = Retargeter(r_config)
                logger.info("Retargeting stage enabled.")
            except Exception as e:
                logger.warning(
                    "Failed to initialize Retargeter (%s). Retargeting stage will be skipped cleanly.",
                    e,
                )
                self.enable_retargeting = False
                self.retargeter = None

        # 10b. Physics verification gate (Phase E, Stage 9b; opt-in, default off).
        # Runs AFTER export: replays the exported HDF5 in MuJoCo with weld-grasp
        # trials and sets the physics_verified label. Never drops episodes.
        pv_cfg = ret_cfg.get("physics_verify") or {}
        self.physics_verify_enabled = bool(pv_cfg.get("enable", False))
        self.episode_verifier = None
        if self.physics_verify_enabled:
            if not self.enable_retargeting:
                logger.warning(
                    "[Pipeline] physics_verify enabled but retargeting is disabled — "
                    "the exported HDF5 will lack robot fields and verification "
                    "will fail loudly per episode."
                )
            try:
                from .retargeting.episode_verifier import EpisodeVerifier, VerifierConfig
                self.episode_verifier = EpisodeVerifier(
                    VerifierConfig(
                        tracking_err_rad_max=float(pv_cfg.get("tracking_err_rad_max", 0.15)),
                        lift_min_m=float(pv_cfg.get("lift_min_m", 0.02)),
                        grasp_proximity_m=float(pv_cfg.get("grasp_proximity_m", 0.09)),
                        min_reachable_pct=float(pv_cfg.get("min_reachable_pct", 50.0)),
                        urdf_path=(self.retargeter.config.urdf_path
                                   if self.retargeter is not None else None),
                        ee_link_name=(self.retargeter.config.end_effector_link
                                      if self.retargeter is not None else "panda_link8"),
                    )
                )
                if not self.episode_verifier.is_available():
                    logger.warning(
                        "[Pipeline] physics_verify enabled but mujoco is not installed — "
                        "verification will report failure (never a silent pass)."
                    )
                else:
                    logger.info("[Pipeline] Physics verification gate enabled.")
            except Exception as e:
                logger.warning(
                    "[Pipeline] Failed to initialize EpisodeVerifier (%s). "
                    "Physics verification will be skipped.", e,
                )
                self.episode_verifier = None

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
        # Phase D: show which VLM backend each stage actually got.
        print(f"  V. VLM backend (global default): {self.vlm_config.backend}")
        if self.enable_retargeting:
            print(" 10. Retargeter (Human-to-Robot Kinematic Retargeting)")
        # Phase C optional perception (None/degraded backends are skipped silently here;
        # each factory already logged why).
        if self.detector_2d is not None:
            print(f"  P. 2D detector backend: {self.detector_2d.backend_name}")
        if self.sam2 is not None:
            print("  P. SAM 2 masks: enabled")
        if self.depth_estimator is not None and self.depth_estimator.is_available():
            print(f"  P. Metric depth backend: {self.depth_estimator.backend_name}")
        if self.pnp_enabled and self.intrinsics is not None:
            print("  P. PnP hand-pose refinement: enabled")
        print("=" * 50 + "\n")

    def _vlm_for_stage(self, stage_cfg: dict, required: bool) -> Optional[VLMBackend]:
        """Resolve the VLM backend for one pipeline stage (Phase D).

        A per-stage ``vlm_backend`` key overrides the global ``vlm.backend``.
        Returns ``None`` when unavailable (stages with their own fallbacks
        degrade); raises for required stages so the pipeline fails loud at
        startup, exactly like the old stage constructors did.
        """
        override = stage_cfg.get("vlm_backend") if isinstance(stage_cfg, dict) else None
        cfg = replace(self.vlm_config, backend=str(override)) if override else self.vlm_config
        backend = create_vlm_backend(cfg)
        if backend is None and required:
            hint = (
                "download the SmolVLM weights manually from "
                "https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct"
                if cfg.backend == "local"
                else "install google-genai and set GEMINI_API_KEY"
            )
            raise RuntimeError(
                f"No VLM backend available for stage (vlm.backend={cfg.backend!r}). {hint}."
            )
        if backend is None:
            logger.warning(
                "[pipeline] No VLM backend available (vlm.backend=%r); "
                "stage will use its degraded fallback.",
                cfg.backend,
            )
        else:
            logger.debug("[pipeline] Stage VLM backend: %s", backend.name)
        return backend

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

        # Stage 3b (Phase C): 3D object localization on detection keyframes.
        # Attaches metric position_3d (+ optional SAM 2 mask_rle) to keyframe
        # ObjectAnnotations. Skipped gracefully without depth/intrinsics.
        self._localize_objects_3d(video_name, image_paths, per_frame_objects)

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

        # Stage 5b: Temporal Grasp Majority Voting & Hold Inheritance
        self.grasp_classifier.smooth_sequence(frames)

        # Stage 5c (Phase C): PnP hand-pose refinement — metric wrist translation
        # per hand via solvePnPRansac against the anthropometric hand model.
        # Per-frame failures fall back to the existing estimates (never raises).
        if self.pnp_enabled:
            if self.intrinsics is None:
                logger.warning(
                    "[Pipeline] pnp_refinement.enabled=true but camera intrinsics "
                    "are null — skipping PnP refinement. Fill in the 'camera:' "
                    "section of the config to enable it."
                )
            else:
                frame_hw = self._probe_frame_resolution(image_paths)
                if frame_hw is not None:
                    fh, fw = frame_hw
                    refine_poses_pnp(frames, self.intrinsics, fw, fh, enabled=True)

        # Build signal timelines for segmentation and compute episode dwell counts
        left_contact_timeline = [f.left_contact for f in frames]
        right_contact_timeline = [f.right_contact for f in frames]
        left_grasp_timeline = [f.left_grasp for f in frames]
        right_grasp_timeline = [f.right_grasp for f in frames]
        frame_timestamps = [f.timestamp for f in frames]

        dwell: dict = {}
        for f in frames:
            if f.left_contact and f.left_contact.in_contact and f.left_contact.object_name:
                obj = f.left_contact.object_name
                dwell[obj] = dwell.get(obj, 0) + 1
            if f.right_contact and f.right_contact.in_contact and f.right_contact.object_name:
                obj = f.right_contact.object_name
                dwell[obj] = dwell.get(obj, 0) + 1

        # Stage 6a: Signal-based segment boundary detection (deterministic)
        candidates = self.signal_segmenter.get_candidates(
            left_contact_timeline,
            right_contact_timeline,
            left_grasp_timeline,
            right_grasp_timeline,
            frame_timestamps,
            dwell=dwell,
        )
        print(f"[Pipeline] SignalSegmenter found {len(candidates)} candidate segments")

        # Stage 6b: VLM labeling of pre-computed segments
        segments = []
        if self.segment_labeler is not None:
            try:
                segments = self.segment_labeler.label_segments(candidates, video_path)
                print(f"[Pipeline] SegmentLabeler labeled {len(segments)} segments")
            except Exception as e:
                print(f"[Pipeline] SegmentLabeler failed: {e}, falling back to auto-labeling")
                self.segment_labeler = None
        
        if self.segment_labeler is None:
            # Fallback: create default segments from candidates
            segments = []
            for cand in candidates:
                hands = cand.hands if hasattr(cand, "hands") and cand.hands else ["right"]
                hand_used = cand.hand_used if hasattr(cand, "hand_used") else ("both" if len(hands) == 2 else hands[0])
                act_name = cand.transition_type if cand.transition_type != "full_video" else "idle"
                obj_name = cand.object_name if (act_name != "idle" and cand.object_name != "unknown") else None
                desc = build_instruction(act_name, obj_name, hands)
                segments.append(ActionSegment(
                    name=act_name,
                    start_time=cand.start_time,
                    end_time=cand.end_time,
                    object_name=cand.object_name or "unknown",
                    hand_used=hand_used,
                    hands=hands,
                    description=desc,
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
        for seg in segments:
            obj_name = seg.object_name if (seg.name != "idle" and seg.object_name != "unknown") else None
            seg.description = build_instruction(seg.name, obj_name, seg.hands)

        for frame in frames:
            assigned = False
            for seg in segments:
                if seg.start_time <= frame.timestamp <= seg.end_time:
                    obj_name = seg.object_name if (seg.name != "idle" and seg.object_name != "unknown") else None
                    frame.frame_description = build_instruction(seg.name, obj_name, seg.hands)
                    assigned = True
                    break
            if not assigned:
                frame.frame_description = build_instruction("idle", None, ["right"])

        # Stage 8: VLA Action Primitives Computation
        self.action_computer.compute_actions(frames)

        # Stage 9: Kinematic Retargeting (Optional)
        target_robot_name = "human_egocentric"
        if self.enable_retargeting and self.retargeter is not None:
            # Fail-loud: if retargeting is explicitly enabled, failures are not silently swallowed.
            # A broken URDF or config should surface immediately rather than producing silent
            # human_egocentric output that passes as retargeted data downstream.
            print("\n[Pipeline] Running Stage 9: Human-to-Robot Kinematic Retargeting...")
            kin = self.retargeter.load_kinematics()
            target_robot_name = kin.robot_name

            target_poses = PoseMapper(self.retargeter.config.pose_mapper).map_frames(frames)
            with create_ik_solver(
                self.retargeter.config.ik_backend,
                kin,
                self.retargeter.config.end_effector_link,
                self.retargeter.config.ik_solver,
                self.retargeter.config.mink_ik,
            ) as solver:
                ik_results = solver.solve_sequence(target_poses)
            gripper_mapper = GripperMapper(kin, self.retargeter.config.gripper_mapper)
            gripper_commands = gripper_mapper.map_frames(frames)

            # Post-processing gates — the SAME shared function
            # Retargeter.run_from_annotations uses (R3 source gate, R4 velocity
            # feasibility, self-collision gate). A past real-verification
            # failure came from this stage inlining retargeting without them.
            from .retargeting.retargeter import apply_retargeting_gates
            import numpy as np
            joint_traj = np.stack([r.joint_angles for r in ik_results], axis=0)
            vel_limits = getattr(kin, "joint_velocity_limits", None)
            gates = apply_retargeting_gates(
                ik_results,
                target_poses,
                gripper_commands,
                joint_traj,
                dt=1.0 / 30.0,
                vel_limits=vel_limits,
            )
            reachability = gates["reachability"]
            gripper_traj = gates["gripper_trajectory"]

            for i, f in enumerate(frames):
                f.robot_joint_angles = ik_results[i].joint_angles.tolist()
                f.robot_gripper_opening_m = float(gripper_traj[i])
                f.robot_gripper_method = gripper_commands[i].gripper_mapping_method
                f.robot_reachable = bool(reachability[i])

            n_reach = int(reachability.sum())
            print(
                f"[Pipeline] Retargeting complete ({target_robot_name}): "
                f"{n_reach}/{len(frames)} frames reachable ({100.0 * n_reach / max(len(frames), 1):.1f}%)"
                f" — {gates['n_velocity_infeasible']} velocity-gated,"
                f" {gates['n_collision_gated']} self-collision-gated"
            )

            if self.save_retargeting_proof_video:
                out_ep_dir = Path(self.dataset_exporter.output_path) / (episode_id or video_name)
                out_ep_dir.mkdir(parents=True, exist_ok=True)
                sbs_path = out_ep_dir / "side_by_side.mp4"
                self._generate_proof_video(
                    video_path, frames, ik_results, gripper_commands, kin, sbs_path
                )

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
            target_robot=target_robot_name,
        )

        if hasattr(self.hand_tracker, "last_metrics") and self.hand_tracker.last_metrics:
            m = self.hand_tracker.last_metrics
            episode.tracking_loss_pct = m.get("lost_pct", 0.0)
            episode.tracking_loss_pct_active = m.get("lost_pct_active", 0.0)
            episode.tracking_rescued_pct = m.get("rescued_pct", 0.0)
            episode.tracking_interpolated_pct = m.get("interpolated_pct", 0.0)

        # Stage 10: Export Episode
        self.dataset_exporter.export_episode(episode)

        # Stage 9b: Physics verification gate (Phase E; opt-in, default off).
        # Replays the exported HDF5 in MuJoCo with weld-grasp trials and sets
        # the physics_verified label. On ANY failure the episode is KEPT —
        # the verifier gates only the label, never the data.
        self._maybe_verify_physics(episode)

        print("\n" + "=" * 50)
        print("EPISODE ANNOTATION COMPLETE")
        print("=" * 50)
        print(f"Episode ID:       {episode.episode_id}")
        print(f"Task Desc:        {episode.task_description}")
        print(f"Total Frames:     {episode.num_frames}")
        print(f"Duration:         {episode.duration_seconds:.2f} seconds")
        print(f"Action Segments:  {len(episode.segments)}")
        print(f"Target Robot:     {episode.target_robot}")
        print("=" * 50 + "\n")

        return episode

    def _maybe_verify_physics(self, episode: AnnotatedEpisode) -> None:
        """Stage 9b: physics-verification gate (Phase E).

        Replays the exported ``episode_rlds.hdf5`` in MuJoCo with weld-grasp
        trials and sets ``episode.physics_verified`` / ``physics_report_path``.
        The verifier gates ONLY the label: on any failure or error the episode
        is kept and the report is still written — data is never dropped.
        """
        if not self.physics_verify_enabled:
            return
        if self.episode_verifier is None:
            logger.warning(
                "[Pipeline] physics_verify enabled but the verifier failed to "
                "initialize — episode %s kept, label withheld.", episode.episode_id)
            return
        out_ep_dir = Path(self.dataset_exporter.output_path) / episode.episode_id
        hdf5_path = out_ep_dir / "episode_rlds.hdf5"
        try:
            # Prefer the retargeter's RESOLVED URDF: config.urdf_path may be
            # None when URDFLoader fell back to its bundled model.
            urdf_path = None
            if self.retargeter is not None:
                try:
                    urdf_path = self.retargeter.load_kinematics().urdf_path
                except Exception as e:
                    logger.warning("[Pipeline] Could not resolve retargeter URDF (%s).", e)
                    urdf_path = self.retargeter.config.urdf_path
            report = self.episode_verifier.verify_episode(
                str(hdf5_path), episode.episode_id, urdf_path=urdf_path)
            report_path = out_ep_dir / "physics_verification.json"
            report.save_json(str(report_path))
            episode.physics_verified = bool(report.passed)
            episode.physics_report_path = str(report_path)
            self.dataset_exporter.persist_physics_verification(episode, out_ep_dir)
            if report.passed:
                logger.info("[Pipeline] Physics verification PASSED for episode %s.",
                            episode.episode_id)
                print(f"[Pipeline] Physics verification PASSED ({episode.episode_id})")
            else:
                logger.warning(
                    "[Pipeline] Physics verification FAILED for episode %s — "
                    "episode kept, label withheld. Report: %s",
                    episode.episode_id, report_path)
                print(f"[Pipeline] Physics verification FAILED ({episode.episode_id}) — "
                      f"episode kept, see {report_path}")
        except Exception as e:  # noqa: BLE001 - the gate must never lose the episode
            logger.warning(
                "[Pipeline] Physics verification errored for episode %s (%s) — "
                "episode kept, label withheld.", episode.episode_id, e)

    @staticmethod
    def _probe_frame_resolution(image_paths: List[str]) -> Optional[tuple]:
        """Return ``(h, w)`` of the extracted frames hand landmarks are normalized to."""
        for p in image_paths:
            img = cv2.imread(p)
            if img is not None:
                return img.shape[:2]
        return None

    def _localize_objects_3d(
        self,
        video_name: str,
        image_paths: List[str],
        per_frame_objects: List[List],
    ) -> None:
        """Phase-C 3D object localization on detection keyframes.

        For each keyframe: estimate metric depth (cached to
        ``<output>/<episode>/depth/depth_<idx>.npy`` as float16 when
        ``depth.keyframe_only``), optionally predict SAM 2 masks, then attach
        ``position_3d`` (meters, camera frame) and ``mask_rle`` to the keyframe
        ``ObjectAnnotation`` objects in place. Never raises: any failure skips
        the 3D pass with a warning, leaving 2D annotations intact.
        """
        try:
            depth_est = getattr(self, "depth_estimator", None)
            if depth_est is None or not depth_est.is_available():
                return
            if self.intrinsics is None:
                logger.warning(
                    "[Pipeline] Metric depth is available but camera intrinsics "
                    "are null — skipping 3D object localization. Fill in the "
                    "'camera:' section of the config to enable it."
                )
                return

            from .perception.detector import Detection  # local: keep import light

            interval = max(1, int(self.object_detector.config.bbox_keyframe_interval))
            keyframes = list(range(0, len(image_paths), interval))
            if keyframes and keyframes[-1] != len(image_paths) - 1:
                keyframes.append(len(image_paths) - 1)

            depth_dir = Path(self.dataset_exporter.output_path) / video_name / "depth"
            n_localized = 0
            for idx in keyframes:
                img = cv2.imread(image_paths[idx])
                if img is None:
                    continue
                try:
                    import numpy as _np

                    depth_map = depth_est.estimate_depth(img)
                except Exception as e:
                    logger.warning(
                        "[Pipeline] Depth estimation failed on keyframe %d: %s "
                        "— skipping remaining 3D localization.",
                        idx,
                        e,
                    )
                    break
                if depth_est.config.keyframe_only:
                    depth_dir.mkdir(parents=True, exist_ok=True)
                    _np.save(
                        depth_dir / f"depth_{idx:06d}.npy",
                        _np.asarray(depth_map, dtype=_np.float16),
                    )

                objs = per_frame_objects[idx] if idx < len(per_frame_objects) else []
                detections: List[Detection] = []
                det_obj_idx: List[int] = []
                for j, obj in enumerate(objs):
                    if obj.bbox is not None:
                        detections.append(
                            Detection(
                                bbox_xyxy_norm=_np.asarray(obj.bbox, dtype=_np.float32),
                                label=obj.name,
                                score=1.0,
                            )
                        )
                        det_obj_idx.append(j)
                if not detections:
                    continue

                masks = None
                sam2 = getattr(self, "sam2", None)
                if sam2 is not None:
                    try:
                        masks = sam2.predict_masks(
                            img, [d.bbox_xyxy_norm for d in detections]
                        )
                    except Exception as e:
                        logger.warning(
                            "[Pipeline] SAM 2 masking failed on keyframe %d: %s "
                            "— continuing box-only for this episode.",
                            idx,
                            e,
                        )
                        self.sam2 = None

                points = localize_objects_3d(
                    detections, depth_map, self.intrinsics, masks
                )
                for di, oi in enumerate(det_obj_idx):
                    if points[di] is not None:
                        objs[oi].position_3d = points[di]
                        n_localized += 1
                    if masks is not None and masks[di] is not None:
                        objs[oi].mask_rle = encode_mask_rle(masks[di])
            if n_localized:
                print(
                    f"[Pipeline] 3D localization: {n_localized} object positions "
                    f"attached across {len(keyframes)} keyframes"
                )
        except Exception as e:
            logger.warning("[Pipeline] 3D object localization skipped: %s", e)

    def _generate_proof_video(
        self,
        video_path: str,
        frames: List[AnnotationFrame],
        ik_results: List[Any],
        gripper_commands: List[Any],
        kin: Any,
        output_path: Path,
    ) -> None:
        """Render side-by-side proof video of human vs robot simulation."""
        from scripts.validate_retargeting import (
            render_robot_frame_pybullet,
            render_robot_skeleton_2d,
            compose_side_by_side,
            RENDER_W,
            RENDER_H,
        )
        import pybullet as pb
        import pybullet_data

        client_id = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        robot_id = pb.loadURDF(
            kin.urdf_path, basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=client_id
        )

        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_duration = src_n / src_fps if (src_n > 0 and src_fps > 0) else (len(frames) / 30.0)

        # Output video FPS matches the sampling rate of the processed frames so the side_by_side duration matches source video duration
        writer_fps = len(frames) / src_duration if src_duration > 0 else src_fps

        test_img = render_robot_frame_pybullet(
            pb, client_id, robot_id, kin, kin.rest_poses, 0.02
        )
        using_pybullet = test_img is not None

        # Write frames to a temporary mp4v file first (OpenCV limitation: cannot write H.264 directly)
        tmp_path = output_path.with_suffix(".tmp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None
        out_w, out_h = None, None

        for i in range(len(frames)):
            frame_obj = frames[i]
            human_img = None
            if hasattr(frame_obj, "image_path") and frame_obj.image_path and Path(frame_obj.image_path).exists():
                human_img = cv2.imread(frame_obj.image_path)
            if human_img is None:
                f_idx = getattr(frame_obj, "frame_idx", i)
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, human_img = cap.read()
                if not ret or human_img is None:
                    human_img = np.zeros((480, 640, 3), np.uint8)

            ik_r = ik_results[i]
            gc = gripper_commands[i]
            g_m = float(gc.opening_m)

            if using_pybullet:
                r_img = render_robot_frame_pybullet(pb, client_id, robot_id, kin, ik_r.joint_angles, g_m)
                if r_img is None:
                    r_img = render_robot_skeleton_2d(ik_r.joint_angles, kin)
            else:
                r_img = render_robot_skeleton_2d(ik_r.joint_angles, kin)

            comp = compose_side_by_side(
                human_img,
                r_img,
                ik_r.frame_idx,
                ik_r.timestamp,
                ik_r.reachable,
                g_m,
                gc.gripper_mapping_method,
                robot_name=kin.robot_name,
            )

            if writer is None:
                out_h, out_w = comp.shape[:2]
                writer = cv2.VideoWriter(str(tmp_path), fourcc, writer_fps, (out_w, out_h))

            if comp.shape[0] != out_h or comp.shape[1] != out_w:
                comp = cv2.resize(comp, (out_w, out_h))

            writer.write(comp)

        if writer is not None:
            writer.release()
        cap.release()
        pb.disconnect(client_id)

        # Re-encode mp4v → H.264 (libx264 + yuv420p + faststart) for browser/QuickTime compatibility.
        # Uses atomic write: encode to output_path, then remove the tmp file.
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(tmp_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", "fast",
            str(output_path),
        ]
        try:
            import subprocess as _sp
            result = _sp.run(ffmpeg_cmd, stdout=_sp.PIPE, stderr=_sp.PIPE, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg H.264 re-encode failed (returncode={result.returncode}):\n"
                    f"{result.stderr.decode()}"
                )
            tmp_path.unlink(missing_ok=True)
            logger.info("Retargeting proof video (H.264) saved to: %s", output_path)
            print(f"[Pipeline] side_by_side.mp4 (H.264/libx264): {output_path}")
        except Exception as e:
            # If ffmpeg fails, keep the mp4v file so the user is not left with nothing,
            # but log a loud warning.
            tmp_path.rename(output_path)
            logger.warning(
                "H.264 re-encode of side_by_side.mp4 failed (%s). "
                "Kept legacy mp4v version at %s. Install ffmpeg with libx264 to fix.",
                e, output_path,
            )

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
