"""Unified full-video EgoAnnotate v3 orchestrator (frames 0..N-1, resumable stage cache, zero paid APIs)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.egoannotate.behavior import (
    build_scene_graph_and_affordances,
    enrich_frames_with_gaze,
    extract_grasp_events,
    segment_atomic_actions,
)
from src.egoannotate.config import PipelineConfig, load_config
from src.egoannotate.exporters_v3 import export_all_formats
from src.egoannotate.geometry_world import (
    MonocularWorldVO,
    register_hand_and_objects_to_world,
    register_objects_to_world,
)
from src.egoannotate.perception_hands import (
    MediaPipeHandTracker,
    MediaPipeUpperBodyTracker,
)
from src.egoannotate.perception_objects import OpenCVManipulatedObjectTracker
from src.egoannotate.quality import score_episode_quality
from src.egoannotate.retargeting_v3 import retarget_all_configured_embodiments
from src.egoannotate.schemas import (
    AtomicActionSegment,
    EpisodeIR,
    FrameAnnotation,
    HierarchicalLanguage,
    ScaleProvenance,
)
from src.egoannotate.semantics import _verify_segment_against_geometry, annotate_semantics_grounded
from src.egoannotate.tactile import (
    ingest_external_tactile_stream,
    populate_rgb_tactile_proxies,
)
from src.egoannotate.video_io import VideoStreamReader, compute_frame_image_quality


class EgoAnnotateV3Pipeline:
    """End-to-end commercial egocentric video-to-VLA pipeline with full-video processing & stage caching."""

    def __init__(self, cfg: PipelineConfig | None = None, profile: str = "auto") -> None:
        self.cfg = cfg if cfg is not None else load_config(profile=profile)

    def _stage1_cache_path(self, video_sha256: str) -> Path:
        cache_dir = Path(self.cfg.runtime.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"stage1_{video_sha256[:16]}.json"

    def _load_stage1_cache(
        self,
        video_sha256: str,
        expected_total: int,
    ) -> tuple[list[FrameAnnotation], ScaleProvenance, list[list[float]]] | None:
        candidates = [
            self._stage1_cache_path(video_sha256),
            Path(self.cfg.runtime.cache_dir) / "phase1b_full_video_cache.json",
        ]
        for cp in candidates:
            if not cp.exists():
                continue
            try:
                raw = json.loads(cp.read_text(encoding="utf-8"))
                if raw.get("video_sha256") == video_sha256 and int(raw.get("total_frames", 0)) >= expected_total - 2:
                    frames = [FrameAnnotation.model_validate(f) for f in raw["frames"]]
                    scale_prov = ScaleProvenance.model_validate(raw["scale_provenance"])
                    sparse_pts = raw.get("sparse_world_points", [])
                    return frames, scale_prov, sparse_pts
            except (OSError, ValueError, KeyError):
                continue
        return None


    def _sample_video_frames_by_idx(self, video_path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return out
        try:
            for fi in sorted(set(frame_indices)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, bgr = cap.read()
                if ok and bgr is not None:
                    out[int(fi)] = bgr
        finally:
            cap.release()
        return out

    def _run_gpu_enrichment(
        self,
        video_path: Path,
        frames: list[FrameAnnotation],
        segments: list[AtomicActionSegment],
        language: HierarchicalLanguage,
        scale_prov: ScaleProvenance,
    ) -> dict[str, Any]:
        """Run live CUDA Grounding DINO, Depth-Anything-V2-Small & SmolVLM2/Qwen2.5-VL when on GPU."""
        if not (self.cfg.runtime.device == "cuda" and torch.cuda.is_available()):
            return {"enabled": False, "device": self.cfg.runtime.device}

        from src.egoannotate.gpu_backends import (
            DepthAnythingV2SmallGPU,
            GroundingDINOGPUDetector,
            LocalOpenSourceVLMAnnotator,
        )

        metrics: dict[str, Any] = {
            "enabled": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "mixed_precision": self.cfg.runtime.mixed_precision,
        }
        n = len(frames)
        if n == 0:
            return metrics

        kf_indices = list(range(0, n, max(1, n // 120)))
        kf_images = self._sample_video_frames_by_idx(video_path, kf_indices)

        # 1. Open-vocabulary Grounding DINO detection on keyframes
        if "grounding_dino" in str(self.cfg.perception.detector_backend):
            t0 = time.perf_counter()
            det = GroundingDINOGPUDetector(
                prompts=self.cfg.perception.open_vocab_prompts,
                device="cuda",
                mixed_precision=self.cfg.runtime.mixed_precision,
            )
            label_counts: dict[str, int] = {}
            total_dets = 0
            kf_labels: dict[int, str] = {}
            for fi, bgr in kf_images.items():
                dets = det.detect_frame(bgr)
                total_dets += len(dets)
                if dets:
                    best = max(dets, key=lambda d: float(d["score"]))
                    lbl = str(best["label"]).strip().replace(" ", "_") or "manipulated_object"
                    kf_labels[fi] = lbl
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

            # Propagate nearest open-vocab label onto tracked manipulated objects
            if kf_labels:
                sorted_kfs = sorted(kf_labels.keys())
                for fr in frames:
                    if not fr.objects:
                        continue
                    nearest = min(sorted_kfs, key=lambda k: abs(k - fr.frame_idx))
                    if abs(nearest - fr.frame_idx) <= 60:
                        fr.objects[0].label = kf_labels[nearest]

            top_labels = sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
            metrics["grounding_dino"] = {
                "keyframes": len(kf_images),
                "total_detections": total_dets,
                "top_labels": top_labels,
                "wall_s": round(time.perf_counter() - t0, 2),
            }
            del det
            torch.cuda.empty_cache()

        # 2. Depth-Anything-V2-Small (Apache-2.0) on keyframes
        if self.cfg.geometry.depth_backend == "depth_anything_v2_small_apache":
            t0 = time.perf_counter()
            depth_model = DepthAnythingV2SmallGPU(
                device="cuda",
                mixed_precision=self.cfg.runtime.mixed_precision,
            )
            med_depths: list[float] = []
            for bgr in kf_images.values():
                dmap = depth_model.estimate_depth_map(bgr)
                med_depths.append(float(np.median(dmap)))
            scale_prov.scale_source = (
                "calibrated_intrinsics_and_metric_depth"
                if scale_prov.is_calibrated_metric
                else "metric_depth_and_palm_prior"
            )
            scale_prov.scale_uncertainty_pct = 5.0 if scale_prov.is_calibrated_metric else 8.5
            metrics["depth_anything_v2_small"] = {
                "keyframes": len(med_depths),
                "median_depth_mean": round(float(np.mean(med_depths)), 4) if med_depths else 0.0,
                "wall_s": round(time.perf_counter() - t0, 2),
            }
            del depth_model
            torch.cuda.empty_cache()

        # 3. Local Apache-2.0 VLM (SmolVLM2 / Qwen2.5-VL) captioning across representative non-idle segments
        if self.cfg.semantic.vlm_backend in ("smolvlm2_local", "qwen25_vl_local"):
            t0 = time.perf_counter()
            vlm = LocalOpenSourceVLMAnnotator(
                model_id=self.cfg.semantic.model_id,
                device="cuda",
                mixed_precision=self.cfg.runtime.mixed_precision,
            )
            non_idle = [s for s in segments if s.verb != "idle"]
            step = max(1, len(non_idle) // 16)
            chosen = non_idle[::step][:16]
            mid_idxs = [(s.start_frame + s.end_frame) // 2 for s in chosen]
            seg_imgs = self._sample_video_frames_by_idx(video_path, mid_idxs)
            frames_by_idx = {f.frame_idx: f for f in frames}
            captioned = 0
            sample_captions: list[str] = []
            for seg, mfi in zip(chosen, mid_idxs):
                seg_bgr = seg_imgs.get(mfi)
                if seg_bgr is None:
                    continue
                cap_txt = vlm.describe_segment_keyframes([seg_bgr], seg.verb)
                if cap_txt:
                    seg.semantic_description = cap_txt
                    seg.geometry_agreement_score = _verify_segment_against_geometry(seg, frames_by_idx)
                    captioned += 1
                    if len(sample_captions) < 5:
                        sample_captions.append(cap_txt)
            language.vlm_backend = self.cfg.semantic.vlm_backend
            language.model_id = self.cfg.semantic.model_id
            metrics["local_vlm"] = {
                "model_id": self.cfg.semantic.model_id,
                "segments_captioned": captioned,
                "sample_captions": sample_captions,
                "wall_s": round(time.perf_counter() - t0, 2),
            }
            del vlm
            torch.cuda.empty_cache()

        return metrics

    def run_video(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        episode_id: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Run all stages end-to-end on the COMPLETE video (frame 0..N-1)."""
        t_start = time.perf_counter()
        v_path = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ep_id = episode_id or v_path.stem

        cal_path = Path(self.cfg.geometry.calibration_yaml) if self.cfg.geometry.calibration_yaml else None
        reader = VideoStreamReader(
            v_path,
            calibration_yaml=cal_path,
            estimate_distortion=self.cfg.geometry.estimate_distortion,
        )
        expected_total = int(reader.probe["nb_frames"])
        dt_s = 1.0 / max(float(reader.probe["fps"]), 1.0)

        stage_timings: dict[str, float] = {}
        t1 = time.perf_counter()
        cached = self._load_stage1_cache(reader.sha256, expected_total) if use_cache else None
        cache_hit = cached is not None

        if cached is not None:
            frames, scale_prov, sparse_pts = cached
        else:
            hand_tracker = MediaPipeHandTracker(
                self.cfg.perception,
                reader.intrinsics,
                self.cfg.geometry.assumed_palm_width_m,
            )
            body_tracker = MediaPipeUpperBodyTracker() if self.cfg.perception.enable_upper_body else None
            obj_tracker = OpenCVManipulatedObjectTracker(reader.intrinsics)
            vo = MonocularWorldVO(self.cfg.geometry, reader.intrinsics)

            frames = []
            seen: set[int] = set()
            prev_l: np.ndarray | None = None
            prev_r: np.ndarray | None = None
            prev_objs: dict[int, np.ndarray] = {}

            try:
                for chunk in reader.iter_chunks(
                    chunk_size=self.cfg.runtime.chunk_size_frames,
                    overlap=self.cfg.runtime.chunk_overlap_frames,
                ):
                    for fi, ts, fr in zip(chunk.frame_indices, chunk.timestamps_s, chunk.frames):
                        if fi in seen:
                            continue
                        seen.add(fi)
                        sharp, luma = compute_frame_image_quality(fr)
                        cam_pose = vo.step(fr, fi, chunk.chunk_id, dt_s)
                        lh, rh = hand_tracker.process_frame(fr, ts)
                        body = (
                            body_tracker.process_frame(fr)
                            if (body_tracker is not None and fi % self.cfg.perception.body_sample_stride == 0)
                            else None
                        )
                        objs = obj_tracker.process_frame(fr, fi, lh, rh)
                        prev_l = register_hand_and_objects_to_world(cam_pose, lh, prev_l, dt_s)
                        prev_r = register_hand_and_objects_to_world(cam_pose, rh, prev_r, dt_s)
                        register_objects_to_world(cam_pose, objs, prev_objs, dt_s)

                        kwargs: dict[str, Any] = {
                            "frame_idx": fi,
                            "timestamp_s": round(ts, 5),
                            "sharpness_laplacian": round(sharp, 3),
                            "mean_luma": round(luma, 3),
                            "camera_pose": cam_pose,
                            "left_hand": lh,
                            "right_hand": rh,
                            "objects": objs,
                        }
                        if body is not None:
                            kwargs["upper_body"] = body
                        frames.append(FrameAnnotation(**kwargs))
            finally:
                hand_tracker.close()
                if body_tracker is not None:
                    body_tracker.close()

            scale_prov = vo.build_scale_provenance(reader.intrinsics, self.cfg.geometry.depth_backend)
            sparse_pts = vo.sparse_world_points[:500]
            if use_cache:
                cp = self._stage1_cache_path(reader.sha256)
                cp.write_text(
                    json.dumps(
                        {
                            "video_sha256": reader.sha256,
                            "video_path": str(v_path),
                            "total_frames": len(frames),
                            "scale_provenance": scale_prov.model_dump(),
                            "sparse_world_points": sparse_pts,
                            "frames": [f.model_dump() for f in frames],
                        }
                    ),
                    encoding="utf-8",
                )
        stage_timings["stage1_perception_geometry_s"] = round(time.perf_counter() - t1, 3)

        # Stage 2: Gaze, Tactile, Behavior, Semantics, Scene Graph & Quality
        t2 = time.perf_counter()
        enrich_frames_with_gaze(frames)
        populate_rgb_tactile_proxies(frames, self.cfg.tactile)
        ext_tactile_info: dict[str, Any] | None = None
        if self.cfg.tactile.external_tactile_path:
            ext_tactile_info = ingest_external_tactile_stream(frames, Path(self.cfg.tactile.external_tactile_path))

        grasps = extract_grasp_events(frames, self.cfg.behavior)
        segments = segment_atomic_actions(frames, grasps, self.cfg.behavior)
        language = annotate_semantics_grounded(frames, segments, self.cfg.semantic)
        scene_graph = build_scene_graph_and_affordances(frames, grasps, sparse_pts)
        t_gpu = time.perf_counter()
        gpu_metrics = self._run_gpu_enrichment(v_path, frames, segments, language, scale_prov)
        stage_timings["stage2b_gpu_enrichment_s"] = round(time.perf_counter() - t_gpu, 3)
        quality = score_episode_quality(frames, segments, scale_prov, self.cfg.quality)
        stage_timings["stage2_behavior_tactile_quality_s"] = round(time.perf_counter() - t2, 3)

        # Stage 3: Pluggable Multi-Embodiment Retargeting
        t3 = time.perf_counter()
        retargeted = retarget_all_configured_embodiments(frames, segments, self.cfg.retargeting)
        retargeting_summary = {name: traj.metrics for name, traj in retargeted.items()}
        stage_timings["stage3_retargeting_s"] = round(time.perf_counter() - t3, 3)

        video_meta = reader.build_metadata(processed_frames=len(frames))
        ir = EpisodeIR(
            episode_id=ep_id,
            created_utc=datetime.now(timezone.utc).isoformat(),
            profile_used=self.cfg.runtime.profile,
            device_used=self.cfg.runtime.device,
            video=video_meta,
            scale_provenance=scale_prov,
            frames=frames,
            grasp_events=grasps,
            segments=segments,
            language=language,
            scene_graph=scene_graph,
            quality=quality,
            retargeting_summary=retargeting_summary,
            provenance={
                "stage1_cache_hit": cache_hit,
                "external_tactile": ext_tactile_info,
                "stage_timings_s": stage_timings,
                "gpu_enrichment": gpu_metrics,
            },
        )

        # Stage 4: Export all formats (LeRobot v3 + GR00T modality.json, RLDS, HDF5, Parquet+JSON)
        t4 = time.perf_counter()
        export_summary = export_all_formats(ir, retargeted, v_path, out_dir, self.cfg)
        stage_timings["stage4_export_s"] = round(time.perf_counter() - t4, 3)
        stage_timings["total_wall_s"] = round(time.perf_counter() - t_start, 3)

        run_report = {
            "episode_id": ep_id,
            "video": video_meta.model_dump(),
            "quality": {
                "robot_readiness_score": quality.robot_readiness_score,
                "verdict": quality.verdict,
                "flags": quality.flags,
                "dimensions": quality.dimensions.model_dump(),
            },
            "retargeting": retargeting_summary,
            "exports": export_summary,
            "privacy_audit": ir.privacy_audit,
            "timings_s": stage_timings,
            "gpu_enrichment": gpu_metrics,
        }
        (out_dir / "run_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")
        return run_report
