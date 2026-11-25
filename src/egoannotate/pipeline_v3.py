"""Unified full-video EgoAnnotate v3 orchestrator (frames 0..N-1, resumable stage cache, zero paid APIs)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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
from src.egoannotate.schemas import EpisodeIR, FrameAnnotation, ScaleProvenance
from src.egoannotate.semantics import annotate_semantics_grounded
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
        }
        (out_dir / "run_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")
        return run_report
