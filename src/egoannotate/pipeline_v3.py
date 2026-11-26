"""Unified full-video EgoAnnotate v4 orchestrator (30Hz CFR, RTMW, MoGe-2, Qwen3-VL, SAM2.1, batched IK)."""

from __future__ import annotations

import json
import os
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
from src.egoannotate.perception_hands_rtmw import (
    RTMWWholeBodyHandTracker,
    interpolate_gloved_hand_dropouts,
)
from src.egoannotate.perception_objects import OpenCVManipulatedObjectTracker
from src.egoannotate.quality import score_episode_quality
from src.egoannotate.retargeting_v3 import retarget_all_configured_embodiments
from src.egoannotate.robot_contracts import (
    annotate_contact_and_tactile_events,
    enforce_gap_free_hierarchical_segments,
    populate_canonical_delta_actions,
)
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


def _hand_bbox_mask(fr: FrameAnnotation, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for hand in (fr.left_hand, fr.right_hand):
        if not hand.detected or not hand.landmarks_2d_px:
            continue
        pts = np.asarray(hand.landmarks_2d_px, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            continue
        x1 = max(0, min(width - 1, round(float(np.min(pts[:, 0])) - 16.0)))
        y1 = max(0, min(height - 1, round(float(np.min(pts[:, 1])) - 16.0)))
        x2 = max(x1 + 1, min(width, round(float(np.max(pts[:, 0])) + 16.0)))
        y2 = max(y1 + 1, min(height, round(float(np.max(pts[:, 1])) + 16.0)))
        mask[y1:y2, x1:x2] = True
    return mask


class EgoAnnotateV3Pipeline:
    """End-to-end commercial egocentric video-to-VLA pipeline with full-video processing & stage caching."""

    def __init__(self, cfg: PipelineConfig | None = None, profile: str = "auto") -> None:
        self.cfg = cfg if cfg is not None else load_config(profile=profile)

    def _stage1_cache_path(self, video_sha256: str, target_fps_tag: str = "") -> Path:
        cache_dir = Path(self.cfg.runtime.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{target_fps_tag}" if target_fps_tag else ""
        return cache_dir / f"stage1_{video_sha256[:16]}{suffix}.json"

    def _load_stage1_cache(
        self,
        video_sha256: str,
        expected_total: int,
        target_fps_tag: str = "",
    ) -> tuple[list[FrameAnnotation], ScaleProvenance, list[list[float]]] | None:
        candidates = [
            self._stage1_cache_path(video_sha256, target_fps_tag),
            self._stage1_cache_path(video_sha256),
            Path(self.cfg.runtime.cache_dir) / "phase1b_full_video_cache.json",
        ]
        for cp in candidates:
            if not cp.exists():
                continue
            try:
                raw = json.loads(cp.read_text(encoding="utf-8"))
                total_cached = int(raw.get("total_frames", 0))
                if raw.get("video_sha256") == video_sha256 and abs(total_cached - expected_total) <= 3:
                    frames = [FrameAnnotation.model_validate(f) for f in raw["frames"]]
                    scale_prov = ScaleProvenance.model_validate(raw["scale_provenance"])
                    sparse_pts = raw.get("sparse_world_points", [])
                    return frames, scale_prov, sparse_pts
            except (OSError, ValueError, KeyError):
                continue
        return None

    def _sample_video_frames_by_idx(
        self,
        video_path: Path,
        frame_indices: list[int],
        fps_ratio: float = 1.0,
    ) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return out
        total_src = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        try:
            for fi in sorted(set(frame_indices)):
                src_fi = max(0, min(total_src - 1, round(float(fi) * fps_ratio)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_fi)
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
        fps_ratio: float = 1.0,
    ) -> dict[str, Any]:
        """Run cached, batched CUDA Qwen3-VL, Grounding DINO, SAM 2.1, and MoGe-2/DA3Metric."""
        use_vllm_remote = self.cfg.semantic.vlm_backend == "qwen3_vl_vllm" and bool(
            self.cfg.semantic.vllm_base_url
        )
        cuda_ok = self.cfg.runtime.device == "cuda" and torch.cuda.is_available()
        if not cuda_ok and not use_vllm_remote:
            return {"enabled": False, "device": self.cfg.runtime.device}

        from src.egoannotate.gpu_backends import (
            DepthAnythingV2SmallGPU,
            GroundingDINOGPUDetector,
            LocalOpenSourceVLMAnnotator,
        )
        from src.egoannotate.gpu_backends_v4 import (
            DA3MetricLargeGPU,
            MoGe2MetricDepthGPU,
            Qwen3VLFastAnnotator,
            SAM21MaskPropagatorGPU,
            lift_box_to_3d_centroid,
        )

        metrics: dict[str, Any] = {
            "enabled": True,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_ok else "remote_vllm",
            "mixed_precision": self.cfg.runtime.mixed_precision,
        }
        n = len(frames)
        if n == 0:
            return metrics

        kf_indices = list(range(0, n, max(1, n // 120)))
        kf_images = self._sample_video_frames_by_idx(video_path, kf_indices, fps_ratio=fps_ratio)
        kf_items = sorted(kf_images.items(), key=lambda kv: kv[0])
        kf_keys = [fi for fi, _ in kf_items]
        kf_bgrs = [bgr for _, bgr in kf_items]
        frames_by_idx = {f.frame_idx: f for f in frames}

        batch_scale = max(0.25, float(os.environ.get("EGOANNOTATE_BATCH_SCALE", "1.0")))
        dino_bs = max(1, round(min(8, self.cfg.perception.gpu_batch_size) * batch_scale))
        depth_bs = max(1, round(self.cfg.perception.gpu_batch_size * batch_scale))

        # Pass 1: Qwen3-VL Scene Discovery + Dynamic Open-Vocab Object Inventory
        active_prompts = list(self.cfg.perception.open_vocab_prompts)
        discovered_roles: dict[str, str] = {}
        qwen_annotator: Qwen3VLFastAnnotator | None = None
        if self.cfg.semantic.vlm_backend in ("qwen3_vl_local", "qwen3_vl_vllm"):
            t0 = time.perf_counter()
            vllm_url = (
                self.cfg.semantic.vllm_base_url
                if self.cfg.semantic.vlm_backend == "qwen3_vl_vllm"
                else None
            )
            qwen_annotator = Qwen3VLFastAnnotator(
                model_id=self.cfg.semantic.model_id,
                device="cuda" if cuda_ok else "cpu",
                mixed_precision=self.cfg.runtime.mixed_precision,
                vllm_base_url=vllm_url,
                vllm_api_key=self.cfg.semantic.vllm_api_key,
            )
            stride_disc = max(1, len(kf_bgrs) // 4)
            disc_frames = kf_bgrs[::stride_disc][:4]
            scene_info = qwen_annotator.discover_scene_and_objects(disc_frames)
            if scene_info.get("task_goal"):
                language.task_goal = str(scene_info["task_goal"])
            language.environment_tag = str(
                scene_info.get("environment_tag") or language.environment_tag
            )
            language.domain_taxonomy = str(
                scene_info.get("domain_taxonomy") or language.domain_taxonomy
            )
            language.operator_ehs_metadata = dict(scene_info.get("operator_ehs_metadata") or {})
            disc_labels: list[str] = []
            for obj in scene_info.get("objects", []):
                lbl = str(obj["label"]).strip().lower()
                if lbl and lbl not in disc_labels:
                    disc_labels.append(lbl)
                if lbl and lbl not in active_prompts:
                    active_prompts.append(lbl)
                if lbl:
                    discovered_roles[lbl.replace(" ", "_")] = str(obj.get("role", "workpiece"))
            language.discovered_objects = disc_labels
            metrics["qwen3_scene_discovery"] = {
                "environment_tag": language.environment_tag,
                "domain_taxonomy": language.domain_taxonomy,
                "discovered_objects": disc_labels,
                "wall_s": round(time.perf_counter() - t0, 2),
            }

        # 1. Batched Grounding DINO Detection on Keyframes (with VLM-discovered prompts merged)
        kf_labels: dict[int, str] = {}
        kf_boxes: dict[int, list[float]] = {}
        if cuda_ok and "grounding_dino" in str(self.cfg.perception.detector_backend):
            t0 = time.perf_counter()
            det = GroundingDINOGPUDetector(
                prompts=active_prompts,
                device="cuda",
                mixed_precision=self.cfg.runtime.mixed_precision,
            )
            label_counts: dict[str, int] = {}
            total_dets = 0
            batch_dets = det.detect_batch(kf_bgrs, batch_size=dino_bs)
            for fi, dets in zip(kf_keys, batch_dets):
                total_dets += len(dets)
                if dets:
                    best = max(dets, key=lambda d: float(d["score"]))
                    lbl = str(best["label"]).strip().replace(" ", "_") or "manipulated_object"
                    kf_labels[fi] = lbl
                    kf_boxes[fi] = [float(v) for v in best["bbox_xyxy"]]
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

            if kf_labels:
                sorted_kfs = sorted(kf_labels.keys())
                for fr in frames:
                    if not fr.objects:
                        continue
                    nearest = min(sorted_kfs, key=lambda k: abs(k - fr.frame_idx))
                    if abs(nearest - fr.frame_idx) <= 60:
                        lbl = kf_labels[nearest]
                        fr.objects[0].label = lbl
                        role_val = discovered_roles.get(lbl, "workpiece")
                        if role_val in (
                            "held_object",
                            "target_object",
                            "tool",
                            "container",
                            "workpiece",
                        ):
                            fr.objects[0].role = role_val  # type: ignore[assignment]

            top_labels = sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
            metrics["grounding_dino"] = {
                "keyframes": len(kf_images),
                "batch_size": dino_bs,
                "total_detections": total_dets,
                "top_labels": top_labels,
                "wall_s": round(time.perf_counter() - t0, 2),
            }

        # 2. SAM 2.1 Pixel Masks + Hand-Object Mask IoU
        if cuda_ok and self.cfg.perception.segmenter_backend == "sam21_apache" and kf_boxes:
            t0 = time.perf_counter()
            sam21 = SAM21MaskPropagatorGPU(
                device="cuda",
                mixed_precision=self.cfg.runtime.mixed_precision,
            )
            kf_mask_info: dict[int, tuple[float, float, list[list[float]]]] = {}
            for fi, box in kf_boxes.items():
                bgr = kf_images.get(fi)
                fr_opt = frames_by_idx.get(fi)
                if bgr is None or fr_opt is None:
                    continue
                h_im, w_im = bgr.shape[:2]
                seg_res = sam21.segment_boxes(bgr, [box])
                if not seg_res:
                    continue
                m_obj = seg_res[0]["mask"]
                h_mask = _hand_bbox_mask(fr_opt, w_im, h_im)
                iou_val = sam21.compute_mask_iou(m_obj, h_mask)
                kf_mask_info[fi] = (
                    float(seg_res[0]["area_frac"]),
                    float(iou_val),
                    list(seg_res[0]["polygon_norm"]),
                )
            if kf_mask_info:
                sorted_mkfs = sorted(kf_mask_info.keys())
                for fr in frames:
                    if not fr.objects:
                        continue
                    nearest = min(sorted_mkfs, key=lambda k: abs(k - fr.frame_idx))
                    if abs(nearest - fr.frame_idx) <= 45:
                        area_f, iou_v, poly = kf_mask_info[nearest]
                        fr.objects[0].mask_area_frac = area_f
                        fr.objects[0].mask_iou_with_hand = iou_v
                        fr.objects[0].mask_polygon_norm = poly
            metrics["sam21"] = {
                "available": sam21.available,
                "keyframes_segmented": len(kf_mask_info),
                "wall_s": round(time.perf_counter() - t0, 2),
            }

        # 3. Metric 3D Depth, Surface Normals, Intrinsics & 3D Object Centroid Lifting
        depth_backend = str(self.cfg.geometry.depth_backend)
        if cuda_ok and depth_backend in (
            "moge2_mit",
            "da3_metric_large_apache",
            "depth_anything_v2_small_apache",
        ):
            t0 = time.perf_counter()
            if depth_backend == "moge2_mit":
                moge_runner = MoGe2MetricDepthGPU(
                    device="cuda",
                    mixed_precision=self.cfg.runtime.mixed_precision,
                )
                depth_outs = moge_runner.infer_batch(kf_bgrs, batch_size=depth_bs)
                actual_backend = moge_runner.backend_used
            elif depth_backend == "da3_metric_large_apache":
                da3_runner = DA3MetricLargeGPU(
                    device="cuda",
                    mixed_precision=self.cfg.runtime.mixed_precision,
                )
                depth_outs = da3_runner.infer_batch(kf_bgrs, batch_size=depth_bs)
                actual_backend = da3_runner.backend_used
            else:
                dav2 = DepthAnythingV2SmallGPU(
                    device="cuda",
                    mixed_precision=self.cfg.runtime.mixed_precision,
                )
                dmaps = dav2.estimate_depth_batch(kf_bgrs, batch_size=depth_bs)
                depth_outs = [
                    {
                        "depth_m": dm,
                        "mean_normal": [0.0, 0.0, 1.0],
                        "intrinsics_3x3": np.array(
                            [
                                [0.85 * max(b.shape[:2]), 0.0, b.shape[1] * 0.5],
                                [0.0, 0.85 * max(b.shape[:2]), b.shape[0] * 0.5],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=np.float32,
                        ),
                        "median_depth_m": round(float(np.median(dm)), 4),
                        "is_metric": False,
                        "backend": "depth_anything_v2_small_apache",
                    }
                    for b, dm in zip(kf_bgrs, dmaps)
                ]
                actual_backend = "depth_anything_v2_small_apache"

            med_depths = [float(o["median_depth_m"]) for o in depth_outs]
            is_metric = any(bool(o.get("is_metric", False)) for o in depth_outs)
            if depth_outs:
                k_mean = np.mean([o["intrinsics_3x3"] for o in depth_outs], axis=0)
                scale_prov.estimated_intrinsics_3x3 = [
                    [round(float(v), 3) for v in row] for row in k_mean.tolist()
                ]
            kf_normals: dict[int, list[float]] = {}
            for fi, bgr, dout in zip(kf_keys, kf_bgrs, depth_outs):
                kf_normals[fi] = [float(v) for v in dout["mean_normal"]]
                fr_opt = frames_by_idx.get(fi)
                box_opt = kf_boxes.get(fi)
                if fr_opt is not None and fr_opt.objects and box_opt is not None:
                    h_im, w_im = bgr.shape[:2]
                    xyz_cam, _ = lift_box_to_3d_centroid(
                        box_opt,
                        dout["depth_m"],
                        dout["intrinsics_3x3"],
                        w_im,
                        h_im,
                    )
                    fr_opt.objects[0].pos_world_m = xyz_cam

            if kf_normals:
                sorted_nkfs = sorted(kf_normals.keys())
                for fr in frames:
                    nearest = min(sorted_nkfs, key=lambda k: abs(k - fr.frame_idx))
                    fr.depth_surface_normal_mean = kf_normals[nearest]

            scale_prov.scale_source = (
                "calibrated_intrinsics_and_metric_depth"
                if scale_prov.is_calibrated_metric
                else "metric_depth_and_palm_prior"
            )
            scale_prov.depth_provenance = (
                "calibrated_stereo_imu"
                if scale_prov.is_calibrated_metric
                else "estimated_metric_monocular"
            )
            scale_prov.scale_uncertainty_pct = (
                4.5 if scale_prov.is_calibrated_metric else (6.5 if is_metric else 8.5)
            )
            depth_metric_entry = {
                "backend_used": actual_backend,
                "is_metric": is_metric,
                "keyframes": len(med_depths),
                "batch_size": depth_bs,
                "median_depth_mean": round(float(np.mean(med_depths)), 4) if med_depths else 0.0,
                "wall_s": round(time.perf_counter() - t0, 2),
            }
            metrics["metric_depth"] = depth_metric_entry
            metrics["depth_anything_v2_small"] = depth_metric_entry

        # 4. Local/vLLM Apache-2.0 VLM Segment Captioning
        if self.cfg.semantic.vlm_backend in (
            "smolvlm2_local",
            "qwen25_vl_local",
            "qwen3_vl_local",
            "qwen3_vl_vllm",
        ):
            t0 = time.perf_counter()
            vlm_runner: Any = (
                qwen_annotator
                if qwen_annotator is not None
                else LocalOpenSourceVLMAnnotator(
                    model_id=self.cfg.semantic.model_id,
                    device="cuda",
                    mixed_precision=self.cfg.runtime.mixed_precision,
                )
            )
            non_idle = [s for s in segments if s.verb != "idle"]
            step = max(1, len(non_idle) // 16)
            chosen = non_idle[::step][:16]
            mid_idxs = [(s.start_frame + s.end_frame) // 2 for s in chosen]
            seg_imgs = self._sample_video_frames_by_idx(
                video_path,
                mid_idxs,
                fps_ratio=fps_ratio,
            )
            captioned = 0
            sample_captions: list[str] = []
            for seg, mfi in zip(chosen, mid_idxs):
                seg_bgr = seg_imgs.get(mfi)
                if seg_bgr is None:
                    continue
                cap_txt = vlm_runner.describe_segment_keyframes([seg_bgr], seg.verb)
                if cap_txt:
                    seg.semantic_description = cap_txt
                    seg.geometry_agreement_score = _verify_segment_against_geometry(
                        seg, frames_by_idx
                    )
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

        cal_path = (
            Path(self.cfg.geometry.calibration_yaml)
            if self.cfg.geometry.calibration_yaml
            else None
        )
        reader = VideoStreamReader(
            v_path,
            calibration_yaml=cal_path,
            estimate_distortion=self.cfg.geometry.estimate_distortion,
            target_fps=self.cfg.runtime.target_fps,
            normalize_fps_cfr=self.cfg.runtime.normalize_fps_cfr,
            use_nvdec=self.cfg.runtime.use_nvdec,
        )
        fps_ratio = float(reader.source_fps) / max(float(reader.processing_fps), 1e-6)
        if self.cfg.runtime.normalize_fps_cfr and abs(reader.source_fps - reader.processing_fps) > 1e-3:
            expected_total = max(1, round(float(reader.probe["duration_s"]) * float(reader.processing_fps)))
        else:
            expected_total = int(reader.probe["nb_frames"])
        dt_s = 1.0 / max(float(reader.processing_fps), 1.0)
        fps_tag = f"fps{round(reader.processing_fps)}"

        stage_timings: dict[str, float] = {}
        t1 = time.perf_counter()
        cached = (
            self._load_stage1_cache(reader.sha256, expected_total, target_fps_tag=fps_tag)
            if use_cache
            else None
        )
        cache_hit = cached is not None
        gloved_filled = 0
        hand_backend_used = self.cfg.perception.hand_backend

        if cached is not None:
            frames, scale_prov, sparse_pts = cached
        else:
            use_rtmw = self.cfg.perception.hand_backend == "rtmw_133_apache"
            rtmw_tracker: RTMWWholeBodyHandTracker | None = (
                RTMWWholeBodyHandTracker(
                    self.cfg.perception,
                    reader.intrinsics,
                    self.cfg.geometry.assumed_palm_width_m,
                    device=self.cfg.runtime.device,
                )
                if use_rtmw
                else None
            )
            if rtmw_tracker is not None:
                hand_backend_used = rtmw_tracker.backend_used
            hand_tracker = (
                None
                if use_rtmw
                else MediaPipeHandTracker(
                    self.cfg.perception,
                    reader.intrinsics,
                    self.cfg.geometry.assumed_palm_width_m,
                )
            )
            body_tracker = (
                None
                if use_rtmw
                else (
                    MediaPipeUpperBodyTracker()
                    if self.cfg.perception.enable_upper_body
                    else None
                )
            )
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
                        if rtmw_tracker is not None:
                            lh, rh, body = rtmw_tracker.process_frame(fr, ts)
                        else:
                            assert hand_tracker is not None
                            lh, rh = hand_tracker.process_frame(fr, ts)
                            body = (
                                body_tracker.process_frame(fr)
                                if (
                                    body_tracker is not None
                                    and fi % self.cfg.perception.body_sample_stride == 0
                                )
                                else None
                            )
                        objs = obj_tracker.process_frame(fr, fi, lh, rh)
                        prev_l = register_hand_and_objects_to_world(cam_pose, lh, prev_l, dt_s)
                        prev_r = register_hand_and_objects_to_world(cam_pose, rh, prev_r, dt_s)
                        register_objects_to_world(cam_pose, objs, prev_objs, dt_s)

                        kwargs: dict[str, Any] = {
                            "frame_idx": fi,
                            "timestamp_s": round(ts, 6),
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
                if rtmw_tracker is not None:
                    rtmw_tracker.close()
                if hand_tracker is not None:
                    hand_tracker.close()
                if body_tracker is not None:
                    body_tracker.close()

            gloved_filled = interpolate_gloved_hand_dropouts(frames, max_gap_frames=18)
            scale_prov = vo.build_scale_provenance(
                reader.intrinsics,
                self.cfg.geometry.depth_backend,
            )
            sparse_pts = vo.sparse_world_points[:500]
            if use_cache:
                cp = self._stage1_cache_path(reader.sha256, target_fps_tag=fps_tag)
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

        # Stage 2: Gaze, Tactile, Behavior, Semantics, GPU Enrichment, Canonical Contracts & Quality
        t2 = time.perf_counter()
        enrich_frames_with_gaze(frames)
        populate_rgb_tactile_proxies(frames, self.cfg.tactile)
        ext_tactile_info: dict[str, Any] | None = None
        if self.cfg.tactile.external_tactile_path:
            ext_tactile_info = ingest_external_tactile_stream(
                frames,
                Path(self.cfg.tactile.external_tactile_path),
            )

        grasps = extract_grasp_events(frames, self.cfg.behavior)
        segments = segment_atomic_actions(frames, grasps, self.cfg.behavior)
        language = annotate_semantics_grounded(frames, segments, self.cfg.semantic)
        scene_graph = build_scene_graph_and_affordances(frames, grasps, sparse_pts)

        t_gpu = time.perf_counter()
        gpu_metrics = self._run_gpu_enrichment(
            v_path,
            frames,
            segments,
            language,
            scale_prov,
            fps_ratio=fps_ratio,
        )
        stage_timings["stage2b_gpu_enrichment_s"] = round(time.perf_counter() - t_gpu, 3)

        if self.cfg.geometry.depth_provenance != "auto":
            scale_prov.depth_provenance = self.cfg.geometry.depth_provenance
        elif scale_prov.is_calibrated_metric:
            scale_prov.depth_provenance = "calibrated_stereo_imu"

        contact_counts = annotate_contact_and_tactile_events(frames)
        populate_canonical_delta_actions(frames)
        segments = enforce_gap_free_hierarchical_segments(frames, segments, language)

        quality = score_episode_quality(frames, segments, scale_prov, self.cfg.quality)
        stage_timings["stage2_behavior_tactile_quality_s"] = round(time.perf_counter() - t2, 3)

        # Stage 3: Vectorized Batched GPU/CPU Multi-Embodiment Retargeting
        t3 = time.perf_counter()
        retargeted = retarget_all_configured_embodiments(frames, segments, self.cfg.retargeting)
        retargeting_summary = {name: traj.metrics for name, traj in retargeted.items()}
        stage_timings["stage3_retargeting_s"] = round(time.perf_counter() - t3, 3)

        video_meta = reader.build_metadata(processed_frames=len(frames))
        ir = EpisodeIR(
            schema_version="4.0.0",
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
                "hand_backend_used": hand_backend_used,
                "gloved_hand_dropouts_filled": gloved_filled,
                "contact_events": contact_counts,
                "depth_provenance": scale_prov.depth_provenance,
                "external_tactile": ext_tactile_info,
                "stage_timings_s": stage_timings,
                "gpu_enrichment": gpu_metrics,
            },
        )

        # Stage 4: Export all formats
        t4 = time.perf_counter()
        export_summary = export_all_formats(ir, retargeted, v_path, out_dir, self.cfg)
        stage_timings["stage4_export_s"] = round(time.perf_counter() - t4, 3)
        stage_timings["total_wall_s"] = round(time.perf_counter() - t_start, 3)

        run_report = {
            "episode_id": ep_id,
            "schema_version": ir.schema_version,
            "video": video_meta.model_dump(),
            "depth_provenance": scale_prov.depth_provenance,
            "hand_backend_used": hand_backend_used,
            "gloved_hand_dropouts_filled": gloved_filled,
            "contact_events": contact_counts,
            "language": {
                "task_goal": language.task_goal,
                "environment_tag": language.environment_tag,
                "domain_taxonomy": language.domain_taxonomy,
                "subtasks_count": len(language.subtasks),
                "segments_count": len(segments),
                "discovered_objects": language.discovered_objects,
            },
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
