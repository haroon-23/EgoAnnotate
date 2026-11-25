"""Calibrated 10-dimension data quality scoring & PASS/WARN/QUARANTINE gating per segment and episode."""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.egoannotate.config import QualityConfig
from src.egoannotate.schemas import (
    AtomicActionSegment,
    EpisodeQualityReport,
    FrameAnnotation,
    QualityDimensionScores,
    ScaleProvenance,
    SegmentQualityReport,
)

DIMENSION_WEIGHTS = {
    "hand_visibility": 0.14,
    "pose_tracking_confidence": 0.13,
    "motion_blur_quality": 0.10,
    "slam_stability": 0.12,
    "scale_confidence": 0.08,
    "occlusion_score": 0.09,
    "lighting_quality": 0.07,
    "task_coverage": 0.08,
    "semantic_geometry_agreement": 0.09,
    "trajectory_plausibility": 0.10,
}


def _score_frame_slice(
    sub_frames: list[FrameAnnotation],
    scale_prov: ScaleProvenance,
    cfg: QualityConfig,
    semantic_agreement: float,
    task_coverage_val: float,
) -> tuple[QualityDimensionScores, float, list[str]]:
    flags: list[str] = []
    if not sub_frames:
        zero = QualityDimensionScores(
            hand_visibility=0.0,
            pose_tracking_confidence=0.0,
            motion_blur_quality=0.0,
            slam_stability=0.0,
            scale_confidence=0.0,
            occlusion_score=0.0,
            lighting_quality=0.0,
            task_coverage=0.0,
            semantic_geometry_agreement=0.0,
            trajectory_plausibility=0.0,
        )
        return zero, 0.0, ["empty_frame_slice"]

    hand_present = [1.0 if (f.left_hand.detected or f.right_hand.detected) else 0.0 for f in sub_frames]
    hand_vis = float(np.mean(hand_present))
    if hand_vis < cfg.min_hand_visibility:
        flags.append("low_hand_visibility")

    confs = []
    untrunc = []
    for f in sub_frames:
        for h in (f.left_hand, f.right_hand):
            if h.detected:
                confs.append(h.confidence)
                untrunc.append(0.0 if h.truncated else 1.0)
    pose_conf = float(np.mean(confs)) if confs else 0.0
    occlusion_score = float(np.mean(untrunc)) if untrunc else 0.0
    if occlusion_score < 0.70 and confs:
        flags.append("frequent_hand_truncation_or_occlusion")

    sharp_vals = np.array([f.sharpness_laplacian for f in sub_frames], dtype=np.float64)
    med_sharp = float(np.median(sharp_vals))
    blur_score = float(np.clip(med_sharp / max(cfg.min_sharpness_laplacian * 4.0, 1.0), 0.0, 1.0))
    if med_sharp < cfg.min_sharpness_laplacian:
        flags.append("severe_motion_blur")

    luma_vals = np.array([f.mean_luma for f in sub_frames], dtype=np.float64)
    med_luma = float(np.median(luma_vals))
    if med_luma < cfg.min_exposure_luma:
        lighting_score = float(np.clip(med_luma / max(cfg.min_exposure_luma, 1.0), 0.0, 0.6))
        flags.append("underexposed_low_light")
    elif med_luma > cfg.max_exposure_luma:
        lighting_score = float(np.clip((255.0 - med_luma) / max(255.0 - cfg.max_exposure_luma, 1.0), 0.0, 0.6))
        flags.append("overexposed_clipping")
    else:
        lighting_score = 1.0

    vo_ok_ratio = float(np.mean([1.0 if f.camera_pose.tracking_ok else 0.0 for f in sub_frames]))
    cam_speeds = [float(np.linalg.norm(f.camera_pose.lin_vel_world_mps)) for f in sub_frames]
    cam_spike_frac = float(np.mean([1.0 if s > 2.5 else 0.0 for s in cam_speeds]))
    slam_stability = float(np.clip(vo_ok_ratio * (1.0 - 0.5 * cam_spike_frac), 0.0, 1.0))
    if vo_ok_ratio < 0.80:
        flags.append("unstable_camera_tracking")

    scale_conf = float(np.clip(1.0 - (scale_prov.scale_uncertainty_pct / 50.0), 0.2, 1.0))
    if not scale_prov.is_calibrated_metric:
        flags.append("monocular_scale_uncalibrated_prior")

    wrist_speeds: list[float] = []
    for f in sub_frames:
        if f.left_hand.detected:
            wrist_speeds.append(float(np.linalg.norm(f.left_hand.wrist_vel_world_mps)))
        if f.right_hand.detected:
            wrist_speeds.append(float(np.linalg.norm(f.right_hand.wrist_vel_world_mps)))
    if wrist_speeds:
        spike_rate = float(np.mean([1.0 if v > 3.0 else 0.0 for v in wrist_speeds]))
        traj_plaus = float(np.clip(1.0 - 2.0 * spike_rate, 0.0, 1.0))
        if spike_rate > 0.10:
            flags.append("wrist_velocity_spikes")
    else:
        traj_plaus = 0.2

    dims = QualityDimensionScores(
        hand_visibility=round(hand_vis, 4),
        pose_tracking_confidence=round(pose_conf, 4),
        motion_blur_quality=round(blur_score, 4),
        slam_stability=round(slam_stability, 4),
        scale_confidence=round(scale_conf, 4),
        occlusion_score=round(occlusion_score, 4),
        lighting_quality=round(lighting_score, 4),
        task_coverage=round(float(np.clip(task_coverage_val, 0.0, 1.0)), 4),
        semantic_geometry_agreement=round(float(np.clip(semantic_agreement, 0.0, 1.0)), 4),
        trajectory_plausibility=round(traj_plaus, 4),
    )

    agg = sum(getattr(dims, k) * w for k, w in DIMENSION_WEIGHTS.items())
    if hand_vis < 0.05 or med_luma < 10.0 or med_sharp < 5.0:
        agg = min(agg, 0.35)
    return dims, round(float(np.clip(agg, 0.0, 1.0)), 4), flags


def _verdict_for_score(score: float, cfg: QualityConfig) -> Literal["PASS", "WARN", "QUARANTINE"]:
    if score >= cfg.warn_overall_score:
        return "PASS"
    if score >= cfg.min_overall_score:
        return "WARN"
    return "QUARANTINE"


def score_episode_quality(
    frames: list[FrameAnnotation],
    segments: list[AtomicActionSegment],
    scale_prov: ScaleProvenance,
    cfg: QualityConfig,
) -> EpisodeQualityReport:
    """Compute calibrated per-segment and episode-level 10-dimension quality scores."""
    frames_by_idx = {f.frame_idx: f for f in frames}
    seg_reports: list[SegmentQualityReport] = []

    for seg in segments:
        sub = [frames_by_idx[i] for i in range(seg.start_frame, seg.end_frame + 1) if i in frames_by_idx]
        cov = 1.0 if seg.verb != "idle" else 0.4
        dims, score, flags = _score_frame_slice(sub, scale_prov, cfg, seg.geometry_agreement_score, cov)
        seg_reports.append(
            SegmentQualityReport(
                segment_id=seg.segment_id,
                start_s=seg.start_s,
                end_s=seg.end_s,
                dimensions=dims,
                robot_readiness_score=score,
                verdict=_verdict_for_score(score, cfg),
                flags=flags,
            )
        )

    non_idle_frames = sum((s.end_frame - s.start_frame + 1) for s in segments if s.verb != "idle")
    ep_cov = float(np.clip(non_idle_frames / max(len(frames), 1), 0.0, 1.0))
    mean_sem = float(np.mean([s.geometry_agreement_score for s in segments])) if segments else 0.5

    ep_dims, ep_score, ep_flags = _score_frame_slice(frames, scale_prov, cfg, mean_sem, ep_cov)
    return EpisodeQualityReport(
        dimensions=ep_dims,
        robot_readiness_score=ep_score,
        verdict=_verdict_for_score(ep_score, cfg),
        flags=ep_flags,
        segment_reports=seg_reports,
    )
