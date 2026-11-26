"""Canonical Pydantic v2 schemas for the Embodiment-Agnostic Intermediate Representation (IR)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CameraIntrinsicsModel(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0.0)
    fy: float = Field(gt=0.0)
    cx: float
    cy: float
    dist_coeffs: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    fov_horizontal_deg: float = Field(gt=10.0, lt=179.0)
    lens_type: Literal["pinhole", "wide_fov", "fisheye"] = "pinhole"
    source: Literal["calibrated_yaml", "exif_metadata", "auto_estimated"] = "auto_estimated"
    uncertainty_pct: float = Field(default=15.0, ge=0.0, le=100.0)


class VideoMetadata(BaseModel):
    video_path: str
    video_sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_fps: float = Field(gt=0.0)
    processing_fps: float = Field(gt=0.0)
    total_frames: int = Field(gt=0)
    processed_frames: int = Field(gt=0)
    duration_s: float = Field(gt=0.0)
    codec: str = "unknown"
    has_audio_source: bool = False
    vfr_detected: bool = False
    is_full_video_processed: bool = True
    intrinsics: CameraIntrinsicsModel


class HandState(BaseModel):
    handedness: Literal["left", "right"]
    detected: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    visibility: float = Field(default=0.0, ge=0.0, le=1.0)
    truncated: bool = False
    landmarks_2d_px: list[list[float]] = Field(default_factory=list)
    landmarks_3d_cam_m: list[list[float]] = Field(default_factory=list)
    landmarks_3d_world_m: list[list[float]] = Field(default_factory=list)
    wrist_pos_cam_m: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    wrist_pos_world_m: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    wrist_quat_xyzw_world: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    wrist_rot6d_world: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    wrist_vel_world_mps: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    fingertips_world_m: list[list[float]] = Field(default_factory=list)
    gripper_aperture_m: float = Field(default=0.0, ge=0.0)
    finger_curls_5d: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    mask_area_frac: float = Field(default=0.0, ge=0.0, le=1.0)


class UpperBodyState(BaseModel):
    detected: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    left_shoulder_cam: list[float] | None = None
    left_elbow_cam: list[float] | None = None
    left_wrist_cam: list[float] | None = None
    right_shoulder_cam: list[float] | None = None
    right_elbow_cam: list[float] | None = None
    right_wrist_cam: list[float] | None = None
    torso_pitch_deg: float | None = None


class ManipulatedObjectState(BaseModel):
    track_id: int
    label: str
    score: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: list[float]
    mask_area_frac: float = Field(default=0.0, ge=0.0, le=1.0)
    pos_world_m: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    vel_world_mps: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    in_hand: Literal["none", "left", "right", "both"] = "none"
    object_state: str = "stationary"
    state_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    role: Literal[
        "held_object",
        "target_object",
        "tool",
        "container",
        "workpiece",
        "unknown",
    ] = "unknown"
    mask_iou_with_hand: float = Field(default=0.0, ge=0.0, le=1.0)
    mask_polygon_norm: list[list[float]] = Field(default_factory=list)


class CameraWorldPose(BaseModel):
    pos_world_m: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    quat_xyzw_world: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    lin_vel_world_mps: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    ang_vel_world_radps: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    tracking_ok: bool = True
    inlier_count: int = Field(default=0, ge=0)
    chunk_id: int = Field(default=0, ge=0)
    loop_closure_adjusted: bool = False


class TactileHandChannel(BaseModel):
    source: Literal["derived_rgb_proxy", "hardware_sensor"] = "derived_rgb_proxy"
    contact_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    contact_zones_6d: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        description="[thumb, index, middle, ring, pinky, palm] estimated contact probability",
    )
    normal_force_proxy_01: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized [0,1] force proxy from aperture deformation (NOT physical Newtons unless source=hardware_sensor).",
    )
    slip_proxy_01: float = Field(default=0.0, ge=0.0, le=1.0)
    impact_proxy_01: float = Field(default=0.0, ge=0.0, le=1.0)
    compliance_prior: Literal["rigid", "semi_rigid", "deformable", "unknown"] = "unknown"
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    raw_sensor_values: list[float] | None = None


class FrameAnnotation(BaseModel):
    frame_idx: int = Field(ge=0)
    timestamp_s: float = Field(ge=0.0)
    sharpness_laplacian: float = Field(default=0.0, ge=0.0)
    mean_luma: float = Field(default=0.0, ge=0.0, le=255.0)
    camera_pose: CameraWorldPose = Field(default_factory=CameraWorldPose)
    left_hand: HandState = Field(default_factory=lambda: HandState(handedness="left"))
    right_hand: HandState = Field(default_factory=lambda: HandState(handedness="right"))
    upper_body: UpperBodyState = Field(default_factory=UpperBodyState)
    objects: list[ManipulatedObjectState] = Field(default_factory=list)
    tactile_left: TactileHandChannel = Field(default_factory=TactileHandChannel)
    tactile_right: TactileHandChannel = Field(default_factory=TactileHandChannel)
    gaze_world_dir: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0])
    proprio_valid: bool = True
    canonical_delta_action: list[float] = Field(default_factory=list)
    contact_event_tags: list[str] = Field(default_factory=list)
    depth_surface_normal_mean: list[float] = Field(default_factory=list)


class GraspEvent(BaseModel):
    hand: Literal["left", "right"]
    onset_frame: int
    release_frame: int
    onset_s: float
    release_s: float
    feix_taxonomy: str
    coarse_grasp: Literal["power", "precision", "pinch", "lateral", "non_prehensile", "unknown"]
    target_object: str = "manipulated_object"
    contact_points_world_m: list[list[float]] = Field(default_factory=list)
    stability_proxy: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class AtomicActionSegment(BaseModel):
    segment_id: int
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    verb: str
    acting_hand: Literal["left", "right", "bimanual", "none"]
    target_objects: list[str] = Field(default_factory=list)
    bimanual_coordination: Literal[
        "unimanual_left",
        "unimanual_right",
        "symmetric_cooperative",
        "asymmetric_support",
        "handover",
        "independent_parallel",
        "idle",
    ] = "idle"
    grasp_type: str = "none"
    manipulation_phase: Literal["approach", "contact_grasp", "transport_manipulate", "place_release", "idle"] = "idle"
    object_state_before: str = "unknown"
    object_state_after: str = "unknown"
    approach_direction_world: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    peak_hand_speed_mps: float = 0.0
    gaze_alignment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_description: str = ""
    vla_paraphrases: list[str] = Field(default_factory=list)
    geometry_agreement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    outcome_tag: Literal["success", "retry", "slip_recovery", "idle"] = "success"
    subtask_id: int = 0


class HierarchicalLanguage(BaseModel):
    task_goal: str
    subtasks: list[dict[str, Any]] = Field(default_factory=list)
    vla_instruction_paraphrases: list[str] = Field(default_factory=list)
    vlm_backend: str
    model_id: str
    grounding_verified: bool = True
    environment_tag: str = "general_workspace"
    domain_taxonomy: str = "general_manipulation"
    operator_ehs_metadata: dict[str, Any] = Field(default_factory=dict)
    discovered_objects: list[str] = Field(default_factory=list)


class ScaleProvenance(BaseModel):
    scale_meters_per_unit: float = Field(gt=0.0)
    scale_source: Literal[
        "calibrated_intrinsics_and_metric_depth",
        "metric_depth_and_palm_prior",
        "anthropometric_palm_prior_only",
    ] = "anthropometric_palm_prior_only"
    scale_uncertainty_pct: float = Field(ge=0.0, le=100.0)
    is_calibrated_metric: bool = False
    gravity_vector_cam0: list[float] = Field(default_factory=lambda: [0.0, 1.0, 0.0])
    table_plane_z_world_m: float | None = None
    depth_provenance: Literal[
        "estimated_metric_monocular",
        "calibrated_stereo_imu",
        "anthropometric_palm_prior",
    ] = "anthropometric_palm_prior"
    estimated_intrinsics_3x3: list[list[float]] = Field(default_factory=list)


class SceneGraphSummary(BaseModel):
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    workspace_bounds_world_m: dict[str, list[float]] = Field(default_factory=dict)
    affordance_hotspots_world_m: list[dict[str, Any]] = Field(default_factory=list)
    point_cloud_sample_world_m: list[list[float]] = Field(default_factory=list)


class QualityDimensionScores(BaseModel):
    hand_visibility: float = Field(ge=0.0, le=1.0)
    pose_tracking_confidence: float = Field(ge=0.0, le=1.0)
    motion_blur_quality: float = Field(ge=0.0, le=1.0)
    slam_stability: float = Field(ge=0.0, le=1.0)
    scale_confidence: float = Field(ge=0.0, le=1.0)
    occlusion_score: float = Field(ge=0.0, le=1.0)
    lighting_quality: float = Field(ge=0.0, le=1.0)
    task_coverage: float = Field(ge=0.0, le=1.0)
    semantic_geometry_agreement: float = Field(ge=0.0, le=1.0)
    trajectory_plausibility: float = Field(ge=0.0, le=1.0)


class SegmentQualityReport(BaseModel):
    segment_id: int
    start_s: float
    end_s: float
    dimensions: QualityDimensionScores
    robot_readiness_score: float = Field(ge=0.0, le=1.0)
    verdict: Literal["PASS", "WARN", "QUARANTINE"]
    flags: list[str] = Field(default_factory=list)


class EpisodeQualityReport(BaseModel):
    dimensions: QualityDimensionScores
    robot_readiness_score: float = Field(ge=0.0, le=1.0)
    verdict: Literal["PASS", "WARN", "QUARANTINE"]
    flags: list[str] = Field(default_factory=list)
    segment_reports: list[SegmentQualityReport] = Field(default_factory=list)


class EpisodeIR(BaseModel):
    schema_version: str = "3.0.0"
    episode_id: str
    created_utc: str
    profile_used: str
    device_used: str
    video: VideoMetadata
    scale_provenance: ScaleProvenance
    frames: list[FrameAnnotation]
    grasp_events: list[GraspEvent] = Field(default_factory=list)
    segments: list[AtomicActionSegment] = Field(default_factory=list)
    language: HierarchicalLanguage
    scene_graph: SceneGraphSummary = Field(default_factory=SceneGraphSummary)
    quality: EpisodeQualityReport
    retargeting_summary: dict[str, Any] = Field(default_factory=dict)
    privacy_audit: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
