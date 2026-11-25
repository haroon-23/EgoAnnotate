"""Strictly typed Pydantic v2 configuration for CPU smoke and CUDA GPU production profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from src.egoannotate.registry import CommercialLicenseError

DISALLOWED_COMMERCIAL_TOKENS = (
    "gemini",
    "openai",
    "anthropic",
    "unidepth",
    "depth-anything-v2-base",
    "depth-anything-v2-large",
    "depth-anything-v2-giant",
    "mano",
    "hamer",
    "wilor",
    "mast3r",
    "dust3r",
    "vggt",
    "yolo-world",
    "sam3",
)


class RuntimeConfig(BaseModel):
    profile: Literal["auto", "cpu_smoke", "gpu_prod"] = "auto"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    mixed_precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    num_workers: int = Field(default=2, ge=1, le=64)
    chunk_size_frames: int = Field(default=300, ge=30, le=3600)
    chunk_overlap_frames: int = Field(default=15, ge=0, le=120)
    target_fps: float = Field(default=30.0, gt=1.0, le=120.0)
    process_full_video: bool = Field(
        default=True,
        description="Must remain True for full-length video certification (frames 0..N-1).",
    )
    cache_dir: str = ".ego_cache"
    offline_mode: bool = False
    allow_non_commercial_plugins: bool = False


class PerceptionConfig(BaseModel):
    hand_backend: str = "mediapipe_hands"
    body_backend: str = "mediapipe_pose"
    detector_backend: str = "opencv_motion_roi"
    segmenter_backend: str = "opencv_roi_mask"
    hand_min_detection_conf: float = Field(default=0.35, ge=0.05, le=0.99)
    hand_min_tracking_conf: float = Field(default=0.45, ge=0.05, le=0.99)
    max_num_hands: int = Field(default=2, ge=1, le=2)
    enable_upper_body: bool = True
    body_sample_stride: int = Field(default=1, ge=1, le=10)
    smoothing_method: Literal["one_euro", "savgol", "none"] = "one_euro"
    one_euro_min_cutoff: float = Field(default=1.5, gt=0.0)
    one_euro_beta: float = Field(default=0.05, ge=0.0)
    open_vocab_prompts: list[str] = Field(
        default_factory=lambda: [
            "bottle",
            "cup",
            "mug",
            "bowl",
            "box",
            "tool",
            "container",
            "cloth",
            "utensil",
            "lid",
            "object",
        ]
    )


class GeometryConfig(BaseModel):
    vo_backend: str = "opencv_essential_vo"
    depth_backend: str = "none"
    auto_estimate_intrinsics: bool = True
    estimate_distortion: bool = True
    assumed_palm_width_m: float = Field(default=0.085, gt=0.05, lt=0.12)
    gravity_align: bool = True
    loop_closure_enabled: bool = True
    vo_max_features: int = Field(default=600, ge=100, le=4000)
    vo_resize_width: int = Field(default=640, ge=160, le=1920)
    depth_sample_stride: int = Field(default=5, ge=1, le=60)
    calibration_yaml: str | None = None


class BehaviorConfig(BaseModel):
    min_segment_duration_s: float = Field(default=0.35, gt=0.05, le=10.0)
    merge_gap_s: float = Field(default=0.25, ge=0.0, le=5.0)
    contact_proximity_m: float = Field(default=0.09, gt=0.01, lt=0.5)
    grasp_pinch_aperture_m: float = Field(default=0.045, gt=0.005, lt=0.15)
    grasp_power_aperture_m: float = Field(default=0.11, gt=0.03, lt=0.25)
    compute_scene_graph: bool = True
    compute_affordances: bool = True


class TactileConfig(BaseModel):
    enable_rgb_proxies: bool = True
    external_tactile_path: str | None = None
    slip_velocity_threshold_mps: float = Field(default=0.12, gt=0.01)
    impact_accel_threshold_mps2: float = Field(default=4.5, gt=0.5)


class SemanticConfig(BaseModel):
    vlm_backend: str = "geometric_grounded"
    model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    max_keyframes_per_segment: int = Field(default=4, ge=1, le=16)
    generate_paraphrases: bool = True
    verify_against_geometry: bool = True


class QualityConfig(BaseModel):
    min_overall_score: float = Field(default=0.65, ge=0.0, le=1.0)
    warn_overall_score: float = Field(default=0.80, ge=0.0, le=1.0)
    min_hand_visibility: float = Field(default=0.30, ge=0.0, le=1.0)
    min_sharpness_laplacian: float = Field(default=25.0, ge=0.0)
    min_exposure_luma: float = Field(default=28.0, ge=0.0, le=255.0)
    max_exposure_luma: float = Field(default=235.0, ge=0.0, le=255.0)


class RetargetingConfig(BaseModel):
    enabled: bool = True
    embodiments: list[str] = Field(
        default_factory=lambda: [
            "franka_panda",
            "dual_arm_franka",
            "unitree_g1_upper",
            "allegro_hand_right",
            "teleop_glove",
        ]
    )
    embodiment_config_dir: str = "configs/embodiments"
    enforce_joint_limits: bool = True
    enforce_table_clearance: bool = True
    table_z_margin_m: float = Field(default=0.01, ge=0.0, le=0.2)
    smoothness_window: int = Field(default=7, ge=3, le=31)


ExportFormat = Literal["lerobot_v3", "rlds", "hdf5", "parquet_json"]


def _default_export_formats() -> list[ExportFormat]:
    return ["lerobot_v3", "rlds", "hdf5", "parquet_json"]


class ExportConfig(BaseModel):
    formats: list[ExportFormat] = Field(default_factory=_default_export_formats)
    include_groot_modality_json: bool = True
    video_crf: int = Field(default=24, ge=14, le=35)
    render_qa_overlay: bool = True
    export_3d_viewer: bool = True
    generate_html_report: bool = True


class PrivacyConfig(BaseModel):
    enable_face_blur: bool = True
    strip_audio: bool = True
    verify_zero_unblurred_faces: bool = True
    blur_sigma: float = Field(default=28.0, ge=5.0, le=80.0)


class PipelineConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    tactile: TactileConfig = Field(default_factory=TactileConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    retargeting: RetargetingConfig = Field(default_factory=RetargetingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)

    @model_validator(mode="after")
    def _enforce_commercial_license_firewall(self) -> PipelineConfig:
        if self.runtime.allow_non_commercial_plugins:
            return self
        fields_to_check = [
            ("perception.hand_backend", self.perception.hand_backend),
            ("perception.body_backend", self.perception.body_backend),
            ("perception.detector_backend", self.perception.detector_backend),
            ("perception.segmenter_backend", self.perception.segmenter_backend),
            ("geometry.vo_backend", self.geometry.vo_backend),
            ("geometry.depth_backend", self.geometry.depth_backend),
            ("semantic.vlm_backend", self.semantic.vlm_backend),
            ("semantic.model_id", self.semantic.model_id),
        ]
        for field_name, val in fields_to_check:
            low = str(val).lower()
            for banned in DISALLOWED_COMMERCIAL_TOKENS:
                if banned in low:
                    raise CommercialLicenseError(
                        f"Commercial license firewall blocked '{field_name}={val}' "
                        f"(matched restricted token '{banned}'). Use Apache-2.0/MIT/BSD defaults "
                        f"or explicitly set runtime.allow_non_commercial_plugins=True for R&D only."
                    )
        return self


def has_cuda_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError, OSError):
        return False


def resolve_profile_path(
    profile: str = "auto",
    profiles_dir: Path = Path("configs/profiles"),
) -> Path:
    if profile == "auto":
        chosen = "gpu_prod" if has_cuda_gpu() else "cpu_smoke"
    else:
        chosen = profile
    path = profiles_dir / f"{chosen}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile config not found: {path}")
    return path


def load_config(
    config_path: Path | None = None,
    profile: str = "auto",
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    path = Path(config_path) if config_path is not None else resolve_profile_path(profile)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(raw.get(k), dict):
                raw[k].update(v)
            else:
                raw[k] = v
    cfg = PipelineConfig.model_validate(raw)
    if cfg.runtime.device == "auto":
        cfg.runtime.device = "cuda" if has_cuda_gpu() else "cpu"
    return cfg
