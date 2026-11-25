"""Clean top-level Python API for embedding EgoAnnotate v3 inside customer robotics pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.egoannotate.config import PipelineConfig, load_config
from src.egoannotate.observability import (
    export_interactive_3d_viewer_html,
    generate_html_qa_report,
    render_full_qa_overlay_video,
)
from src.egoannotate.pipeline_v3 import EgoAnnotateV3Pipeline
from src.egoannotate.policy_train_v3 import run_policy_training_smoke
from src.egoannotate.schemas import EpisodeIR


def annotate_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    profile: str = "auto",
    config_path: str | Path | None = None,
    external_tactile_path: str | Path | None = None,
    calibration_yaml: str | Path | None = None,
    episode_id: str | None = None,
    generate_qa_artifacts: bool = True,
) -> dict[str, Any]:
    """Run EgoAnnotate v3 end-to-end on a complete video and optionally render QA HTML, 3D viewer & overlay."""
    overrides: dict[str, Any] = {}
    if external_tactile_path is not None:
        overrides.setdefault("tactile", {})["external_tactile_path"] = str(external_tactile_path)
    if calibration_yaml is not None:
        overrides.setdefault("geometry", {})["calibration_yaml"] = str(calibration_yaml)

    cfg: PipelineConfig = load_config(
        config_path=Path(config_path) if config_path else None,
        profile=profile,
        overrides=overrides or None,
    )
    pipe = EgoAnnotateV3Pipeline(cfg=cfg)
    out_p = Path(output_dir)
    report = pipe.run_video(Path(video_path), out_p, episode_id=episode_id)

    if generate_qa_artifacts:
        report["qa_artifacts"] = export_qa_artifacts(out_p, render_overlay=cfg.export.render_qa_overlay)
    return report


def export_qa_artifacts(run_dir: str | Path, *, render_overlay: bool = True) -> dict[str, Any]:
    """Generate qa_report.html, interactive offline viewer_3d.html, and full-video qa_overlay.mp4 from a run directory."""
    r_dir = Path(run_dir)
    ir_path = r_dir / "parquet_json" / "episode_ir.json"
    if not ir_path.exists():
        raise FileNotFoundError(f"Episode IR not found: {ir_path}")
    ir = EpisodeIR.model_validate_json(ir_path.read_text(encoding="utf-8"))

    out: dict[str, Any] = {}
    qa_html = generate_html_qa_report(ir, r_dir / "qa_report.html")
    out["qa_report_html"] = str(qa_html)
    viewer_html = export_interactive_3d_viewer_html(ir, r_dir / "viewer_3d.html")
    out["viewer_3d_html"] = str(viewer_html)

    sanitized_mp4 = r_dir / "lerobot_v3" / "videos" / "observation.images.ego" / "chunk-000" / "file-000.mp4"
    if render_overlay and sanitized_mp4.exists():
        out["overlay"] = render_full_qa_overlay_video(ir, sanitized_mp4, r_dir / "qa_overlay.mp4")
    return out


def verify_policy_training(lerobot_dir: str | Path, epochs: int = 6) -> dict[str, Any]:
    """Run CPU PyTorch policy-training smoke check over an exported LeRobot v3 directory."""
    return run_policy_training_smoke(Path(lerobot_dir), epochs=epochs)
