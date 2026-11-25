"""Phase 1A verification: dual-profile configs, license firewall, schemas, and full-video streaming reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.egoannotate.config import load_config
from src.egoannotate.registry import CommercialLicenseError, PluginRegistry
from src.egoannotate.video_io import VideoStreamReader, compute_frame_image_quality


def test_load_both_profiles_and_commercial_defaults() -> None:
    cpu_cfg = load_config(profile="cpu_smoke")
    gpu_cfg = load_config(profile="gpu_prod")
    assert cpu_cfg.runtime.process_full_video is True
    assert gpu_cfg.runtime.process_full_video is True
    assert cpu_cfg.runtime.allow_non_commercial_plugins is False
    assert gpu_cfg.runtime.allow_non_commercial_plugins is False
    assert gpu_cfg.geometry.depth_backend == "depth_anything_v2_small_apache"
    assert gpu_cfg.semantic.vlm_backend in ("smolvlm2_local", "qwen25_vl_local")


@pytest.mark.parametrize(
    "override",
    [
        {"semantic": {"vlm_backend": "gemini_pro"}},
        {"geometry": {"depth_backend": "unidepth_v2"}},
        {"geometry": {"depth_backend": "depth-anything-v2-large"}},
        {"perception": {"hand_backend": "hamer_mano"}},
        {"geometry": {"vo_backend": "mast3r_slam"}},
    ],
)
def test_commercial_license_firewall_blocks_restricted_models(override: dict) -> None:
    with pytest.raises(CommercialLicenseError):
        load_config(profile="cpu_smoke", overrides=override)


def test_plugin_registry_blocks_non_commercial_and_paid_backends() -> None:
    reg = PluginRegistry()
    reg.register("depth", "dav2_small", lambda: "ok", code_license="Apache-2.0", weights_license="Apache-2.0")
    reg.register("depth", "unidepth", lambda: "bad", code_license="Apache-2.0", weights_license="CC-BY-NC-4.0")
    reg.register("vlm", "paid_cloud", lambda: "bad", code_license="MIT", is_paid_api=True)

    assert reg.create("depth", "dav2_small") == "ok"
    with pytest.raises(CommercialLicenseError):
        reg.create("depth", "unidepth")
    assert reg.create("depth", "unidepth", allow_non_commercial=True) == "bad"
    with pytest.raises(CommercialLicenseError):
        reg.create("vlm", "paid_cloud")


def test_full_real_drive_video_streaming_and_intrinsics() -> None:
    sample_mp4 = Path("data/sample_raw/ad521ca5-0017-475b-9dd8-1fdb423a66b7.mp4")
    if not sample_mp4.exists():
        pytest.skip("Real Drive sample video not present locally")

    reader = VideoStreamReader(sample_mp4)
    total_streamed = 0
    chunk_count = 0
    last_idx = -1
    first_sharp, first_luma = 0.0, 0.0

    for chunk in reader.iter_chunks(chunk_size=120, overlap=0):
        chunk_count += 1
        assert chunk.frames
        if chunk_count == 1:
            first_sharp, first_luma = compute_frame_image_quality(chunk.frames[0])
        assert chunk.start_frame == last_idx + 1
        last_idx = chunk.end_frame
        total_streamed += len(chunk.frames)

    meta = reader.build_metadata(processed_frames=total_streamed)
    assert meta.is_full_video_processed is True
    assert total_streamed > 100
    assert meta.intrinsics.fx > 0 and meta.intrinsics.fy > 0
    assert first_sharp > 0.0 and 0.0 < first_luma < 255.0
    print(
        f"\n[REAL FULL VIDEO STREAMED] {sample_mp4.name}: "
        f"{meta.width}x{meta.height} @ {meta.source_fps:.2f} fps, "
        f"frames={total_streamed}/{meta.total_frames}, duration={meta.duration_s:.2f}s, "
        f"chunks={chunk_count}, lens={meta.intrinsics.lens_type} ({meta.intrinsics.fov_horizontal_deg:.1f} deg)"
    )
