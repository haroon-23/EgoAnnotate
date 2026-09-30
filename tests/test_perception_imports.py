"""Regression test: `import src.perception` must never raise for missing heavy deps.

Simulates a minimal environment by blocking groundingdino / onnxruntime / sam2 /
transformers via sys.modules, then asserts every lazy export resolves and every
factory degrades gracefully (None / unavailable) instead of raising.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_BLOCKED_TOP = ("groundingdino", "onnxruntime", "sam2", "transformers")


@pytest.fixture
def minimal_env(monkeypatch):
    """Block heavy third-party deps and force fresh perception imports."""
    for name in list(sys.modules):
        top = name.split(".")[0]
        if name == "src.perception" or name.startswith("src.perception."):
            monkeypatch.delitem(sys.modules, name, raising=False)
        elif top in _BLOCKED_TOP:
            monkeypatch.delitem(sys.modules, name, raising=False)
    for top in _BLOCKED_TOP:
        monkeypatch.setitem(sys.modules, top, None)
    yield


def test_import_package_never_raises(minimal_env):
    import src.perception  # noqa: F401  -- must not raise


def test_lazy_exports_resolve(minimal_env):
    from src.perception import (  # noqa: F401
        Detection,
        Detector2D,
        Detector2DConfig,
        OwlViTDetector,
        GroundingDinoDetector,
        GroundingDINODetector,
        create_detector_2d,
        bbox_to_location_description,
        DepthEstimator,
        DepthConfig,
        UniDepthEstimator,
        DisabledDepth,
        create_depth_estimator,
        localize_objects_3d,
        Sam2Segmenter,
        Sam2Config,
        create_sam2_segmenter,
        encode_mask_rle,
        decode_mask_rle,
        CameraIntrinsics,
        PnPResult,
        solve_pnp_pose,
        backproject_to_3d,
        backproject_pixel,
        HandPnPRefiner,
        refine_poses_pnp,
    )


def test_factories_degrade_gracefully(minimal_env):
    from src.perception import (
        Detector2DConfig,
        DepthConfig,
        Sam2Config,
        create_detector_2d,
        create_depth_estimator,
        create_sam2_segmenter,
    )

    # No heavy deps -> factories return None / unavailable objects, never raise.
    assert create_detector_2d(Detector2DConfig(backend="owlvit")) is None
    assert create_detector_2d(Detector2DConfig(backend="grounding_dino")) is None

    depth = create_depth_estimator(DepthConfig(backend="unidepth"))
    assert depth.is_available() is False
    assert create_depth_estimator(DepthConfig(backend="none")).is_available() is False

    assert create_sam2_segmenter(Sam2Config(enabled=True)) is None
    assert create_sam2_segmenter(Sam2Config(enabled=False)) is None


def test_unknown_backends_fail_loud(minimal_env):
    from src.perception import (
        Detector2DConfig,
        DepthConfig,
        create_detector_2d,
        create_depth_estimator,
    )

    with pytest.raises(ValueError):
        create_detector_2d(Detector2DConfig(backend="yolo_nas"))
    with pytest.raises(ValueError):
        create_depth_estimator(DepthConfig(backend="midas"))
