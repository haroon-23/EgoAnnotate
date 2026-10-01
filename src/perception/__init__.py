"""Phase-C perception subpackage: 2D detection, metric depth, SAM 2 masks, PnP pose.

Heavy third-party backends (Grounding DINO, UniDepth ONNX, SAM 2) are imported
**lazily** via :func:`__getattr__` (PEP 562), so ``import src.perception`` never
raises when optional dependencies are missing. Each backend exposes
``is_available()`` and the factories return ``None`` (or an explicit disabled
object) instead of raising when a backend cannot be built.

Box convention everywhere in this subpackage is **normalized [0, 1] xyxy**,
matching :class:`src.datatypes.ObjectAnnotation`.
"""

from __future__ import annotations

import importlib
from typing import Any

# attribute -> defining module (imported lazily on first access)
_LAZY_EXPORTS = {
    # --- 2D detection (src/perception/detector.py) ---
    "Detection": "src.perception.detector",
    "Detector2D": "src.perception.detector",
    "Detector2DConfig": "src.perception.detector",
    "OwlViTDetector": "src.perception.detector",
    "GroundingDinoDetector": "src.perception.detector",
    "LocateAnythingDetector": "src.perception.detector",
    "create_detector_2d": "src.perception.detector",
    "bbox_to_location_description": "src.perception.detector",
    # NOTE: "GroundingDINODetector" here is the REAL Grounding DINO
    # (IDEA-Research) backend. The legacy OWL-ViT class of the same name lives
    # at src.grounding_detector.GroundingDINODetector (back-compat shim).
    "GroundingDINODetector": "src.perception.grounding_dino_detector",
    # --- metric depth (src/perception/depth.py) ---
    "DepthEstimator": "src.perception.depth",
    "DepthConfig": "src.perception.depth",
    "UniDepthEstimator": "src.perception.depth",
    "DisabledDepth": "src.perception.depth",
    "create_depth_estimator": "src.perception.depth",
    "localize_objects_3d": "src.perception.depth",
    # --- SAM 2 masks (src/perception/sam2_segmenter.py) ---
    "Sam2Segmenter": "src.perception.sam2_segmenter",
    "Sam2Config": "src.perception.sam2_segmenter",
    "create_sam2_segmenter": "src.perception.sam2_segmenter",
    "encode_mask_rle": "src.perception.sam2_segmenter",
    "decode_mask_rle": "src.perception.sam2_segmenter",
    # --- PnP pose refinement (src/perception/pnp_pose.py) ---
    "CameraIntrinsics": "src.perception.pnp_pose",
    "PnPResult": "src.perception.pnp_pose",
    "solve_pnp_pose": "src.perception.pnp_pose",
    "backproject_to_3d": "src.perception.pnp_pose",
    "backproject_pixel": "src.perception.pnp_pose",
    "HandPnPRefiner": "src.perception.pnp_pose",
    "refine_poses_pnp": "src.perception.pnp_pose",
}


def __getattr__(name: str) -> Any:
    """Lazily import perception backends on first attribute access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, name)
    except AttributeError as e:
        raise AttributeError(
            f"module {module_name!r} has no attribute {name!r}"
        ) from e


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = sorted(_LAZY_EXPORTS)
