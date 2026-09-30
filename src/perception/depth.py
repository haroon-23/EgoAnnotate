"""Swappable metric-depth backends.

Backends
--------
* ``"unidepth"`` — :class:`UniDepthEstimator`, monocular metric depth via an
  ONNX export of UniDepth. Real implementation (mm -> m); guarded import, lazy
  session, no auto-download.
* ``"none"`` — :class:`DisabledDepth`, an explicit no-op documenting "no depth"
  as a configuration choice rather than a broken backend.

The old ``project_to_3d`` 60-degree-FOV approximation was removed in Phase C;
use :func:`src.perception.pnp_pose.backproject_to_3d` (real intrinsics) or
:func:`localize_objects_3d` below.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .detector import Detection
from .pnp_pose import CameraIntrinsics, backproject_pixel

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort

    _ORT_IMPORT_OK = True
except ImportError:  # pragma: no cover - depends on installed env
    ort = None  # type: ignore[assignment]
    _ORT_IMPORT_OK = False
    logger.debug("onnxruntime not installed — UniDepth backend unavailable.")

# No auto-download anywhere: missing weights fail loud with this URL.
UNIDEPTH_DOWNLOAD_URL = "https://huggingface.co/ibaiGorordo/ONNX-UniDepth-V1"


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


@dataclass
class DepthConfig:
    backend: str = "unidepth"  # "unidepth" | "none"
    model_path: str = "models/unidepth_v1.onnx"
    keyframe_only: bool = True  # estimate on detection keyframes, cache to disk


class DepthEstimator:
    """Base class: ``estimate_depth`` returns float32 meters, shape (H, W)."""

    backend_name = "base"

    def __init__(self, config: Optional[DepthConfig] = None):
        self.config = config or DepthConfig()
        self._warned: dict[str, bool] = {}

    def _warn_once(self, key: str, msg: str, *args) -> None:
        if not self._warned.get(key):
            logger.warning(msg, *args)
            self._warned[key] = True

    def is_available(self) -> bool:
        raise NotImplementedError

    def estimate_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        """Estimate metric depth (meters), shape ``(H, W)``, float32."""
        raise NotImplementedError


class DisabledDepth(DepthEstimator):
    """Explicit no-op backend: documents 'no depth' vs 'broken depth'."""

    backend_name = "none"

    def is_available(self) -> bool:
        return False

    def estimate_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "Depth backend is 'none': no depth estimation is configured. "
            "Set 'depth.backend: unidepth' (and provide models/unidepth_v1.onnx) "
            "to enable metric depth."
        )


# ---------------------------------------------------------------------------
# UniDepth (ONNX)
# ---------------------------------------------------------------------------


class UniDepthEstimator(DepthEstimator):
    """Monocular metric depth via the UniDepth ONNX export.

    Lazy: importing this module and constructing the class never touch the
    model file. The ONNX session builds on first :meth:`is_available` /
    :meth:`estimate_depth`. Missing weights → ``is_available()`` False with a
    loud message naming the exact download URL (no auto-download).
    """

    backend_name = "unidepth"

    def __init__(self, config: Optional[DepthConfig] = None):
        super().__init__(config)
        self._session = None
        self._input_name: Optional[str] = None
        self._input_hw: Optional[tuple[int, int]] = None

    def _ensure_session(self) -> bool:
        if self._session is not None:
            return True
        if not _ORT_IMPORT_OK:
            self._warn_once(
                "ort-missing",
                "UniDepth backend requested but 'onnxruntime' is not installed "
                "(pip install onnxruntime>=1.17.0). Continuing without depth.",
            )
            return False
        model_path = self.config.model_path
        if not model_path or not os.path.exists(model_path):
            self._warn_once(
                "unidepth-weights",
                "UniDepth ONNX weights not found at '%s'. Download manually "
                "(no auto-download): %s — then set 'depth.model_path' in "
                "configs/default.yaml. Continuing without depth.",
                model_path,
                UNIDEPTH_DOWNLOAD_URL,
            )
            return False
        try:
            logger.info("Loading UniDepth ONNX session: %s", model_path)
            # Providers default: CPUExecutionProvider (CPU-only mandate).
            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            model_input = self._session.get_inputs()[0]
            self._input_name = model_input.name
            shape = model_input.shape  # e.g. [1, 3, 448, 448]
            try:
                self._input_hw = (int(shape[2]), int(shape[3]))
            except (TypeError, ValueError, IndexError):
                self._input_hw = (448, 448)
            logger.info(
                "UniDepth ONNX session ready (input %s, %dx%d)",
                self._input_name,
                self._input_hw[1],
                self._input_hw[0],
            )
        except Exception as e:
            logger.error("Failed to create UniDepth ONNX session: %s", e)
            self._session = None
        return self._session is not None

    def is_available(self) -> bool:
        return self._ensure_session()

    def estimate_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self._ensure_session():
            raise RuntimeError(
                "UniDepthEstimator is not available (missing onnxruntime or "
                f"weights at '{self.config.model_path}'). See the warning log "
                "for the manual download URL."
            )
        import cv2

        image_rgb = image_bgr[:, :, ::-1]
        h_orig, w_orig = image_rgb.shape[:2]
        in_h, in_w = self._input_hw or (448, 448)

        resized = cv2.resize(image_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        tensor = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, :]

        outputs = self._session.run(None, {self._input_name: tensor})
        depth_pred = np.asarray(outputs[0][0], dtype=np.float32)  # drop batch dim

        depth_map = cv2.resize(
            depth_pred, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
        )
        # UniDepth outputs millimetres -> convert to meters.
        return (depth_map / 1000.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_depth_estimator(config: Optional[DepthConfig] = None) -> DepthEstimator:
    """Build the configured depth backend.

    Always returns an estimator object (``DisabledDepth`` for ``"none"``);
    callers check :meth:`is_available`. Unknown backend names raise
    :class:`ValueError` (fail loud on config typos).
    """
    cfg = config or DepthConfig()
    backend = (cfg.backend or "unidepth").strip().lower()
    if backend == "unidepth":
        return UniDepthEstimator(cfg)
    if backend == "none":
        return DisabledDepth(cfg)
    raise ValueError(
        f"Unknown depth backend {cfg.backend!r}; expected 'unidepth' or 'none'."
    )


# ---------------------------------------------------------------------------
# 3D object localization (composes detection + depth + optional SAM2 masks)
# ---------------------------------------------------------------------------


def localize_objects_3d(
    detections: List[Detection],
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    masks: Optional[List[Optional[np.ndarray]]] = None,
) -> List[Optional[np.ndarray]]:
    """Localize each detection as a 3D point in the camera frame (meters).

    Depth sampling strategy (robust to noisy single pixels):
    * with a SAM2 mask: **median depth over the mask** (finite, positive
      samples only), back-projected at the mask centroid;
    * without: median over a 3x3 patch at the box center, back-projected at
      the box center.

    Args:
        detections: 2D detections with normalized [0, 1] xyxy boxes.
        depth_map: ``(H, W)`` float32 metric depth in meters.
        intrinsics: Real camera intrinsics (never a FOV guess).
        masks: Optional bool ``(H, W)`` masks aligned with ``detections``.

    Returns:
        List of ``(3,)`` float64 points (camera frame, meters), or ``None``
        per detection when no valid depth sample exists.
    """
    h, w = depth_map.shape[:2]
    points: List[Optional[np.ndarray]] = []
    for i, det in enumerate(detections):
        x1n, y1n, x2n, y2n = (float(v) for v in det.bbox_xyxy_norm)
        # Epsilon-tolerant pixel conversion: normalized boxes are float32, so
        # e.g. 0.4 * 100 can be 40.0000006 and a naive ceil would add a pixel.
        x1 = max(0, int(x1n * w + 1e-6))
        y1 = max(0, int(y1n * h + 1e-6))
        x2 = min(w, int(np.ceil(x2n * w - 1e-6)))
        y2 = min(h, int(np.ceil(y2n * h - 1e-6)))
        if x2 <= x1 or y2 <= y1:
            points.append(None)
            continue

        mask: Optional[np.ndarray] = None
        if masks is not None and i < len(masks) and masks[i] is not None:
            m = np.asarray(masks[i], dtype=bool)
            if m.shape == (h, w) and m.any():
                mask = m

        if mask is not None:
            vals = depth_map[mask]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size == 0:
                points.append(None)
                continue
            z = float(np.median(vals))
            ys, xs = np.nonzero(mask)
            u, v = float(xs.mean()), float(ys.mean())
        else:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ix0, ix1 = max(0, int(cx) - 1), min(w, int(cx) + 2)
            iy0, iy1 = max(0, int(cy) - 1), min(h, int(cy) + 2)
            patch = depth_map[iy0:iy1, ix0:ix1]
            vals = patch[np.isfinite(patch) & (patch > 0)]
            if vals.size == 0:
                points.append(None)
                continue
            z = float(np.median(vals))
            u, v = cx, cy

        points.append(backproject_pixel(u, v, z, intrinsics))
    return points


__all__ = [
    "DepthEstimator",
    "DepthConfig",
    "UniDepthEstimator",
    "DisabledDepth",
    "create_depth_estimator",
    "localize_objects_3d",
    "UNIDEPTH_DOWNLOAD_URL",
]
