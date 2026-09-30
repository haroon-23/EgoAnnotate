"""SAM 2 box-prompted instance masks (optional, off by default).

:class:`Sam2Segmenter` wraps ``SAM2ImagePredictor`` with the same guarded-import
+ lazy-build + ``is_available()`` pattern used by the other perception backends.
Masks are returned as in-memory bool arrays; :func:`encode_mask_rle` serializes
them to a COCO-style RLE dict implemented in pure numpy (no new dependency).

CPU note: even the tiny ``sam2_hiera_t`` checkpoint is tens of seconds per
image on the 2017 Intel Mac — keep ``sam2.enabled: false`` unless masks are
needed, and run on detection keyframes only.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _SAM2_IMPORT_OK = True
except ImportError:  # pragma: no cover - depends on installed env
    _SAM2_IMPORT_OK = False
    logger.debug("sam2 not installed — SAM 2 backend unavailable (pip install sam2).")

# No auto-download anywhere: missing weights fail loud with this URL.
SAM2_TINY_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)


@dataclass
class Sam2Config:
    enabled: bool = False  # master switch; default off (heavy on CPU)
    checkpoint: str = "models/sam2.1_hiera_tiny.pt"
    # Either a config yaml path or a config name resolved inside the installed
    # sam2 package (e.g. "sam2_hiera_t" -> .../configs/sam2.1/sam2.1_hiera_t.yaml).
    config: str = "sam2_hiera_t"
    device: str = "cpu"  # "cpu" recommended; "cuda" if available


# ---------------------------------------------------------------------------
# COCO-style RLE in pure numpy (no pycocotools dependency)
# ---------------------------------------------------------------------------


def encode_mask_rle(mask: np.ndarray) -> dict:
    """Encode a bool ``(H, W)`` mask as a COCO-style RLE dict.

    Returns ``{"size": [h, w], "counts": [...]}`` — JSON-safe. Column-major
    (Fortran order) run lengths starting with the background (0) run, per the
    COCO mask API convention.
    """
    m = np.ascontiguousarray(mask, dtype=bool)
    h, w = m.shape
    flat = np.asarray(m, dtype=bool).reshape(-1, order="F").astype(np.uint8)  # COCO: column-major
    # Run boundaries: indices where the value changes.
    change = np.nonzero(np.diff(flat) != 0)[0] + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    counts = np.diff(bounds).astype(np.int64)
    if flat.size and flat[0] == 1:
        # COCO RLE always starts with the 0-run; prepend an empty one.
        counts = np.concatenate(([0], counts))
    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}


def decode_mask_rle(rle: dict) -> np.ndarray:
    """Decode a :func:`encode_mask_rle` dict back to a bool ``(H, W)`` mask."""
    h, w = (int(v) for v in rle["size"])
    counts = [int(c) for c in rle["counts"]]
    if sum(counts) != h * w:
        raise ValueError(
            f"RLE counts sum to {sum(counts)} but size is [{h}, {w}] "
            f"({h * w} pixels)."
        )
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for c in counts:
        if val == 1 and c:
            flat[pos : pos + c] = 1
        pos += c
        val = 1 - val
    return flat.reshape((h, w), order="F").astype(bool)


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------


class Sam2Segmenter:
    """Box-prompted SAM 2 masks. Lazy build; never raises on missing pieces."""

    def __init__(self, config: Optional[Sam2Config] = None):
        self.config = config or Sam2Config()
        self._predictor = None
        self._warned: dict[str, bool] = {}

    def _warn_once(self, key: str, msg: str, *args) -> None:
        if not self._warned.get(key):
            logger.warning(msg, *args)
            self._warned[key] = True

    def _resolve_config_path(self) -> Optional[str]:
        cfg = self.config.config
        if cfg and os.path.exists(cfg):
            return cfg
        if _SAM2_IMPORT_OK:
            try:
                import sam2

                cfg_root = Path(sam2.__file__).parent / "configs"
                if cfg_root.is_dir():
                    stem = Path(cfg).stem
                    matches = sorted(cfg_root.rglob(f"*{stem}*.yaml"))
                    if matches:
                        return str(matches[0])
            except Exception:
                pass
        return None

    def _ensure_predictor(self) -> bool:
        if self._predictor is not None:
            return True
        if not _SAM2_IMPORT_OK:
            self._warn_once(
                "sam2-missing",
                "sam2.enabled=true but the 'sam2' package is not installed "
                "(pip install sam2; CPU inference works but is slow). "
                "Continuing box-only.",
            )
            return False
        ckpt = self.config.checkpoint
        if not ckpt or not os.path.exists(ckpt):
            self._warn_once(
                "sam2-weights",
                "sam2.enabled=true but checkpoint not found at '%s'. Download "
                "manually (no auto-download): %s — then set 'sam2.checkpoint' "
                "in configs/default.yaml. Continuing box-only.",
                ckpt,
                SAM2_TINY_CHECKPOINT_URL,
            )
            return False
        cfg_path = self._resolve_config_path()
        if cfg_path is None:
            self._warn_once(
                "sam2-config",
                "sam2.enabled=true but config '%s' could not be resolved "
                "(expected a sam2 model-config yaml). Continuing box-only.",
                self.config.config,
            )
            return False
        try:
            logger.info("Building SAM 2 predictor (%s on %s)", cfg_path, self.config.device)
            model = build_sam2(cfg_path, ckpt, device=self.config.device)
            self._predictor = SAM2ImagePredictor(model)
            logger.info("SAM 2 predictor ready")
        except Exception as e:
            logger.error("Failed to build SAM 2 predictor: %s", e)
            self._predictor = None
        return self._predictor is not None

    def is_available(self) -> bool:
        return self._ensure_predictor()

    def predict_masks(
        self, image_bgr: np.ndarray, boxes_xyxy_norm: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Predict one bool ``(H, W)`` mask per normalized box.

        Raises:
            RuntimeError: when the backend is not available.
        """
        if not self.is_available():
            raise RuntimeError(
                "Sam2Segmenter is not available (missing sam2 package, "
                "checkpoint, or config). Gate calls with is_available()."
            )
        h, w = image_bgr.shape[:2]
        image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        self._predictor.set_image(image_rgb)

        masks: List[np.ndarray] = []
        for box in boxes_xyxy_norm:
            x1n, y1n, x2n, y2n = (float(v) for v in box)
            x1 = float(np.clip(x1n * w, 0, w - 1))
            y1 = float(np.clip(y1n * h, 0, h - 1))
            x2 = float(np.clip(x2n * w, 0, w - 1))
            y2 = float(np.clip(y2n * h, 0, h - 1))
            if x2 <= x1 or y2 <= y1:
                masks.append(np.zeros((h, w), dtype=bool))
                continue
            try:
                out_masks, _scores, _logits = self._predictor.predict(
                    box=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                    multimask_output=False,
                )
            except Exception as e:
                logger.error("SAM 2 predict failed for box %s: %s", box, e)
                masks.append(np.zeros((h, w), dtype=bool))
                continue
            m = np.asarray(out_masks[0])
            if m.ndim == 3:  # (1, H, W) -> (H, W)
                m = m[0]
            if m.shape != (h, w):
                logger.warning(
                    "SAM 2 returned mask shape %s for image (%d, %d); dropping.",
                    m.shape,
                    h,
                    w,
                )
                masks.append(np.zeros((h, w), dtype=bool))
                continue
            masks.append(m.astype(bool))
        return masks


def create_sam2_segmenter(config: Optional[Sam2Config] = None) -> Optional[Sam2Segmenter]:
    """Build the SAM 2 segmenter.

    Returns ``None`` when disabled (the default) or when unavailable — the
    pipeline continues box-only in both cases.
    """
    cfg = config or Sam2Config()
    if not cfg.enabled:
        return None
    seg = Sam2Segmenter(cfg)
    if seg.is_available():
        logger.info("SAM 2 segmenter ready")
        return seg
    logger.warning("SAM 2 requested but unavailable — continuing box-only.")
    return None


__all__ = [
    "Sam2Segmenter",
    "Sam2Config",
    "create_sam2_segmenter",
    "encode_mask_rle",
    "decode_mask_rle",
    "SAM2_TINY_CHECKPOINT_URL",
]
