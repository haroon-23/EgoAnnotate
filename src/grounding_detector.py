"""Back-compatibility shim for the OWL-ViT detector.

The detector that historically lived here (misnamed ``GroundingDINODetector`` —
it is OWL-ViT under the hood, not Grounding DINO) now lives in
:mod:`src.perception.detector` as :class:`OwlViTDetector`. This module keeps the
old import path, class name, config name, factory, and ``detect()`` signature
working. New code should import from ``src.perception`` instead.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .datatypes import ObjectAnnotation
from .perception.detector import (
    Detector2DConfig,
    OwlViTDetector,
    bbox_to_location_description,
)

logger = logging.getLogger(__name__)

# Legacy name: this config always configured OWL-ViT (model_name defaults to
# "google/owlvit-base-patch32"), never the real Grounding DINO.
GroundingDINOConfig = Detector2DConfig


class GroundingDINODetector(OwlViTDetector):
    """Legacy name for :class:`OwlViTDetector` (kept for API compatibility).

    Despite the name, this backend is OWL-ViT. For the real Grounding DINO
    (IDEA-Research) backend use ``src.perception.create_detector_2d`` with
    ``backend="grounding_dino"``.
    """

    def detect(
        self,
        image: np.ndarray,
        object_names: List[str],
        threshold: Optional[float] = None,
    ) -> List[ObjectAnnotation]:
        """Detect objects; returns :class:`ObjectAnnotation` (legacy contract).

        Args:
            image: Input image as numpy array (BGR or RGB, HxWx3).
            object_names: List of object names to search for (text prompts).
            threshold: Optional per-call confidence-threshold override.
        """
        # Route through OwlViTDetector.detect (the real implementation), NOT
        # through detect_annotations -> self.detect, which would recurse into
        # this override forever.
        if threshold is not None:
            prev = self.config.confidence_threshold
            self.config.confidence_threshold = float(threshold)
            try:
                dets = super().detect(image, object_names)
            finally:
                self.config.confidence_threshold = prev
        else:
            dets = super().detect(image, object_names)
        return [
            ObjectAnnotation(
                name=d.label,
                location_description=bbox_to_location_description(d.bbox_xyxy_norm),
                touched=False,
                bbox=np.asarray(d.bbox_xyxy_norm, dtype=np.float32),
                state="idle",
            )
            for d in dets
        ]

    # Kept because src/object_detector.py calls this private helper.
    def _bbox_to_location(self, bbox: np.ndarray) -> str:
        return bbox_to_location_description(bbox)


def create_grounding_detector(
    config: Optional[GroundingDINOConfig] = None,
) -> Optional[GroundingDINODetector]:
    """Factory with graceful degradation (returns None, never raises)."""
    if config is None:
        config = GroundingDINOConfig()
    try:
        detector = GroundingDINODetector(config)
    except Exception as e:
        logger.warning("Failed to create OWL-ViT detector: %s", e)
        return None
    if detector.is_available():
        return detector
    logger.warning("OWL-ViT detector created but model not loaded")
    return None


__all__ = [
    "GroundingDINOConfig",
    "GroundingDINODetector",
    "create_grounding_detector",
]
