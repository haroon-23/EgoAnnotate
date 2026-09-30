"""Back-compat alias: the hardened Grounding DINO backend moved.

:class:`GroundingDinoDetector` now lives in :mod:`src.perception.detector`
(lazy imports, no eager weight loading, normalized-box convention). This module
keeps the old import path working.
"""

from .detector import (
    GROUNDING_DINO_CONFIG_URL,
    GROUNDING_DINO_WEIGHTS_URL,
    Detector2DConfig,
    GroundingDinoDetector,
)

# Legacy class name used by the pre-Phase-C module. Note: the *other*
# ``GroundingDINODetector`` at ``src.grounding_detector`` is the OWL-ViT shim —
# prefer the unambiguous names above.
GroundingDINODetector = GroundingDinoDetector

__all__ = [
    "GroundingDinoDetector",
    "GroundingDINODetector",
    "Detector2DConfig",
    "GROUNDING_DINO_WEIGHTS_URL",
    "GROUNDING_DINO_CONFIG_URL",
]
