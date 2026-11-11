from __future__ import annotations
"""VLM Client Interface with Mandatory Privacy Face Blurring / Redaction.

Every frame sent to any external VLM API passes through face blurring / redaction
prior to network egress, ensuring zero raw unblurred facial biometrics leave the machine.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from .privacy import blur_faces_in_frame

logger = logging.getLogger(__name__)


def prepare_frame_for_vlm(
    frame: Union[np.ndarray, Image.Image, Path, str]
) -> Tuple[np.ndarray, int]:
    """Ensure frame is face-blurred and redacted before passing to VLM client.
    
    Returns:
        Tuple of (blurred_bgr_frame, num_faces_blurred)
    """
    if isinstance(frame, (str, Path)):
        p = Path(frame)
        if not p.exists():
            raise FileNotFoundError(f"Frame not found: {p}")
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise ValueError(f"Could not decode image at {p}")
    elif isinstance(frame, Image.Image):
        rgb = np.array(frame)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(frame, np.ndarray):
        bgr = frame.copy()
    else:
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    blurred_bgr, num_faces = blur_faces_in_frame(bgr)
    if num_faces > 0:
        logger.info("[vlm_client] Blurred/redacted %d face(s) before VLM transmission", num_faces)
    return blurred_bgr, num_faces


def encode_image_for_vlm(frame: Union[np.ndarray, Image.Image]) -> bytes:
    """Blur faces in image and encode to JPEG bytes for VLM upload."""
    blurred_bgr, _ = prepare_frame_for_vlm(frame)
    success, encoded = cv2.imencode(".jpg", blurred_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not success:
        raise RuntimeError("Failed to encode blurred frame to JPEG")
    return encoded.tobytes()
