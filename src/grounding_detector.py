"""Zero-shot object detection for real bounding boxes.

Uses OWL-ViT (owlvit-base-patch32) for zero-shot object detection
to localize objects in egocentric video frames. OWL-ViT is compatible
with transformers 4.37.0 and torch 2.2.2, unlike Grounding DINO which
requires newer versions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch

from .datatypes import ObjectAnnotation

logger = logging.getLogger(__name__)

try:
    from transformers import AutoProcessor, OwlViTForObjectDetection
    OWL_VIT_AVAILABLE = True
except ImportError:
    OWL_VIT_AVAILABLE = False
    logger.warning(
        "transformers not installed. OWL-ViT will not be available. "
        "Install via: pip install transformers>=4.30.0 torch>=2.0.0"
    )


@dataclass
class GroundingDINOConfig:
    """Configuration for OWL-ViT Detector (kept name for API compatibility)."""
    model_name: str = "google/owlvit-base-patch32"
    confidence_threshold: float = 0.3
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    device: str = "auto"  # "auto", "cpu", "cuda"


class GroundingDINODetector:
    """Zero-shot object detection using OWL-ViT (API compatible with Grounding DINO).
    
    Takes an image and a list of object names (text prompts) and returns
    bounding boxes for each detected object in normalized [0, 1] coordinates.
    """
    
    def __init__(self, config: GroundingDINOConfig):
        """Initialize the OWL-ViT detector.
        
        Args:
            config: GroundingDINOConfig with model settings
        """
        self.config = config
        self._processor = None
        self._model = None
        self._device = self._resolve_device()
        self._init_model()
    
    def _resolve_device(self) -> str:
        """Determine the device to use."""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    def _init_model(self) -> None:
        """Load the OWL-ViT model and processor."""
        if not OWL_VIT_AVAILABLE:
            logger.warning("OWL-ViT unavailable - transformers not installed")
            return
        
        try:
            logger.info(f"Loading OWL-ViT model: {self.config.model_name} on {self._device}")
            self._processor = AutoProcessor.from_pretrained(self.config.model_name)
            self._model = OwlViTForObjectDetection.from_pretrained(
                self.config.model_name
            ).to(self._device)
            self._model.eval()
            logger.info("OWL-ViT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load OWL-ViT model: {e}")
            self._processor = None
            self._model = None
    
    def is_available(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model is not None and self._processor is not None
    
    def detect(
        self,
        image: np.ndarray,
        object_names: List[str],
    ) -> List[ObjectAnnotation]:
        """Detect objects in an image using OWL-ViT.
        
        Args:
            image: Input image as numpy array (BGR or RGB, HxWx3)
            object_names: List of object names to search for (text prompts)
            
        Returns:
            List of ObjectAnnotation with populated bbox in normalized [0, 1] coordinates
        """
        if not self.is_available():
            logger.warning("OWL-ViT not available, returning empty list")
            return []
        
        if not object_names:
            return []
        
        # Convert BGR to RGB if needed
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Prepare text prompts - OWL-ViT expects list of text queries
        text_prompts = object_names
        
        try:
            # Process inputs
            inputs = self._processor(
                images=image_rgb,
                text=text_prompts,
                return_tensors="pt"
            ).to(self._device)
            
            # Run inference
            with torch.no_grad():
                outputs = self._model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image_rgb.shape[:2]]).to(self._device)
            results = self._processor.post_process_object_detection(
                outputs,
                threshold=self.config.confidence_threshold,
                target_sizes=target_sizes
            )[0]
            
            # Convert results to ObjectAnnotation list
            annotations = []
            for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
                if score < self.config.confidence_threshold:
                    continue
                
                # Map label index to object name
                label = text_prompts[label_idx.item()] if label_idx.item() < len(text_prompts) else "unknown"
                
                # Box is in [x_min, y_min, x_max, y_max] in pixel coordinates
                # Convert to normalized [0, 1]
                h, w = image_rgb.shape[:2]
                bbox_norm = np.array([
                    box[0].item() / w,
                    box[1].item() / h,
                    box[2].item() / w,
                    box[3].item() / h
                ], dtype=np.float32)
                
                # Clamp to [0, 1]
                bbox_norm = np.clip(bbox_norm, 0.0, 1.0)
                
                annotations.append(ObjectAnnotation(
                    name=label,
                    location_description=self._bbox_to_location(bbox_norm),
                    touched=False,  # Will be determined by contact detector
                    bbox=bbox_norm,
                    state="idle"
                ))
            
            return annotations
            
        except Exception as e:
            logger.error(f"OWL-ViT detection failed: {e}")
            return []
    
    def _bbox_to_location(self, bbox: np.ndarray) -> str:
        """Convert normalized bbox to rough location description."""
        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        # Horizontal position
        if cx < 0.33:
            h_pos = "left"
        elif cx < 0.66:
            h_pos = "center"
        else:
            h_pos = "right"
        
        # Vertical position
        if cy < 0.33:
            v_pos = "top"
        elif cy < 0.66:
            v_pos = "middle"
        else:
            v_pos = "bottom"
        
        return f"{v_pos}-{h_pos}"


def create_grounding_detector(config: Optional[GroundingDINOConfig] = None) -> Optional[GroundingDINODetector]:
    """Factory function to create GroundingDINODetector with graceful degradation.
    
    Returns None if model cannot be loaded (instead of raising).
    """
    if config is None:
        config = GroundingDINOConfig()
    
    try:
        detector = GroundingDINODetector(config)
        if detector.is_available():
            return detector
        else:
            logger.warning("OWL-ViT detector created but model not loaded")
            return None
    except Exception as e:
        logger.warning(f"Failed to create OWL-ViT detector: {e}")
        return None