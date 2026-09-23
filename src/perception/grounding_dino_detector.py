"""Grounding DINO: Open-source text-conditioned object detection.
Replaces Gemini VLM for consistent, fast object detection."""
import os
import torch
import numpy as np
from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops
import supervision as sv

class GroundingDINODetector:
    def __init__(self, 
                 model_config="groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 model_weights="groundingdino_swint_ogc.pth",
                 device="cpu"):
        self.device = device
        if not os.path.exists(model_config):
            try:
                import groundingdino
                pkg_config = os.path.join(os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinT_OGC.py")
                if os.path.exists(pkg_config):
                    model_config = pkg_config
            except Exception:
                pass
        self.model = load_model(model_config, model_weights)
        self.BOX_THRESHOLD = 0.30
        self.TEXT_THRESHOLD = 0.25
        
        # Vocabulary for this project (can be extended)
        self.VOCABULARY = [
            "bottle", "container", "case", "mouse", "keyboard", 
            "phone", "charger", "cable", "keys", "notebook",
            "pen", "cup", "mug", "remote", "headphones"
        ]
    
    def detect(self, image_bgr: np.ndarray, text_prompt: str = None) -> list:
        """
        Detect objects in image using Grounding DINO.
        
        Args:
            image_bgr: OpenCV BGR image
            text_prompt: Custom text prompt (uses VOCABULARY if None)
        
        Returns:
            List of detections: [{"bbox": [x1,y1,x2,y2], "name": str, "score": float}]
        """
        if text_prompt is None:
            text_prompt = ". ".join(self.VOCABULARY) + "."
        
        # Convert to RGB for Grounding DINO (ensure contiguous memory for PyTorch)
        image_rgb = image_bgr[:, :, ::-1].copy()
        
        # Predict
        boxes, logits, phrases = predict(
            model=self.model,
            image=torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0,
            caption=text_prompt,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
            device=self.device
        )
        
        # Convert boxes to [x1, y1, x2, y2] format
        h, w = image_bgr.shape[:2]
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h])
        
        detections = []
        for box, phrase, score in zip(boxes_xyxy, phrases, logits):
            detections.append({
                "bbox": box.tolist(),
                "name": self._canonicalize_name(phrase),
                "score": float(score)
            })
        
        return self._nms(detections)
    
    def _canonicalize_name(self, name: str) -> str:
        """Map detected name to canonical vocabulary."""
        name_lower = name.lower().strip()
        
        # Direct match
        if name_lower in self.VOCABULARY:
            return name_lower
        
        # Fuzzy match
        from difflib import SequenceMatcher
        best_match = None
        best_score = 0.5
        
        for vocab_item in self.VOCABULARY:
            score = SequenceMatcher(None, name_lower, vocab_item).ratio()
            if score > best_score:
                best_score = score
                best_match = vocab_item
        
        return best_match if best_match else "unknown"
    
    def _nms(self, detections: list, iou_threshold: float = 0.5) -> list:
        """Non-maximum suppression to remove duplicate boxes."""
        if not detections:
            return []
        
        # Sort by score descending
        detections = sorted(detections, key=lambda d: d["score"], reverse=True)
        
        keep = []
        for det in detections:
            should_keep = True
            for kept in keep:
                if self._iou(det["bbox"], kept["bbox"]) > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(det)
        
        return keep
    
    @staticmethod
    def _iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
