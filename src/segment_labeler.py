"""Segment labeling using VLM (Gemini).

Takes pre-computed CandidateSegments from SignalSegmenter and labels each
with an action category using Gemini Vision. Does NOT derive boundaries —
only labels existing segments.
"""

from __future__ import annotations

import logging
import os
import time
import re
import json
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .datatypes import ActionSegment, CandidateSegment

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 3
VIDEO_UPLOAD_TIMEOUT = 30


@dataclass
class SegmentLabelerConfig:
    """Configuration for the SegmentLabeler."""
    gemini_model: str = "gemini-1.5-flash"
    prompt_template: str = (
        "You are labeling a segment from an egocentric manipulation video.\n\n"
        "Segment info:\n"
        "  Time: {start_time:.2f}s - {end_time:.2f}s (duration {duration:.2f}s)\n"
        "  Contact state: {contact_state}\n"
        "  Grasp type: {grasp_type}\n"
        "  Object: {object_name}\n\n"
        "The segment starts with a transition: {transition_type}.\n\n"
        "Choose ONE action category:\n"
        "  approach   - hand moving toward object, no contact yet\n"
        "  contact    - first moment of touch\n"
        "  grasp      - fingers closing around object\n"
        "  manipulate - object being moved/used while grasped\n"
        "  release    - fingers opening, object let go\n"
        "  retreat    - hand moving away after release\n"
        "  idle       - no contact, no manipulation\n\n"
        "Respond in JSON format:\n"
        "{{\n"
        "  \"action\": \"category_name\",\n"
        "  \"description\": \"short caption\"\n"
        "}}"
    )


class SegmentLabeler:
    """Labels pre-computed segments with action categories using Gemini."""
    
    def __init__(self, config: SegmentLabelerConfig):
        self.config = config
        self._model = None
        self._init_model()
    
    def _init_model(self):
        if not GEMINI_AVAILABLE:
            raise RuntimeError("google-generativeai not installed.")
        
        if not os.environ.get("GEMINI_API_KEY"):
            for env_path in [Path(__file__).resolve().parent.parent / ".env", Path(".env"), Path.home() / "sia_agent" / ".env"]:
                if env_path.exists():
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("GEMINI_API_KEY="):
                                val = line.split("=", 1)[1].strip("'\"")
                                if val:
                                    os.environ["GEMINI_API_KEY"] = val
                                    break
                    if os.environ.get("GEMINI_API_KEY"):
                        break

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        
        genai.configure(api_key=api_key, transport='rest')
        
        model_name = self.config.gemini_model
        if model_name in ["gemini-1.5-flash", "gemini-flash-latest", "models/gemini-1.5-flash", "models/gemini-flash-latest"]:
            model_name = "gemini-flash-lite-latest"
            
        if not model_name.startswith("models/") and model_name != "gemini-1.5-pro-latest":
            model_name = f"models/{model_name}"
        
        self._model = genai.GenerativeModel(model_name)
        print(f"[SegmentLabeler] Using {model_name}")
    
    def label_segments(
        self,
        candidates: List[CandidateSegment],
        video_path: str,
    ) -> List[ActionSegment]:
        """Label each candidate segment with an action category.
        
        Args:
            candidates: List of CandidateSegment from SignalSegmenter.
            video_path: Path to the source video file.
            
        Returns:
            List of ActionSegment with action labels and descriptions.
        """
        if not candidates:
            return []
        
        labeled_segments = []
        
        for seg in candidates:
            label = self._label_single_segment(seg, video_path)
            
            labeled_segments.append(ActionSegment(
                name=label["action"],
                start_time=seg.start_time,
                end_time=seg.end_time,
                object_name=seg.object_name or "unknown",
                hand_used="right",  # Could be enhanced to track hand
                description=label["description"],
            ))
        
        return labeled_segments
    
    def _label_single_segment(
        self,
        seg: CandidateSegment,
        video_path: str,
    ) -> dict:
        """Label a single segment using a keyframe and context."""
        
        # Extract keyframe at segment midpoint
        keyframe = self._extract_keyframe(video_path, seg)
        
        if keyframe is None:
            logger.warning(f"Could not extract keyframe for segment {seg.start_time}-{seg.end_time}")
            return {"action": "unlabeled", "description": "keyframe extraction failed"}
        
        # Build prompt
        prompt = self.config.prompt_template.format(
            start_time=seg.start_time,
            end_time=seg.end_time,
            duration=seg.end_time - seg.start_time,
            contact_state=seg.contact_state,
            grasp_type=seg.grasp_type,
            object_name=seg.object_name or "unknown",
            transition_type=seg.transition_type,
        )
        
        # Call Gemini with retries
        result = self._call_gemini_with_retry(keyframe, prompt)
        
        if result is None:
            logger.warning(f"Gemini call failed for segment {seg.start_time}-{seg.end_time}, using default")
            return self._default_label(seg)
        
        return result
    
    def _extract_keyframe(self, video_path: str, seg: CandidateSegment) -> Optional[Image.Image]:
        """Extract a frame at segment midpoint."""
        mid_time = (seg.start_time + seg.end_time) / 2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        
        frame_idx = int(mid_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize if too large
        if pil_image.width > 1024 or pil_image.height > 1024:
            pil_image.thumbnail((1024, 1024))
        
        return pil_image
    
    def _call_gemini_with_retry(self, image: Image.Image, prompt: str) -> Optional[dict]:
        """Call Gemini with retry logic."""
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self._model.generate_content(
                    [prompt, image],
                    generation_config={"temperature": 0.1},
                    request_options={"timeout": 5.0}
                )
                return self._parse_response(response.text)
            
            except Exception as e:
                err = str(e).lower()
                
                # Fatal errors
                if any(k in err for k in ["404", "not found", "no longer available", 
                                          "invalid model", "api key not valid", "permission denied"]):
                    logger.error(f"[FATAL] {e}")
                    raise
                
                # Rate limit
                if any(k in err for k in ["rate limit", "quota", "429", "resource exhausted"]):
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"[RETRY] Rate limit. Wait {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[RETRY] {attempt+1}/{MAX_RETRIES}: {e}")
                    time.sleep(RETRY_DELAY)
                
                if attempt == MAX_RETRIES - 1:
                    print(f"[FAILED] Segment labeling failed after retries")
                    return None
        
        return None
    
    def _parse_response(self, text: str) -> Optional[dict]:
        """Parse Gemini response for action label and description."""
        if not text:
            return None
        
        text = text.strip()
        
        # Try JSON parsing
        try:
            # Extract JSON from markdown if wrapped
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if m:
                text = m.group(1)
            elif text.startswith("```"):
                text = re.sub(r"^```(?:json)?\n", "", text)
                text = re.sub(r"\n```$", "", text)
            
            data = json.loads(text.strip())
            
            action = data.get("action", "unlabeled").strip().lower()
            desc = data.get("description", "").strip()
            
            # Validate action
            valid_actions = {"approach", "contact", "grasp", "manipulate", "release", "retreat", "idle"}
            if action not in valid_actions:
                action = "unlabeled"
            
            return {"action": action, "description": desc}
        
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _default_label(self, seg: CandidateSegment) -> dict:
        """Generate a default label based on signal properties."""
        # Heuristic mapping from signals to action
        if seg.contact_state == "no_contact":
            if seg.transition_type == "start":
                action = "idle"
            elif seg.transition_type == "contact_off":
                action = "retreat"
            else:
                action = "approach"
        else:  # contact
            if seg.transition_type == "contact_on":
                action = "contact"
            elif seg.grasp_type in ("precision_pinch", "power_wrap", "hook"):
                action = "grasp"
            else:
                action = "manipulate"
        
        return {
            "action": action,
            "description": f"auto-labeled: {action} ({seg.transition_type})"
        }


def create_segment_labeler(config: Optional[SegmentLabelerConfig] = None) -> Optional[SegmentLabeler]:
    """Factory function with graceful degradation."""
    if config is None:
        config = SegmentLabelerConfig()
    
    try:
        return SegmentLabeler(config)
    except Exception as e:
        logger.warning(f"Failed to create SegmentLabeler: {e}")
        return None