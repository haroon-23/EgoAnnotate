"""Temporal action segmentation using Gemini Vision API."""
import os
import time
import re
import json
import cv2
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .datatypes import ActionSegment


MAX_RETRIES = 2
RETRY_DELAY = 3
VIDEO_UPLOAD_TIMEOUT = 30  # seconds max wait for video processing


@dataclass
class ActionSegmenterConfig:
    """Configuration for the GeminiActionSegmenter."""
    gemini_model: str = "gemini-1.5-pro-latest"
    prompt: str = (
        "Analyze the actions performed in this egocentric video and segment it into contiguous temporal segments.\n"
        "For each segment, provide:\n"
        "1. 'name': The action verb/label (e.g., 'pick_up', 'pour', 'place', 'stir').\n"
        "2. 'start_time': Starting timestamp (e.g. '0:02.5' or 2.5 seconds).\n"
        "3. 'end_time': Ending timestamp (e.g. '0:08.0' or 8.0 seconds).\n"
        "4. 'object_name': The object being manipulated (e.g. 'cup', 'spoon').\n"
        "5. 'hand_used': The hand performing the action ('left', 'right', or 'both').\n"
        "6. 'description': A brief natural language description of what is happening.\n\n"
        "Format the response strictly as a JSON list of segments, like this:\n"
        "[\n"
        "  {\n"
        "    \"name\": \"pick_up\",\n"
        "    \"start_time\": \"0:02.0\",\n"
        "    \"end_time\": \"0:05.5\",\n"
        "    \"object_name\": \"cup\",\n"
        "    \"hand_used\": \"right\",\n"
        "    \"description\": \"picking up the red cup\"\n"
        "  }\n"
        "]"
    )


class GeminiActionSegmenter:
    """Segments video into manipulation primitives."""
    
    def __init__(self, config: ActionSegmenterConfig):
        self.config = config
        self._model = None
        self._init_model()
    
    def _init_model(self):
        if not GEMINI_AVAILABLE:
            raise RuntimeError("google-generativeai not installed.")
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to use GeminiActionSegmenter.")
        
        genai.configure(api_key=api_key)
        model_name = self.config.gemini_model
        if model_name == "gemini-1.5-flash":
            model_name = "gemini-flash-latest"
        elif model_name == "models/gemini-1.5-flash":
            model_name = "models/gemini-flash-latest"
            
        if not model_name.startswith("models/") and model_name != "gemini-1.5-pro-latest":
            model_name = f"models/{model_name}"
        self._model = genai.GenerativeModel(model_name)
        print(f"[ActionSegmenter] Using {model_name}")
    
    def segment_video(self, video_path: str) -> List[ActionSegment]:
        """Segment video. Fast path with timeout."""
        video_path = Path(video_path)
        
        # Try video upload with strict timeout
        segments = self._try_video_upload(video_path)
        if segments:
            return segments
        
        # Fast fallback: single segment covering full duration
        print("[ActionSegmenter] Using fallback single segment")
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 300
        duration = frames / fps if fps > 0 else 10.0
        cap.release()
        
        return [ActionSegment(
            name="manipulate",
            start_time=0.0,
            end_time=duration,
            object_name="unknown",
            hand_used="right",
            description="performing manipulation task"
        )]
    
    def _try_video_upload(self, video_path: Path) -> Optional[List[ActionSegment]]:
        for attempt in range(MAX_RETRIES):
            try:
                print(f"[ActionSegmenter] Uploading video...")
                video_file = genai.upload_file(str(video_path))
                
                # Wait with timeout
                waited = 0
                while video_file.state.name == "PROCESSING" and waited < VIDEO_UPLOAD_TIMEOUT:
                    time.sleep(2)
                    waited += 2
                    try:
                        video_file = genai.get_file(video_file.name)
                    except Exception:
                        pass
                
                if video_file.state.name != "ACTIVE":
                    print(f"[ActionSegmenter] Upload timeout/failed: {video_file.state.name}")
                    return None
                
                prompt = self.config.prompt or """Analyze this egocentric video. Segment into manipulation primitives with timestamps: approach, contact, grasp, manipulate, release, retreat, idle. For each: action name, start time, end time, object, hand. Format as JSON."""
                
                response = self._model.generate_content([video_file, prompt], generation_config={"temperature": 0.1})
                response_text = response.text
                
                # Try JSON parsing
                segments = self._parse_json_response(response_text)
                if not segments:
                    # Fallback to regex text parsing
                    segments = self._parse_text_fallback(response_text)
                
                # Clean up
                try:
                    genai.delete_file(video_file.name)
                except Exception:
                    pass
                
                if segments:
                    print(f"[ActionSegmenter] {len(segments)} segments found")
                    return segments
                return None
            
            except Exception as e:
                err = str(e).lower()
                
                if any(k in err for k in ["404", "not found", "no longer available", "invalid model", "api key not valid"]):
                    print(f"[FATAL] {e}")
                    raise
                
                if any(k in err for k in ["rate limit", "quota", "429", "resource exhausted"]):
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"[RETRY] Rate limit. Wait {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[RETRY] {attempt+1}/{MAX_RETRIES}: {e}")
                    time.sleep(RETRY_DELAY)
                
                if attempt == MAX_RETRIES - 1:
                    print(f"[FAILED] Segmentation failed")
                    return None
        
        return None
    
    def _parse_time(self, time_str: str) -> float:
        """Parse time string formats (MM:SS.ms, MM:SS, SS.ms, or seconds) into float seconds."""
        s = time_str.strip()
        if not s:
            return 0.0
        
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 2:
                # MM:SS or MM:SS.ms
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60.0 + seconds
            elif len(parts) == 3:
                # HH:MM:SS
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600.0 + minutes * 60.0 + seconds
        try:
            return float(s)
        except ValueError:
            return 0.0
            
    def _parse_json_response(self, text: str) -> List[ActionSegment]:
        """Attempt to parse the response text as a JSON list of segments."""
        text = text.strip()
        try:
            cleaned = text
            # Extract JSON block between markdown markers or brackets
            json_match = re.search(r'```(?:json)?\n(.*?)\n```', text, re.DOTALL)
            if json_match:
                cleaned = json_match.group(1).strip()
            else:
                start_idx = text.find('[')
                end_idx = text.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    cleaned = text[start_idx:end_idx+1]
            
            data = json.loads(cleaned)
            segment_list = []
            if isinstance(data, list):
                segment_list = data
            elif isinstance(data, dict):
                # Search for lists inside dictionary values
                for val in data.values():
                    if isinstance(val, list):
                        segment_list = val
                        break
            
            segments = []
            for item in segment_list:
                if isinstance(item, dict):
                    name = item.get("name") or "manipulate"
                    start_str = str(item.get("start_time") or "0.0")
                    end_str = str(item.get("end_time") or "10.0")
                    object_name = item.get("object_name") or "unknown"
                    hand_used = item.get("hand_used") or "right"
                    description = item.get("description") or ""
                    
                    start_time = self._parse_time(start_str)
                    end_time = self._parse_time(end_str)
                    
                    segments.append(
                        ActionSegment(
                            name=str(name).strip(),
                            start_time=start_time,
                            end_time=end_time,
                            object_name=str(object_name).strip(),
                            hand_used=str(hand_used).strip().lower(),
                            description=str(description).strip(),
                        )
                    )
            if segments:
                return segments
        except Exception:
            pass
        return []
        
    def _parse_text_fallback(self, text: str) -> List[ActionSegment]:
        """Parse structured text lines using regex matching fallback."""
        segments = []
        # Support colon, dash, tilde
        pattern = r'(\d+[:\.]\d+(?:\.\d+)?)\s*[-~]\s*(\d+[:\.]\d+(?:\.\d+)?)\s*[:\-]?\s*(\w+)\s*(.*)'
        matches = re.finditer(pattern, text)
        
        stops = r'(?:the|a|an|down|up|with|using|both|left|right|hand|hands|of|in|on|at|to|into|for|using)\b'
        
        for match in matches:
            start_str = match.group(1)
            end_str = match.group(2)
            name = match.group(3)
            rest = match.group(4).strip()
            
            start_time = self._parse_time(start_str)
            end_time = self._parse_time(end_str)
            
            hand_used = "right"
            rest_lower = rest.lower()
            if "left" in rest_lower:
                hand_used = "left"
            elif "both" in rest_lower:
                hand_used = "both"
                
            object_name = "unknown"
            
            # Preposition-based search
            prep_match = re.search(
                r'\b(?:into|in|to|on|onto|at)\s+(?:the\s+|a\s+|an\s+)?(?:(?!' + stops + r')[a-zA-Z_]+\s+)?((?!' + stops + r')[a-zA-Z_]+)',
                rest_lower
            )
            if prep_match:
                object_name = prep_match.group(1)
                
            # Verb-based search
            if object_name == "unknown":
                obj_match = re.search(
                    r'\b(?:pick|picking|place|placing|grasp|grasping|pour|pouring|hold|holding|manipulate|manipulating|reach|reaching|touch|touching|grab|grabbing)\b\s+(?:up\s+|for\s+|down\s+)?(?:the\s+|a\s+|an\s+)?(?:(?!' + stops + r')[a-zA-Z_]+\s+)?((?!' + stops + r')[a-zA-Z_]+)',
                    rest_lower,
                )
                if obj_match:
                    object_name = obj_match.group(1)
                    
            segments.append(
                ActionSegment(
                    name=str(name).strip(),
                    start_time=start_time,
                    end_time=end_time,
                    object_name=object_name,
                    hand_used=hand_used,
                    description=rest,
                )
            )
        return segments
