"""Object detection in egocentric video frames using Gemini Vision API."""
import os
import time
import json
import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from PIL import Image
import cv2
import numpy as np

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .datatypes import ObjectAnnotation


MAX_RETRIES = 2
RETRY_DELAY = 3


@dataclass
class ObjectDetectorConfig:
    keyframes_per_video: int = 3
    gemini_model: str = "gemini-1.5-pro-latest"
    prompt: str = "Identify all objects in the frame. Respond in JSON format."


class GeminiObjectDetector:
    """Detects objects in egocentric video frames using Gemini."""
    
    def __init__(self, config: ObjectDetectorConfig):
        self.config = config
        self._model = None
        self._init_model()
    
    def _init_model(self):
        if not GEMINI_AVAILABLE:
            raise RuntimeError("google-generativeai not installed.")
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to use GeminiObjectDetector.")
        
        genai.configure(api_key=api_key)
        
        model_name = self.config.gemini_model
        if model_name == "gemini-1.5-flash":
            model_name = "gemini-flash-latest"
        elif model_name == "models/gemini-1.5-flash":
            model_name = "models/gemini-flash-latest"
            
        if not model_name.startswith("models/") and model_name != "gemini-1.5-pro-latest":
            model_name = f"models/{model_name}"
        
        self._model = genai.GenerativeModel(model_name)
        print(f"[ObjectDetector] Using {model_name}")
    
    def detect_objects(self, video_path: str) -> List[ObjectAnnotation]:
        """Detect objects from keyframes. Fast path: skip if no API key."""
        video_path = Path(video_path)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []
        
        indices = np.linspace(0, total_frames - 1, self.config.keyframes_per_video, dtype=int)
        all_objects = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            if pil_image.width > 1024 or pil_image.height > 1024:
                pil_image.thumbnail((1024, 1024))
            
            objects = self._call_gemini_fast(pil_image)
            all_objects.extend(objects)
        
        cap.release()
        
        # Deduplicate
        deduped = {}
        for obj in all_objects:
            if obj.name in deduped:
                if obj.touched and not deduped[obj.name].touched:
                    deduped[obj.name] = obj
            else:
                deduped[obj.name] = obj
        
        result = list(deduped.values())
        print(f"[ObjectDetector] {len(result)} objects: {[o.name for o in result]}")
        return result
    
    def _call_gemini_fast(self, pil_image: Image.Image) -> List[ObjectAnnotation]:
        """Call Gemini with strict retry limits. Fast fail on fatal errors."""
        prompt = self.config.prompt or """Analyze this egocentric video frame. List ALL objects the hands are interacting with or could interact with. For each: name, location (top-left/center/bottom-right), touched (yes/no). Format as JSON list. Example: [{"name":"red_cup","location":"center","touched":true}]"""
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self._model.generate_content([prompt, pil_image], generation_config={"temperature": 0.1})
                return self._parse_response(response.text)
            
            except Exception as e:
                err = str(e).lower()
                
                # FATAL: wrong model, auth failure — fail immediately
                if any(k in err for k in ["404", "not found", "no longer available", "invalid model", "api key not valid", "permission denied"]):
                    print(f"[FATAL] {e}")
                    print(f"[FATAL] Fix: change model in configs/default.yaml to gemini-1.5-flash")
                    raise
                
                # Rate limit — short wait then retry
                if any(k in err for k in ["rate limit", "quota", "429", "resource exhausted"]):
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"[RETRY] Rate limit. Wait {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[RETRY] {attempt+1}/{MAX_RETRIES}: {e}")
                    time.sleep(RETRY_DELAY)
                
                if attempt == MAX_RETRIES - 1:
                    print(f"[FAILED] Object detection failed after retries")
                    return []
        
        return []
    
    def _parse_response(self, text: str) -> List[ObjectAnnotation]:
        if not text:
            return []
        
        text_clean = text.strip()
        
        # 1. Extract JSON from markdown if wrapped
        m = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text_clean, re.DOTALL)
        if m:
            text_clean = m.group(1)
        elif text_clean.startswith("```"):
            text_clean = re.sub(r"^```(?:json)?\n", "", text_clean)
            text_clean = re.sub(r"\n```$", "", text_clean)
        
        text_clean = text_clean.strip()
        
        # 2. Try JSON parsing
        try:
            data = json.loads(text_clean)
            object_list = []
            if isinstance(data, list):
                object_list = data
            elif isinstance(data, dict):
                # Search for any list value inside the dict
                for val in data.values():
                    if isinstance(val, list):
                        object_list = val
                        break
            
            annotations = []
            for item in object_list:
                if isinstance(item, dict):
                    name = item.get("name", item.get("object", "unknown"))
                    loc = item.get("location", item.get("location_description", "unknown"))
                    touched = bool(item.get("touched", False))
                    annotations.append(ObjectAnnotation(name=name, location_description=loc, touched=touched))
            if annotations:
                return annotations
        except json.JSONDecodeError:
            pass
        
        # 3. Fallback: regex search for JSON-like lines in broken JSON
        annotations = []
        for line in text.split("\n"):
            if "name" in line or "object" in line:
                name_match = re.search(r'["\']?(?:name|object)["\']?\s*[:=]\s*["\']?([^"\'\r\n,}]+)', line)
                if name_match:
                    name = name_match.group(1).strip().strip('"\'')
                    
                    loc_match = re.search(r'["\']?(?:location|location_description)["\']?\s*[:=]\s*["\']?([^"\'\r\n,}]+)', line)
                    loc = loc_match.group(1).strip().strip('"\'') if loc_match else "unknown"
                    
                    touch_match = re.search(r'["\']?touched["\']?\s*[:=]\s*(true|false|1|0)', line, re.IGNORECASE)
                    touched = False
                    if touch_match:
                        touched = touch_match.group(1).lower() in ["true", "1"]
                        
                    annotations.append(ObjectAnnotation(name=name, location_description=loc, touched=touched))
        
        if annotations:
            return annotations
        
        # 4. Fallback: regex search for plain text bullet lists like "- Knife: on the cutting board (touched: true)"
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                match = re.match(r'^[-*]\s*([^:]+)\s*:\s*([^(]+)(?:\(touched:\s*(true|false)\))?', line, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    loc = match.group(2).strip()
                    touched_str = match.group(3)
                    touched = False
                    if touched_str:
                        touched = touched_str.lower() == "true"
                    annotations.append(ObjectAnnotation(name=name, location_description=loc, touched=touched))
                    
        return annotations
