"""Object detection in egocentric video frames using Gemini Vision API + Grounding DINO.

Two-stage approach:
1. Gemini identifies WHAT objects are present and their touch state (semantic understanding)
2. Grounding DINO localizes each object with REAL bounding boxes (spatial precision)
"""
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
from .grounding_detector import GroundingDINODetector, GroundingDINOConfig, create_grounding_detector


MAX_RETRIES = 2
RETRY_DELAY = 3


@dataclass
class ObjectDetectorConfig:
    keyframes_per_video: int = 3
    bbox_keyframe_interval: int = 15
    gemini_model: str = "gemini-1.5-pro-latest"
    prompt: str = "Identify all objects in the frame. Respond in JSON format."
    # Grounding DINO settings (optional, can be overridden by pipeline config)
    grounding_dino_model: str = "IDEA-Research/grounding-dino-tiny"
    grounding_dino_confidence: float = 0.3
    grounding_dino_box_threshold: float = 0.3
    grounding_dino_text_threshold: float = 0.25


# Auto-load GEMINI_API_KEY from .env at module import if not already in environment
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


class GeminiObjectDetector:
    """Detects objects in egocentric video frames using Gemini + Grounding DINO.
    
    Two-stage detection:
    1. Gemini Vision identifies object names and touch state (semantic)
    2. Grounding DINO provides precise bounding boxes for each object (spatial)
    """
    
    def __init__(self, config: ObjectDetectorConfig):
        self.config = config
        self._model = None
        self._grounding_detector = None
        self._init_model()
        self._init_grounding_dino()
    
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
        print(f"[ObjectDetector] Using Gemini: {model_name}")
    
    def _init_grounding_dino(self):
        """Initialize Grounding DINO detector for bbox localization."""
        gd_config = GroundingDINOConfig(
            model_name=self.config.grounding_dino_model,
            confidence_threshold=self.config.grounding_dino_confidence,
            box_threshold=self.config.grounding_dino_box_threshold,
            text_threshold=self.config.grounding_dino_text_threshold,
        )
        self._grounding_detector = create_grounding_detector(gd_config)
        if self._grounding_detector:
            print(f"[ObjectDetector] Grounding DINO: {self.config.grounding_dino_model} (bbox localization enabled)")
        else:
            print("[ObjectDetector] Grounding DINO: unavailable (will use Gemini location descriptions only)")
    
    def detect_objects(self, video_path: str) -> List[ObjectAnnotation]:
        """Detect objects from keyframes using Gemini only (no bboxes).
        
        Use detect_objects_with_bboxes() for full two-stage detection with bounding boxes.
        """
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
        print(f"[ObjectDetector] {len(result)} objects (Gemini only): {[o.name for o in result]}")
        return result
    
    def detect_objects_with_bboxes(self, video_path: str) -> List[ObjectAnnotation]:
        """Two-stage detection: Gemini for semantics + Grounding DINO for bboxes.
        
        Returns ObjectAnnotation list with bbox field populated for each object.
        """
        # Stage 1: Gemini detects object names + touch state
        gemini_objects = self.detect_objects(video_path)
        
        if not gemini_objects:
            return []
        
        # Stage 2: Grounding DINO localizes each object
        if self._grounding_detector is None:
            print("[ObjectDetector] Grounding DINO not available, returning Gemini results without bboxes")
            return gemini_objects
        
        # Extract object names for Grounding DINO
        object_names = [obj.name for obj in gemini_objects]
        
        # We need to run Grounding DINO on a representative frame
        # Use the middle keyframe
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("[ObjectDetector] Could not read frame for Grounding DINO, returning Gemini results")
            return gemini_objects
        
        # Run Grounding DINO detection
        dino_results = self._grounding_detector.detect(frame, object_names)
        
        # Merge: match Grounding DINO results to Gemini objects by name
        dino_by_name = {obj.name: obj for obj in dino_results}
        
        merged = []
        for gemini_obj in gemini_objects:
            if gemini_obj.name in dino_by_name:
                dino_obj = dino_by_name[gemini_obj.name]
                # Create merged object with Gemini semantics + DINO bbox
                merged.append(ObjectAnnotation(
                    name=gemini_obj.name,
                    location_description=dino_obj.location_description,  # Use DINO's spatial desc
                    touched=gemini_obj.touched,  # Keep Gemini's touch judgment
                    bbox=dino_obj.bbox,  # Real bbox from DINO
                    state=gemini_obj.state
                ))
            else:
                # Object detected by Gemini but not localized by DINO
                merged.append(gemini_obj)
        
        print(f"[ObjectDetector] {len(merged)} objects with bboxes: {[o.name for o in merged]}")
        return merged

    def detect_per_frame_objects_with_bboxes(
        self, video_path: str, image_paths: List[str]
    ) -> List[List[ObjectAnnotation]]:
        """Two-stage detection with per-frame bbox tracking across all frames.
        
        1. Gemini detects object names + touch state on keyframes (semantics)
        2. Grounding DINO runs on keyframes at config.bbox_keyframe_interval
        3. Linearly interpolates / propagates bboxes between keyframes for 100% per-frame coverage
        
        Returns:
            List of ObjectAnnotation lists, one per frame in image_paths.
        """
        gemini_objects = self.detect_objects(video_path)
        if not gemini_objects:
            return [[] for _ in image_paths]

        if self._grounding_detector is None or not image_paths:
            print("[ObjectDetector] Grounding DINO not available, returning Gemini results for all frames")
            return [gemini_objects for _ in image_paths]

        object_names = [obj.name for obj in gemini_objects]
        interval = max(1, self.config.bbox_keyframe_interval)
        num_frames = len(image_paths)

        # Step 1: Detect DINO bboxes on keyframe images
        keyframe_bboxes: dict = {}
        for idx in range(0, num_frames, interval):
            img_path = image_paths[idx]
            frame_img = cv2.imread(img_path)
            if frame_img is not None:
                dino_results = self._grounding_detector.detect(frame_img, object_names)
                if dino_results:
                    keyframe_bboxes[idx] = {obj.name: obj.bbox for obj in dino_results if obj.bbox is not None}

        # Ensure last frame is also sampled if needed
        last_idx = num_frames - 1
        if last_idx not in keyframe_bboxes and last_idx > 0:
            frame_img = cv2.imread(image_paths[last_idx])
            if frame_img is not None:
                dino_results = self._grounding_detector.detect(frame_img, object_names)
                if dino_results:
                    keyframe_bboxes[last_idx] = {obj.name: obj.bbox for obj in dino_results if obj.bbox is not None}

        # Step 2: Interpolate/propagate per-frame bboxes
        sorted_k_indices = sorted(keyframe_bboxes.keys())
        per_frame_results: List[List[ObjectAnnotation]] = []

        for i in range(num_frames):
            k_left = max([k for k in sorted_k_indices if k <= i], default=None)
            k_right = min([k for k in sorted_k_indices if k >= i], default=None)

            frame_objects = []
            for g_obj in gemini_objects:
                name = g_obj.name
                box_left = keyframe_bboxes.get(k_left, {}).get(name) if k_left is not None else None
                box_right = keyframe_bboxes.get(k_right, {}).get(name) if k_right is not None else None

                computed_bbox = None
                if box_left is not None and box_right is not None:
                    if k_left == k_right:
                        computed_bbox = box_left.copy()
                    else:
                        alpha = (i - k_left) / (k_right - k_left)
                        computed_bbox = (1.0 - alpha) * box_left + alpha * box_right
                elif box_left is not None:
                    computed_bbox = box_left.copy()
                elif box_right is not None:
                    computed_bbox = box_right.copy()

                loc_desc = (
                    self._grounding_detector._bbox_to_location(computed_bbox)
                    if computed_bbox is not None
                    else g_obj.location_description
                )

                frame_objects.append(
                    ObjectAnnotation(
                        name=g_obj.name,
                        location_description=loc_desc,
                        touched=g_obj.touched,
                        bbox=computed_bbox,
                        state=g_obj.state,
                    )
                )

            per_frame_results.append(frame_objects)

        print(
            f"[ObjectDetector] Generated per-frame bboxes for {num_frames} frames "
            f"({len(keyframe_bboxes)} DINO keyframe samples at interval {interval})"
        )
        return per_frame_results
    
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
