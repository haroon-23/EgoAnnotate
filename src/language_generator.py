"""Natural language instruction generation using Gemini Vision API."""
import os
import time
import re
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
VIDEO_UPLOAD_TIMEOUT = 30


@dataclass
class LanguageGeneratorConfig:
    """Configuration for the GeminiLanguageGenerator."""
    gemini_model: str = "gemini-1.5-pro-latest"
    episode_prompt: str = (
        "Summarize the overall task performed in this egocentric video in one concise sentence (e.g. 'cooking pasta' or 'assembling a table')."
    )
    segment_prompt: str = (
        "Based on the video and the provided temporal segments, generate a single concise natural-language description for each segment. Respond strictly in the format:\nSegment 1: <description>\nSegment 2: <description>"
    )


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


class GeminiLanguageGenerator:
    """Generates natural language descriptions."""
    
    def __init__(self, config: LanguageGeneratorConfig):
        self.config = config
        self.model = None
        self._init_model()
    
    def _init_model(self):
        if not GEMINI_AVAILABLE:
            raise RuntimeError("google-generativeai not installed.")
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to use GeminiLanguageGenerator.")
        
        genai.configure(api_key=api_key, transport='rest')
        model_name = self.config.gemini_model
        if model_name in ["gemini-1.5-flash", "gemini-flash-latest", "models/gemini-1.5-flash", "models/gemini-flash-latest"]:
            model_name = "gemini-flash-lite-latest"
            
        if not model_name.startswith("models/") and model_name != "gemini-1.5-pro-latest":
            model_name = f"models/{model_name}"
        self.model = genai.GenerativeModel(model_name)
        print(f"[LanguageGenerator] Using {model_name}")
    
    def generate_episode_description(self, video_path: str) -> str:
        """Generate one-sentence task description. Fast fallback on failure."""
        prompt = self.config.episode_prompt or "Describe the physical task in this video in one sentence. Be specific. Return ONLY the sentence."
        
        try:
            result = self._call_with_video(Path(video_path), prompt)
            if result:
                result = result.strip().strip('"').strip("'")
                result = re.sub(r'\*\*', '', result)
                
                # Truncate to max 50 tokens (words)
                words = result.split()
                if len(words) > 50:
                    result = " ".join(words[:50])
                return result
        except Exception as e:
            print(f"[LanguageGenerator] Gemini call failed ({e}), using default task description")
        
        return "manipulating object"
    
    def generate_segment_descriptions(self, video_path: str, segments: List[ActionSegment]) -> List[str]:
        """Generate descriptions for segments."""
        if not segments:
            return []
        
        segment_info = "\n".join([
            f"Segment {i+1}: name='{seg.name}', start_time={seg.start_time}s, "
            f"end_time={seg.end_time}s, object_name='{seg.object_name}', hand_used='{seg.hand_used}'"
            for i, seg in enumerate(segments)
        ])
        
        prompt = f"{self.config.segment_prompt}\n\nHere are the segments to describe:\n{segment_info}"
        
        descriptions_map = {}
        try:
            result = self._call_with_video(Path(video_path), prompt)
            if result:
                lines = result.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    match = re.match(r'Segment\s+(\d+)\s*:\s*(.*)', line, re.IGNORECASE)
                    if match:
                        seg_num = int(match.group(1))
                        desc = match.group(2).strip()
                        descriptions_map[seg_num] = desc
        except Exception as e:
            print(f"[LanguageGenerator] Gemini call failed for segment descriptions ({e}), using default fallback")
        
        # Build final descriptions list, falling back to "{name_clean} the {object_name}"
        descriptions = []
        for idx, seg in enumerate(segments):
            seg_num = idx + 1
            if seg_num in descriptions_map and descriptions_map[seg_num]:
                descriptions.append(descriptions_map[seg_num])
            else:
                name_clean = seg.name.replace('_', ' ')
                descriptions.append(f"{name_clean} the {seg.object_name}")
                
        return descriptions
    
    def _call_with_video(self, video_path: Path, prompt: str) -> Optional[str]:
        for attempt in range(MAX_RETRIES):
            try:
                video_file = genai.upload_file(str(video_path))
                
                waited = 0
                while video_file.state.name == "PROCESSING" and waited < 5:
                    time.sleep(1)
                    waited += 1
                    try:
                        video_file = genai.get_file(video_file.name)
                    except Exception:
                        pass
                
                if video_file.state.name != "ACTIVE":
                    return None
                
                response = self.model.generate_content([video_file, prompt], generation_config={"temperature": 0.1}, request_options={"timeout": 5.0})
                
                try:
                    genai.delete_file(video_file.name)
                except Exception:
                    pass
                
                return response.text if response and response.text else None
            
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
                    return None
        
        return None
