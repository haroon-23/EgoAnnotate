"""Natural language instruction generation using a VLM backend (Phase D dual backend)."""
import time
import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# Phase D: VLM dual backend (gemini default, local SmolVLM opt-in).
from .vlm_backend import (
    VLMBackend,
    VLMRequest,
    ensure_api_key_from_dotenv,
    resolve_stage_backend,
)

# Legacy behavior: pick up GEMINI_API_KEY from .env at import time.
ensure_api_key_from_dotenv()

from .datatypes import ActionSegment


MAX_RETRIES = 2
RETRY_DELAY = 3
VIDEO_UPLOAD_TIMEOUT = 30


@dataclass
class LanguageGeneratorConfig:
    """Configuration for the GeminiLanguageGenerator."""
    gemini_model: str = "gemini-3.8-flash"
    # Phase D: VLM backend for this stage ("gemini" | "local" | None).
    # None -> the pipeline's global `vlm.backend` (default "gemini"; the
    # video stages keep the gemini default because they rely on native
    # video understanding).
    vlm_backend: Optional[str] = None
    episode_prompt: str = (
        "Summarize the overall task performed in this egocentric video in one concise sentence (e.g. 'cooking pasta' or 'assembling a table')."
    )
    segment_prompt: str = (
        "Based on the video and the provided temporal segments, generate a single concise natural-language description for each segment. Respond strictly in the format:\nSegment 1: <description>\nSegment 2: <description>"
    )


class GeminiLanguageGenerator:
    """Generates natural language descriptions."""
    
    def __init__(self, config: LanguageGeneratorConfig, vlm: Optional[VLMBackend] = None):
        self.config = config
        self.model = None
        # Phase D: VLM backend. Injected by the pipeline, or resolved here
        # (strict — raises with the legacy messages when unavailable).
        self._vlm: VLMBackend = (
            vlm
            if vlm is not None
            else resolve_stage_backend(config.vlm_backend, config.gemini_model)
        )
        print(f"[LanguageGenerator] Using VLM backend: {self._vlm.name}")
    
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
            print(f"[LanguageGenerator] VLM call failed ({e}), using default task description")
        
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
            print(f"[LanguageGenerator] VLM call failed for segment descriptions ({e}), using default fallback")
        
        # Build final descriptions list, falling back to build_instruction
        descriptions = []
        from .segment_labeler import build_instruction
        for idx, seg in enumerate(segments):
            seg_num = idx + 1
            if seg_num in descriptions_map and descriptions_map[seg_num]:
                descriptions.append(descriptions_map[seg_num])
            else:
                obj_name = seg.object_name if (seg.name != "idle" and seg.object_name != "unknown") else None
                descriptions.append(build_instruction(seg.name, obj_name, seg.hands))
                
        return descriptions
    
    def _call_with_video(self, video_path: Path, prompt: str) -> Optional[str]:
        for attempt in range(MAX_RETRIES):
            try:
                # Phase D: the backend owns video handling — Gemini uploads the
                # file natively, the local backend samples timestamped stills.
                response_text = self._vlm.generate(
                    VLMRequest(
                        video_path=str(video_path),
                        prompt=prompt,
                        temperature=0.1,
                        video_upload_timeout_s=10.0,
                    )
                ).text

                return response_text if response_text else None

            except RuntimeError:
                # Fatal backend errors (auth, missing weights, upload failure)
                # propagate immediately — never retried.
                raise
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
