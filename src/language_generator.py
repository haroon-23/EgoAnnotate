"""Natural language instruction generation using a VLM backend (Fail-loud, zero canned text)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .datatypes import ActionSegment
from .vlm_backend import (
    VLMBackend,
    VLMRequest,
    ensure_api_key_from_dotenv,
    resolve_stage_backend,
)

ensure_api_key_from_dotenv()

MAX_RETRIES = 2
VIDEO_UPLOAD_TIMEOUT = 10.0


@dataclass
class LanguageGeneratorConfig:
    """Configuration for the GeminiLanguageGenerator."""
    gemini_model: str = "gemini-2.5-flash"
    vlm_backend: Optional[str] = None
    episode_prompt: str = (
        "Summarize the overall physical manipulation task performed in this egocentric video in one concise sentence."
    )
    segment_prompt: str = (
        "Based on the video and the provided temporal segments, generate a single concise natural-language description "
        "for each segment. Respond strictly in the format:\nSegment 1: <description>\nSegment 2: <description>"
    )


class GeminiLanguageGenerator:
    """Generates natural language descriptions without silent canned fallbacks."""

    def __init__(self, config: LanguageGeneratorConfig, vlm: Optional[VLMBackend] = None):
        self.config = config
        self.model = None
        self._vlm: VLMBackend = (
            vlm
            if vlm is not None
            else resolve_stage_backend(config.vlm_backend, config.gemini_model)
        )
        print(f"[LanguageGenerator] Using VLM backend: {self._vlm.name}")

    def generate_episode_description(self, video_path: str) -> str:
        """Generate one-sentence task description. Raises RuntimeError on failure (never canned text)."""
        prompt = (
            self.config.episode_prompt
            or "Describe the physical task in this video in one sentence. Be specific. Return ONLY the sentence."
        )
        result = self._call_with_video(Path(video_path), prompt)
        if not result or not result.strip():
            raise RuntimeError(
                f"[LanguageGenerator] VLM returned empty episode description for {video_path}. "
                "Refusing to substitute canned placeholder text."
            )
        cleaned = result.strip().strip('"').strip("'")
        cleaned = re.sub(r"\*\*", "", cleaned).strip()
        if not cleaned or cleaned.lower() in ("manipulating object", "unknown", "n/a"):
            raise RuntimeError(
                f"[LanguageGenerator] VLM returned invalid placeholder description {cleaned!r} for {video_path}."
            )
        words = cleaned.split()
        if len(words) > 50:
            cleaned = " ".join(words[:50])
        return cleaned

    def generate_segment_descriptions(self, video_path: str, segments: List[ActionSegment]) -> List[str]:
        """Generate descriptions for segments. Raises RuntimeError on VLM failure."""
        if not segments:
            return []

        segment_info = "\n".join([
            f"Segment {i+1}: name='{seg.name}', start_time={seg.start_time}s, "
            f"end_time={seg.end_time}s, object_name='{seg.object_name}', hand_used='{seg.hand_used}'"
            for i, seg in enumerate(segments)
        ])
        prompt = f"{self.config.segment_prompt}\n\nHere are the segments to describe:\n{segment_info}"
        result = self._call_with_video(Path(video_path), prompt)
        if not result or not result.strip():
            raise RuntimeError(
                f"[LanguageGenerator] VLM returned empty segment descriptions for {video_path}."
            )

        descriptions_map = {}
        for line in result.strip().split("\n"):
            line = line.strip()
            match = re.match(r"Segment\s+(\d+)\s*:\s*(.*)", line, re.IGNORECASE)
            if match:
                seg_num = int(match.group(1))
                desc = match.group(2).strip()
                if desc:
                    descriptions_map[seg_num] = desc

        if not descriptions_map and len(segments) == 1:
            single = result.strip().splitlines()[0].strip()
            if single:
                descriptions_map[1] = single

        if len(descriptions_map) < len(segments):
            missing = [i + 1 for i in range(len(segments)) if (i + 1) not in descriptions_map]
            raise RuntimeError(
                f"[LanguageGenerator] VLM response missing descriptions for segment(s) {missing}."
            )

        return [descriptions_map[i + 1] for i in range(len(segments))]

    def _call_with_video(self, video_path: Path, prompt: str) -> Optional[str]:
        last_exc: Optional[BaseException] = None
        for attempt in range(MAX_RETRIES):
            try:
                response_text = self._vlm.generate(
                    VLMRequest(
                        video_path=str(video_path),
                        prompt=prompt,
                        temperature=0.1,
                        video_upload_timeout_s=VIDEO_UPLOAD_TIMEOUT,
                    )
                ).text
                if response_text and response_text.strip():
                    return response_text
                raise RuntimeError("Empty response text from VLM backend")
            except RuntimeError:
                raise
            except BaseException as e:
                last_exc = e
        raise RuntimeError(f"[LanguageGenerator] VLM call failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc
