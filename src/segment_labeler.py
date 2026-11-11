from __future__ import annotations
"""Segment labeling using a VLM backend (Phase D dual backend).

Takes pre-computed CandidateSegments from SignalSegmenter and labels each
with an action category. Does NOT derive boundaries — only labels existing
segments.
"""


import logging
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

# Phase D: VLM dual backend (gemini default, local SmolVLM opt-in).
from .vlm_backend import (
    VLMBackend,
    VLMRequest,
    ensure_api_key_from_dotenv,
    parse_json_strict,
    resolve_stage_backend,
)

# Legacy behavior: pick up GEMINI_API_KEY from .env at import time.
ensure_api_key_from_dotenv()

from .datatypes import ActionSegment, CandidateSegment

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 3
VIDEO_UPLOAD_TIMEOUT = 30


@dataclass
class SegmentLabelerConfig:
    """Configuration for the SegmentLabeler."""
    gemini_model: str = "gemini-3.8-flash"
    # Phase D: VLM backend for this stage ("gemini" | "local" | None).
    # None -> the pipeline's global `vlm.backend` (default "gemini").
    vlm_backend: Optional[str] = None
    no_vlm: bool = False
    require_vlm: bool = False
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


def build_instruction(segment_name: str, object_name: str, hands: list[str]) -> str:
    """hands: list like ["left"], ["right"], ["left","right"]. Never emits
    'the unknown' or 'idle the <object>'."""
    if segment_name == "idle" or not object_name:
        return "idle (no object)"
    obj = object_name if object_name != "unknown" else "an unidentified object"
    who = "both hands" if len(hands) == 2 else f"{hands[0]} hand"
    return f"{segment_name} {obj} with {who}"


class SegmentLabeler:
    """Labels pre-computed segments with action categories using a VLM backend."""

    def __init__(self, config: SegmentLabelerConfig, vlm: Optional[VLMBackend] = None):
        self.config = config
        self._model = None
        # Phase D: VLM backend. Injected by the pipeline, or resolved here
        # (strict — raises with the legacy messages when unavailable).
        self._vlm: VLMBackend = (
            vlm
            if vlm is not None
            else resolve_stage_backend(config.vlm_backend, config.gemini_model)
        )
        print(f"[SegmentLabeler] Using VLM backend: {self._vlm.name}")
    
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
        
        # Mission 3.5 Item 6: Sub-split at signal transitions / max duration (<= 10.0s)
        # BEFORE labeling so each sub-window extracts its own keyframe and makes its own VLM call.
        sub_candidates: List[tuple[CandidateSegment, int, int]] = []
        max_duration = 10.0
        for cand in candidates:
            dur = cand.end_time - cand.start_time
            if dur <= max_duration:
                sub_candidates.append((cand, 1, 1))
            else:
                n_chunks = int(np.ceil(dur / max_duration))
                chunk_dur = dur / float(n_chunks)
                chunk_frames = max(1, (cand.end_frame - cand.start_frame) // n_chunks)
                for c_i in range(n_chunks):
                    t0 = round(cand.start_time + c_i * chunk_dur, 3)
                    t1 = round(cand.start_time + (c_i + 1) * chunk_dur if c_i < n_chunks - 1 else cand.end_time, 3)
                    f0 = cand.start_frame + c_i * chunk_frames
                    f1 = cand.start_frame + (c_i + 1) * chunk_frames if c_i < n_chunks - 1 else cand.end_frame
                    sub_cand = CandidateSegment(
                        start_frame=f0,
                        end_frame=f1,
                        start_time=t0,
                        end_time=t1,
                        transition_type=cand.transition_type,
                        contact_state=cand.contact_state,
                        grasp_type=cand.grasp_type,
                        object_name=cand.object_name,
                        hands=cand.hands,
                        hand_used=cand.hand_used,
                    )
                    sub_candidates.append((sub_cand, c_i + 1, n_chunks))

        labeled_segments = []
        for seg, phase_idx, total_phases in sub_candidates:
            label = self._label_single_segment(seg, video_path, phase_idx=phase_idx, total_phases=total_phases)
            hands = getattr(seg, "hands", ["right"])
            hand_used = getattr(seg, "hand_used", "both" if len(hands) == 2 else (hands[0] if hands else "right"))
            action_name = label.get("action", "idle")
            obj_name = seg.object_name if (action_name != "idle" and seg.contact_state != "no_contact") else None
            
            base_desc = build_instruction(action_name, obj_name, hands)
            if total_phases > 1:
                phase_verb = "initiating " if phase_idx == 1 else ("completing " if phase_idx == total_phases else "continuing ")
                default_desc = f"{phase_verb}{base_desc}"
            else:
                default_desc = base_desc
                
            desc = label.get("description") or default_desc
            if total_phases > 1 and label.get("description") and str(phase_idx) not in desc and "phase" not in desc.lower():
                phase_suffix = f" (phase {phase_idx}/{total_phases})"
                desc = f"{desc}{phase_suffix}"
                
            caption_src = label.get("caption_source", "template")

            labeled_segments.append(ActionSegment(
                name=action_name,
                start_time=seg.start_time,
                end_time=seg.end_time,
                object_name=seg.object_name or "unknown",
                hand_used=hand_used,
                description=desc,
                hands=hands,
                caption_source=caption_src,
            ))
        
        # Enforce: no repeated identical text across phases or consecutive segments
        for i in range(1, len(labeled_segments)):
            prev_txt = labeled_segments[i - 1].description.strip().lower()
            curr_txt = labeled_segments[i].description.strip().lower()
            if curr_txt == prev_txt:
                labeled_segments[i].description = f"continuing {labeled_segments[i].description}"

        # Merge any segments < 0.50s per Gate 11 requirement
        cleaned: List[ActionSegment] = []
        for s in labeled_segments:
            if not cleaned:
                cleaned.append(s)
            elif s.duration < 0.50:
                prev = cleaned[-1]
                prev.end_time = s.end_time
                if s.caption_source == "vlm" and prev.caption_source != "vlm":
                    prev.caption_source = "vlm"
            else:
                cleaned.append(s)
        if len(cleaned) > 1 and cleaned[0].duration < 0.50:
            cleaned[1].start_time = cleaned[0].start_time
            cleaned.pop(0)

        return cleaned
    
    def _label_single_segment(
        self,
        seg: CandidateSegment,
        video_path: str,
        phase_idx: int = 1,
        total_phases: int = 1,
    ) -> dict:
        """Label a single segment using its own keyframe and context."""
        if getattr(self.config, "no_vlm", False):
            return self._default_label(seg)

        # Extract keyframe at this sub-window's midpoint
        keyframe = self._extract_keyframe(video_path, seg)
        
        if keyframe is None:
            if getattr(self.config, "require_vlm", False):
                raise RuntimeError(f"Could not extract keyframe for segment {seg.start_time}-{seg.end_time} (--require-vlm)")
            logger.warning(f"Could not extract keyframe for segment {seg.start_time}-{seg.end_time}")
            return self._default_label(seg)
        
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
        if total_phases > 1:
            prompt += f"\n  Phase progression: phase {phase_idx} of {total_phases} (describe the distinct motion in this phase without repeating identical descriptions across phases)\n"

        
        # Call the VLM backend with retries
        try:
            if getattr(self._vlm, "name", None) == "local":
                if getattr(self.config, "require_vlm", False):
                    raise RuntimeError(f"VLM required but local backend unavailable (--require-vlm)")
                logger.warning(f"VLM call failed for segment {seg.start_time}-{seg.end_time}, using default")
                return self._default_label(seg)
            result = self._call_vlm_with_retry(keyframe, prompt)
        except Exception as e:
            if getattr(self.config, "require_vlm", False):
                raise RuntimeError(f"Segment labeling failed with --require-vlm: {e}") from e
            logger.warning(f"VLM call exception for segment {seg.start_time}-{seg.end_time}: {e}, using default")
            result = None

        if result is None:
            if getattr(self.config, "require_vlm", False):
                raise RuntimeError(f"Segment labeling failed after retries for segment {seg.start_time}-{seg.end_time} (--require-vlm)")
            logger.warning(f"VLM call failed for segment {seg.start_time}-{seg.end_time}, using default")
            return self._default_label(seg)

        return result
    
    def _extract_keyframe(self, video_path: str, seg: CandidateSegment) -> Optional[Image.Image]:
        """Extract a frame at segment midpoint with privacy face blurring."""
        from .privacy import blur_faces_in_frame

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
        
        blurred_bgr, _ = blur_faces_in_frame(frame)
        frame_rgb = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize if too large
        if pil_image.width > 1024 or pil_image.height > 1024:
            pil_image.thumbnail((1024, 1024))
        
        return pil_image
    
    def _call_vlm_with_retry(self, image: Image.Image, prompt: str) -> Optional[dict]:
        """Call the VLM backend with retry logic.

        Output contract: strict JSON parse -> one repair re-prompt; failure
        returns None and the caller falls back to the heuristic default label.
        """

        for attempt in range(MAX_RETRIES):
            try:
                response_text = self._vlm.generate(
                    VLMRequest(images=[image], prompt=prompt, temperature=0.1)
                ).text
                parsed = parse_json_strict(
                    response_text,
                    repair_fn=lambda rp: self._vlm.generate(
                        VLMRequest(
                            images=[image], prompt=rp,
                            max_new_tokens=128, temperature=0.1,
                        )
                    ).text,
                    expect="JSON object",
                )
                if parsed is None:
                    return None
                return self._validate_label_dict(parsed)

            except RuntimeError:
                # Fatal backend errors (auth, missing weights) propagate
                # immediately — never retried.
                raise
            except Exception as e:
                err = str(e).lower()

                # Fatal or network unreachable errors
                if any(k in err for k in ["404", "not found", "no longer available",
                                          "invalid model", "api key not valid", "permission denied",
                                          "dns", "could not contact dns", "address lookup failed"]):
                    logger.error(f"[FATAL/NETWORK UNREACHABLE] {e}")
                    return None

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
        """Parse VLM response for action label and description (pure; no repair).

        The live path (:meth:`_call_vlm_with_retry`) adds one repair re-prompt
        before giving up.
        """
        if not text:
            return None
        parsed = parse_json_strict(text, repair_fn=None)
        if parsed is None or not isinstance(parsed, dict):
            return None
        return self._validate_label_dict(parsed)

    @staticmethod
    def _validate_label_dict(data: dict) -> dict:
        """Validate a parsed label dict against the 7-category schema."""
        action = str(data.get("action", "unlabeled")).strip().lower()
        desc = str(data.get("description", "")).strip()

        # Validate action
        valid_actions = {"approach", "contact", "grasp", "manipulate", "release", "retreat", "idle"}
        if action not in valid_actions:
            action = "unlabeled"

        return {"action": action, "description": desc, "caption_source": "vlm"}
    
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
        
        hands = getattr(seg, "hands", ["right"])
        obj_name = seg.object_name if (action != "idle" and seg.contact_state != "no_contact") else None
        desc = build_instruction(action, obj_name, hands)
        return {
            "action": action,
            "description": desc,
            "caption_source": "template",
        }


def create_segment_labeler(
    config: Optional[SegmentLabelerConfig] = None,
    vlm: Optional[VLMBackend] = None,
) -> Optional[SegmentLabeler]:
    """Factory function with graceful degradation."""
    if config is None:
        config = SegmentLabelerConfig()

    try:
        return SegmentLabeler(config, vlm=vlm)
    except Exception as e:
        logger.warning(f"Failed to create SegmentLabeler: {e}")
        return None
