"""Segment naming and task instruction from ONE batched VLM request per episode.

Segments (boundaries, contact, grasp, hands, object) are computed
deterministically upstream. The VLM only names them and writes the task
instruction. Without a usable VLM, or once the request budget is spent, the
same fields come from a deterministic template built from the measured
signals, and every string carries its source (``vlm:<backend>:<model>`` or
``template``) so template text is never presented as model output.
"""

from __future__ import annotations

import logging
import time
import weakref
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import cv2
from PIL import Image

from .vlm_backend import (
    BudgetExceeded,
    VLMBackend,
    VLMRequest,
    ensure_api_key_from_dotenv,
    parse_json_strict,
    resolve_stage_backend,
    sample_video_frames,
)

# Legacy behavior: pick up GEMINI_API_KEY from .env at import time.
ensure_api_key_from_dotenv()

from .datatypes import ActionSegment, CandidateSegment

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 3
VIDEO_UPLOAD_TIMEOUT = 30

#: Bump when ``build_episode_prompt`` changes; feed it to ``make_stage_key(prompt=...)``.
EPISODE_PROMPT_VERSION = "episode-v1"
VALID_ACTIONS = ("approach", "contact", "grasp", "manipulate", "release", "retreat", "idle")
TEMPLATE_SOURCE = "template"
_PLACEHOLDERS = {"manipulating object", "unknown", "n/a", "none", ""}


def vlm_source(vlm: Any) -> str:
    """Source label for text produced by ``vlm``: ``vlm:<backend>:<model>``."""
    model = str(getattr(vlm, "model_id", "") or "")
    return f"vlm:{getattr(vlm, 'name', 'unknown')}" + (f":{model}" if model else "")


def is_null_backend(vlm: Any) -> bool:
    return getattr(vlm, "name", None) == "none"


@dataclass
class SegmentLabelerConfig:
    """Configuration for the SegmentLabeler."""
    gemini_model: str = "gemini-2.5-flash"
    # VLM backend for this stage when none is injected
    # ("auto" | "gemini" | "openai_compatible" | "local" | "none" | None).
    vlm_backend: Optional[str] = None
    # True: name all segments + write the task instruction in ONE request.
    # False: legacy one request per segment. None: batch iff the backend is a
    # ManagedBackend (i.e. it has a request budget).
    batch_requests: Optional[bool] = None
    keyframes_per_request: int = 5
    keyframe_max_side: int = 448
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


# ---------------------------------------------------------------------------
# Deterministic templates (no model)
# ---------------------------------------------------------------------------


def template_label(seg: CandidateSegment) -> dict:
    """Heuristic action name + description from the measured signals only."""
    if seg.contact_state == "no_contact":
        if seg.transition_type == "start":
            action = "idle"
        elif seg.transition_type == "contact_off":
            action = "retreat"
        else:
            action = "approach"
    else:
        if seg.transition_type == "contact_on":
            action = "contact"
        elif seg.grasp_type in ("precision_pinch", "power_wrap", "hook"):
            action = "grasp"
        else:
            action = "manipulate"
    hands = getattr(seg, "hands", None) or ["right"]
    obj_name = seg.object_name if (action != "idle" and seg.contact_state != "no_contact") else None
    return {"action": action, "description": build_instruction(action, obj_name, hands)}


def _with_article(name: str) -> str:
    n = name.strip()
    if n.lower().split(" ", 1)[0] in ("the", "a", "an"):
        return n
    return f"the {n}"


def template_instruction(
    candidates: Sequence[CandidateSegment], object_names: Optional[Sequence[str]] = None
) -> str:
    """Task instruction from measured object / grasp / contact / hand only."""
    contact = [c for c in candidates if c.contact_state == "contact"]
    if not contact:
        return "No hand-object interaction detected."
    dur: Dict[str, float] = {}
    for c in contact:
        name = (c.object_name or "").strip()
        if name and name != "unknown":
            dur[name] = dur.get(name, 0.0) + max(0.0, c.end_time - c.start_time)
    if dur:
        obj = _with_article(max(dur.items(), key=lambda kv: kv[1])[0])
    elif object_names:
        obj = _with_article(str(object_names[0]))
    else:
        obj = "an unidentified object"
    hands = set()
    for c in contact:
        hands.update(getattr(c, "hands", None) or ["right"])
    if len(hands) >= 2:
        who = "both hands"
    else:
        who = f"the {next(iter(hands))} hand"
    grasps = Counter(c.grasp_type for c in contact if c.grasp_type and c.grasp_type != "none")
    grasp = f" using a {grasps.most_common(1)[0][0].replace('_', ' ')} grasp" if grasps else ""
    return f"Manipulate {obj} with {who}{grasp}."


# ---------------------------------------------------------------------------
# Batched episode annotation: one request -> segment names + task instruction
# ---------------------------------------------------------------------------


@dataclass
class EpisodeAnnotation:
    task_instruction: str
    instruction_source: str  # "vlm:<backend>:<model>" | "template"
    segment_labels: List[dict] = field(default_factory=list)  # {"action","description"}
    segment_sources: List[str] = field(default_factory=list)
    degraded_reason: str = ""  # why a template was used, "" when fully VLM
    requests_used: int = 0
    cache_hit: bool = False

    @property
    def language_source(self) -> str:
        return "template" if self.instruction_source == TEMPLATE_SOURCE else "vlm"

    @property
    def from_model(self) -> bool:
        """True only if every produced string came from the VLM."""
        return self.instruction_source != TEMPLATE_SOURCE and all(
            s != TEMPLATE_SOURCE for s in self.segment_sources
        )

    def to_dict(self) -> dict:
        return {
            "task_instruction": self.task_instruction,
            "language_source": self.language_source,
            "instruction_source": self.instruction_source,
            "segment_sources": list(self.segment_sources),
            "degraded_reason": self.degraded_reason,
            "requests_used": self.requests_used,
            "cache_hit": self.cache_hit,
        }


def _segment_table(candidates: Sequence[CandidateSegment]) -> str:
    rows = []
    for i, c in enumerate(candidates, 1):
        hands = ",".join(getattr(c, "hands", None) or ["right"])
        rows.append(
            f"{i}. {c.start_time:.2f}-{c.end_time:.2f}s | contact={c.contact_state} | "
            f"grasp={c.grasp_type} | hands={hands} | object={c.object_name or 'unknown'} | "
            f"starts_with={c.transition_type}"
        )
    return "\n".join(rows)


def build_episode_prompt(
    candidates: Sequence[CandidateSegment],
    frame_times: Sequence[float],
    object_names: Optional[Sequence[str]] = None,
) -> str:
    n = len(candidates)
    frames = "; ".join(f"image {i + 1} at t={t:.1f}s" for i, t in enumerate(frame_times))
    parts = [
        "You are annotating an egocentric (first-person) video of a person using their hands.",
        f"You are given {len(frame_times)} keyframes in temporal order: {frames}.",
    ]
    if n:
        parts += [
            "The segments below were MEASURED automatically from hand tracking. Do not change "
            "their times, hands, contact or grasp. Only name each segment and write one task "
            "instruction for the whole video.",
            "",
            "Segments:",
            _segment_table(candidates),
        ]
    else:
        parts += ["Write one task instruction for the whole video."]
    if object_names:
        parts += ["", "Objects detected in the scene: " + ", ".join(map(str, object_names)) + "."]
    schema = (
        '{"task_instruction": "<imperative sentence, at most 20 words, e.g. '
        "'Fold the shirt on the bed'>\", \"segments\": "
    )
    if n:
        schema += (
            '[{"index": 1, "action": "<one of: ' + ", ".join(VALID_ACTIONS) + '>", '
            '"description": "<at most 12 words>"}]}'
        )
    else:
        schema += "[]}"
    parts += [
        "",
        "Return ONLY strict JSON, no markdown, no commentary:",
        schema,
    ]
    if n:
        parts.append(f'Include exactly {n} entries in "segments", with index 1..{n}.')
    parts.append("Describe only what is visible; do not invent objects.")
    return "\n".join(parts)


def _clean_instruction(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None
    t = text.strip().strip('"').strip("'").replace("**", "").strip()
    if t.lower() in _PLACEHOLDERS:
        return None
    words = t.split()
    return " ".join(words[:40]) if words else None


class EpisodeAnnotator:
    """Builds one request per episode and degrades to templates, with sources.

    Results are memoized per (video, segments) so the segment labeler and the
    language generator share a single request when they share a backend
    (see :meth:`for_backend`).
    """

    _registry: "weakref.WeakKeyDictionary[Any, EpisodeAnnotator]" = weakref.WeakKeyDictionary()

    def __init__(
        self,
        vlm: VLMBackend,
        keyframes: int = 5,
        keyframe_max_side: int = 448,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        self.vlm = vlm
        self.keyframes = max(1, min(int(keyframes), 8))
        self.keyframe_max_side = int(keyframe_max_side)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self._memo: Dict[Any, EpisodeAnnotation] = {}
        self._latest: Dict[str, EpisodeAnnotation] = {}

    @classmethod
    def for_backend(cls, vlm: Any, **kwargs) -> "EpisodeAnnotator":
        try:
            ann = cls._registry.get(vlm)
            if ann is None:
                ann = cls(vlm, **kwargs)
                cls._registry[vlm] = ann
            return ann
        except TypeError:  # backend not weak-referenceable / hashable
            return cls(vlm, **kwargs)

    def latest(self, video_path: str) -> Optional[EpisodeAnnotation]:
        return self._latest.get(str(video_path))

    @staticmethod
    def _signature(candidates: Sequence[CandidateSegment]) -> tuple:
        return tuple(
            (round(c.start_time, 3), round(c.end_time, 3), c.contact_state, c.grasp_type,
             c.object_name, tuple(getattr(c, "hands", None) or ()), c.transition_type)
            for c in candidates
        )

    def annotate(
        self,
        candidates: Sequence[CandidateSegment],
        video_path: str,
        object_names: Optional[Sequence[str]] = None,
    ) -> EpisodeAnnotation:
        key = (str(video_path), self._signature(candidates), tuple(object_names or ()))
        if key in self._memo:
            self._latest[str(video_path)] = self._memo[key]
            return self._memo[key]
        ann = self._annotate_uncached(list(candidates), str(video_path), object_names)
        self._memo[key] = ann
        self._latest[str(video_path)] = ann
        return ann

    def _template(
        self, candidates, object_names, reason: str, used: int = 0
    ) -> EpisodeAnnotation:
        labels = [template_label(c) for c in candidates]
        return EpisodeAnnotation(
            task_instruction=template_instruction(candidates, object_names),
            instruction_source=TEMPLATE_SOURCE,
            segment_labels=labels,
            segment_sources=[TEMPLATE_SOURCE] * len(labels),
            degraded_reason=reason,
            requests_used=used,
        )

    def _budget_used(self) -> Optional[int]:
        budget = getattr(self.vlm, "budget", None)
        return getattr(budget, "used", None)

    def _annotate_uncached(self, candidates, video_path, object_names) -> EpisodeAnnotation:
        if is_null_backend(self.vlm):
            reason = getattr(self.vlm, "selection_reason", "") or "no VLM backend (vlm.backend=none)"
            return self._template(candidates, object_names, reason)

        used0 = self._budget_used()
        budget = getattr(self.vlm, "budget", None)
        denied0 = getattr(budget, "denied", 0)

        def used_now() -> int:
            u = self._budget_used()
            return (u - used0) if (u is not None and used0 is not None) else 1

        try:
            frames = sample_video_frames(
                video_path, k=self.keyframes, max_size=self.keyframe_max_side
            )
        except Exception as e:
            return self._template(candidates, object_names, f"keyframe extraction failed: {e}")

        prompt = build_episode_prompt(candidates, [t for t, _ in frames], object_names)
        request = VLMRequest(
            images=[img for _, img in frames], prompt=prompt,
            max_new_tokens=self.max_new_tokens, temperature=self.temperature,
        )
        try:
            resp = self.vlm.generate(request)
        except BudgetExceeded as e:
            return self._template(candidates, object_names, f"request budget exhausted: {e}", used_now())
        except Exception as e:
            return self._template(
                candidates, object_names, f"VLM request failed: {type(e).__name__}: {str(e)[:200]}",
                used_now(),
            )

        def repair(rp: str) -> str:
            return self.vlm.generate(
                VLMRequest(prompt=rp, max_new_tokens=self.max_new_tokens, temperature=0.0)
            ).text

        data = parse_json_strict(resp.text, repair_fn=repair, expect="JSON object")
        if not isinstance(data, dict) and not candidates and resp.text:
            # Instruction-only request: accept a single plain sentence.
            plain = _clean_instruction(resp.text.strip().splitlines()[0]) if resp.text.strip() else None
            if plain and "{" not in plain:
                data = {"task_instruction": plain, "segments": []}
        if not isinstance(data, dict):
            discard = getattr(self.vlm, "discard", None)
            if discard:
                discard(request)
            exhausted = getattr(budget, "denied", 0) > denied0
            reason = (
                "request budget exhausted before the JSON repair retry"
                if exhausted else "VLM reply was not valid JSON"
            )
            return self._template(candidates, object_names, reason, used_now())

        src = vlm_source(self.vlm)
        instruction = _clean_instruction(data.get("task_instruction"))
        reasons: List[str] = []
        if instruction is None:
            instruction = template_instruction(candidates, object_names)
            instruction_source = TEMPLATE_SOURCE
            reasons.append("VLM gave no usable task_instruction")
        else:
            instruction_source = src

        by_index: Dict[int, dict] = {}
        raw_segments = data.get("segments")
        if isinstance(raw_segments, list):
            for pos, item in enumerate(raw_segments, 1):
                if isinstance(item, dict):
                    idx = item.get("index", pos)
                    try:
                        by_index[int(idx)] = item
                    except (TypeError, ValueError):
                        continue
        labels: List[dict] = []
        sources: List[str] = []
        for i, c in enumerate(candidates, 1):
            item = by_index.get(i)
            action = str(item.get("action", "")).strip().lower() if item else ""
            if item and action in VALID_ACTIONS:
                desc = str(item.get("description", "")).strip() or template_label(c)["description"]
                labels.append({"action": action, "description": desc})
                sources.append(src)
            else:
                labels.append(template_label(c))
                sources.append(TEMPLATE_SOURCE)
        if any(s == TEMPLATE_SOURCE for s in sources):
            reasons.append(
                f"{sum(s == TEMPLATE_SOURCE for s in sources)} segment(s) missing/invalid in VLM reply"
            )
        return EpisodeAnnotation(
            task_instruction=instruction,
            instruction_source=instruction_source,
            segment_labels=labels,
            segment_sources=sources,
            degraded_reason="; ".join(reasons),
            requests_used=used_now(),
            cache_hit=bool(getattr(resp, "cached", False)),
        )


# ---------------------------------------------------------------------------
# Segment labeler
# ---------------------------------------------------------------------------


class SegmentLabeler:
    """Labels pre-computed segments with action categories.

    ``label_segments`` returns ``ActionSegment`` objects, each carrying
    ``label_source`` ("vlm:<backend>:<model>" or "template"). ``label_sources``
    and ``last_annotation`` expose the same for the whole call.
    """

    def __init__(
        self,
        config: SegmentLabelerConfig,
        vlm: Optional[VLMBackend] = None,
        annotator: Optional[EpisodeAnnotator] = None,
    ):
        self.config = config
        self._model = None
        self._vlm: VLMBackend = (
            vlm
            if vlm is not None
            else resolve_stage_backend(config.vlm_backend, config.gemini_model)
        )
        self._annotator = annotator
        self.label_sources: List[str] = []
        self.last_annotation: Optional[EpisodeAnnotation] = None
        print(f"[SegmentLabeler] Using VLM backend: {self._vlm.name}")

    @property
    def annotator(self) -> EpisodeAnnotator:
        if self._annotator is None:
            self._annotator = EpisodeAnnotator.for_backend(
                self._vlm,
                keyframes=self.config.keyframes_per_request,
                keyframe_max_side=self.config.keyframe_max_side,
            )
        return self._annotator

    def _batched(self) -> bool:
        if self.config.batch_requests is not None:
            return bool(self.config.batch_requests)
        return hasattr(self._vlm, "budget")  # ManagedBackend => budgeted => batch

    def label_segments(
        self,
        candidates: List[CandidateSegment],
        video_path: str,
        object_names: Optional[Sequence[str]] = None,
    ) -> List[ActionSegment]:
        """Label each candidate segment with an action category.

        With a managed backend this is one request for all segments (shared
        with the task instruction); with ``none`` it is purely template.
        """
        self.label_sources = []
        if not candidates:
            return []

        if is_null_backend(self._vlm) or self._batched():
            ann = self.annotator.annotate(candidates, video_path, object_names)
            self.last_annotation = ann
            labels, sources = ann.segment_labels, ann.segment_sources
            if ann.degraded_reason:
                logger.warning("[SegmentLabeler] template used: %s", ann.degraded_reason)
        else:
            labels, sources = self._label_individually(candidates, video_path)

        out: List[ActionSegment] = []
        for seg, label, source in zip(candidates, labels, sources):
            hands = getattr(seg, "hands", ["right"])
            hand_used = getattr(
                seg, "hand_used", "both" if len(hands) == 2 else (hands[0] if hands else "right")
            )
            action_name = label.get("action", "idle")
            obj_name = seg.object_name if (action_name != "idle" and seg.contact_state != "no_contact") else None
            # Model text is kept; the structured template description is used when empty.
            desc = label.get("description") if source != TEMPLATE_SOURCE and label.get("description") else None
            if source == TEMPLATE_SOURCE or not desc:
                desc = build_instruction(action_name, obj_name, hands)
            a = ActionSegment(
                name=action_name,
                start_time=seg.start_time,
                end_time=seg.end_time,
                object_name=seg.object_name or "unknown",
                hand_used=hand_used,
                description=desc,
                hands=hands,
            )
            a.label_source = source  # not a dataclass field: datatypes.py is untouched
            out.append(a)
            self.label_sources.append(source)
        return out

    def _label_individually(self, candidates, video_path):
        """Legacy: one request per segment. Stops calling once the budget is spent."""
        labels, sources = [], []
        src = vlm_source(self._vlm)
        stop = False
        for seg in candidates:
            label = None
            if not stop:
                try:
                    label = self._label_single_segment(seg, video_path)
                except BudgetExceeded as e:
                    logger.warning("[SegmentLabeler] %s; remaining segments use the template", e)
                    stop = True
            if label is None:
                labels.append(template_label(seg))
                sources.append(TEMPLATE_SOURCE)
            else:
                labels.append(label)
                sources.append(src)
        return labels, sources

    def _label_single_segment(self, seg: CandidateSegment, video_path: str) -> Optional[dict]:
        """One VLM label for one segment; ``None`` means "use the template".

        Re-raises :class:`BudgetExceeded`; any other backend failure returns
        ``None`` after logging the real reason.
        """
        keyframe = self._extract_keyframe(video_path, seg)
        if keyframe is None:
            logger.warning(f"Could not extract keyframe for segment {seg.start_time}-{seg.end_time}")
            return None
        prompt = self.config.prompt_template.format(
            start_time=seg.start_time,
            end_time=seg.end_time,
            duration=seg.end_time - seg.start_time,
            contact_state=seg.contact_state,
            grasp_type=seg.grasp_type,
            object_name=seg.object_name or "unknown",
            transition_type=seg.transition_type,
        )
        try:
            result = self._call_vlm_with_retry(keyframe, prompt)
        except BudgetExceeded:
            raise
        except Exception as e:
            logger.warning(
                f"VLM backend {self._vlm.name!r} failed for segment "
                f"{seg.start_time}-{seg.end_time}: {e}; using template"
            )
            return None
        if result is None or result.get("action") == "unlabeled":
            logger.warning(
                f"VLM backend {self._vlm.name!r} gave no valid label for segment "
                f"{seg.start_time}-{seg.end_time}; using template"
            )
            return None
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

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if pil_image.width > 1024 or pil_image.height > 1024:
            pil_image.thumbnail((1024, 1024))
        return pil_image

    def _call_vlm_with_retry(self, image: Image.Image, prompt: str) -> Optional[dict]:
        """Call the VLM backend with retry logic.

        Output contract: strict JSON parse -> one repair re-prompt; failure
        returns None and the caller falls back to the template label.
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
                # Fatal backend errors (auth, missing weights, exhausted
                # budget) propagate immediately — never retried.
                raise
            except Exception as e:
                err = str(e).lower()

                if any(k in err for k in ["404", "not found", "no longer available",
                                          "invalid model", "api key not valid", "permission denied",
                                          "dns", "could not contact dns", "address lookup failed"]):
                    logger.error(f"[FATAL/NETWORK UNREACHABLE] {e}")
                    return None

                if any(k in err for k in ["rate limit", "quota", "429", "resource exhausted"]):
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"[RETRY] Rate limit. Wait {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[RETRY] {attempt+1}/{MAX_RETRIES}: {e}")
                    time.sleep(RETRY_DELAY)

                if attempt == MAX_RETRIES - 1:
                    print("[FAILED] Segment labeling failed after retries")
                    return None

        return None

    def _parse_response(self, text: str) -> Optional[dict]:
        """Parse VLM response for action label and description (pure; no repair)."""
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
        if action not in VALID_ACTIONS:
            action = "unlabeled"
        return {"action": action, "description": desc}

    def _default_label(self, seg: CandidateSegment) -> dict:
        """Template label from signal properties (kept for compatibility)."""
        return template_label(seg)


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
