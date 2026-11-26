"""Commercially-licensed CUDA GPU inference runners (DINO, Depth, Local VLM)."""

from __future__ import annotations

import gc
import inspect
import os
import re
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

_MODEL_CACHE: dict[tuple[str, str, str, str], tuple[Any, Any]] = {}

_ACTION_STOP_VERBS = (
    r"grasps?|picks?|places?|manipulates?|holds?|lifts?|moves?|reaches?|"
    r"pulls?|pushes?|cuts?|presses?|spreads?|welds?|sews?|folds?|aligns?|"
    r"rotates?|inserts?|uses?|is|are"
)
_CLOTHING_NOUNS = (
    r"(?:t-?shirt|shirt|sleeves?|gloves?|flip[\s-]*flops|shoes?|boots?|"
    r"pants|jeans|trousers|jacket|hoodie|vest|apron|clothes|uniform|hat|cap)"
)
_CLOTHING_PHRASE_RE = re.compile(
    rf"\b(?:wearing|in\s+a|with\s+a)\s+"
    rf"(?:(?!\b(?:{_ACTION_STOP_VERBS})\b)[^.,;])*{_CLOTHING_NOUNS}\b"
    rf"(?:\s+and\s+(?:(?!\b(?:{_ACTION_STOP_VERBS})\b)[^.,;])*"
    rf"{_CLOTHING_NOUNS}\b)*",
    flags=re.IGNORECASE,
)
_OF_PERSON_RE = re.compile(
    r"\bof\s+(?:a\s+|the\s+)?(?:person|human|operator|worker|robot)\b",
    flags=re.IGNORECASE,
)
_ACTOR_PREFIX_RE = re.compile(
    r"^(?:(?:the|a)\s+)?"
    r"(?:robot'?s?|person'?s?|operator'?s?|human'?s?|first[- ]person)\s+",
    flags=re.IGNORECASE,
)
_LEADING_ARTICLE_HAND_RE = re.compile(
    r"^the\s+(?=(?:left|right|both)?\s*hands?\b)",
    flags=re.IGNORECASE,
)


def _cache_enabled() -> bool:
    return os.environ.get("EGOANNOTATE_CACHE_GPU_MODELS", "1") != "0"


def clear_gpu_model_cache() -> None:
    """Release cached GPU weights from memory between distinct jobs or tests."""
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sanitize_vla_caption(raw: str, verb_hint: str) -> str:
    """Clean VLM output into a crisp VLA instruction without leakage."""
    raw_s = str(raw).strip()
    line = raw_s.splitlines()[0].strip() if raw_s else ""
    line = line.strip("\"'` ")
    line = _CLOTHING_PHRASE_RE.sub("", line)
    line = _OF_PERSON_RE.sub("", line)
    line = _ACTOR_PREFIX_RE.sub("", line)
    line = re.sub(r"\brobot'?s?\b", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\s+", " ", line).strip(" ,;:-")
    line = _LEADING_ARTICLE_HAND_RE.sub("", line)
    line = re.sub(r"\s+", " ", line).strip(" ,;:-")
    if not line or len(line) < 4:
        verb = (verb_hint or "manipulate").strip().capitalize()
        return f"{verb} target object with hand."
    line = line[0].upper() + line[1:]
    if not line.endswith("."):
        line += "."
    return line


def _resolve_torch_dtype(mixed_precision: str, device: str) -> torch.dtype:
    if device != "cuda" or not torch.cuda.is_available():
        return torch.float32
    if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if mixed_precision in ("fp16", "bf16"):
        return torch.float16
    return torch.float32


def _move_inputs(
    inputs: Any,
    device: str,
    float_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if float_dtype is not None and v.is_floating_point():
                moved[k] = v.to(device=device, dtype=float_dtype)
            else:
                moved[k] = v.to(device=device)
        else:
            moved[k] = v
    return moved


class GroundingDINOGPUDetector:
    """Apache-2.0 IDEA-Research/grounding-dino-base open-vocab CUDA detector."""

    def __init__(
        self,
        prompts: list[str],
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        mixed_precision: str = "fp16",
        box_threshold: float = 0.25,
    ) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = device if use_cuda else "cpu"
        self.dtype = _resolve_torch_dtype(mixed_precision, self.device)
        self.prompts = [p.strip().lower() for p in prompts if p.strip()]
        self.text_prompt = ". ".join(self.prompts) + "."
        self.box_threshold = float(box_threshold)

        cache_key = ("grounding_dino", model_id, self.device, str(self.dtype))
        if _cache_enabled() and cache_key in _MODEL_CACHE:
            self.processor, self.model = _MODEL_CACHE[cache_key]
        else:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id
            ).to(self.device)
            self.model.eval()
            if _cache_enabled() and isinstance(self.model, torch.nn.Module):
                _MODEL_CACHE[cache_key] = (self.processor, self.model)

    def detect_batch(
        self,
        frames_bgr: list[np.ndarray],
        batch_size: int = 8,
    ) -> list[list[dict[str, Any]]]:
        """Run batched FP16/AMP open-vocabulary detection across BGR frames."""
        if not frames_bgr:
            return []
        bs = max(1, int(batch_size))
        if (
            self.device.startswith("cuda")
            and torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory < 22 * (1024**3)
        ):
            bs = min(bs, 2)
        use_amp = self.device == "cuda" and self.dtype != torch.float32
        post_fn = self.processor.post_process_grounded_object_detection
        sig_params = inspect.signature(post_fn).parameters
        all_out: list[list[dict[str, Any]]] = []

        def _run_chunk(sub_frames: list[np.ndarray]) -> list[list[dict[str, Any]]]:
            pil_imgs = [
                Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) for fr in sub_frames
            ]
            texts = [self.text_prompt] * len(pil_imgs)
            raw_inputs = self.processor(
                images=pil_imgs,
                text=texts,
                padding=True,
                return_tensors="pt",
            )
            inputs = _move_inputs(raw_inputs, self.device)
            with (
                torch.inference_mode(),
                torch.amp.autocast("cuda", dtype=self.dtype, enabled=use_amp),
            ):
                outputs = self.model(**inputs)
            target_sizes = torch.tensor(
                [img.size[::-1] for img in pil_imgs],
                device=self.device,
            )
            kwargs: dict[str, Any] = {
                "text_threshold": 0.22,
                "target_sizes": target_sizes,
            }
            if "threshold" in sig_params:
                kwargs["threshold"] = self.box_threshold
            else:
                kwargs["box_threshold"] = self.box_threshold
            results = post_fn(outputs, inputs["input_ids"], **kwargs)
            chunk_out: list[list[dict[str, Any]]] = []
            for idx in range(len(pil_imgs)):
                frame_dets: list[dict[str, Any]] = []
                if idx < len(results):
                    res = results[idx]
                    label_key = "text_labels" if "text_labels" in res else "labels"
                    for box, score, label in zip(
                        res["boxes"], res["scores"], res[label_key]
                    ):
                        b = [round(float(v), 2) for v in box.detach().cpu().tolist()]
                        frame_dets.append(
                            {
                                "bbox_xyxy": b,
                                "score": round(float(score.item()), 4),
                                "label": str(label),
                            }
                        )
                chunk_out.append(frame_dets)
            del raw_inputs, inputs, outputs, target_sizes, results
            return chunk_out

        for start in range(0, len(frames_bgr), bs):
            chunk = frames_bgr[start : start + bs]
            try:
                all_out.extend(_run_chunk(chunk))
            except torch.cuda.OutOfMemoryError:
                if len(chunk) == 1:
                    raise
                torch.cuda.empty_cache()
                for single in chunk:
                    all_out.extend(_run_chunk([single]))
        return all_out

    def detect_frame(self, bgr: np.ndarray) -> list[dict[str, Any]]:
        return self.detect_batch([bgr], batch_size=1)[0]


class DepthAnythingV2SmallGPU:
    """Apache-2.0 depth-anything/Depth-Anything-V2-Small-hf depth prior."""

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cuda",
        mixed_precision: str = "fp16",
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = device if use_cuda else "cpu"
        self.dtype = _resolve_torch_dtype(mixed_precision, self.device)

        cache_key = ("depth_v2_small", model_id, self.device, str(self.dtype))
        if _cache_enabled() and cache_key in _MODEL_CACHE:
            self.processor, self.model = _MODEL_CACHE[cache_key]
        else:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(
                self.device
            )
            self.model.eval()
            if _cache_enabled() and isinstance(self.model, torch.nn.Module):
                _MODEL_CACHE[cache_key] = (self.processor, self.model)

    def estimate_depth_batch(
        self,
        frames_bgr: list[np.ndarray],
        batch_size: int = 16,
    ) -> list[np.ndarray]:
        """Run batched FP16/AMP depth inference across multiple BGR frames."""
        if not frames_bgr:
            return []
        bs = max(1, int(batch_size))
        use_amp = self.device == "cuda" and self.dtype != torch.float32
        out: list[np.ndarray] = []

        for start in range(0, len(frames_bgr), bs):
            chunk = frames_bgr[start : start + bs]
            pil_imgs = [
                Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) for fr in chunk
            ]
            inputs = _move_inputs(
                self.processor(images=pil_imgs, return_tensors="pt"),
                self.device,
            )
            with (
                torch.inference_mode(),
                torch.amp.autocast("cuda", dtype=self.dtype, enabled=use_amp),
            ):
                outputs = self.model(**inputs)
                pred = outputs.predicted_depth
            pred_np = pred.detach().float().cpu().numpy()
            if pred_np.ndim == 2:
                out.append(np.asarray(pred_np, dtype=np.float32))
            else:
                for arr in pred_np:
                    out.append(np.asarray(arr, dtype=np.float32))
        return out

    def estimate_depth_map(self, bgr: np.ndarray) -> np.ndarray:
        return self.estimate_depth_batch([bgr], batch_size=1)[0]


class LocalOpenSourceVLMAnnotator:
    """Apache-2.0 local VLM keyframe captioner (SmolVLM2 / Qwen3-VL)."""

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        device: str = "cuda",
        mixed_precision: str = "fp16",
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = device if use_cuda else "cpu"
        self.dtype = _resolve_torch_dtype(mixed_precision, self.device)

        cache_key = ("local_vlm", model_id, self.device, str(self.dtype))
        if _cache_enabled() and cache_key in _MODEL_CACHE:
            self.processor, self.model = _MODEL_CACHE[cache_key]
        else:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=self.dtype,
            ).to(self.device)
            self.model.eval()
            if _cache_enabled() and isinstance(self.model, torch.nn.Module):
                _MODEL_CACHE[cache_key] = (self.processor, self.model)

    def describe_segment_keyframes(
        self,
        keyframes_bgr: list[np.ndarray],
        verb_hint: str,
    ) -> str:
        if not keyframes_bgr:
            return ""
        pil_imgs = [
            Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            for fr in keyframes_bgr[:2]
        ]
        prompt = (
            "Write one short imperative instruction describing the hand-object "
            f"action (verb hint: {verb_hint}). Example: 'Grasp green fabric with "
            "right hand.' Name only the action, hand (left, right, or both), and "
            "target object or tool. Never mention a robot, camera, person, or "
            "operator clothing."
        )
        content: list[dict[str, Any]] = [{"type": "image"} for _ in pil_imgs]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text_in = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        raw_inputs = self.processor(
            text=text_in,
            images=pil_imgs,
            return_tensors="pt",
        )
        inputs = _move_inputs(raw_inputs, self.device, float_dtype=self.dtype)
        with torch.inference_mode():
            gen_ids = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
        prompt_len = int(inputs["input_ids"].shape[-1])
        new_tokens = gen_ids[:, prompt_len:]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        if not decoded:
            return ""
        return sanitize_vla_caption(str(decoded[0]), verb_hint)
