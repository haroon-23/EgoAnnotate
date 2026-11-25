"""Commercially-licensed CUDA GPU inference runners (Grounding DINO, Depth-Anything-V2-Small, Local VLM)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


def _resolve_torch_dtype(mixed_precision: str, device: str) -> torch.dtype:
    if device != "cuda" or not torch.cuda.is_available():
        return torch.float32
    if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if mixed_precision in ("fp16", "bf16"):
        return torch.float16
    return torch.float32


class GroundingDINOGPUDetector:
    """Apache-2.0 IDEA-Research/grounding-dino-base open-vocab detector for CUDA GPU production."""

    def __init__(
        self,
        prompts: list[str],
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        mixed_precision: str = "fp16",
        box_threshold: float = 0.28,
    ) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.dtype = _resolve_torch_dtype(mixed_precision, self.device)
        self.prompts = [p.strip().lower() for p in prompts if p.strip()]
        self.text_prompt = ". ".join(self.prompts) + "."
        self.box_threshold = float(box_threshold)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def detect_frame(self, bgr: np.ndarray) -> list[dict[str, Any]]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self.processor(images=pil_img, text=self.text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([pil_img.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=0.25,
            target_sizes=target_sizes,
        )
        out: list[dict[str, Any]] = []
        if not results:
            return out
        res0 = results[0]
        for box, score, label in zip(res0["boxes"], res0["scores"], res0["labels"]):
            b = [round(float(v), 2) for v in box.detach().cpu().tolist()]
            out.append({"bbox_xyxy": b, "score": round(float(score.item()), 4), "label": str(label)})
        return out


class DepthAnythingV2SmallGPU:
    """Apache-2.0 depth-anything/Depth-Anything-V2-Small-hf metric/relative depth prior."""

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cuda",
        mixed_precision: str = "fp16",
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.dtype = _resolve_torch_dtype(mixed_precision, self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def estimate_depth_map(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            pred = outputs.predicted_depth
        depth = pred.squeeze().detach().float().cpu().numpy()
        return np.asarray(depth, dtype=np.float32)


class LocalOpenSourceVLMAnnotator:
    """Apache-2.0 local VLM (SmolVLM2-2.2B / Qwen2.5-VL-7B) keyframe captioner with zero paid APIs."""

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        device: str = "cuda",
        mixed_precision: str = "bf16",
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.dtype = _resolve_torch_dtype(mixed_precision, self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def describe_segment_keyframes(self, keyframes_bgr: list[np.ndarray], verb_hint: str) -> str:
        if not keyframes_bgr:
            return ""
        pil_imgs = [Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) for fr in keyframes_bgr[:4]]
        prompt = (
            f"Describe the egocentric hand-object manipulation in one concise robotics instruction "
            f"(kinematic verb hint: {verb_hint}). Mention the acting hand and manipulated object."
        )
        content: list[dict[str, Any]] = [{"type": "image"} for _ in pil_imgs]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text_in = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_in, images=pil_imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=48, do_sample=False)
        decoded = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        if not decoded:
            return ""
        return str(decoded[0]).strip().splitlines()[-1].strip()
