"""Commercial-safe v4 GPU backends: MoGe-2, DA3Metric, SAM 2.1, and Qwen3-VL (+ vLLM)."""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from src.egoannotate.gpu_backends import (
    _MODEL_CACHE,
    DepthAnythingV2SmallGPU,
    _cache_enabled,
    _move_inputs,
    _resolve_torch_dtype,
    sanitize_vla_caption,
)
from src.egoannotate.registry import register_backend

_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(text)
        if not m:
            return {}
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _encode_bgr_jpeg_b64(bgr: np.ndarray, max_dim: int = 768, quality: int = 85) -> str:
    h, w = bgr.shape[:2]
    scale = min(1.0, float(max_dim) / max(float(max(h, w)), 1.0))
    resized = (
        cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))))
        if scale < 0.99
        else bgr
    )
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _compute_normals_from_depth(
    depth_m: np.ndarray,
    intrinsics_3x3: np.ndarray,
) -> np.ndarray:
    """Derive unit surface normals (H, W, 3) from a metric depth map and intrinsics."""
    h, w = depth_m.shape[:2]
    fx = max(float(intrinsics_3x3[0, 0]), 1.0)
    fy = max(float(intrinsics_3x3[1, 1]), 1.0)
    cx = float(intrinsics_3x3[0, 2])
    cy = float(intrinsics_3x3[1, 2])
    us, vs = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    z = np.clip(depth_m.astype(np.float32), 0.05, 50.0)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)
    dx = np.gradient(pts, axis=1)
    dy = np.gradient(pts, axis=0)
    normals = np.cross(dx, dy)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)
    return normals.astype(np.float32)


def lift_box_to_3d_centroid(
    bbox_xyxy: list[float],
    depth_map_m: np.ndarray,
    intrinsics_3x3: np.ndarray,
    img_w: int,
    img_h: int,
) -> tuple[list[float], float]:
    """Lift a 2D pixel bounding box into a metric 3D camera-frame centroid (X, Y, Z)."""
    dh, dw = depth_map_m.shape[:2]
    sx = dw / max(float(img_w), 1.0)
    sy = dh / max(float(img_h), 1.0)
    x1 = max(0, min(dw - 1, round(bbox_xyxy[0] * sx)))
    y1 = max(0, min(dh - 1, round(bbox_xyxy[1] * sy)))
    x2 = max(x1 + 1, min(dw, round(bbox_xyxy[2] * sx)))
    y2 = max(y1 + 1, min(dh, round(bbox_xyxy[3] * sy)))
    patch = depth_map_m[y1:y2, x1:x2]
    finite = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 20.0)]
    z_m = float(np.median(finite)) if finite.size > 0 else 0.75
    u_c = 0.5 * (bbox_xyxy[0] + bbox_xyxy[2])
    v_c = 0.5 * (bbox_xyxy[1] + bbox_xyxy[3])
    fx = max(float(intrinsics_3x3[0, 0]), 1.0)
    fy = max(float(intrinsics_3x3[1, 1]), 1.0)
    cx = float(intrinsics_3x3[0, 2])
    cy = float(intrinsics_3x3[1, 2])
    x_m = (u_c - cx) * z_m / fx
    y_m = (v_c - cy) * z_m / fy
    return [round(x_m, 4), round(y_m, 4), round(z_m, 4)], round(z_m, 4)


class MoGe2MetricDepthGPU:
    """MIT-licensed MoGe-2 metric depth, surface normals & camera intrinsics runner."""

    def __init__(
        self,
        model_id: str = "Ruicheng/moge-2-vitb-normal",
        device: str = "cuda",
        mixed_precision: str = "fp16",
    ) -> None:
        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = device if use_cuda else "cpu"
        self.dtype = _resolve_torch_dtype(mixed_precision, self.device)
        self.model_id = model_id
        self.backend_used = "moge2_mit"
        self._moge_model: Any | None = None
        self._fallback_depth: DepthAnythingV2SmallGPU | None = None

        cache_key = ("moge2", model_id, self.device, str(self.dtype))
        if _cache_enabled() and cache_key in _MODEL_CACHE:
            self._moge_model = _MODEL_CACHE[cache_key][0]
            return

        try:
            from moge.model.v2 import MoGeModel

            model = MoGeModel.from_pretrained(model_id).to(self.device)
            model.eval()
            self._moge_model = model
            if _cache_enabled():
                _MODEL_CACHE[cache_key] = (model, None)
        except (ImportError, RuntimeError, OSError, ValueError):
            self.backend_used = "depth_anything_v2_small_fallback"
            self._fallback_depth = DepthAnythingV2SmallGPU(
                device=self.device,
                mixed_precision=mixed_precision,
            )

    def infer_batch(
        self,
        frames_bgr: list[np.ndarray],
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """Predict metric depth, surface normals, validity mask, and 3x3 intrinsics."""
        if not frames_bgr:
            return []
        results: list[dict[str, Any]] = []
        if self._moge_model is not None:
            use_amp = self.device == "cuda" and self.dtype != torch.float32
            for bgr in frames_bgr:
                h, w = bgr.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                inp = (
                    torch.from_numpy(rgb)
                    .permute(2, 0, 1)
                    .to(device=self.device, dtype=torch.float32)
                )
                with (
                    torch.inference_mode(),
                    torch.amp.autocast("cuda", dtype=self.dtype, enabled=use_amp),
                ):
                    out = self._moge_model.infer(inp)
                depth_np = out["depth"].detach().float().cpu().numpy().astype(np.float32)
                mask_np = (
                    out["mask"].detach().cpu().numpy().astype(bool)
                    if "mask" in out
                    else np.isfinite(depth_np)
                )
                k_norm = out["intrinsics"].detach().float().cpu().numpy().astype(np.float32)
                k_px = np.array(
                    [
                        [float(k_norm[0, 0]) * w, 0.0, float(k_norm[0, 2]) * w],
                        [0.0, float(k_norm[1, 1]) * h, float(k_norm[1, 2]) * h],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )
                if "normal" in out and out["normal"] is not None:
                    normal_np = out["normal"].detach().float().cpu().numpy().astype(np.float32)
                else:
                    normal_np = _compute_normals_from_depth(depth_np, k_px)
                valid_vals = depth_np[mask_np & np.isfinite(depth_np) & (depth_np > 0.05)]
                med = float(np.median(valid_vals)) if valid_vals.size > 0 else 1.0
                mean_n = (
                    normal_np[mask_np].mean(axis=0).tolist()
                    if np.any(mask_np)
                    else [0.0, 0.0, 1.0]
                )
                results.append(
                    {
                        "depth_m": depth_np,
                        "normal_map": normal_np,
                        "valid_mask": mask_np,
                        "intrinsics_3x3": k_px,
                        "median_depth_m": round(med, 4),
                        "mean_normal": [round(float(v), 4) for v in mean_n],
                        "is_metric": True,
                        "backend": self.backend_used,
                    }
                )
            return results

        assert self._fallback_depth is not None
        dmaps = self._fallback_depth.estimate_depth_batch(frames_bgr, batch_size=batch_size)
        for bgr, dmap in zip(frames_bgr, dmaps):
            h, w = bgr.shape[:2]
            f_px = 0.85 * float(max(h, w))
            k_px = np.array(
                [[f_px, 0.0, w * 0.5], [0.0, f_px, h * 0.5], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            normals = _compute_normals_from_depth(dmap, k_px)
            valid = np.isfinite(dmap) & (dmap > 0.05)
            med = float(np.median(dmap[valid])) if np.any(valid) else 1.0
            mean_n = normals[valid].mean(axis=0).tolist() if np.any(valid) else [0.0, 0.0, 1.0]
            results.append(
                {
                    "depth_m": dmap,
                    "normal_map": normals,
                    "valid_mask": valid,
                    "intrinsics_3x3": k_px,
                    "median_depth_m": round(med, 4),
                    "mean_normal": [round(float(v), 4) for v in mean_n],
                    "is_metric": False,
                    "backend": self.backend_used,
                }
            )
        return results


class DA3MetricLargeGPU:
    """Apache-2.0 depth-anything/DA3METRIC-LARGE metric depth runner with safe fallback."""

    def __init__(
        self,
        model_id: str = "depth-anything/DA3METRIC-LARGE",
        device: str = "cuda",
        mixed_precision: str = "fp16",
    ) -> None:
        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = device if use_cuda else "cpu"
        self.dtype = _resolve_torch_dtype(mixed_precision, self.device)
        self.model_id = model_id
        self.backend_used = "da3_metric_large_apache"
        self._da3_model: Any | None = None
        self._fallback: DepthAnythingV2SmallGPU | None = None

        cache_key = ("da3_metric", model_id, self.device, str(self.dtype))
        if _cache_enabled() and cache_key in _MODEL_CACHE:
            self._da3_model = _MODEL_CACHE[cache_key][0]
            return

        try:
            from depth_anything_3.api import DepthAnything3

            model = DepthAnything3.from_pretrained(model_id).to(self.device)
            model.eval()
            self._da3_model = model
            if _cache_enabled():
                _MODEL_CACHE[cache_key] = (model, None)
        except (ImportError, RuntimeError, OSError, ValueError):
            self.backend_used = "depth_anything_v2_small_fallback"
            self._fallback = DepthAnythingV2SmallGPU(
                device=self.device,
                mixed_precision=mixed_precision,
            )

    def infer_batch(
        self,
        frames_bgr: list[np.ndarray],
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        if not frames_bgr:
            return []
        out_list: list[dict[str, Any]] = []
        if self._da3_model is not None:
            pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
            for start in range(0, len(pil_imgs), max(1, batch_size)):
                chunk_imgs = pil_imgs[start : start + batch_size]
                chunk_bgr = frames_bgr[start : start + batch_size]
                pred = self._da3_model.inference(chunk_imgs)
                depths = np.asarray(pred.depth, dtype=np.float32)
                for idx, bgr in enumerate(chunk_bgr):
                    h, w = bgr.shape[:2]
                    dmap = depths[idx] if depths.ndim == 3 else depths
                    f_px = 0.85 * float(max(h, w))
                    k_px = np.array(
                        [[f_px, 0.0, w * 0.5], [0.0, f_px, h * 0.5], [0.0, 0.0, 1.0]],
                        dtype=np.float32,
                    )
                    normals = _compute_normals_from_depth(dmap, k_px)
                    valid = np.isfinite(dmap) & (dmap > 0.05)
                    med = float(np.median(dmap[valid])) if np.any(valid) else 1.0
                    mean_n = (
                        normals[valid].mean(axis=0).tolist() if np.any(valid) else [0.0, 0.0, 1.0]
                    )
                    out_list.append(
                        {
                            "depth_m": dmap,
                            "normal_map": normals,
                            "valid_mask": valid,
                            "intrinsics_3x3": k_px,
                            "median_depth_m": round(med, 4),
                            "mean_normal": [round(float(v), 4) for v in mean_n],
                            "is_metric": True,
                            "backend": self.backend_used,
                        }
                    )
            return out_list

        assert self._fallback is not None
        dmaps = self._fallback.estimate_depth_batch(frames_bgr, batch_size=batch_size)
        for bgr, dmap in zip(frames_bgr, dmaps):
            h, w = bgr.shape[:2]
            f_px = 0.85 * float(max(h, w))
            k_px = np.array(
                [[f_px, 0.0, w * 0.5], [0.0, f_px, h * 0.5], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            normals = _compute_normals_from_depth(dmap, k_px)
            valid = np.isfinite(dmap) & (dmap > 0.05)
            med = float(np.median(dmap[valid])) if np.any(valid) else 1.0
            mean_n = normals[valid].mean(axis=0).tolist() if np.any(valid) else [0.0, 0.0, 1.0]
            out_list.append(
                {
                    "depth_m": dmap,
                    "normal_map": normals,
                    "valid_mask": valid,
                    "intrinsics_3x3": k_px,
                    "median_depth_m": round(med, 4),
                    "mean_normal": [round(float(v), 4) for v in mean_n],
                    "is_metric": False,
                    "backend": self.backend_used,
                }
            )
        return out_list


def _mask_to_norm_polygon(mask_bool: np.ndarray, max_pts: int = 16) -> list[list[float]]:
    h, w = mask_bool.shape[:2]
    cnts, _ = cv2.findContours(
        mask_bool.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not cnts:
        return []
    largest = max(cnts, key=cv2.contourArea)
    eps = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True).reshape(-1, 2)
    step = max(1, len(approx) // max(1, max_pts))
    pts = approx[::step][:max_pts]
    return [[round(float(x) / max(w, 1), 4), round(float(y) / max(h, 1), 4)] for x, y in pts]


class SAM21MaskPropagatorGPU:
    """Apache-2.0 facebook/sam2.1-hiera-small mask predictor for hands/tools/workpieces."""

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-small",
        device: str = "cuda",
        mixed_precision: str = "fp16",
    ) -> None:
        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = device if use_cuda else "cpu"
        self.dtype = _resolve_torch_dtype(mixed_precision, self.device)
        self.model_id = model_id
        self.processor: Any | None = None
        self.model: Any | None = None
        self.available = False

        cache_key = ("sam21", model_id, self.device, str(self.dtype))
        if _cache_enabled() and cache_key in _MODEL_CACHE:
            self.processor, self.model = _MODEL_CACHE[cache_key]
            self.available = self.model is not None
            return

        try:
            from transformers import Sam2Model, Sam2Processor

            self.processor = Sam2Processor.from_pretrained(model_id)
            self.model = Sam2Model.from_pretrained(model_id).to(self.device)
            self.model.eval()
            self.available = True
            if _cache_enabled() and isinstance(self.model, torch.nn.Module):
                _MODEL_CACHE[cache_key] = (self.processor, self.model)
        except (ImportError, RuntimeError, OSError, ValueError):
            self.available = False

    def segment_boxes(
        self,
        bgr: np.ndarray,
        boxes_xyxy: list[list[float]],
    ) -> list[dict[str, Any]]:
        """Predict pixel-accurate binary masks, area fractions, and normalized polygons."""
        h, w = bgr.shape[:2]
        if not boxes_xyxy:
            return []

        if self.available and self.processor is not None and self.model is not None:
            pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            inputs = _move_inputs(
                self.processor(
                    images=pil_img,
                    input_boxes=[boxes_xyxy],
                    return_tensors="pt",
                ),
                self.device,
            )
            use_amp = self.device == "cuda" and self.dtype != torch.float32
            with (
                torch.inference_mode(),
                torch.amp.autocast("cuda", dtype=self.dtype, enabled=use_amp),
            ):
                outputs = self.model(**inputs)
            masks_t = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"],
            )[0]
            out_masks: list[dict[str, Any]] = []
            for idx in range(len(boxes_xyxy)):
                m_arr = masks_t[idx, 0].numpy() > 0.0
                area_frac = float(np.mean(m_arr))
                out_masks.append(
                    {
                        "mask": m_arr,
                        "area_frac": round(area_frac, 5),
                        "polygon_norm": _mask_to_norm_polygon(m_arr),
                        "backend": "sam21_apache",
                    }
                )
            return out_masks

        fallback_out: list[dict[str, Any]] = []
        for box in boxes_xyxy:
            m = np.zeros((h, w), dtype=bool)
            x1 = max(0, min(w - 1, round(box[0])))
            y1 = max(0, min(h - 1, round(box[1])))
            x2 = max(x1 + 1, min(w, round(box[2])))
            y2 = max(y1 + 1, min(h, round(box[3])))
            m[y1:y2, x1:x2] = True
            fallback_out.append(
                {
                    "mask": m,
                    "area_frac": round(float(np.mean(m)), 5),
                    "polygon_norm": _mask_to_norm_polygon(m),
                    "backend": "box_mask_fallback",
                }
            )
        return fallback_out

    @staticmethod
    def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        inter = float(np.logical_and(mask_a, mask_b).sum())
        union = float(np.logical_or(mask_a, mask_b).sum())
        return round(inter / union, 4) if union > 0.0 else 0.0


class Qwen3VLFastAnnotator:
    """Apache-2.0 Qwen3-VL annotator supporting both local transformers and self-hosted vLLM."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "cuda",
        mixed_precision: str = "fp16",
        vllm_base_url: str | None = None,
        vllm_api_key: str = "EMPTY",
    ) -> None:
        self.model_id = model_id
        self.vllm_base_url = vllm_base_url.rstrip("/") if vllm_base_url else None
        self.vllm_api_key = vllm_api_key
        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = device if use_cuda else "cpu"
        self.dtype = _resolve_torch_dtype(mixed_precision, self.device)
        self.processor: Any | None = None
        self.model: Any | None = None

        if self.vllm_base_url is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            cache_key = ("qwen3_vl", model_id, self.device, str(self.dtype))
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

    def _generate_vllm(
        self,
        frames_bgr: list[np.ndarray],
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        assert self.vllm_base_url is not None
        content: list[dict[str, Any]] = []
        for fr in frames_bgr:
            b64 = _encode_bgr_jpeg_b64(fr)
            if b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": int(max_new_tokens),
        }
        req = urllib.request.Request(
            url=f"{self.vllm_base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.vllm_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if not choices:
                return ""
            return str(choices[0].get("message", {}).get("content", "")).strip()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, OSError):
            return ""

    def _generate_local(
        self,
        frames_bgr: list[np.ndarray],
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        assert self.processor is not None and self.model is not None
        pil_imgs: list[Image.Image] = []
        for fr in frames_bgr:
            h, w = fr.shape[:2]
            scale = min(1.0, 512.0 / max(float(max(h, w)), 1.0))
            resized = (
                cv2.resize(fr, (max(1, round(w * scale)), max(1, round(h * scale))))
                if scale < 0.99
                else fr
            )
            pil_imgs.append(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        content: list[dict[str, Any]] = [{"type": "image"} for _ in pil_imgs]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text_in = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        raw_inputs = self.processor(text=text_in, images=pil_imgs, return_tensors="pt")
        inputs = _move_inputs(raw_inputs, self.device, float_dtype=self.dtype)
        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
            )
        prompt_len = int(inputs["input_ids"].shape[-1])
        decoded = self.processor.batch_decode(
            gen_ids[:, prompt_len:],
            skip_special_tokens=True,
        )
        return str(decoded[0]).strip() if decoded else ""

    def generate(
        self,
        frames_bgr: list[np.ndarray],
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        if not frames_bgr:
            return ""
        if self.vllm_base_url is not None:
            return self._generate_vllm(frames_bgr, prompt, max_new_tokens=max_new_tokens)
        return self._generate_local(frames_bgr, prompt, max_new_tokens=max_new_tokens)

    def discover_scene_and_objects(
        self,
        keyframes_bgr: list[np.ndarray],
    ) -> dict[str, Any]:
        """Pass 1: Discover active tools, workpieces (0..1000 boxes), domain, and EHS metadata."""
        sample = keyframes_bgr[:4]
        prompt = (
            "Analyze these first-person manipulation frames and return ONLY valid JSON with keys: "
            '{"task_goal": str, "environment_tag": str, "domain_taxonomy": str, '
            '"operator_ehs_metadata": {"gloves_worn": bool, "ppe_visible": [str], "hazards": [str]}, '
            '"objects": [{"label": str, "role": "tool|workpiece|container|target_object", '
            '"bbox_1000": [x1, y1, x2, y2]}]}. '
            "Never mention a robot or camera in task_goal."
        )
        raw = self.generate(sample, prompt, max_new_tokens=320)
        parsed = _extract_json_object(raw)
        objects_raw = parsed.get("objects", [])
        clean_objs: list[dict[str, Any]] = []
        if isinstance(objects_raw, list):
            for obj in objects_raw:
                if not isinstance(obj, dict):
                    continue
                lbl = str(obj.get("label", "")).strip().lower()
                if not lbl:
                    continue
                role = str(obj.get("role", "workpiece")).strip().lower()
                if role not in ("held_object", "target_object", "tool", "container", "workpiece"):
                    role = "workpiece"
                box = obj.get("bbox_1000")
                clean_box = (
                    [float(v) for v in box[:4]]
                    if isinstance(box, list) and len(box) >= 4
                    else None
                )
                clean_objs.append({"label": lbl, "role": role, "bbox_1000": clean_box})
        task_goal = sanitize_vla_caption(str(parsed.get("task_goal", "")), "manipulate")
        return {
            "task_goal": task_goal,
            "environment_tag": str(parsed.get("environment_tag", "industrial_workspace")).strip(),
            "domain_taxonomy": str(parsed.get("domain_taxonomy", "industrial_manipulation")).strip(),
            "operator_ehs_metadata": (
                parsed.get("operator_ehs_metadata", {})
                if isinstance(parsed.get("operator_ehs_metadata"), dict)
                else {}
            ),
            "objects": clean_objs,
        }

    def describe_segment_keyframes(
        self,
        keyframes_bgr: list[np.ndarray],
        verb_hint: str,
    ) -> str:
        prompt = (
            "Write one short imperative instruction describing the hand-object action "
            f"(verb hint: {verb_hint}). Example: 'Spread mortar on floor with right hand using trowel.' "
            "Name only the action, hand (left, right, or both), and object/tool. "
            "Never mention a robot, camera, person, or operator clothing."
        )
        raw = self.generate(keyframes_bgr[:2], prompt, max_new_tokens=48)
        return sanitize_vla_caption(raw, verb_hint) if raw else ""


@register_backend(
    stage="depth",
    name="moge2_mit",
    code_license="MIT",
    weights_license="MIT",
    requires_gpu=True,
    notes="MoGe-2 metric depth, surface normals, validity mask, and normalized camera intrinsics.",
)
def build_moge2_mit(model_id: str = "Ruicheng/moge-2-vitb-normal") -> dict[str, str]:
    return {"model_id": model_id, "license": "MIT", "runtime": "torch_cuda"}


@register_backend(
    stage="depth",
    name="da3_metric_large_apache",
    code_license="Apache-2.0",
    weights_license="Apache-2.0",
    requires_gpu=True,
    notes="Depth Anything 3 Metric-Large (Apache-2.0) monocular metric depth estimator.",
)
def build_da3_metric_large_apache(model_id: str = "depth-anything/DA3METRIC-LARGE") -> dict[str, str]:
    return {"model_id": model_id, "license": "Apache-2.0", "runtime": "torch_cuda"}
