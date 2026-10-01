"""Unified 2D object-detection interface with swappable backends.

Backends
--------
* ``"owlvit"`` (default) — :class:`OwlViTDetector`, the zero-shot detector the
  pipeline has used all along (previously misnamed ``GroundingDINODetector`` in
  ``src/grounding_detector.py``). Installed, working, no weight download.
* ``"grounding_dino"`` — :class:`GroundingDinoDetector`, the real Grounding DINO
  (IDEA-Research, SwinT_OGC). Opt-in; needs a manual weight download.
* ``"locate_anything"`` — :class:`LocateAnythingDetector`, NVIDIA's
  LocateAnything-3B open-vocabulary grounding VLM. Opt-in research backend
  (non-commercial weights); needs a manual weight download. Serves as the
  pipeline's accuracy oracle on cluttered scenes.

  NOTE on "MM-Grounding-DINO": the standing plan once named OpenMMLab's
  MM-Grounding-DINO, but its mmcv/mmdet dependency has no usable macOS-Intel
  wheels, so it is uninstallable on the 2017 Intel Mac dev machine. The
  IDEA-Research variant is the implementable choice here; it is documented as
  such wherever the backend is referenced.

Box convention: **normalized [0, 1] xyxy** everywhere, matching
:class:`src.datatypes.ObjectAnnotation`. The real-GDINO backend converts its
native pixel boxes to this convention.

All heavy imports are guarded: importing this module never raises for missing
optional dependencies. Factories return ``None`` instead of raising when a
backend cannot be built.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from ..datatypes import ObjectAnnotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded optional imports
# ---------------------------------------------------------------------------

try:
    import torch
    from transformers import AutoProcessor, OwlViTForObjectDetection

    _OWL_VIT_IMPORT_OK = True
except ImportError:  # pragma: no cover - depends on installed env
    _OWL_VIT_IMPORT_OK = False
    logger.debug("transformers/torch not installed — OWL-ViT backend unavailable.")

try:
    import torch as _torch_gd  # noqa: F401  (kept for device probing)
    from groundingdino.util.inference import load_model, predict
    from groundingdino.util import box_ops

    _GDINO_IMPORT_OK = True
except ImportError:  # pragma: no cover - depends on installed env
    _GDINO_IMPORT_OK = False
    logger.debug(
        "groundingdino not installed — Grounding DINO backend unavailable. "
        "Install (CPU-only): "
        "pip install --no-build-isolation -e "
        "git+https://github.com/IDEA-Research/GroundingDINO.git"
    )

# Download locations (no auto-download anywhere — fail loud with these URLs).
GROUNDING_DINO_WEIGHTS_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)
GROUNDING_DINO_CONFIG_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/blob/main/"
    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)

# NVIDIA LocateAnything-3B (no auto-download — fail loud with this URL).
# LICENSE CAVEAT: NVIDIA non-commercial, research-only weights. This backend
# is the pipeline's accuracy oracle for eval/research; OWL-ViT stays the
# commercial default.
LOCATE_ANYTHING_MODEL_ID = "nvidia/LocateAnything-3B"
LOCATE_ANYTHING_WEIGHTS_URL = "https://huggingface.co/nvidia/LocateAnything-3B"

try:
    import transformers as _transformers_la  # noqa: F401  (version checked at runtime)
    _LOCATE_ANYTHING_IMPORT_OK = True
except ImportError:  # pragma: no cover - depends on installed env
    _LOCATE_ANYTHING_IMPORT_OK = False
    logger.debug("transformers not installed — LocateAnything backend unavailable.")


def _locate_anything_transformers_status() -> Tuple[bool, str, bool]:
    """Return ``(import_ok, version_str, matches_upstream_pin)``.

    Upstream pins ``transformers==4.57.1`` (5.x breaks its custom modeling
    code). We warn — never hard-fail — on other versions, since the dev
    machine may carry a different 4.x for the Phase-D local VLM.
    """
    try:
        import transformers

        ver = getattr(transformers, "__version__", "unknown")
    except ImportError:
        return False, "not installed", False
    return True, ver, ver.startswith("4.57.")


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """One 2D detection. Box is normalized [0, 1] ``xyxy``."""

    bbox_xyxy_norm: np.ndarray  # shape (4,), float32, [x1, y1, x2, y2] in [0, 1]
    label: str
    score: float


@dataclass
class Detector2DConfig:
    """Configuration for :func:`create_detector_2d`.

    ``backend``: ``"owlvit"`` (default, installed), ``"grounding_dino"``
    (opt-in, needs manual weight download), or ``"locate_anything"``
    (opt-in research backend, needs manual weight download).
    """

    backend: str = "owlvit"
    # OWL-ViT settings
    model_name: str = "google/owlvit-base-patch32"
    confidence_threshold: float = 0.3
    # Grounding DINO (IDEA-Research) settings
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    grounding_dino_weights: str = "models/groundingdino_swint_ogc.pth"
    grounding_dino_config: str = "models/GroundingDINO_SwinT_OGC.py"
    # LocateAnything (NVIDIA, opt-in research backend) settings
    locate_anything_weights: str = "models/locate-anything-3b"
    locate_anything_max_new_tokens: int = 256
    device: str = "auto"  # "auto", "cpu", "cuda"


def bbox_to_location_description(bbox: np.ndarray) -> str:
    """Convert a normalized bbox to a rough location description."""
    x_min, y_min, x_max, y_max = (float(v) for v in bbox)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    h_pos = "left" if cx < 0.33 else ("center" if cx < 0.66 else "right")
    v_pos = "top" if cy < 0.33 else ("middle" if cy < 0.66 else "bottom")
    return f"{v_pos}-{h_pos}"


class Detector2D:
    """Base class for 2D detectors. Boxes are normalized [0, 1] xyxy."""

    backend_name = "base"

    def __init__(self, config: Optional[Detector2DConfig] = None):
        self.config = config or Detector2DConfig()
        self._device = self._resolve_device(self.config.device)
        self._warned: dict[str, bool] = {}

    # -- plumbing ---------------------------------------------------------
    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        if _OWL_VIT_IMPORT_OK or _GDINO_IMPORT_OK or _LOCATE_ANYTHING_IMPORT_OK:
            try:
                import torch as _t

                return "cuda" if _t.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return "cpu"

    def _warn_once(self, key: str, msg: str, *args) -> None:
        if not self._warned.get(key):
            logger.warning(msg, *args)
            self._warned[key] = True

    # -- interface ---------------------------------------------------------
    def is_available(self) -> bool:
        """True when the backend is loaded and ready to detect."""
        raise NotImplementedError

    def detect(self, image_bgr: np.ndarray, prompts: List[str]) -> List[Detection]:
        """Detect ``prompts`` in a BGR image.

        Returns:
            List of :class:`Detection` with normalized [0, 1] xyxy boxes.
        """
        raise NotImplementedError

    def detect_annotations(
        self,
        image_bgr: np.ndarray,
        prompts: List[str],
        threshold: Optional[float] = None,
    ) -> List[ObjectAnnotation]:
        """Detect and return :class:`ObjectAnnotation` (legacy-friendly).

        ``threshold`` temporarily overrides the backend's confidence threshold
        for this call only.
        """
        if threshold is not None:
            prev = self.config.confidence_threshold
            self.config.confidence_threshold = float(threshold)
            try:
                return self._detect_annotations_inner(image_bgr, prompts)
            finally:
                self.config.confidence_threshold = prev
        return self._detect_annotations_inner(image_bgr, prompts)

    def _detect_annotations_inner(
        self, image_bgr: np.ndarray, prompts: List[str]
    ) -> List[ObjectAnnotation]:
        annotations = []
        for det in self.detect(image_bgr, prompts):
            annotations.append(
                ObjectAnnotation(
                    name=det.label,
                    location_description=bbox_to_location_description(
                        det.bbox_xyxy_norm
                    ),
                    touched=False,  # determined downstream by the contact detector
                    bbox=np.asarray(det.bbox_xyxy_norm, dtype=np.float32),
                    state="idle",
                )
            )
        return annotations


# ---------------------------------------------------------------------------
# OWL-ViT backend (default)
# ---------------------------------------------------------------------------


class OwlViTDetector(Detector2D):
    """Zero-shot detection with OWL-ViT (``google/owlvit-base-patch32``).

    This is the detector the pipeline has used since the beginning (it was
    historically misnamed ``GroundingDINODetector`` in ``src/grounding_detector.py``).
    """

    backend_name = "owlvit"

    def __init__(self, config: Optional[Detector2DConfig] = None):
        super().__init__(config)
        self._processor = None
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        if not _OWL_VIT_IMPORT_OK:
            self._warn_once(
                "owlvit-missing",
                "OWL-ViT unavailable — transformers/torch not installed. "
                "Install via: pip install transformers>=4.30.0 torch>=2.0.0",
            )
            return
        try:
            logger.info(
                "Loading OWL-ViT model: %s on %s",
                self.config.model_name,
                self._device,
            )
            self._processor = AutoProcessor.from_pretrained(self.config.model_name)
            self._model = OwlViTForObjectDetection.from_pretrained(
                self.config.model_name
            ).to(self._device)
            self._model.eval()
            logger.info("OWL-ViT model loaded successfully")
        except Exception as e:
            logger.error("Failed to load OWL-ViT model: %s", e)
            self._processor = None
            self._model = None

    def is_available(self) -> bool:
        return self._model is not None and self._processor is not None

    def detect(self, image_bgr: np.ndarray, prompts: List[str]) -> List[Detection]:
        import cv2

        if not self.is_available():
            self._warn_once(
                "owlvit-unavailable", "OWL-ViT not available, returning empty list"
            )
            return []
        if not prompts:
            return []

        image_rgb = (
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if image_bgr.shape[2] == 3
            else image_bgr
        )
        try:
            inputs = self._processor(
                images=image_rgb, text=list(prompts), return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            target_sizes = torch.tensor([image_rgb.shape[:2]]).to(self._device)
            results = self._processor.post_process_object_detection(
                outputs,
                threshold=self.config.confidence_threshold,
                target_sizes=target_sizes,
            )[0]
        except Exception as e:
            logger.error("OWL-ViT detection failed: %s", e)
            return []

        h, w = image_rgb.shape[:2]
        detections = []
        for box, score, label_idx in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            score_f = float(score)
            if score_f < self.config.confidence_threshold:
                continue
            li = int(label_idx.item()) if hasattr(label_idx, "item") else int(label_idx)
            label = prompts[li] if 0 <= li < len(prompts) else "unknown"
            bbox_norm = np.array(
                [
                    float(box[0]) / w,
                    float(box[1]) / h,
                    float(box[2]) / w,
                    float(box[3]) / h,
                ],
                dtype=np.float32,
            )
            detections.append(
                Detection(
                    bbox_xyxy_norm=np.clip(bbox_norm, 0.0, 1.0),
                    label=label,
                    score=score_f,
                )
            )
        return detections


# ---------------------------------------------------------------------------
# Grounding DINO backend (IDEA-Research, opt-in)
# ---------------------------------------------------------------------------


class GroundingDinoDetector(Detector2D):
    """Real Grounding DINO (IDEA-Research, SwinT_OGC) text-conditioned detector.

    Lazy: importing this module and constructing the class never touch the
    weights. The model loads on first :meth:`is_available` / :meth:`detect`.
    Missing weights → ``is_available()`` is False with a loud message naming
    the exact download URL (no auto-download).
    """

    backend_name = "grounding_dino"

    def __init__(self, config: Optional[Detector2DConfig] = None):
        super().__init__(config)
        self._model = None

    # -- lazy model ------------------------------------------------------
    def _ensure_model(self) -> bool:
        """Load the model on first use. Returns True when ready."""
        if self._model is not None:
            return True
        if not _GDINO_IMPORT_OK:
            self._warn_once(
                "gdino-missing",
                "Grounding DINO backend requested but the 'groundingdino' package "
                "is not installed. CPU-only install: pip install --no-build-isolation "
                "-e git+https://github.com/IDEA-Research/GroundingDINO.git "
                "(falling back to no detector).",
            )
            return False
        weights = self.config.grounding_dino_weights
        if not weights or not os.path.exists(weights):
            self._warn_once(
                "gdino-weights",
                "Grounding DINO weights not found at '%s'. Download manually "
                "(no auto-download): %s — and the config %s — then set "
                "'perception.grounding_dino_weights' in configs/default.yaml. "
                "(falling back to no detector).",
                weights,
                GROUNDING_DINO_WEIGHTS_URL,
                GROUNDING_DINO_CONFIG_URL,
            )
            return False
        cfg_path = self.config.grounding_dino_config
        if not os.path.exists(cfg_path):
            # Fall back to the config shipped inside the installed package.
            try:
                import groundingdino

                pkg_cfg = os.path.join(
                    os.path.dirname(groundingdino.__file__),
                    "config",
                    "GroundingDINO_SwinT_OGC.py",
                )
                if os.path.exists(pkg_cfg):
                    cfg_path = pkg_cfg
            except Exception:
                pass
        try:
            logger.info(
                "Loading Grounding DINO (SwinT_OGC) weights: %s on %s",
                weights,
                self._device,
            )
            self._model = load_model(cfg_path, weights, device=self._device)
            logger.info("Grounding DINO model loaded successfully")
        except Exception as e:
            logger.error("Failed to load Grounding DINO model: %s", e)
            self._model = None
        return self._model is not None

    def is_available(self) -> bool:
        return self._ensure_model()

    # -- detection ---------------------------------------------------------
    def detect(self, image_bgr: np.ndarray, prompts: List[str]) -> List[Detection]:
        if not self.is_available():
            return []
        if not prompts:
            return []
        caption = ". ".join(p.strip() for p in prompts if p and p.strip())
        if not caption:
            return []
        caption += "."

        import torch as _t

        image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        try:
            boxes, logits, phrases = predict(
                model=self._model,
                image=_t.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0,
                caption=caption,
                box_threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                device=self._device,
            )
        except Exception as e:
            logger.error("Grounding DINO detection failed: %s", e)
            return []

        h, w = image_bgr.shape[:2]
        # boxes are normalized cxcywh -> pixel xyxy -> normalized xyxy
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes).cpu() * _t.tensor(
            [w, h, w, h], dtype=_t.float32
        )
        detections = []
        for box, phrase, logit in zip(boxes_xyxy, phrases, logits):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            bbox_norm = np.clip(
                np.array([x1 / w, y1 / h, x2 / w, y2 / h], dtype=np.float32),
                0.0,
                1.0,
            )
            detections.append(
                Detection(
                    bbox_xyxy_norm=bbox_norm,
                    label=self._canonicalize_name(str(phrase), prompts),
                    score=float(logit),
                )
            )
        return self._nms(detections)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _canonicalize_name(phrase: str, prompts: List[str]) -> str:
        """Map a model phrase back onto one of the requested prompts."""
        pl = phrase.lower().strip()
        for p in prompts:
            if p.lower().strip() == pl:
                return p
        best, best_score = phrase, 0.5
        for p in prompts:
            s = SequenceMatcher(None, pl, p.lower().strip()).ratio()
            if s > best_score:
                best, best_score = p, s
        return best

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def _nms(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        if not detections:
            return []
        ordered = sorted(detections, key=lambda d: d.score, reverse=True)
        keep: List[Detection] = []
        for det in ordered:
            if all(self._iou(det.bbox_xyxy_norm, k.bbox_xyxy_norm) <= iou_threshold for k in keep):
                keep.append(det)
        return keep


# ---------------------------------------------------------------------------
# LocateAnything output parsing
# ---------------------------------------------------------------------------

_LA_BOX_RE = re.compile(r"<box>(.*?)</box>", re.DOTALL | re.IGNORECASE)
_LA_INT_RE = re.compile(r"-?\d+")


def parse_locate_anything_boxes(text: str) -> List[np.ndarray]:
    """Parse LocateAnything ``<box>`` token spans into normalized [0, 1] xyxy boxes.

    The model emits coordinates as integers in [0, 1000]; each is divided by
    1000 and clipped to [0, 1] (corner order is normalized too, in case the
    model emits ``x2 < x1``). Point spans (``<box> x, y </box>``, 2 ints)
    carry no box geometry and are ignored. Malformed spans and empty input
    yield no boxes — never raises.
    """
    boxes: List[np.ndarray] = []
    if not text:
        return boxes
    for match in _LA_BOX_RE.finditer(text):
        nums = [int(n) for n in _LA_INT_RE.findall(match.group(1))]
        if len(nums) == 4:
            x1, y1, x2, y2 = (n / 1000.0 for n in nums)
            x_lo, x_hi = min(x1, x2), max(x1, x2)
            y_lo, y_hi = min(y1, y2), max(y1, y2)
            boxes.append(
                np.clip(
                    np.array([x_lo, y_lo, x_hi, y_hi], dtype=np.float32),
                    0.0,
                    1.0,
                )
            )
        elif len(nums) == 2:
            logger.debug(
                "LocateAnything point token ignored (no box geometry): %s",
                match.group(0)[:64],
            )
        else:
            logger.debug(
                "LocateAnything malformed <box> span ignored: %s",
                match.group(0)[:64],
            )
    return boxes


# ---------------------------------------------------------------------------
# LocateAnything backend (NVIDIA, opt-in research)
# ---------------------------------------------------------------------------


class LocateAnythingDetector(Detector2D):
    """NVIDIA LocateAnything-3B open-vocabulary visual grounding (opt-in).

    MoonViT + Qwen2.5-3B-Instruct with Parallel Box Decoding: given an image
    and a text query it emits ``<box> x1, y1, x2, y2 </box>`` tokens with
    coordinates in [0, 1000]. Stronger than OWL-ViT on cluttered scenes, so it
    serves as the pipeline's accuracy oracle — but note the caveats:

    * LICENSE: NVIDIA **non-commercial, research-only** weights. Eval/research
      use; OWL-ViT stays the commercial default.
    * Slow: off-CUDA it runs in autoregressive ``slow`` mode — tens of seconds
      per frame on old Intel CPUs. Offline annotation only.
    * Upstream pins ``transformers==4.57.1`` (5.x breaks its custom modeling
      code) and requires ``trust_remote_code=True``.

    Lazy: importing this module and constructing the class never touch the
    weights. The model loads on first :meth:`is_available` / :meth:`detect`.
    Missing transformers/weights → ``is_available()`` False, factory ``None``.

    One query per label (mirrors OWL-ViT semantics; more reliable than a
    multi-category single shot on a 3B model). The model emits no confidence
    scores, so every parsed box gets ``score=1.0``.
    """

    backend_name = "locate_anything"

    def __init__(self, config: Optional[Detector2DConfig] = None):
        super().__init__(config)
        self._processor = None
        self._tokenizer = None
        self._model = None

    # -- lazy model ------------------------------------------------------
    def _ensure_model(self) -> bool:
        """Load the model on first use. Returns True when ready."""
        if self._model is not None:
            return True
        if not _LOCATE_ANYTHING_IMPORT_OK:
            self._warn_once(
                "la-no-transformers",
                "LocateAnything backend requested but 'transformers' is not "
                "installed. Upstream pins transformers==4.57.1: "
                "pip install 'transformers==4.57.1'. (falling back to no detector).",
            )
            return False
        _ok, ver, ver_ok = _locate_anything_transformers_status()
        if not ver_ok:
            self._warn_once(
                "la-version",
                "LocateAnything upstream pins transformers==4.57.1 (5.x breaks "
                "its custom modeling code); detected %s. Continuing anyway — "
                "inference may fail.",
                ver,
            )
        weights = self.config.locate_anything_weights
        if not weights or not os.path.isdir(weights):
            self._warn_once(
                "la-weights",
                "LocateAnything weights not found at '%s'. Download manually "
                "(no auto-download): %s -> '%s/' — then set "
                "'perception.locate_anything_weights' in configs/default.yaml. "
                "(falling back to no detector).",
                weights,
                LOCATE_ANYTHING_WEIGHTS_URL,
                weights,
            )
            return False
        try:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
            import torch as _t

            logger.info(
                "Loading LocateAnything-3B weights: %s on %s (slow CPU mode)",
                weights,
                self._device,
            )
            # trust_remote_code=True is required by upstream (custom Eagle
            # modeling code); local_files_only=True enforces the project
            # no-auto-download rule — a partial dir fails loud instead of
            # silently fetching from the Hub.
            self._processor = AutoProcessor.from_pretrained(
                weights, trust_remote_code=True, local_files_only=True
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                weights, trust_remote_code=True, local_files_only=True
            )
            self._model = AutoModel.from_pretrained(
                weights,
                torch_dtype=_t.bfloat16 if self._device == "cuda" else _t.float32,
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation="sdpa",  # magi backend is CUDA-only
            )
            self._model.to(self._device)
            self._model.eval()
            logger.info("LocateAnything-3B model loaded successfully")
        except Exception as e:
            logger.error("Failed to load LocateAnything-3B model: %s", e)
            self._processor = None
            self._tokenizer = None
            self._model = None
        return self._model is not None

    def is_available(self) -> bool:
        return self._ensure_model()

    # -- detection ---------------------------------------------------------
    def _generate_text(self, image_pil: Image.Image, label: str) -> str:
        """Run one grounding query; returns raw decoded text ("" on failure)."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_pil},
                        {
                            "type": "text",
                            "text": (
                                "Please provide the bounding box of the "
                                f"<ref>{label}</ref>."
                            ),
                        },
                    ],
                }
            ]
            chat_text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[chat_text], images=[image_pil], return_tensors="pt"
            )
            inputs = {
                k: (v.to(self._device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }
            import torch as _t

            with _t.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.locate_anything_max_new_tokens,
                    do_sample=False,
                )
            input_len = inputs["input_ids"].shape[-1]
            # skip_special_tokens=False keeps the <box> coordinate tokens.
            text = self._processor.batch_decode(
                out[:, input_len:], skip_special_tokens=False
            )[0]
            return text if isinstance(text, str) else ""
        except Exception as e:
            logger.error("LocateAnything query failed for %r: %s", label, e)
            return ""

    def detect(self, image_bgr: np.ndarray, prompts: List[str]) -> List[Detection]:
        import cv2

        if not self.is_available():
            self._warn_once(
                "la-unavailable", "LocateAnything not available, returning empty list"
            )
            return []
        if not prompts:
            return []
        image_rgb = (
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if image_bgr.shape[2] == 3
            else image_bgr
        )
        pil_image = Image.fromarray(np.ascontiguousarray(image_rgb))
        detections: List[Detection] = []
        for prompt in prompts:
            label = (prompt or "").strip()
            if not label:
                continue
            text = self._generate_text(pil_image, label)
            for box in parse_locate_anything_boxes(text):
                # The model emits no confidence scores.
                detections.append(
                    Detection(bbox_xyxy_norm=box, label=label, score=1.0)
                )
        return detections


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_detector_2d(config: Optional[Detector2DConfig] = None) -> Optional[Detector2D]:
    """Build the configured 2D detector backend.

    Returns ``None`` (with a warning) when the backend cannot be built —
    e.g. missing optional dependency or missing weight files — instead of
    raising. Unknown backend names raise :class:`ValueError` (fail loud on
    config typos).
    """
    cfg = config or Detector2DConfig()
    backend = (cfg.backend or "owlvit").strip().lower()
    if backend == "owlvit":
        detector: Detector2D = OwlViTDetector(cfg)
    elif backend in ("grounding_dino", "groundingdino", "gdino"):
        detector = GroundingDinoDetector(cfg)
    elif backend in ("locate_anything", "locate-anything", "locateanything"):
        detector = LocateAnythingDetector(cfg)
    else:
        raise ValueError(
            f"Unknown detector backend {cfg.backend!r}; "
            "expected 'owlvit', 'grounding_dino' or 'locate_anything'."
        )
    if detector.is_available():
        logger.info("2D detector backend ready: %s", detector.backend_name)
        return detector
    logger.warning(
        "2D detector backend '%s' unavailable — continuing without 2D boxes.",
        backend,
    )
    return None
