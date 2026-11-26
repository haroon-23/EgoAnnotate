"""Full-video privacy redaction (face + optional PP-OCRv3 blur, CUDA blur, NVENC/x264, audio strip)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

from src.egoannotate.config import PrivacyConfig

_NVENC_PROBED: bool | None = None


def probe_nvenc_available(ffmpeg_bin: str | None = None) -> bool:
    """Verify live h264_nvenc hardware encoding works on this host (cached)."""
    global _NVENC_PROBED
    if _NVENC_PROBED is not None:
        return _NVENC_PROBED
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg")
    if not ffmpeg or not torch.cuda.is_available():
        _NVENC_PROBED = False
        return False
    try:
        enc_list = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if "h264_nvenc" not in (enc_list.stdout or ""):
            _NVENC_PROBED = False
            return False
        test_cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:d=0.1:r=30",
            "-frames:v",
            "2",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ]
        probe = subprocess.run(test_cmd, capture_output=True, timeout=15, check=False)
        _NVENC_PROBED = probe.returncode == 0
    except (OSError, subprocess.SubprocessError):
        _NVENC_PROBED = False
    return bool(_NVENC_PROBED)


def _build_ffmpeg_video_cmd(
    ffmpeg: str,
    width: int,
    height: int,
    fps: float,
    dst_video: Path,
    cfg: PrivacyConfig,
    crf: int,
) -> tuple[list[str], str]:
    use_nvenc = bool(cfg.use_nvenc and probe_nvenc_available(ffmpeg))
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
    ]
    if use_nvenc:
        encoder = "h264_nvenc"
        cmd.extend(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-rc",
                "vbr",
                "-cq",
                str(crf),
                "-bf",
                "0",
                "-g",
                "30",
                "-pix_fmt",
                "yuv420p",
            ]
        )
    else:
        encoder = "libx264"
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-bf",
                "0",
                "-g",
                "30",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
            ]
        )
    if cfg.strip_audio:
        cmd.append("-an")
    cmd.append(str(dst_video))
    return cmd, encoder


def _gaussian_blur_cuda(roi_f32: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 1D Gaussian blur on CUDA for ROI (H, W, 3)."""
    rad = max(1, min(31, round(sigma * 1.5)))
    xs = torch.arange(-rad, rad + 1, device="cuda", dtype=torch.float32)
    k1d = torch.exp(-0.5 * (xs / max(float(sigma), 1.0)) ** 2)
    k1d = k1d / k1d.sum()
    t = (
        torch.from_numpy(np.ascontiguousarray(roi_f32))
        .to(device="cuda", dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    kh = k1d.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    kv = k1d.view(1, 1, -1, 1).expand(3, 1, -1, 1)
    with torch.inference_mode():
        t = F.pad(t, (rad, rad, 0, 0), mode="reflect")
        t = F.conv2d(t, kh, groups=3)
        t = F.pad(t, (0, 0, rad, rad), mode="reflect")
        t = F.conv2d(t, kv, groups=3)
    return t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def _apply_redaction(
    img: np.ndarray,
    box: tuple[int, int, int, int],
    sigma: float,
    pad: int = 32,
    mean_blend: float = 0.35,
    use_gpu: bool = False,
) -> None:
    """Apply feathered Gaussian blur blended with local mean color."""
    h, w = img.shape[:2]
    x0 = max(0, box[0] - pad)
    y0 = max(0, box[1] - pad)
    x1 = min(w, box[2] + pad)
    y1 = min(h, box[3] + pad)
    rw, rh = x1 - x0, y1 - y0
    if rw < 6 or rh < 6:
        return

    roi = img[y0:y1, x0:x1].astype(np.float32)
    if use_gpu and torch.cuda.is_available() and rw * rh >= 4096:
        blurred = _gaussian_blur_cuda(roi, sigma=sigma)
    else:
        blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mean_col = roi.mean(axis=(0, 1), keepdims=True)
    redacted = blurred * (1.0 - mean_blend) + mean_col * mean_blend

    b = min(10, rw // 4, rh // 4)
    mask: np.ndarray = np.ones((rh, rw), dtype=np.float32)
    if b > 0:
        mask[:b, :] = 0.0
        mask[-b:, :] = 0.0
        mask[:, :b] = 0.0
        mask[:, -b:] = 0.0
        k = (b * 2) | 1
        mask = cv2.GaussianBlur(mask, (k, k), sigmaX=max(1.5, b * 0.6))
    alpha = mask[..., None]
    img[y0:y1, x0:x1] = np.clip(roi * (1.0 - alpha) + redacted * alpha, 0.0, 255.0).astype(
        np.uint8
    )


def _load_optional_ppocr_detector() -> Any | None:
    """Load Apache-2.0 PP-OCRv3/v4 via rapidocr_onnxruntime if installed."""
    try:
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR()
    except ImportError:
        return None


def _detect_text_boxes_ppocr(
    bgr: np.ndarray,
    ocr_engine: Any | None,
) -> list[tuple[int, int, int, int]]:
    if ocr_engine is None:
        return []
    h, w = bgr.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    try:
        res, _ = ocr_engine(bgr)
        for item in res or []:
            poly = item[0]
            xs = [round(pt[0]) for pt in poly]
            ys = [round(pt[1]) for pt in poly]
            x1 = max(0, min(xs))
            y1 = max(0, min(ys))
            x2 = min(w, max(xs))
            y2 = min(h, max(ys))
            if (x2 - x1) >= 10 and (y2 - y1) >= 8:
                boxes.append((x1, y1, x2, y2))
    except (RuntimeError, ValueError, TypeError, IndexError):
        return []
    return boxes


def _detect_face_boxes_fullres(
    bgr: np.ndarray,
    detector: Any,
) -> list[tuple[int, int, int, int]]:
    h, w = bgr.shape[:2]
    res = detector.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    boxes: list[tuple[int, int, int, int]] = []
    for d in res.detections or []:
        bb = d.location_data.relative_bounding_box
        x1 = max(0, round(bb.xmin * w))
        y1 = max(0, round(bb.ymin * h))
        x2 = min(w, round((bb.xmin + bb.width) * w))
        y2 = min(h, round((bb.ymin + bb.height) * h))
        if (x2 - x1) >= 8 and (y2 - y1) >= 8:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _box_laplacian_var(bgr: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    roi = bgr[y1:y2, x1:x2]
    if roi.size <= 100:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _find_unblurred_face_hits(
    mp4_path: Path,
    sample_count: int = 25,
    max_lap_var: float = 120.0,
) -> dict[int, list[tuple[int, int, int, int]]]:
    """Return {frame_idx: [boxes]} for sampled frames with sharp unblurred faces."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return {}
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
    idxs = np.linspace(0, max(0, total - 1), min(total, sample_count), dtype=int)
    det = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.50,
    )
    hits: dict[int, list[tuple[int, int, int, int]]] = {}
    try:
        for fi in idxs:
            fi_int = int(fi)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi_int)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            bad_boxes = [
                b
                for b in _detect_face_boxes_fullres(bgr, det)
                if _box_laplacian_var(bgr, b) > max_lap_var
            ]
            if bad_boxes:
                hits[fi_int] = bad_boxes
    finally:
        cap.release()
        det.close()
    return hits


def _heal_surviving_hits(
    mp4_path: Path,
    hits: dict[int, list[tuple[int, int, int, int]]],
    cfg: PrivacyConfig,
    crf: int,
) -> None:
    """Surgical re-encode strongly redacting post-compression surviving frames (+/-2)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg binary required for privacy healing pass.")

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    target_map: dict[int, list[tuple[int, int, int, int]]] = {}
    for fi, bxs in hits.items():
        for delta in (-2, -1, 0, 1, 2):
            nfi = fi + delta
            if 0 <= nfi < max(n_total, fi + 3):
                target_map.setdefault(nfi, []).extend(bxs)

    tmp_path = mp4_path.with_suffix(".heal.mp4")
    cmd, _ = _build_ffmpeg_video_cmd(ffmpeg, w, h, fps, tmp_path, cfg, crf)

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    fi = 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            if fi in target_map:
                for bx in target_map[fi]:
                    _apply_redaction(
                        fr,
                        bx,
                        sigma=45.0,
                        pad=48,
                        mean_blend=0.85,
                        use_gpu=cfg.use_gpu_blur,
                    )
            proc.stdin.write(fr.tobytes())
            fi += 1
    finally:
        cap.release()
        proc.stdin.close()
        rc = proc.wait()

    if rc != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg healing pass failed ({rc}) on {mp4_path}")
    tmp_path.replace(mp4_path)


def export_sanitized_mp4(
    src_video: Path,
    dst_video: Path,
    cfg: PrivacyConfig,
    crf: int = 25,
    detect_stride: int = 2,
    target_fps: float | None = None,
) -> dict[str, Any]:
    """Stream video 0..N-1 with face/OCR redaction, CFR resample, audio strip, and verification."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg binary required for sanitized MP4 export.")

    dst_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {src_video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    do_resample = (
        target_fps is not None
        and target_fps > 1.0
        and src_fps > 1.0
        and abs(src_fps - float(target_fps)) / float(target_fps) > 0.002
    )
    out_fps = float(target_fps) if (do_resample and target_fps is not None) else src_fps

    cmd, encoder_used = _build_ffmpeg_video_cmd(ffmpeg, w, h, out_fps, dst_video, cfg, crf)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    face_det = (
        mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.25,
        )
        if cfg.enable_face_blur
        else None
    )
    ocr_engine = _load_optional_ppocr_detector() if cfg.enable_ocr_blur else None
    active_boxes: list[tuple[int, int, int, int]] = []
    active_ocr_boxes: list[tuple[int, int, int, int]] = []
    hold_counter = 0
    ocr_hold_counter = 0
    total_written = 0
    blurred_frames = 0
    text_regions_blurred = 0
    base_sigma = max(float(cfg.blur_sigma), 32.0)

    src_idx = 0
    next_emit_t = 0.0

    try:
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            src_t = src_idx / max(src_fps, 1e-6)
            src_idx += 1

            if do_resample and src_t + 1e-9 < next_emit_t:
                continue

            emit_count = 1
            if do_resample:
                next_emit_t = (total_written + emit_count) / out_fps
                while src_t - next_emit_t >= (0.5 / out_fps):
                    emit_count += 1
                    next_emit_t = (total_written + emit_count) / out_fps

            if face_det is not None:
                if total_written % max(1, detect_stride) == 0:
                    found = _detect_face_boxes_fullres(fr, face_det)
                    if found:
                        active_boxes = found
                        hold_counter = max(3, detect_stride + 2)
                    elif hold_counter > 0:
                        hold_counter -= 1
                    else:
                        active_boxes = []

                if active_boxes:
                    for bx in active_boxes:
                        _apply_redaction(
                            fr,
                            bx,
                            sigma=base_sigma,
                            pad=36,
                            mean_blend=0.40,
                            use_gpu=cfg.use_gpu_blur,
                        )
                    blurred_frames += emit_count

            if ocr_engine is not None:
                if total_written % max(1, detect_stride * 3) == 0:
                    found_ocr = _detect_text_boxes_ppocr(fr, ocr_engine)
                    if found_ocr:
                        active_ocr_boxes = found_ocr
                        ocr_hold_counter = max(4, detect_stride * 3)
                    elif ocr_hold_counter > 0:
                        ocr_hold_counter -= 1
                    else:
                        active_ocr_boxes = []
                if active_ocr_boxes:
                    for obx in active_ocr_boxes:
                        _apply_redaction(
                            fr,
                            obx,
                            sigma=max(18.0, base_sigma * 0.7),
                            pad=8,
                            mean_blend=0.65,
                            use_gpu=cfg.use_gpu_blur,
                        )
                    text_regions_blurred += len(active_ocr_boxes) * emit_count

            raw_bytes = fr.tobytes()
            for _ in range(emit_count):
                proc.stdin.write(raw_bytes)
                total_written += 1
    finally:
        cap.release()
        proc.stdin.close()
        rc = proc.wait()
        if face_det is not None:
            face_det.close()

    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc} exporting {dst_video}")

    unblurred_audit = 0
    heal_passes = 0
    if cfg.enable_face_blur and cfg.verify_zero_unblurred_faces:
        for _ in range(2):
            hits = _find_unblurred_face_hits(dst_video, sample_count=25)
            unblurred_audit = sum(len(bxs) for bxs in hits.values())
            if unblurred_audit == 0:
                break
            _heal_surviving_hits(dst_video, hits, cfg=cfg, crf=crf)
            heal_passes += 1
        unblurred_audit = verify_zero_unblurred_faces(dst_video, sample_count=25)

    return {
        "output_video": str(dst_video),
        "frames_written": total_written,
        "output_fps": round(out_fps, 4),
        "encoder_used": encoder_used,
        "frames_blurred": blurred_frames,
        "ocr_blur_enabled": bool(cfg.enable_ocr_blur and ocr_engine is not None),
        "text_regions_blurred": text_regions_blurred,
        "heal_passes": heal_passes,
        "audio_stripped": cfg.strip_audio,
        "verified_unblurred_faces": unblurred_audit,
        "privacy_pass": bool(unblurred_audit == 0),
    }


def verify_zero_unblurred_faces(
    mp4_path: Path,
    sample_count: int = 25,
    max_lap_var: float = 120.0,
) -> int:
    """Sample evenly across the exported MP4 and verify zero sharp unblurred faces remain."""
    hits = _find_unblurred_face_hits(
        mp4_path,
        sample_count=sample_count,
        max_lap_var=max_lap_var,
    )
    return sum(len(bxs) for bxs in hits.values())
