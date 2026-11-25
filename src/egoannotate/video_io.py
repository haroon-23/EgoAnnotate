"""Robust full-length streaming video reader, chunk iterator, and auto-intrinsics estimator."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import yaml

from src.egoannotate.schemas import CameraIntrinsicsModel, VideoMetadata


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_video_ffprobe(video_path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found on PATH; install FFmpeg.")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    v_streams = [s for s in streams if s.get("codec_type") == "video"]
    a_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not v_streams:
        raise ValueError(f"No video stream found in {video_path}")
    vs = v_streams[0]

    def _parse_rate(rate_str: str) -> float:
        if not rate_str or rate_str in ("0/0", "N/A"):
            return 0.0
        if "/" in rate_str:
            num, den = rate_str.split("/", 1)
            return float(num) / max(float(den), 1e-9)
        return float(rate_str)

    r_fps = _parse_rate(str(vs.get("r_frame_rate", "0/0")))
    avg_fps = _parse_rate(str(vs.get("avg_frame_rate", "0/0")))
    fps = avg_fps if avg_fps > 1.0 else (r_fps if r_fps > 1.0 else 30.0)
    vfr = bool(r_fps > 1.0 and avg_fps > 1.0 and abs(r_fps - avg_fps) / max(r_fps, avg_fps) > 0.02)

    width = int(vs.get("width", 0))
    height = int(vs.get("height", 0))
    duration_s = float(vs.get("duration") or data.get("format", {}).get("duration") or 0.0)
    nb_frames_raw = vs.get("nb_frames")
    if nb_frames_raw and str(nb_frames_raw).isdigit():
        nb_frames = int(nb_frames_raw)
    else:
        nb_frames = max(1, round(duration_s * fps))

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "r_fps": r_fps,
        "avg_fps": avg_fps,
        "vfr_detected": vfr,
        "nb_frames": nb_frames,
        "duration_s": duration_s if duration_s > 0 else nb_frames / max(fps, 1.0),
        "codec": str(vs.get("codec_name", "unknown")),
        "has_audio": len(a_streams) > 0,
        "tags": vs.get("tags", {}),
    }


def estimate_camera_intrinsics(
    width: int,
    height: int,
    sample_frames: list[np.ndarray] | None = None,
    calibration_yaml: Path | None = None,
    estimate_distortion: bool = True,
) -> CameraIntrinsicsModel:
    """Load calibrated intrinsics if provided, else auto-estimate FOV & radial distortion."""
    if calibration_yaml is not None and Path(calibration_yaml).exists():
        cal = yaml.safe_load(Path(calibration_yaml).read_text(encoding="utf-8")) or {}
        fx = float(cal.get("fx", 0.85 * max(width, height)))
        fy = float(cal.get("fy", fx))
        cx = float(cal.get("cx", width / 2.0))
        cy = float(cal.get("cy", height / 2.0))
        dist = [float(x) for x in cal.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])]
        fov_h = 2.0 * math.degrees(math.atan2(width / 2.0, fx))
        lens: Literal["pinhole", "wide_fov", "fisheye"] = "fisheye" if fov_h > 115.0 else ("wide_fov" if fov_h > 85.0 else "pinhole")
        return CameraIntrinsicsModel(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            dist_coeffs=dist,
            fov_horizontal_deg=float(fov_h),
            lens_type=lens,
            source="calibrated_yaml",
            uncertainty_pct=2.0,
        )

    # Auto-estimation heuristic refined by edge curvature if sample frames are available
    max_dim = float(max(width, height))
    aspect = float(width) / max(float(height), 1.0)
    # Typical egocentric head-cam / action-cam / phone FOVs:
    # 16:9 wide action cams ~90-110 deg, vertical/standard smartphones ~65-78 deg
    hfov_est_deg = 82.0 if aspect >= 1.55 else 68.0
    k1_est = 0.0
    lens_type: Literal["pinhole", "wide_fov", "fisheye"] = "pinhole"

    if estimate_distortion and sample_frames:
        # Measure curved line density near image borders vs center to detect strong wide-FOV/fisheye barrel distortion
        border_curvatures: list[float] = []
        for fr in sample_frames[:6]:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 60, 160)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if len(cnt) < 40:
                    continue
                pts = cnt[:, 0, :].astype(np.float32)
                p0, p1 = pts[0], pts[-1]
                chord = float(np.linalg.norm(p1 - p0))
                if chord < 0.12 * max_dim:
                    continue
                arc = float(cv2.arcLength(cnt, False))
                ratio = arc / max(chord, 1e-6)
                # Border-located smooth curves (1.03..1.18) indicate barrel distortion
                mid = pts[len(pts) // 2]
                norm_r = math.hypot((mid[0] - width / 2.0) / (width / 2.0), (mid[1] - height / 2.0) / (height / 2.0))
                if norm_r > 0.65 and 1.02 < ratio < 1.25:
                    border_curvatures.append(ratio - 1.0)
        if len(border_curvatures) >= 6:
            med_curv = float(np.median(border_curvatures))
            if med_curv > 0.07:
                hfov_est_deg = 118.0
                k1_est = -0.18
                lens_type = "fisheye"
            elif med_curv > 0.035:
                hfov_est_deg = 96.0
                k1_est = -0.08
                lens_type = "wide_fov"

    fx = (width / 2.0) / math.tan(math.radians(hfov_est_deg / 2.0))
    fy = fx
    return CameraIntrinsicsModel(
        width=width,
        height=height,
        fx=float(fx),
        fy=float(fy),
        cx=float(width / 2.0),
        cy=float(height / 2.0),
        dist_coeffs=[float(k1_est), 0.0, 0.0, 0.0, 0.0],
        fov_horizontal_deg=float(hfov_est_deg),
        lens_type=lens_type,
        source="auto_estimated",
        uncertainty_pct=15.0,
    )


def compute_frame_image_quality(bgr: np.ndarray) -> tuple[float, float]:
    """Return (directional_sharpness_var, mean_luma) catching both isotropic defocus and head-pan motion blur."""
    h, w = bgr.shape[:2]
    scale = min(1.0, 480.0 / max(w, 1))
    small = cv2.resize(bgr, (max(16, int(w * scale)), max(16, int(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    kx = np.array([[1.0, -2.0, 1.0]], dtype=np.float64)
    ky = kx.T
    dxx = cv2.filter2D(gray, cv2.CV_64F, kx)
    dyy = cv2.filter2D(gray, cv2.CV_64F, ky)
    sharpness = 2.0 * min(float(dxx.var()), float(dyy.var()))
    mean_luma = float(gray.mean())
    return sharpness, mean_luma


@dataclass
class VideoChunk:
    chunk_id: int
    start_frame: int
    end_frame: int
    frame_indices: list[int]
    timestamps_s: list[float]
    frames: list[np.ndarray]


class VideoStreamReader:
    """Bounded-memory full-video streaming reader (processes frames 0..N-1 completely)."""

    def __init__(
        self,
        video_path: Path,
        calibration_yaml: Path | None = None,
        estimate_distortion: bool = True,
    ) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        self.probe = probe_video_ffprobe(self.video_path)
        self.sha256 = compute_file_sha256(self.video_path)
        self._sample_frames = self._read_evenly_spaced_samples(count=6)
        self.intrinsics = estimate_camera_intrinsics(
            width=int(self.probe["width"]),
            height=int(self.probe["height"]),
            sample_frames=self._sample_frames,
            calibration_yaml=calibration_yaml,
            estimate_distortion=estimate_distortion,
        )

    def _read_evenly_spaced_samples(self, count: int = 6) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return []
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or self.probe["nb_frames"]))
        idxs = np.linspace(0, max(0, total - 1), min(total, count), dtype=int)
        frames: list[np.ndarray] = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            if ok and fr is not None:
                frames.append(fr)
        cap.release()
        return frames

    def build_metadata(self, processed_frames: int) -> VideoMetadata:
        fps = float(self.probe["fps"])
        total_reported = int(self.probe["nb_frames"])
        actual_total = max(processed_frames, total_reported)
        return VideoMetadata(
            video_path=str(self.video_path),
            video_sha256=self.sha256,
            width=int(self.probe["width"]),
            height=int(self.probe["height"]),
            source_fps=fps,
            processing_fps=fps,
            total_frames=actual_total,
            processed_frames=processed_frames,
            duration_s=float(processed_frames / max(fps, 1e-6)),
            codec=str(self.probe["codec"]),
            has_audio_source=bool(self.probe["has_audio"]),
            vfr_detected=bool(self.probe["vfr_detected"]),
            is_full_video_processed=(processed_frames >= max(1, total_reported - 2)),
            intrinsics=self.intrinsics,
        )

    def iter_frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield (frame_idx, timestamp_s, bgr) sequentially across EVERY frame 0..N-1."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {self.video_path}")
        fps = float(self.probe["fps"])
        idx = 0
        try:
            while True:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                ts = idx / max(fps, 1e-6)
                yield idx, ts, fr
                idx += 1
        finally:
            cap.release()

    def iter_chunks(
        self,
        chunk_size: int = 300,
        overlap: int = 15,
    ) -> Iterator[VideoChunk]:
        """Stream complete video in fixed-size overlapping chunks (bounded RAM)."""
        buf_idx: list[int] = []
        buf_ts: list[float] = []
        buf_fr: list[np.ndarray] = []
        chunk_id = 0
        for fi, ts, fr in self.iter_frames():
            buf_idx.append(fi)
            buf_ts.append(ts)
            buf_fr.append(fr)
            if len(buf_fr) >= chunk_size:
                yield VideoChunk(
                    chunk_id=chunk_id,
                    start_frame=buf_idx[0],
                    end_frame=buf_idx[-1],
                    frame_indices=list(buf_idx),
                    timestamps_s=list(buf_ts),
                    frames=list(buf_fr),
                )
                chunk_id += 1
                if overlap > 0:
                    buf_idx = buf_idx[-overlap:]
                    buf_ts = buf_ts[-overlap:]
                    buf_fr = buf_fr[-overlap:]
                else:
                    buf_idx, buf_ts, buf_fr = [], [], []
        if buf_fr:
            yield VideoChunk(
                chunk_id=chunk_id,
                start_frame=buf_idx[0],
                end_frame=buf_idx[-1],
                frame_indices=buf_idx,
                timestamps_s=buf_ts,
                frames=buf_fr,
            )
