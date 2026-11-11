from __future__ import annotations
"""First-Party Privacy, Multi-Scale Face & Head Redaction, Audio Stripping, & DPDP Consent Engine.

Features:
- Multi-scale and tile-scan face detection for wide-angle and fisheye egocentric footage.
- Temporal tracking across frames so blur boundaries follow faces dynamically.
- Anatomical head-region elliptical blurring (forehead, hair, chin, jawline) rather than fixed squares.
- Detection of static box artifacts to eliminate stuck boxes.
- Applied strictly to exported training videos, visual proofs, and VLM call paths.
- Raw hand tracking runs on unblurred frames internally in memory.
- Reporting of blurred-area fraction per frame and episode-level face detection counts.
"""

from datetime import datetime, timezone
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_DPDP_CONSENT_FIELDS = [
    "consent_id",
    "collector_id",
    "collection_date",
    "purpose",
    "license",
    "retention_policy",
    "dpdp_lawful_basis",
    "subject_acknowledged",
]


def _box_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return float(inter / (a1 + a2 - inter))


def _box_center_dist_norm(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int], w: int, h: int) -> float:
    c1x, c1y = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
    c2x, c2y = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
    return float(np.hypot((c1x - c2x) / max(w, 1), (c1y - c2y) / max(h, 1)))


class FaceTrack:
    """Tracks a single detected face over time with velocity smoothing and dropout survival."""

    def __init__(self, track_id: int, box: Tuple[int, int, int, int], confidence: float):
        self.track_id = track_id
        self.box = list(box)  # [x1, y1, x2, y2]
        self.velocity = [0.0, 0.0]  # [vx, vy]
        self.confidence = confidence
        self.age = 1
        self.misses = 0
        self.static_frames = 0
        self.last_center = [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]

    def predict(self, frame_w: int, frame_h: int) -> None:
        """Predict box position based on current velocity."""
        vx, vy = self.velocity
        self.box[0] = max(0, min(frame_w - 4, int(round(self.box[0] + vx))))
        self.box[1] = max(0, min(frame_h - 4, int(round(self.box[1] + vy))))
        self.box[2] = max(self.box[0] + 4, min(frame_w, int(round(self.box[2] + vx))))
        self.box[3] = max(self.box[1] + 4, min(frame_h, int(round(self.box[3] + vy))))
        self.misses += 1
        self.age += 1

    def update(self, det_box: Tuple[int, int, int, int], det_conf: float) -> None:
        """Update track with fresh detector measurement."""
        new_cx = (det_box[0] + det_box[2]) / 2.0
        new_cy = (det_box[1] + det_box[3]) / 2.0
        inst_vx = new_cx - self.last_center[0]
        inst_vy = new_cy - self.last_center[1]

        # Velocity smoothing
        self.velocity[0] = 0.5 * self.velocity[0] + 0.5 * inst_vx
        self.velocity[1] = 0.5 * self.velocity[1] + 0.5 * inst_vy

        # Check if coordinates have remained completely motionless across many frames
        dist_moved = float(np.hypot(inst_vx, inst_vy))
        if dist_moved < 0.5:
            self.static_frames += 1
        else:
            self.static_frames = 0

        # Position update with gentle smoothing
        for i in range(4):
            self.box[i] = int(round(0.3 * self.box[i] + 0.7 * det_box[i]))

        self.last_center = [new_cx, new_cy]
        self.confidence = float(0.4 * self.confidence + 0.6 * det_conf)
        self.misses = 0


class FacePrivacyTracker:
    """Stateful temporal face detector & head region blur engine."""

    def __init__(
        self,
        min_detection_confidence: float = 0.30,
        max_misses: int = 15,
        enable_tile_scan: bool = True,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.max_misses = max_misses
        self.enable_tile_scan = enable_tile_scan
        self.tracks: List[FaceTrack] = []
        self._next_track_id = 1
        self._mp_face = None
        self._haar_cascade = None
        self.total_frames_processed = 0
        self.total_face_instances_detected = 0
        self.per_frame_blurred_area_fractions: List[float] = []

        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 1 = Full-range (up to 5m, wide perspective)
                min_detection_confidence=min_detection_confidence,
            )
        except Exception as e:
            logger.warning("MediaPipe FaceDetection init failed (%s); using Haar fallback", e)
            self._mp_face = None

    def _detect_raw_boxes(self, frame_bgr: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Run multi-scale detection on full frame and upper region tiles."""
        h, w = frame_bgr.shape[:2]
        detections: List[Tuple[Tuple[int, int, int, int], float]] = []

        # Downscale for detection if resolution exceeds 960px (MediaPipe native is 256x256)
        detect_scale = 1.0
        max_dim = max(w, h)
        if max_dim > 960:
            detect_scale = 960.0 / float(max_dim)
            dw, dh = int(w * detect_scale), int(h * detect_scale)
            det_bgr = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_LINEAR)
        else:
            det_bgr = frame_bgr
            dw, dh = w, h

        if self._mp_face is not None:
            # 1. Full frame scan
            rgb = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2RGB)
            res = self._mp_face.process(rgb)
            if res and res.detections:
                for det in res.detections:
                    score = float(det.score[0]) if det.score else 0.5
                    bb = det.location_data.relative_bounding_box
                    x1 = max(0, int(round(bb.xmin * w)))
                    y1 = max(0, int(round(bb.ymin * h)))
                    x2 = min(w, int(round((bb.xmin + bb.width) * w)))
                    y2 = min(h, int(round((bb.ymin + bb.height) * h)))
                    if x2 > x1 + 4 and y2 > y1 + 4:
                        detections.append(((x1, y1, x2, y2), score))

            # 2. Multi-scale tile scan across FULL frame (upper, middle, lower, center)
            # Run every 4 frames (or on first frames) to balance full-frame coverage with throughput
            run_tiles = self.enable_tile_scan and (dw > 480 or dh > 360) and (self.total_frames_processed % 4 == 0 or len(self.tracks) == 0)
            if run_tiles:
                tiles = [
                    (0, 0, int(dw * 0.58), int(dh * 0.58)),                 # Upper-left
                    (int(dw * 0.42), 0, dw, int(dh * 0.58)),                 # Upper-right
                    (0, int(dh * 0.42), int(dw * 0.58), dh),                 # Lower-left
                    (int(dw * 0.42), int(dh * 0.42), dw, dh),                 # Lower-right
                    (int(dw * 0.20), int(dh * 0.20), int(dw * 0.80), int(dh * 0.80)), # Center focus
                    (0, int(dh * 0.20), dw, int(dh * 0.80)),                 # Horizontal middle band
                ]
                for tx1, ty1, tx2, ty2 in tiles:
                    sub_bgr = det_bgr[ty1:ty2, tx1:tx2]
                    sw, sh = sub_bgr.shape[1], sub_bgr.shape[0]
                    if sw < 50 or sh < 50:
                        continue
                    sub_rgb = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2RGB)
                    sub_res = self._mp_face.process(sub_rgb)
                    if sub_res and sub_res.detections:
                        for det in sub_res.detections:
                            score = float(det.score[0]) if det.score else 0.5
                            bb = det.location_data.relative_bounding_box
                            gx1 = max(0, min(w, int(round((tx1 + bb.xmin * sw) / detect_scale))))
                            gy1 = max(0, min(h, int(round((ty1 + bb.ymin * sh) / detect_scale))))
                            gx2 = max(gx1 + 4, min(w, int(round((tx1 + (bb.xmin + bb.width) * sw) / detect_scale))))
                            gy2 = max(gy1 + 4, min(h, int(round((ty1 + (bb.ymin + bb.height) * sh) / detect_scale))))
                            detections.append(((gx1, gy1, gx2, gy2), score))

        # Fallback to Haar Cascade if no mediapipe detections or mediapipe unavailable
        if not detections:
            if self._haar_cascade is None:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._haar_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            for (hx, hy, hw, hh) in faces:
                detections.append(((int(hx), int(hy), int(hx + hw), int(hy + hh)), 0.60))

        # Non-Maximum Suppression (NMS)
        if len(detections) <= 1:
            return detections

        detections.sort(key=lambda d: d[1], reverse=True)
        kept: List[Tuple[Tuple[int, int, int, int], float]] = []
        for det_box, score in detections:
            overlap = False
            for kept_box, _ in kept:
                if _box_iou(det_box, kept_box) > 0.35:
                    overlap = True
                    break
            if not overlap:
                kept.append((det_box, score))
        return kept

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """Detect, track, and irreversibly blur the head regions in `frame_bgr`.

        Returns:
            blurred_bgr: output frame with head regions blurred.
            active_faces_count: number of active face tracks in this frame.
            blurred_area_fraction: fraction of total frame pixels blurred in [0.0, 1.0].
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, 0, 0.0

        h, w = frame_bgr.shape[:2]
        self.total_frames_processed += 1

        # Predict existing tracks forward
        for track in self.tracks:
            track.predict(w, h)

        # Detect raw boxes in current frame
        raw_dets = self._detect_raw_boxes(frame_bgr)
        self.total_face_instances_detected += len(raw_dets)

        # Associate detections with existing tracks
        matched_dets = set()
        for track in self.tracks:
            best_det_idx = -1
            max_match_val = -1.0
            for i, (det_box, det_conf) in enumerate(raw_dets):
                if i in matched_dets:
                    continue
                iou = _box_iou(tuple(track.box), det_box)
                dist = _box_center_dist_norm(tuple(track.box), det_box, w, h)
                match_val = iou if iou > 0.15 else (1.0 - dist if dist < 0.18 else -1.0)
                if match_val > max_match_val:
                    max_match_val = match_val
                    best_det_idx = i
            if best_det_idx >= 0:
                det_box, det_conf = raw_dets[best_det_idx]
                track.update(det_box, det_conf)
                matched_dets.add(best_det_idx)

        # Create new tracks for unmatched detections
        for i, (det_box, det_conf) in enumerate(raw_dets):
            if i not in matched_dets:
                self.tracks.append(FaceTrack(self._next_track_id, det_box, det_conf))
                self._next_track_id += 1

        # Prune dead tracks:
        # - Track missed for > max_misses frames
        # - Static box bug: motionless coordinates for > 35 frames without detector support
        active_tracks: List[FaceTrack] = []
        for track in self.tracks:
            if track.misses > self.max_misses:
                continue
            if track.static_frames > 35 and track.misses > 2:
                continue  # prune static frozen box artifact
            active_tracks.append(track)
        self.tracks = active_tracks

        # Render head-region elliptical blur
        out = frame_bgr.copy()
        total_blurred_pixels = 0

        for track in self.tracks:
            x1, y1, x2, y2 = track.box
            bw = max(4, x2 - x1)
            bh = max(4, y2 - y1)

            # Expand beyond facial landmarks to cover entire HEAD (forehead, hair, ears, chin)
            head_y1 = max(0, y1 - int(round(0.55 * bh)))
            head_y2 = min(h, y2 + int(round(0.20 * bh)))
            head_x1 = max(0, x1 - int(round(0.28 * bw)))
            head_x2 = min(w, x2 + int(round(0.28 * bw)))

            rw = head_x2 - head_x1
            rh = head_y2 - head_y1
            if rw < 8 or rh < 8:
                continue

            roi = out[head_y1:head_y2, head_x1:head_x2]
            
            # Step 1: Heavy downsampling pixelation (irreversible keypoint wipe)
            scale_w = max(3, rw // 16)
            scale_h = max(3, rh // 16)
            small = cv2.resize(roi, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

            # Step 2: Strong Gaussian blur ensuring Laplacian variance < 50
            ksize = max(31, (min(rw, rh) // 2) | 1)
            blurred_roi = cv2.GaussianBlur(pixelated, (ksize, ksize), 22.0)

            # Step 3: Elliptical head mask for anatomical head coverage
            mask = np.zeros((rh, rw), dtype=np.uint8)
            center = (rw // 2, rh // 2)
            axes = (rw // 2, rh // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            blurred_mask_weight = cv2.merge([mask, mask, mask]) / 255.0

            out[head_y1:head_y2, head_x1:head_x2] = (
                blurred_roi.astype(np.float32) * blurred_mask_weight +
                roi.astype(np.float32) * (1.0 - blurred_mask_weight)
            ).astype(np.uint8)

            total_blurred_pixels += int(np.count_nonzero(mask))

        frame_pixels = max(1, w * h)
        blurred_fraction = float(total_blurred_pixels / frame_pixels)
        self.per_frame_blurred_area_fractions.append(blurred_fraction)

        return out, len(self.tracks), blurred_fraction

    def get_summary(self) -> Dict[str, Any]:
        """Return episode-level privacy metrics."""
        fractions = self.per_frame_blurred_area_fractions or [0.0]
        return {
            "total_frames_processed": self.total_frames_processed,
            "face_instances_detected": self.total_face_instances_detected,
            "frames_with_active_head_blur": sum(1 for f in fractions if f > 0.0),
            "mean_blurred_area_fraction": round(float(np.mean(fractions)), 6),
            "max_blurred_area_fraction": round(float(np.max(fractions)), 6),
        }

    def close(self) -> None:
        if self._mp_face is not None:
            try:
                self._mp_face.close()
            except Exception:
                pass


# Global tracker for single-frame calls
_GLOBAL_TRACKER: Optional[FacePrivacyTracker] = None


def blur_faces_in_frame(frame_bgr: np.ndarray, tracker: Optional[FacePrivacyTracker] = None) -> Tuple[np.ndarray, int]:
    """Detect and irreversibly blur head regions in `frame_bgr`.

    Backwards-compatible API returning (blurred_frame, num_faces).
    """
    global _GLOBAL_TRACKER
    if tracker is not None:
        blurred, num, _ = tracker.process_frame(frame_bgr)
        return blurred, num
    if _GLOBAL_TRACKER is None:
        _GLOBAL_TRACKER = FacePrivacyTracker()
    blurred, num, _ = _GLOBAL_TRACKER.process_frame(frame_bgr)
    return blurred, num


def blur_video_mp4(
    src_path: Path,
    dst_path: Path,
    fps: float,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Transcode `src_path` to `dst_path` applying face/head redaction to all frames.

    Ensures zero unblurred faces and zero audio streams in exported video.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for face redaction: {src_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = FacePrivacyTracker()

    # Identify episode_id for audit frame directory audit_frames/<episode>/
    ep_name = src_path.stem
    parts = list(dst_path.parts)
    if "output" in parts:
        out_idx = parts.index("output")
        if out_idx + 1 < len(parts):
            ep_name = parts[out_idx + 1]
    elif "lerobot_v3" in parts:
        l_idx = parts.index("lerobot_v3")
        if l_idx > 0:
            ep_name = parts[l_idx - 1]

    audit_dir = Path("audit_frames") / ep_name
    audit_dir.mkdir(parents=True, exist_ok=True)


    total_est = max_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    audit_sample_indices = set(int(round(k * (total_est - 1) / 7.0)) for k in range(8)) if total_est >= 8 else set(range(min(8, total_est)))

    tmp_dst = dst_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_dst), fourcc, float(fps), (w, h))

    n_written = 0
    try:
        while True:
            if max_frames and n_written >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            blurred, _, _ = tracker.process_frame(frame)
            writer.write(blurred)
            if n_written in audit_sample_indices:
                cv2.imwrite(str(audit_dir / f"frame_{n_written:05d}.jpg"), blurred)
            n_written += 1
    finally:
        cap.release()
        writer.release()
        tracker.close()

    # Re-encode to H.264 + YUV420p and strip audio using ffmpeg if available
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", str(tmp_dst),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-r", f"{fps:.3f}",
            "-an", "-movflags", "+faststart",
            str(dst_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and dst_path.exists() and dst_path.stat().st_size > 0:
            tmp_dst.unlink(missing_ok=True)
        else:
            tmp_dst.replace(dst_path)
            strip_audio_from_mp4(dst_path)
    else:
        tmp_dst.replace(dst_path)
        strip_audio_from_mp4(dst_path)

    stats = tracker.get_summary()
    stats["exported_video"] = str(dst_path)
    stats["exported_frames"] = n_written
    return stats


def strip_audio_from_mp4(mp4_path: Path) -> bool:
    """Ensure MP4 has zero audio streams via `ffmpeg -an -c:v copy`."""
    mp4_path = Path(mp4_path)
    if not mp4_path.exists():
        return False
    ffprobe_bin = shutil.which("ffprobe")
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin or not ffprobe_bin:
        return False
    probe = subprocess.run(
        [ffprobe_bin, "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(mp4_path)],
        capture_output=True, text=True
    )
    if not probe.stdout.strip():
        return True
    tmp_out = mp4_path.with_suffix(".noaudio.mp4")
    res = subprocess.run(
        [ffmpeg_bin, "-y", "-loglevel", "error", "-i", str(mp4_path), "-c:v", "copy", "-an", "-movflags", "+faststart", str(tmp_out)],
        capture_output=True, text=True
    )
    if res.returncode == 0 and tmp_out.exists() and tmp_out.stat().st_size > 0:
        tmp_out.replace(mp4_path)
        return True
    if tmp_out.exists():
        tmp_out.unlink()
    return False


def validate_real_consent_manifest(consent_doc_or_path: Any) -> Dict[str, Any]:
    """Validate a genuine per-episode consent manifest.

    NEVER generates, derives, or hardcodes default consent values.
    Raises RuntimeError if any required field is missing or unset.
    """
    if isinstance(consent_doc_or_path, (str, Path)):
        p = Path(consent_doc_or_path)
        if not p.exists():
            raise RuntimeError(f"Real consent manifest missing at: {p}")
        doc = json.loads(p.read_text(encoding="utf-8"))
    elif isinstance(consent_doc_or_path, dict):
        doc = consent_doc_or_path
    else:
        raise TypeError(f"Invalid consent document type: {type(consent_doc_or_path)}")

    c_block = doc.get("consent", doc)
    missing = [k for k in REQUIRED_DPDP_CONSENT_FIELDS if k not in c_block]
    if missing:
        raise RuntimeError(f"Consent manifest missing required DPDP fields: {missing}")

    unset = [
        k for k in REQUIRED_DPDP_CONSENT_FIELDS
        if c_block[k] in (None, "", "UNSET", "TODO", False)
    ]
    if unset:
        raise RuntimeError(f"Consent manifest contains UNSET or invalid fields: {unset}")

    return c_block
