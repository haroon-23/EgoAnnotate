"""Video loading, frame extraction, and sampling.

Handles both uniform FPS-based extraction and timestamp-targeted extraction
for downstream annotation stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrames:
    """Result of :meth:`VideoProcessor.process_with_metadata`.

    ``timestamps_s[i]`` is the presentation time of ``paths[i]`` in the source
    video; ``effective_fps`` is the measured sampled rate (median spacing), not
    the requested one.
    """

    paths: List[str]
    timestamps_s: List[float]
    source_indices: List[int]
    source_fps: float
    effective_fps: float
    source_size: Tuple[int, int]  # (width, height) of the decoded video
    frame_size: Tuple[int, int]  # (width, height) of the saved frames
    small_paths: List[str] = field(default_factory=list)


def fit_max_side(size: Tuple[int, int], max_side: int) -> Tuple[int, int]:
    """Scale ``(w, h)`` so the long side is <= ``max_side``, keeping aspect."""
    w, h = size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


class VideoProcessor:
    """Extracts and pre-processes frames from video files.

    Saved frames keep the video's native aspect ratio (long side limited to
    ``tracking_max_side``) because hand/camera geometry is computed on them;
    squashing to a square would change the pixel aspect and break intrinsics.
    Small frames for models that want them are written on request via
    ``small_output_dir`` (``image_size``, squashed on purpose).

    Args:
        target_fps: Desired sampling rate in frames per second. Frames are
            chosen by source timestamp so the measured rate matches the target
            as closely as the source allows; if ``target_fps`` ≥ the source FPS
            every frame is kept.
        image_size: Size ``(width, height)`` of the optional small frames and
            of :meth:`extract_frames_at_times` output.
        tracking_max_side: Long-side limit of the native-aspect frames.

    Example::

        vp = VideoProcessor(target_fps=10)
        paths = vp.process("data/raw/video.mp4", "data/frames/video")
        vp.last_extraction.timestamps_s  # true source timestamps
    """

    def __init__(
        self,
        target_fps: float = 30.0,
        image_size: Tuple[int, int] = (224, 224),
        tracking_max_side: int = 960,
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError(f"image_size dimensions must be positive, got {image_size}")
        if tracking_max_side <= 0:
            raise ValueError(f"tracking_max_side must be positive, got {tracking_max_side}")

        self.target_fps = target_fps
        self.image_size = image_size
        self.tracking_max_side = int(tracking_max_side)
        self.last_extraction: Optional[ExtractedFrames] = None

    @property
    def effective_fps(self) -> Optional[float]:
        """Measured sampled fps of the last :meth:`process` call."""
        return None if self.last_extraction is None else self.last_extraction.effective_fps

    @property
    def timestamps_s(self) -> List[float]:
        """Source timestamps of the frames returned by the last :meth:`process`."""
        return [] if self.last_extraction is None else list(self.last_extraction.timestamps_s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        video_path: str,
        output_dir: str,
        small_output_dir: Optional[str] = None,
    ) -> List[str]:
        """Extract frames at the configured target FPS (see :meth:`process_with_metadata`).

        Returns the ordered list of saved native-aspect frame paths. Timestamps
        and the measured fps are on :attr:`last_extraction`.
        """
        return self.process_with_metadata(video_path, output_dir, small_output_dir).paths

    def process_with_metadata(
        self,
        video_path: str,
        output_dir: str,
        small_output_dir: Optional[str] = None,
    ) -> ExtractedFrames:
        """Extract frames, returning paths plus true timestamps and fps.

        Args:
            video_path: Path to the source video file (MP4, AVI, MOV, etc.).
            output_dir: Directory for native-aspect PNGs (created if missing).
            small_output_dir: If given, also write ``image_size`` frames there.

        Raises:
            ValueError: If *video_path* does not exist or cannot be opened.
            IOError: If a frame fails to write to disk.
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.is_file():
            raise ValueError(
                f"Video file not found: {video_path}\n"
                f"Check that the path exists and is a valid video file."
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(
                f"Cannot open video: {video_path}\n"
                f"Ensure the file is a supported format (MP4/AVI/MOV/MKV) "
                f"and that OpenCV was built with the required codec support.\n"
                f"Install codecs: pip install opencv-python-headless"
            )

        paths: List[str] = []
        small_paths: List[str] = []
        stamps: List[float] = []
        indices: List[int] = []
        source_size = (0, 0)
        frame_size = (0, 0)
        try:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if source_fps <= 0:
                logger.warning("Could not determine FPS for %s, defaulting to 30.0", video_path)
                source_fps = 30.0

            keep_all = self.target_fps >= source_fps * 0.999
            period = 1.0 / self.target_fps
            half_frame = 0.5 / source_fps

            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            small_dir = Path(small_output_dir) if small_output_dir else None
            if small_dir is not None:
                small_dir.mkdir(parents=True, exist_ok=True)

            pbar = tqdm(total=total_frames, desc="Extracting frames", unit="fr", leave=True)
            frame_idx = 0
            next_t = 0.0
            prev_ts = -1.0
            while True:
                # POS_MSEC before the read is the presentation time of the frame
                # about to be decoded; fall back to idx/fps if it is not monotonic.
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                ret, frame = cap.read()
                if not ret:
                    break
                ts = pts if pts > prev_ts or frame_idx == 0 else frame_idx / source_fps
                prev_ts = ts

                if keep_all or ts + half_frame >= next_t:
                    if not keep_all:
                        next_t += period
                        if next_t < ts - period:
                            next_t = ts + period
                    h, w = frame.shape[:2]
                    source_size = (w, h)
                    frame_size = fit_max_side((w, h), self.tracking_max_side)
                    out = frame if frame_size == (w, h) else cv2.resize(
                        frame, frame_size, interpolation=cv2.INTER_AREA)

                    name = f"frame_{len(paths):06d}.png"
                    save_path = out_dir / name
                    if not cv2.imwrite(str(save_path), out):
                        raise IOError(
                            f"Failed to write frame to {save_path}. "
                            f"Check disk space and directory permissions."
                        )
                    if small_dir is not None:
                        small = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
                        small_path = small_dir / name
                        if not cv2.imwrite(str(small_path), small):
                            raise IOError(f"Failed to write frame to {small_path}.")
                        small_paths.append(str(small_path))

                    paths.append(str(save_path))
                    stamps.append(float(ts))
                    indices.append(frame_idx)

                frame_idx += 1
                pbar.update(1)
            pbar.close()
        finally:
            cap.release()

        if len(stamps) > 1:
            effective_fps = float(1.0 / np.median(np.diff(stamps)))
        else:
            effective_fps = float(min(self.target_fps, source_fps))

        self.last_extraction = ExtractedFrames(
            paths=paths,
            timestamps_s=stamps,
            source_indices=indices,
            source_fps=float(source_fps),
            effective_fps=effective_fps,
            source_size=source_size,
            frame_size=frame_size,
            small_paths=small_paths,
        )

        duration_sec = total_frames / source_fps if source_fps > 0 else 0.0
        print(
            f"\n{'─' * 50}\n"
            f"Video:          {video_path_obj.name}\n"
            f"Source FPS:     {source_fps:.2f}\n"
            f"Total frames:   {total_frames}\n"
            f"Duration:       {duration_sec:.1f}s\n"
            f"Target FPS:     {self.target_fps:.2f} (effective {effective_fps:.2f})\n"
            f"Frame size:     {frame_size[0]}x{frame_size[1]} (source {source_size[0]}x{source_size[1]})\n"
            f"Saved frames:   {len(paths)}\n"
            f"Output dir:     {output_dir}\n"
            f"{'─' * 50}"
        )
        logger.info("Extracted %d/%d frames from %s -> %s", len(paths), total_frames, video_path, output_dir)
        return self.last_extraction

    def extract_frames_at_times(
        self,
        video_path: str,
        timestamps: List[float],
        keep_aspect: bool = False,
    ) -> List[Tuple[float, np.ndarray]]:
        """Extract frames at specific timestamps (for sensor-sync use cases).

        Seeks to each requested timestamp and returns the nearest frame.
        Useful for aligning video frames with external sensor data (IMU,
        eye-tracker, etc.) that has its own clock.

        Args:
            video_path: Path to the source video file.
            timestamps: List of timestamps in seconds to extract frames at.
                Need not be sorted — they will be processed in order.
            keep_aspect: Resize to native aspect with long side
                ``tracking_max_side`` instead of squashing to :attr:`image_size`.

        Returns:
            List of ``(actual_timestamp, frame_rgb)`` tuples where
            ``frame_rgb`` is an ``np.ndarray`` in RGB format, resized to
            :attr:`image_size`. The actual timestamp may differ slightly
            from the requested one due to keyframe alignment.

        Raises:
            ValueError: If *video_path* cannot be opened.
        """
        if not Path(video_path).is_file():
            raise ValueError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(
                f"Cannot open video: {video_path}\n"
                f"Ensure the file is a supported format and codecs are installed."
            )

        results: List[Tuple[float, np.ndarray]] = []

        try:
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            if original_fps <= 0:
                original_fps = 30.0

            for ts in tqdm(timestamps, desc="Seeking timestamps", unit="ts"):
                # Seek to the target position in milliseconds
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(
                        "Could not read frame at t=%.3fs in %s, skipping",
                        ts,
                        video_path,
                    )
                    continue

                # Record the actual position we ended up at
                actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                actual_ts = actual_ms / 1000.0

                # BGR → RGB and resize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                target = (
                    fit_max_side((frame.shape[1], frame.shape[0]), self.tracking_max_side)
                    if keep_aspect else self.image_size
                )
                frame_resized = cv2.resize(frame_rgb, target, interpolation=cv2.INTER_AREA)

                results.append((actual_ts, frame_resized))

        finally:
            cap.release()

        logger.info(
            "Extracted %d/%d timestamp-targeted frames from %s",
            len(results),
            len(timestamps),
            video_path,
        )

        return results
