"""Video loading, frame extraction, and sampling.

Handles both uniform FPS-based extraction and timestamp-targeted extraction
for downstream annotation stages.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Extracts and pre-processes frames from video files.

    Frames are converted from BGR → RGB, resized with area interpolation,
    and saved as PNG images with zero-padded filenames.

    Args:
        target_fps: Desired sampling rate in frames per second. Frames are
            uniformly sub-sampled from the source video to approximate this
            rate. If ``target_fps`` ≥ the source FPS, every frame is kept.
        image_size: Output spatial dimensions as ``(width, height)``.

    Example::

        vp = VideoProcessor(target_fps=2, image_size=(224, 224))
        paths = vp.process("data/raw/video.mp4", "data/frames/video")
    """

    def __init__(
        self,
        target_fps: float = 30.0,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError(f"image_size dimensions must be positive, got {image_size}")

        self.target_fps = target_fps
        self.image_size = image_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, video_path: str, output_dir: str) -> List[str]:
        """Extract frames from a video at the configured target FPS.

        Args:
            video_path: Path to the source video file (MP4, AVI, MOV, etc.).
            output_dir: Directory where extracted PNGs will be saved.
                Created automatically if it does not exist.

        Returns:
            Ordered list of absolute paths to saved frame images.

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

        try:
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if original_fps <= 0:
                logger.warning(
                    "Could not determine FPS for %s, defaulting to 30.0",
                    video_path,
                )
                original_fps = 30.0

            # Calculate how many source frames to skip between samples.
            # sample_every = 1 means keep every frame.
            sample_every = max(1, int(round(original_fps / self.target_fps)))
            effective_fps = original_fps / sample_every

            output_dir_obj = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)

            saved_paths: List[str] = []
            frame_idx = 0
            save_idx = 0

            pbar = tqdm(
                total=total_frames,
                desc=f"Extracting frames",
                unit="fr",
                leave=True,
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_every == 0:
                    # BGR → RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Resize with INTER_AREA (best for downscaling)
                    frame_resized = cv2.resize(
                        frame_rgb,
                        self.image_size,
                        interpolation=cv2.INTER_AREA,
                    )

                    # Save as PNG — convert back to BGR for cv2.imwrite
                    filename = f"frame_{save_idx:06d}.png"
                    save_path = output_dir_obj / filename
                    frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(str(save_path), frame_bgr)

                    if not success:
                        raise IOError(
                            f"Failed to write frame to {save_path}. "
                            f"Check disk space and directory permissions."
                        )

                    saved_paths.append(str(save_path))
                    save_idx += 1

                frame_idx += 1
                pbar.update(1)

            pbar.close()

        finally:
            cap.release()

        # Summary
        duration_sec = total_frames / original_fps if original_fps > 0 else 0.0
        print(
            f"\n{'─' * 50}\n"
            f"Video:          {video_path_obj.name}\n"
            f"Original FPS:   {original_fps:.1f}\n"
            f"Total frames:   {total_frames}\n"
            f"Duration:       {duration_sec:.1f}s\n"
            f"Sample every:   {sample_every} frames (≈{effective_fps:.1f} FPS)\n"
            f"Saved frames:   {save_idx}\n"
            f"Output dir:     {output_dir}\n"
            f"{'─' * 50}"
        )

        logger.info(
            "Extracted %d/%d frames from %s → %s",
            save_idx,
            total_frames,
            video_path,
            output_dir,
        )

        return saved_paths

    def extract_frames_at_times(
        self,
        video_path: str,
        timestamps: List[float],
    ) -> List[Tuple[float, np.ndarray]]:
        """Extract frames at specific timestamps (for sensor-sync use cases).

        Seeks to each requested timestamp and returns the nearest frame.
        Useful for aligning video frames with external sensor data (IMU,
        eye-tracker, etc.) that has its own clock.

        Args:
            video_path: Path to the source video file.
            timestamps: List of timestamps in seconds to extract frames at.
                Need not be sorted — they will be processed in order.

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
                frame_resized = cv2.resize(
                    frame_rgb,
                    self.image_size,
                    interpolation=cv2.INTER_AREA,
                )

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
