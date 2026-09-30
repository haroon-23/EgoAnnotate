"""Torch Dataset over a merged LeRobot v3 root (Phase B).

Reads ``meta/info.json`` (dims, fps), ``meta/stats.json`` (normalization),
``meta/episodes/...`` (episode bounds + video lookup) and the data chunk
parquets. Videos are decoded LAZILY via one cached ``cv2.VideoCapture`` per
mp4 file — frames are never preloaded (old-Mac RAM discipline).

``torch`` is optional: this module imports cleanly without it; constructing
:class:`LeRobotParquetDataset` without torch raises a helpful error.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from . import require_torch

logger = logging.getLogger(__name__)

_DatasetBase: type
try:
    import torch as _torch  # noqa: F401
    _DatasetBase = _torch.utils.data.Dataset
except ImportError:  # torch optional: module must import cleanly without it
    _DatasetBase = object


def _norm_params(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """(mean, scale) for feature-wise normalization.

    Uses std; falls back to the q01/q99 range when std is ~0 (constant
    feature), floored at 1e-8 so we never divide by zero.
    """
    mean = np.asarray(entry["mean"], dtype=np.float32)
    std = np.asarray(entry["std"], dtype=np.float32)
    q01 = np.asarray(entry["q01"], dtype=np.float32)
    q99 = np.asarray(entry["q99"], dtype=np.float32)
    fallback = np.maximum((q99 - q01) / 2.0, 1e-8).astype(np.float32)
    scale = np.where(std > 1e-8, std, fallback)
    return mean, scale


class LeRobotParquetDataset(_DatasetBase):
    """Random-access (state, image, action-chunk) samples from a merged export.

    Args:
        merged_dir: merged ``lerobot_v3`` root (see scripts/merge_lerobot.py).
        video_key: video feature key, e.g. ``"observation.images.ego"``.
        chunk_size: action chunk length k (tail padded by repeating last frame).
        image_size: frames are resized to (image_size, image_size).

    Sample ``t`` (global frame index) -> ``{"state": (D,), "images": (3,H,W),
    "action_chunk": (k,D), "episode_index": int}`` with state/action
    normalized by the merged ``stats.json``.
    """

    def __init__(self, merged_dir: str | Path, video_key: str = "observation.images.ego",
                 chunk_size: int = 16, image_size: int = 96):
        require_torch()  # helpful error when torch is not installed
        self.root = Path(merged_dir)
        self.chunk_size = int(chunk_size)
        self.image_size = int(image_size)

        info = json.loads((self.root / "meta" / "info.json").read_text())
        features = info["features"]
        self.state_dim = int(features["observation.state"]["shape"][0])
        self.action_dim = int(features["action"]["shape"][0])

        video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
        if video_key not in video_keys:
            raise ValueError(
                f"video_key {video_key!r} not in dataset video features {video_keys}")
        self.video_key = video_key

        stats = json.loads((self.root / "meta" / "stats.json").read_text())
        self._state_mean, self._state_scale = _norm_params(stats["observation.state"])
        self._act_mean, self._act_scale = _norm_params(stats["action"])

        ep_df = pd.read_parquet(
            self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        ep_df = ep_df.sort_values("episode_index").reset_index(drop=True)
        lookup_col = f"videos/{video_key}/file_index"
        self._episodes: List[dict] = []
        for _, row in ep_df.iterrows():
            fi = row[lookup_col]
            if pd.isna(fi):
                raise ValueError(
                    f"Episode {int(row['episode_index'])} has no video "
                    f"(videos/{video_key}/file_index is NaN). Merge allows mixed "
                    "video, but a vision policy cannot train without images — "
                    "re-merge with video for every episode or drop it.")
            self._episodes.append({
                "episode_index": int(row["episode_index"]),
                "from": int(row["dataset_from_index"]),
                "to": int(row["dataset_to_index"]),
                "length": int(row["length"]),
                "video_path": self.root / "videos" / video_key / "chunk-000"
                              / f"file-{int(fi):03d}.mp4",
            })
        self._bounds = np.array([e["from"] for e in self._episodes]
                                + [self._episodes[-1]["to"]])

        chunks = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not chunks:
            raise FileNotFoundError(f"No data chunks under {self.root / 'data'}")
        self._df = pd.concat([pd.read_parquet(c) for c in chunks], ignore_index=True)
        # Per-episode row slices for O(1) access.
        self._ep_frames: List[pd.DataFrame] = []
        for e in self._episodes:
            sl = self._df.iloc[e["from"]:e["to"]].reset_index(drop=True)
            assert len(sl) == e["length"], (len(sl), e["length"])
            self._ep_frames.append(sl)

        self._caps: Dict[str, cv2.VideoCapture] = {}
        logger.info("LeRobotParquetDataset: %d episodes, %d frames, state_dim=%d, "
                    "action_dim=%d, chunk_size=%d",
                    len(self._episodes), len(self._df),
                    self.state_dim, self.action_dim, self.chunk_size)

    # -- torch Dataset API -------------------------------------------------
    def __len__(self) -> int:
        return len(self._df)

    def _read_frame(self, ep: dict, f: int) -> np.ndarray:
        """Decode frame f of an episode's video -> (H, W, 3) RGB uint8."""
        vpath = str(ep["video_path"])
        cap = self._caps.get(vpath)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                raise IOError(f"Cannot open video {vpath}")
            self._caps[vpath] = cap

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f = max(0, min(f, n_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:  # seek hiccup: reopen once and retry
            cap.release()
            cap = cv2.VideoCapture(vpath)
            self._caps[vpath] = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                raise IOError(f"Failed to decode frame {f} of {vpath}")
        got = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if abs(got - f) > 1:
            logger.warning("Video seek landed on frame %d, wanted %d (%s)",
                           got, f, vpath)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame.shape[1], frame.shape[0]) != (self.image_size, self.image_size):
            frame = cv2.resize(frame, (self.image_size, self.image_size),
                               interpolation=cv2.INTER_AREA)
        return frame

    def __getitem__(self, t: int) -> dict:
        torch = require_torch()
        t = int(t)
        if not 0 <= t < len(self):
            raise IndexError(t)
        ei = int(np.searchsorted(self._bounds, t, side="right") - 1)
        ep = self._episodes[ei]
        sl = self._ep_frames[ei]
        f = t - ep["from"]  # local frame index within the episode

        state = np.stack(sl["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(sl["action"].to_numpy()).astype(np.float32)

        s = state[f]
        state_n = (s - self._state_mean) / self._state_scale

        # Action chunk with tail padding by repeating the last frame.
        idx = np.minimum(np.arange(f, f + self.chunk_size), len(sl) - 1)
        chunk = action[idx]
        chunk_n = (chunk - self._act_mean) / self._act_scale

        img = self._read_frame(ep, f).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        return {
            "state": torch.from_numpy(state_n),
            "images": torch.from_numpy(np.ascontiguousarray(img)),
            "action_chunk": torch.from_numpy(np.ascontiguousarray(chunk_n)),
            "episode_index": ei,
        }

    # -- helpers ------------------------------------------------------------
    def normalize_state(self, s: np.ndarray) -> np.ndarray:
        return (s - self._state_mean) / self._state_scale

    def unnormalize_action(self, chunk_n: np.ndarray) -> np.ndarray:
        """Back to raw action units (for sanity checks / rollouts)."""
        return chunk_n * self._act_scale + self._act_mean

    def close(self):
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
