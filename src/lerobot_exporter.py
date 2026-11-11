"""LeRobot v3.0 exporter for annotated egocentric episodes.

Input is an :class:`~src.datatypes.AnnotatedEpisode` (``annotations.json``).
One export per episode, written to ``<episode_dir>/lerobot_v3/``.

Modes (docs/ARCHITECTURE.md section 7):

* ``robot``: ``observation.state[t]`` = joint positions (+ gripper opening in
  metres) at ``t``; ``action[t]`` = the same quantities at ``t+1`` (the last row
  repeats). Only the longest contiguous run of valid frames
  (``robot_reachable`` true, joints present) is exported and the trim is
  recorded. Refused when ``retargeting.report.gate.passed`` is false unless
  ``allow_ungated=True``; then ``gate_passed: false`` is written to info.json.
* ``human``: per hand (left, right) wrist position (3), rotation 6D (6),
  thumb-index aperture in metres (1), five fingertip positions in the hand
  frame (15) and a validity flag (1); positions/rotations are in the world frame
  when ``T_world_cam`` is available (else the camera frame; info.json says
  which). ``action[t]`` = ``state[t+1]``. Untracked hands are zeros with
  validity 0.

Both modes: float32, no normalisation or clipping (``meta/stats.json`` carries
statistics for training-time normalisation), named dimensions, and a video
encoded with system ffmpeg (h264, yuv420p, short GOP) from exactly the frames
that have rows, one image per row, at the episode's effective fps. There is no
fallback encoder: without ffmpeg the export fails with a clear message.

Layout (what LeRobot 0.4.4 itself writes)::

    meta/info.json  meta/stats.json  meta/tasks.parquet (indexed by task text)
    meta/episodes/chunk-000/file-000.parquet  meta/modality.json (GR00T)
    data/chunk-000/file-000.parquet
    videos/observation.images.<camera>/chunk-000/file-000.mp4
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .datatypes import AnnotatedEpisode, AnnotationFrame

logger = logging.getLogger(__name__)

EXPORT_VERSION = 2
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINGERTIP_IDX = (4, 8, 12, 16, 20)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
HAND_NAMES = ("left", "right")
PER_HAND_DIM = 3 + 6 + 1 + 15 + 1
HUMAN_DIM = 2 * PER_HAND_DIM

GRIPPER_NAME = "gripper_opening_m"
ANNOTATION_COLUMN = "annotation.human.task_description"

ACTION_CONVENTION = {
    "human": "next_step_hand_state",
    "robot": "next_step_joint_targets",
}
PLACEHOLDER_TASKS = frozenset({
    "", "partial_run_unlabeled_task", "manipulating object", "todo",
    "placeholder", "canned_default", "unknown", "none", "n/a",
})

VIDEO_CODEC = "h264"
VIDEO_PIX_FMT = "yuv420p"
VIDEO_GOP = 2
VIDEO_CRF = 18
IMAGE_STATS_SAMPLES = 100
IMAGE_STATS_MAX_SIDE = 150

STAT_QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


class ExportRefusedError(ValueError):
    """The export was refused because the data would not be trustworthy."""


class MissingRobotRetargetingError(ExportRefusedError):
    """Robot mode requested but the episode has no usable retargeted frames."""


class FfmpegUnavailableError(RuntimeError):
    """System ffmpeg is required to encode the dataset video."""


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_feature_stats(arrays: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """Per-feature ``min/max/mean/std/count/q01/q10/q50/q90/q99`` over rows.

    Values are per-dimension lists; ``count`` is ``[n]`` as LeRobot writes it.
    Quantiles are pooled over all rows, never averaged across episodes.
    """
    stats: Dict[str, dict] = {}
    for key, arr in arrays.items():
        a = np.asarray(arr, dtype=np.float64)
        flat = a.reshape(a.shape[0], -1)
        entry = {
            "min": flat.min(axis=0).tolist(),
            "max": flat.max(axis=0).tolist(),
            "mean": flat.mean(axis=0).tolist(),
            "std": flat.std(axis=0).tolist(),
            "count": [int(a.shape[0])],
        }
        for name, q in STAT_QUANTILES.items():
            entry[name] = np.quantile(flat, q, axis=0).tolist()
        stats[key] = entry
    return stats


def _count_of(entry: dict) -> int:
    c = entry["count"]
    return int(c[0] if isinstance(c, (list, tuple, np.ndarray)) else c)


def _image_stats(samples: np.ndarray, n_frames: int) -> dict:
    """LeRobot image stats: per-channel over (N,H,W), RGB in [0,1], shape (3,1,1)."""
    a = samples.astype(np.float64)  # (n,3,h,w)
    axes = (0, 2, 3)
    out = {
        "min": a.min(axis=axes, keepdims=True)[0].tolist(),
        "max": a.max(axis=axes, keepdims=True)[0].tolist(),
        "mean": a.mean(axis=axes, keepdims=True)[0].tolist(),
        "std": a.std(axis=axes, keepdims=True)[0].tolist(),
        "count": [int(n_frames)],
    }
    ch = a.transpose(1, 0, 2, 3).reshape(3, -1)
    for name, q in STAT_QUANTILES.items():
        out[name] = np.quantile(ch, q, axis=1).reshape(3, 1, 1).tolist()
    return out


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

def rot6d_from_matrix(R: np.ndarray) -> np.ndarray:
    """First two columns of ``R``, concatenated: [R[:,0], R[:,1]] (Zhou et al.)."""
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[:, 0], R[:, 1]])


# ---------------------------------------------------------------------------
# Human features
# ---------------------------------------------------------------------------

def human_dimension_names() -> List[str]:
    names: List[str] = []
    for hand in HAND_NAMES:
        names += [f"{hand}_wrist_{a}" for a in "xyz"]
        names += [f"{hand}_rot6d_{i}" for i in range(6)]
        names.append(f"{hand}_aperture_m")
        for finger in FINGER_NAMES:
            names += [f"{hand}_{finger}_tip_{a}" for a in "xyz"]
        names.append(f"{hand}_valid")
    return names


def _finite_pose(pose) -> bool:
    return bool(
        np.all(np.isfinite(np.asarray(pose.wrist_position_m, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(pose.wrist_rotation, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(pose.keypoints_m, dtype=np.float64)))
    )


def _tracked(pose) -> bool:
    return pose is not None and not pose.is_interpolated and _finite_pose(pose)


def _world_transform(frame: AnnotationFrame) -> Optional[np.ndarray]:
    T = frame.T_world_cam
    if T is None:
        return None
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return None
    return T


@dataclass
class HumanArrays:
    state: np.ndarray            # (N, 52) float32
    action: np.ndarray           # (N, 52) float32
    frame: str                   # "world" | "camera"
    tracked: np.ndarray          # (N, 2) bool [left, right]
    frames_without_world: int    # tracked hands zeroed for lack of T_world_cam
    names: List[str] = field(default_factory=human_dimension_names)


def _next_step(state: np.ndarray) -> np.ndarray:
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]
    return action


def build_human_arrays(frames: Sequence[AnnotationFrame], human_frame: str = "auto") -> HumanArrays:
    """Section 7 human state/action. ``human_frame``: auto | world | camera."""
    if human_frame not in ("auto", "world", "camera"):
        raise ValueError(f"human_frame must be auto|world|camera, got {human_frame!r}")
    n = len(frames)
    sides = ("left_hand_metric", "right_hand_metric")
    any_tracked = False
    any_world = False
    for fr in frames:
        T = _world_transform(fr)
        for side in sides:
            if _tracked(getattr(fr, side)):
                any_tracked = True
                any_world = any_world or T is not None
    if not any_tracked:
        raise ExportRefusedError(
            "human export needs metric hand poses (left/right_hand_metric, tracked and "
            "not interpolated) but no frame has one; run the metric hand-pose stage.")
    if human_frame == "world" and not any_world:
        raise ExportRefusedError("human_frame='world' requested but no frame has T_world_cam")
    use_world = any_world if human_frame == "auto" else human_frame == "world"

    state = np.zeros((n, HUMAN_DIM), dtype=np.float64)
    tracked = np.zeros((n, 2), dtype=bool)
    without_world = 0
    for t, fr in enumerate(frames):
        T = _world_transform(fr) if use_world else None
        for h, side in enumerate(sides):
            pose = getattr(fr, side)
            if not _tracked(pose):
                continue
            if use_world and T is None:
                without_world += 1
                continue
            R = np.asarray(pose.wrist_rotation, dtype=np.float64)
            p = np.asarray(pose.wrist_position_m, dtype=np.float64)
            kp = np.asarray(pose.keypoints_m, dtype=np.float64)
            tips_h = (kp[list(FINGERTIP_IDX)] - p) @ R  # R^T (k - p), in the hand frame
            if use_world:
                p = T[:3, :3] @ p + T[:3, 3]
                R = T[:3, :3] @ R
            vec = np.concatenate([p, rot6d_from_matrix(R), [pose.aperture_m()],
                                  tips_h.reshape(-1), [1.0]])
            state[t, h * PER_HAND_DIM:(h + 1) * PER_HAND_DIM] = vec
            tracked[t, h] = True
    state32 = state.astype(np.float32)
    return HumanArrays(state32, _next_step(state32), "world" if use_world else "camera",
                       tracked, without_world)


# ---------------------------------------------------------------------------
# Retargeting gate and robot features
# ---------------------------------------------------------------------------

def retargeting_gate(episode: AnnotatedEpisode) -> Tuple[Optional[bool], List[str]]:
    """``(passed, reasons)`` from ``retargeting.report.gate``; passed None if absent."""
    report = (episode.retargeting or {}).get("report")
    if not isinstance(report, dict):
        return None, []
    gate = report.get("gate")
    if not isinstance(gate, dict):
        nested = report.get(episode.target_robot)
        gate = nested.get("gate") if isinstance(nested, dict) else None
    if not isinstance(gate, dict) or "passed" not in gate:
        return None, []
    return bool(gate["passed"]), [str(r) for r in gate.get("reasons") or []]


def _joint_limits(episode: AnnotatedEpisode, names: List[str]) -> Optional[dict]:
    """``retargeting.joint_limits`` = {"names"?, "lower", "upper"} (per joint, rad/m)."""
    src = (episode.retargeting or {}).get("joint_limits")
    if not isinstance(src, dict) or "lower" not in src or "upper" not in src:
        return None
    lower = [float(x) for x in src["lower"]]
    upper = [float(x) for x in src["upper"]]
    lim_names = list(src.get("names") or names)
    if len(lower) != len(upper) or len(lower) != len(lim_names):
        raise ExportRefusedError("retargeting.joint_limits lower/upper/names lengths differ")
    return {"names": lim_names, "lower": lower, "upper": upper}


def longest_true_run(mask: np.ndarray) -> Tuple[int, int]:
    """``(start, end_exclusive)`` of the longest run of True; (0, 0) if none."""
    best = (0, 0)
    start = None
    for i, ok in enumerate(list(mask) + [False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


@dataclass
class RobotArrays:
    state: np.ndarray            # (n, D) float32, trimmed
    action: np.ndarray
    names: List[str]
    row_start: int               # index into episode.frames
    row_end: int                 # exclusive
    has_gripper: bool
    ee_pos_error_m: Optional[np.ndarray]


def build_robot_arrays(episode: AnnotatedEpisode) -> RobotArrays:
    """Section 7 robot state/action over the longest valid contiguous run."""
    frames = episode.frames
    joint_names = list(episode.robot_joint_names or [])
    if not joint_names:
        raise MissingRobotRetargetingError(
            "episode.robot_joint_names is empty; cannot name the robot dimensions.")
    nq = len(joint_names)
    n = len(frames)
    joints = np.full((n, nq), np.nan)
    gripper = np.full(n, np.nan)
    reachable = np.zeros(n, dtype=bool)
    for i, fr in enumerate(frames):
        if fr.robot_reachable is not True or fr.robot_joint_angles is None:
            continue
        q = np.asarray(fr.robot_joint_angles, dtype=np.float64)
        if q.shape != (nq,):
            raise ExportRefusedError(
                f"frame {fr.frame_idx}: {q.size} joint angles but "
                f"robot_joint_names has {nq} entries")
        if not np.all(np.isfinite(q)):
            continue
        joints[i] = q
        reachable[i] = True
        if fr.robot_gripper_opening_m is not None and np.isfinite(fr.robot_gripper_opening_m):
            gripper[i] = float(fr.robot_gripper_opening_m)
    if not reachable.any():
        raise MissingRobotRetargetingError(
            "robot export needs retargeted frames with robot_reachable=True and "
            "joint angles; this episode has none.")
    has_gripper = bool(np.isfinite(gripper[reachable]).any())
    valid = reachable & (np.isfinite(gripper) if has_gripper else True)
    start, end = longest_true_run(valid)
    if end <= start:
        raise MissingRobotRetargetingError("no valid robot frame with all required values")
    state = joints[start:end]
    names = joint_names
    if has_gripper:
        state = np.hstack([state, gripper[start:end, None]])
        names = joint_names + [GRIPPER_NAME]
    state = state.astype(np.float32)
    err = None
    errs = [frames[i].robot_ee_pos_error_m for i in range(start, end)]
    if all(e is not None and np.isfinite(e) for e in errs):
        err = np.asarray(errs, dtype=np.float32)
    return RobotArrays(state, _next_step(state), names, start, end, has_gripper, err)


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def find_ffmpeg() -> str:
    exe = os.environ.get("EGO_FFMPEG") or shutil.which("ffmpeg")
    if not exe:
        raise FfmpegUnavailableError(
            "ffmpeg was not found on PATH. The LeRobot export encodes its video with "
            "system ffmpeg (libx264) and has no fallback encoder. Install ffmpeg "
            "(apt install ffmpeg / brew install ffmpeg) or set EGO_FFMPEG to its path.")
    return exe


def count_decoded_frames(path: Path | str) -> int:
    """Number of frames actually decodable from a video (not the container header)."""
    path = str(path)
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        res = subprocess.run(
            [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=300)
        out = res.stdout.strip().split("\n")[0].strip().rstrip(",")
        if res.returncode == 0 and out.isdigit():
            return int(out)
    cap = cv2.VideoCapture(path)
    try:
        n = 0
        while cap.grab():
            n += 1
        return n
    finally:
        cap.release()


def probe_video_codec(path: Path | str) -> Optional[str]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    res = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,pix_fmt", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        return None
    try:
        s = json.loads(res.stdout)["streams"][0]
    except (KeyError, IndexError, ValueError):
        return None
    return str(s.get("codec_name"))


def resolve_image_path(raw: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    p = Path(raw)
    if p.is_absolute():
        return p if p.exists() else None
    for base in [Path.cwd(), PROJECT_ROOT, *search_dirs]:
        cand = base / p
        if cand.exists():
            return cand
    return None


def encode_frames_to_h264(image_paths: Sequence[Path], dst: Path, fps: int, *,
                          crf: int = VIDEO_CRF, gop: int = VIDEO_GOP) -> dict:
    """Encode exactly ``image_paths`` (one frame each) to h264/yuv420p.

    Returns ``{"width","height","codec","pix_fmt","n_frames","stats"}``; ``stats`` are
    LeRobot-style image stats. Odd dimensions are cropped by one pixel.
    """
    ffmpeg = find_ffmpeg()
    if not image_paths:
        raise ValueError("no frames to encode")
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"cannot read image {image_paths[0]}")
    h, w = first.shape[:2]
    n = len(image_paths)
    sample_at = set(np.unique(np.linspace(0, n - 1, min(n, IMAGE_STATS_SAMPLES)).round().astype(int)))
    scale = min(1.0, IMAGE_STATS_MAX_SIDE / max(h, w))
    samples: List[np.ndarray] = []

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-framerate", str(fps),
        "-i", "-", "-an",
        "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-bf", "0",
        "-g", str(gop), "-pix_fmt", VIDEO_PIX_FMT, "-r", str(fps),
        "-movflags", "+faststart", str(dst),
    ]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        try:
            for i, path in enumerate(image_paths):
                img = first if i == 0 else cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(f"cannot read image {path}")
                if img.shape[:2] != (h, w):
                    raise ExportRefusedError(
                        f"frame {i} ({path}) is {img.shape[1]}x{img.shape[0]} but the first "
                        f"frame is {w}x{h}; all frames of an episode must share one size")
                if i in sample_at:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if scale < 1.0:
                        rgb = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                                         interpolation=cv2.INTER_AREA)
                    samples.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
                proc.stdin.write(np.ascontiguousarray(img).tobytes())
            proc.stdin.close()
            rc = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        if rc != 0:
            err.seek(0)
            raise RuntimeError(f"ffmpeg failed (exit {rc}): {err.read().decode(errors='replace')[-800:]}")
    decoded = count_decoded_frames(dst)
    if decoded != n:
        raise RuntimeError(f"encoded video decodes to {decoded} frames, expected {n}")
    cap = cv2.VideoCapture(str(dst))
    try:
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return {
        "width": vw, "height": vh,
        "codec": probe_video_codec(dst) or VIDEO_CODEC,
        "pix_fmt": VIDEO_PIX_FMT, "n_frames": n,
        "stats": _image_stats(np.stack(samples), n),
    }


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

_ARROW = {"float32": pa.float32(), "float64": pa.float64(), "int64": pa.int64(),
          "int32": pa.int32(), "bool": pa.bool_()}


def frame_table(columns: Dict[str, np.ndarray], features: Dict[str, dict]) -> pa.Table:
    """Arrow table with fixed-size float32 lists for vector features (as LeRobot)."""
    arrays, fields = [], []
    for key, ft in features.items():
        if ft["dtype"] == "video" or key not in columns:
            continue
        typ = _ARROW[ft["dtype"]]
        col = np.asarray(columns[key])
        shape = tuple(ft["shape"])
        if shape == (1,):
            arr = pa.array(col.reshape(-1), type=typ)
        elif len(shape) == 1:
            flat = pa.array(col.reshape(-1), type=typ)
            arr = pa.FixedSizeListArray.from_arrays(flat, shape[0])
        else:
            raise ValueError(f"unsupported frame feature shape {shape} for {key}")
        arrays.append(arr)
        fields.append(pa.field(key, arr.type))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def write_tasks_parquet(path: Path, tasks: Sequence[str]) -> None:
    """Task text is the INDEX, ``task_index`` the column: that is how LeRobot reads it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"task_index": list(range(len(tasks)))}, index=list(tasks)).to_parquet(path)


def build_features(state_dim: int, state_names: List[str], action_names: List[str],
                   *, video_key: str, video_shape: Tuple[int, int, int], video_info: dict,
                   extra: Dict[str, dict]) -> Dict[str, dict]:
    feats: Dict[str, dict] = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": state_names},
        "action": {"dtype": "float32", "shape": [len(action_names)], "names": action_names},
        video_key: {"dtype": "video", "shape": list(video_shape),
                    "names": ["height", "width", "channels"], "info": video_info},
    }
    feats.update(extra)
    feats.update({
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    })
    return feats


# ---------------------------------------------------------------------------
# modality.json (NVIDIA Isaac GR00T)
# ---------------------------------------------------------------------------

def build_modality(mode: str, state_names: List[str], video_key: str, camera: str) -> dict:
    """``meta/modality.json`` in the schema of Isaac-GR00T getting_started/data_preparation.md."""
    if mode == "human":
        slices: Dict[str, Dict[str, int]] = {}
        for h, hand in enumerate(HAND_NAMES):
            o = h * PER_HAND_DIM
            for key, (a, b) in {
                f"{hand}_wrist_position": (0, 3), f"{hand}_rotation_6d": (3, 9),
                f"{hand}_aperture_m": (9, 10), f"{hand}_fingertips_hand_frame": (10, 25),
                f"{hand}_valid": (25, 26),
            }.items():
                slices[key] = {"start": o + a, "end": o + b}
        state = dict(slices)
        action = dict(slices)
    else:
        has_g = state_names[-1] == GRIPPER_NAME
        nj = len(state_names) - (1 if has_g else 0)
        state = {"joint_positions": {"start": 0, "end": nj}}
        action = {"joint_targets": {"start": 0, "end": nj}}
        if has_g:
            state["gripper_opening_m"] = {"start": nj, "end": nj + 1}
            action["gripper_opening_m"] = {"start": nj, "end": nj + 1}
    return {
        "state": state, "action": action,
        "video": {camera: {"original_key": video_key}},
        "annotation": {"human.task_description": {}},
    }


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def _determine_fps(episode: AnnotatedEpisode, frames: Sequence[AnnotationFrame],
                   override: Optional[float]) -> Tuple[float, str]:
    if override:
        return float(override), "override"
    if episode.effective_fps:
        return float(episode.effective_fps), "episode.effective_fps"
    ts = np.array([f.timestamp for f in frames], dtype=np.float64)
    dt = np.diff(ts)
    dt = dt[dt > 0]
    if dt.size:
        return float(1.0 / np.median(dt)), "median_timestamp_delta"
    raise ExportRefusedError(
        "cannot determine fps: episode.effective_fps is unset and frame timestamps give no "
        "positive interval; pass fps=... explicitly")


def export_episode_to_lerobot(
    episode: AnnotatedEpisode,
    lerobot_dir: Path | str,
    *,
    export_mode: str = "human",
    camera: str = "ego",
    fps: Optional[float] = None,
    allow_ungated: bool = False,
    human_frame: str = "auto",
    image_search_dirs: Sequence[Path | str] = (),
    target_embodiment: Optional[str] = None,
) -> Path:
    """Write one episode as a LeRobot v3.0 dataset at ``lerobot_dir`` (replaced atomically)."""
    if export_mode not in ("human", "robot"):
        raise ValueError(f"export_mode must be 'human' or 'robot', got {export_mode!r}")
    if not episode.frames:
        raise ExportRefusedError(f"episode {episode.episode_id} has no frames")
    lerobot_dir = Path(lerobot_dir)
    search = [Path(d) for d in image_search_dirs]
    frames = episode.frames
    n_source = len(frames)

    gate_passed, gate_reasons = retargeting_gate(episode)
    extra_cols: Dict[str, np.ndarray] = {}
    extra_feats: Dict[str, dict] = {}
    info_extra: Dict[str, Any] = {}

    if export_mode == "robot":
        if gate_passed is False and not allow_ungated:
            raise ExportRefusedError(
                f"retargeting gate failed for {episode.episode_id} "
                f"({'; '.join(gate_reasons) or 'no reasons recorded'}); robot export refused. "
                "Pass allow_ungated=True (--allow-ungated) to export anyway; "
                "info.json will then carry gate_passed=false.")
        robot = build_robot_arrays(episode)
        state, action, state_names = robot.state, robot.action, robot.names
        row_start, row_end = robot.row_start, robot.row_end
        state_frame = "joint"
        robot_type = episode.target_robot or target_embodiment or "unknown_robot"
        limits = _joint_limits(episode, list(episode.robot_joint_names))
        info_extra["robot"] = {
            "embodiment": robot_type, "joint_names": list(episode.robot_joint_names),
            "has_gripper": robot.has_gripper,
            "gripper_name": GRIPPER_NAME if robot.has_gripper else None,
            "joint_limits": limits,
        }
        if robot.ee_pos_error_m is not None:
            extra_cols["robot_ee_pos_error_m"] = robot.ee_pos_error_m
            extra_feats["robot_ee_pos_error_m"] = {
                "dtype": "float32", "shape": [1], "names": None}
        extra_info_trim_reason = "longest contiguous run of robot_reachable frames"
    else:
        human = build_human_arrays(frames, human_frame)
        state, action, state_names = human.state, human.action, human.names
        row_start, row_end = 0, n_source
        state_frame = human.frame
        robot_type = "human_egocentric"
        info_extra["human"] = {
            "frame": human.frame,
            "frame_note": ("camera-to-world transforms T_world_cam were applied"
                           if human.frame == "world" else
                           "T_world_cam absent: positions and rotations are in the camera frame"),
            "rotation_6d": "first two columns of the hand rotation matrix, [col0, col1]",
            "tracked_frames": {"left": int(human.tracked[:, 0].sum()),
                               "right": int(human.tracked[:, 1].sum())},
            "tracked_hands_zeroed_without_world_transform": human.frames_without_world,
        }
        extra_info_trim_reason = None

    rows = frames[row_start:row_end]
    n = len(rows)
    assert state.shape[0] == n and action.shape == state.shape

    fps_exact, fps_source = _determine_fps(episode, rows, fps)
    fps_int = max(1, int(round(fps_exact)))
    rounding_err_pct = abs(fps_int - fps_exact) / fps_exact * 100.0

    task = str(episode.task_description or "").strip()
    trim = {
        "applied": bool(n != n_source),
        "reason": extra_info_trim_reason,
        "n_source_rows": n_source,
        "n_exported": n,
        "n_dropped": n_source - n,
        "n_dropped_leading": row_start,
        "n_dropped_trailing": n_source - row_end,
        "source_row_start": row_start,
        "source_row_end_inclusive": row_end - 1,
        "source_frame_idx_start": int(rows[0].frame_idx),
        "source_frame_idx_end": int(rows[-1].frame_idx),
    }

    # Everything is built in a sibling directory and swapped in at the end.
    tmp_dir = lerobot_dir.parent / f".{lerobot_dir.name}.tmp-{os.getpid()}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        video_key = f"observation.images.{camera}"
        image_paths: List[Path] = []
        missing: List[str] = []
        for fr in rows:
            p = resolve_image_path(fr.image_path, search)
            if p is None:
                missing.append(fr.image_path)
            else:
                image_paths.append(p)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {n} frame images not found (first: {missing[:3]}); the "
                "video is encoded from the sampled frames so every row needs its image")
        video_path = tmp_dir / VIDEO_PATH.format(video_key=video_key, chunk_index=0, file_index=0)
        enc = encode_frames_to_h264(image_paths, video_path, fps_int)
        video_info = {
            "video.height": enc["height"], "video.width": enc["width"],
            "video.codec": enc["codec"], "video.pix_fmt": enc["pix_fmt"],
            "video.is_depth_map": False, "video.fps": fps_int, "video.channels": 3,
            "has_audio": False,
        }

        frame_index = np.arange(n, dtype=np.int64)
        columns: Dict[str, np.ndarray] = {
            "observation.state": state, "action": action,
            ANNOTATION_COLUMN: np.zeros(n, dtype=np.int64),
            "source_frame_idx": np.array([f.frame_idx for f in rows], dtype=np.int64),
            "source_timestamp_s": np.array([f.timestamp for f in rows], dtype=np.float64),
            "timestamp": (frame_index / fps_int).astype(np.float32),
            "frame_index": frame_index,
            "episode_index": np.zeros(n, dtype=np.int64),
            "index": frame_index.copy(),
            "task_index": np.zeros(n, dtype=np.int64),
        }
        columns.update(extra_cols)
        extra = {
            ANNOTATION_COLUMN: {"dtype": "int64", "shape": [1], "names": None},
            "source_frame_idx": {"dtype": "int64", "shape": [1], "names": None},
            "source_timestamp_s": {"dtype": "float64", "shape": [1], "names": None},
            **extra_feats,
        }
        features = build_features(
            state.shape[1], state_names, state_names, video_key=video_key,
            video_shape=(enc["height"], enc["width"], 3), video_info=video_info, extra=extra)
        table = frame_table(columns, features)
        data_path = tmp_dir / DATA_PATH.format(chunk_index=0, file_index=0)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path)

        stats = compute_feature_stats({k: v for k, v in columns.items()})
        stats[video_key] = enc["stats"]
        meta = tmp_dir / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "stats.json").write_text(json.dumps(stats, indent=4))
        write_tasks_parquet(meta / "tasks.parquet", [task])
        modality = build_modality(export_mode, state_names, video_key, camera)
        (meta / "modality.json").write_text(json.dumps(modality, indent=4))

        gate_block = {
            "gate_passed": gate_passed, "gate_reasons": gate_reasons,
            "allow_ungated": bool(allow_ungated),
        }
        info = {
            "codebase_version": "v3.0", "robot_type": robot_type,
            "total_episodes": 1, "total_frames": n, "total_tasks": 1,
            "total_videos": 1, "total_chunks": 1, "chunks_size": 1000,
            "data_files_size_in_mb": 100, "video_files_size_in_mb": 200,
            "fps": fps_int, "splits": {"train": "0:1"},
            "data_path": DATA_PATH, "video_path": VIDEO_PATH, "features": features,
            "export_version": EXPORT_VERSION, "export_mode": export_mode,
            "action_convention": ACTION_CONVENTION[export_mode],
            "state_frame": state_frame,
            "fps_exact": fps_exact, "fps_source": fps_source,
            "fps_rounding_error_pct": round(rounding_err_pct, 4),
            "source_episode_id": episode.episode_id,
            "task_description": task,
            "geometry": episode.geometry or {},
            "trim": trim,
            **gate_block,
            **info_extra,
        }
        if gate_passed is False:
            info["WARNING"] = "retargeting gate FAILED; exported with allow_ungated"
        (meta / "info.json").write_text(json.dumps(info, indent=4))

        ep_row: Dict[str, Any] = {
            "episode_index": [0], "tasks": [[task]], "length": [n],
            "data/chunk_index": [0], "data/file_index": [0],
            "dataset_from_index": [0], "dataset_to_index": [n],
            "meta/episodes/chunk_index": [0], "meta/episodes/file_index": [0],
            f"videos/{video_key}/chunk_index": [0], f"videos/{video_key}/file_index": [0],
            f"videos/{video_key}/from_timestamp": [0.0],
            f"videos/{video_key}/to_timestamp": [n / fps_int],
            "export_mode": [export_mode], "action_convention": [ACTION_CONVENTION[export_mode]],
            "robot_type": [robot_type], "state_frame": [state_frame],
            "fps": [fps_int], "fps_source": [fps_source],
            "gate_status": ["unknown" if gate_passed is None
                            else ("passed" if gate_passed else "failed")],
            "source/episode_id": [episode.episode_id],
        }
        for k, v in trim.items():
            if k != "reason":
                ep_row[f"trim/{k}"] = [v]
        for feat, s in stats.items():
            for agg, val in s.items():
                ep_row[f"stats/{feat}/{agg}"] = [val]
        ep_path = meta / "episodes" / "chunk-000" / "file-000.parquet"
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ep_row).to_parquet(ep_path, index=False)

        if lerobot_dir.exists():
            shutil.rmtree(lerobot_dir)
        os.replace(tmp_dir, lerobot_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    logger.info("LeRobot v3 export of %s (%s): %d rows (%d dropped) -> %s",
                episode.episode_id, export_mode, n, n_source - n, lerobot_dir)
    return lerobot_dir


def export_to_lerobot(
    episode_id: str,
    output_dir: str,
    *,
    export_mode: str = "human",
    camera: str = "ego",
    fps: Optional[float] = None,
    target_embodiment: Optional[str] = None,
    allow_ungated: bool = False,
    human_frame: str = "auto",
) -> Path:
    """Export ``<output_dir>/<episode_id>/annotations.json`` to ``.../lerobot_v3``."""
    episode_dir = Path(output_dir) / episode_id
    ann = episode_dir / "annotations.json"
    if not ann.exists():
        raise FileNotFoundError(f"annotations.json not found: {ann}")
    episode = AnnotatedEpisode.load_json(ann)
    return export_episode_to_lerobot(
        episode, episode_dir / "lerobot_v3", export_mode=export_mode, camera=camera,
        fps=fps, allow_ungated=allow_ungated, human_frame=human_frame,
        image_search_dirs=[episode_dir], target_embodiment=target_embodiment)


def export_all_lerobot(output_dir: str = "data/output", *, export_mode: str = "human",
                       **kwargs) -> List[Path]:
    """Export every episode with ``annotations.json``; refused episodes are skipped and logged."""
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning("Output directory %s does not exist. Nothing to export.", output_dir)
        return []
    exported: List[Path] = []
    for path in sorted(output_path.iterdir()):
        if not (path.is_dir() and (path / "annotations.json").exists()):
            continue
        try:
            out = export_to_lerobot(path.name, str(output_path), export_mode=export_mode, **kwargs)
            exported.append(out)
            print(f"Exported LeRobot for episode: {path.name} -> {out.relative_to(output_path)}")
        except ExportRefusedError as e:
            logger.warning("Skipping episode %s (%s mode): %s", path.name, export_mode, e)
            print(f"Skipped episode {path.name} ({export_mode} mode): {e}")
        except FfmpegUnavailableError:
            raise
        except Exception as e:
            logger.error("Failed to export episode %s to LeRobot: %s", path.name, e, exc_info=True)
            print(f"Error exporting LeRobot for episode {path.name}: {e}")
    return exported
