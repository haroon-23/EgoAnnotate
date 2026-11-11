"""RLDS-style HDF5 export of an annotated episode.

This is NOT an RLDS / TFDS dataset. It is a single HDF5 file whose group layout
follows the RLDS episode/step convention (``<episode_id>/steps/{observation,
action,is_first,is_last,is_terminal}``), so RLDS-minded tooling can be adapted
to it, but it has no TFDS builder, no ``dataset_info.json`` and no TFRecords.
Use the LeRobot export (``src/lerobot_exporter.py``) for training; use
``tfds``-based pipelines only after writing a converter.

Layout (N = number of annotated frames; nothing is trimmed)::

    <episode_id>/steps/observation/image            (N,H,W,3) uint8, RGB [optional]
    <episode_id>/steps/observation/image_valid      (N,) bool
    <episode_id>/steps/observation/state            (N,52) float32  human state
    <episode_id>/steps/observation/hands/<side>/{valid,wrist_position,rotation_6d,
                                                  keypoints,aperture_m}
    <episode_id>/steps/observation/robot_joint_angles (N,nq) NaN where unknown
    <episode_id>/steps/observation/robot_gripper_opening_m (N,) NaN where unknown
    <episode_id>/steps/observation/robot_valid      (N,) bool (robot_reachable & data)
    <episode_id>/steps/observation/language_instruction (N,) utf-8
    <episode_id>/steps/action                       (N,52) float32  next-step state
    <episode_id>/steps/{is_first,is_last,is_terminal}
    <episode_id>/metadata/...

Missing data is never fabricated: a frame whose image cannot be read makes the
export fail (``allow_missing_images=True`` records it in ``image_valid``
instead), and frames without hand or robot data are flagged invalid, with NaN
(never zero) in the robot arrays.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import h5py
import numpy as np

from .datatypes import AnnotatedEpisode
from .lerobot_exporter import (
    build_human_arrays, human_dimension_names, resolve_image_path, rot6d_from_matrix,
    _tracked, _world_transform,
)

logger = logging.getLogger(__name__)

FORMAT_NAME = "rlds_style_hdf5"
FORMAT_NOTE = ("RLDS-style HDF5 (RLDS group layout, custom file). Not RLDS/TFDS: no TFRecords, "
               "no tfds builder.")


def export_episode_to_rlds_hdf5(
    episode: AnnotatedEpisode,
    output_path: Path | str,
    *,
    include_images: bool = True,
    image_size: Optional[tuple] = (224, 224),
    allow_missing_images: bool = False,
    human_frame: str = "auto",
    image_search_dirs: Sequence[Path | str] = (),
) -> Path:
    """Write the RLDS-style HDF5 for ``episode`` (atomic replace). ``image_size`` is (w, h) or None."""
    frames = episode.frames
    n = len(frames)
    if n == 0:
        raise ValueError(f"No frames found in episode {episode.episode_id}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    search = [Path(d) for d in image_search_dirs]

    sides = ("left", "right")
    human = None
    try:
        human = build_human_arrays(frames, human_frame)
    except ValueError as exc:
        logger.warning("RLDS HDF5 for %s has no metric hand state: %s", episode.episode_id, exc)
    use_world = human is not None and human.frame == "world"

    hands = {s: {"valid": np.zeros(n, bool), "wrist_position": np.full((n, 3), np.nan, np.float32),
                 "rotation_6d": np.full((n, 6), np.nan, np.float32),
                 "keypoints": np.full((n, 21, 3), np.nan, np.float32),
                 "aperture_m": np.full(n, np.nan, np.float32)} for s in sides}
    for t, fr in enumerate(frames):
        T = _world_transform(fr) if use_world else None
        for s in sides:
            pose = getattr(fr, f"{s}_hand_metric")
            if not _tracked(pose) or (use_world and T is None):
                continue
            R = np.asarray(pose.wrist_rotation, dtype=np.float64)
            p = np.asarray(pose.wrist_position_m, dtype=np.float64)
            kp = np.asarray(pose.keypoints_m, dtype=np.float64)
            if use_world:
                p, R, kp = T[:3, :3] @ p + T[:3, 3], T[:3, :3] @ R, kp @ T[:3, :3].T + T[:3, 3]
            h = hands[s]
            h["valid"][t] = True
            h["wrist_position"][t] = p
            h["rotation_6d"][t] = rot6d_from_matrix(R)
            h["keypoints"][t] = kp
            h["aperture_m"][t] = pose.aperture_m()

    images = image_valid = None
    if include_images:
        w, h = image_size if image_size else (None, None)
        imgs: List[Optional[np.ndarray]] = []
        missing: List[str] = []
        for fr in frames:
            path = resolve_image_path(fr.image_path, search)
            bgr = cv2.imread(str(path)) if path else None
            if bgr is None:
                missing.append(fr.image_path)
                imgs.append(None)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if image_size and rgb.shape[:2] != (h, w):
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            imgs.append(rgb)
        if missing and not allow_missing_images:
            raise FileNotFoundError(
                f"{len(missing)} of {n} frame images are unreadable (first: {missing[:3]}); "
                "refusing to write placeholder images (allow_missing_images=True records absence)")
        shapes = {i.shape for i in imgs if i is not None}
        if len(shapes) > 1:
            raise ValueError(f"frames have differing sizes {sorted(shapes)}; pass image_size")
        image_valid = np.array([i is not None for i in imgs], dtype=bool)
        shape = next(iter(shapes)) if shapes else (0, 0, 3)
        images = np.zeros((n, *shape), dtype=np.uint8)
        for i, img in enumerate(imgs):
            if img is not None:
                images[i] = img

    nq = len(episode.robot_joint_names)
    has_robot = nq > 0 and any(f.robot_joint_angles is not None for f in frames)
    if has_robot:
        q = np.full((n, nq), np.nan, np.float32)
        grip = np.full(n, np.nan, np.float32)
        robot_valid = np.zeros(n, bool)
        for i, fr in enumerate(frames):
            if fr.robot_joint_angles is not None:
                arr = np.asarray(fr.robot_joint_angles, dtype=np.float64)
                if arr.shape != (nq,):
                    raise ValueError(f"frame {fr.frame_idx}: {arr.size} joints, expected {nq}")
                q[i] = arr
            if fr.robot_gripper_opening_m is not None:
                grip[i] = fr.robot_gripper_opening_m
            robot_valid[i] = bool(fr.robot_reachable is True and np.all(np.isfinite(q[i])))

    utf8 = h5py.string_dtype(encoding="utf-8")
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), prefix=f".{out.name}.", suffix=".tmp")
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["format"] = FORMAT_NAME
            f.attrs["format_note"] = FORMAT_NOTE
            f.attrs["rlds_compliant"] = False
            ep = f.create_group(episode.episode_id)
            steps = ep.create_group("steps")
            obs = steps.create_group("observation")
            if images is not None:
                obs.create_dataset("image", data=images, compression="gzip", compression_opts=4)
                obs.create_dataset("image_valid", data=image_valid)
            if human is not None:
                st = obs.create_dataset("state", data=human.state)
                st.attrs["names"] = np.array(human_dimension_names(), dtype=object).astype(utf8)
                st.attrs["frame"] = human.frame
                steps.create_dataset("action", data=human.action)
            elif has_robot:
                nxt = np.concatenate([q, grip[:, None]], axis=1)
                nxt = np.concatenate([nxt[1:], nxt[-1:]], axis=0)
                ac = steps.create_dataset("action", data=nxt)
                ac.attrs["note"] = "robot next-step [joints, gripper_m]; NaN where unknown"
            else:
                raise ValueError(
                    f"episode {episode.episode_id} has neither metric hand poses nor robot data")
            for s in sides:
                g = obs.create_group(f"hands/{s}")
                for k, v in hands[s].items():
                    g.create_dataset(k, data=v)
                g.attrs["frame"] = "world" if use_world else "camera"
            if has_robot:
                ds = obs.create_dataset("robot_joint_angles", data=q)
                ds.attrs["joint_names"] = np.array(list(episode.robot_joint_names), dtype=object).astype(utf8)
                obs.create_dataset("robot_gripper_opening_m", data=grip)
                obs.create_dataset("robot_valid", data=robot_valid)
            lang = obs.create_dataset("language_instruction", (n,), dtype=utf8)
            for i, fr in enumerate(frames):
                lang[i] = fr.frame_description or episode.task_description
            is_first = np.zeros(n, bool)
            is_first[0] = True
            is_last = np.zeros(n, bool)
            is_last[-1] = True
            steps.create_dataset("is_first", data=is_first)
            steps.create_dataset("is_last", data=is_last)
            steps.create_dataset("is_terminal", data=np.zeros(n, bool))
            meta = ep.create_group("metadata")
            meta.create_dataset("task_description", data=episode.task_description, dtype=utf8)
            meta.create_dataset("num_frames", data=n, dtype=np.int32)
            meta.create_dataset("duration_seconds", data=float(episode.duration_seconds), dtype=np.float64)
            meta.create_dataset("effective_fps", data=float(episode.effective_fps or np.nan), dtype=np.float64)
            meta.create_dataset("source_frame_idx", data=np.array([x.frame_idx for x in frames], np.int64))
            meta.create_dataset("source_timestamp_s", data=np.array([x.timestamp for x in frames], np.float64))
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return out


def export_to_rlds(episode_id: str, output_dir: str, **kwargs) -> Path:
    """Export ``<output_dir>/<episode_id>/annotations.json`` to ``episode_rlds.hdf5``."""
    episode_dir = Path(output_dir) / episode_id
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")
    ann = episode_dir / "annotations.json"
    if not ann.exists():
        raise FileNotFoundError(f"annotations.json not found: {ann}")
    episode = AnnotatedEpisode.load_json(ann)
    kwargs.setdefault("image_search_dirs", [episode_dir])
    return export_episode_to_rlds_hdf5(episode, episode_dir / "episode_rlds.hdf5", **kwargs)


def export_all_episodes(output_dir: str = "data/output") -> List[Path]:
    """Export every episode directory that has ``annotations.json``."""
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning("Output directory %s does not exist. Nothing to export.", output_dir)
        return []
    exported = []
    for path in sorted(output_path.iterdir()):
        if path.is_dir() and (path / "annotations.json").exists():
            try:
                rlds_path = export_to_rlds(path.name, str(output_path))
                exported.append(rlds_path)
                print(f"Successfully exported RLDS-style HDF5 for episode: {path.name} -> {rlds_path.name}")
            except Exception as e:
                logger.error("Failed to export episode %s: %s", path.name, e, exc_info=True)
                print(f"Error exporting RLDS-style HDF5 for episode {path.name}: {e}")
    return exported
