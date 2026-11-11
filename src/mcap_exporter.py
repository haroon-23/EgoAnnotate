"""MCAP export of an annotated episode, written with the official ``mcap`` package.

The file is chunked (zstd), message-indexed and has a summary section, so
Foxglove and the ``mcap`` CLI can seek and list channels. Channels use JSON
messages with JSON-schema schemas; where Foxglove defines a schema we use it
(``foxglove.FrameTransform``, ``foxglove.CompressedImage``).

======================  =============================  ==========================
topic                   schema                         one message per
======================  =============================  ==========================
``/episode/metadata``   ``ego.EpisodeMetadata``        episode
``/annotations/frame``  ``ego.FrameAnnotation``        annotated frame (row)
``/tf``                 ``foxglove.FrameTransform``    world->camera, camera->wrist
``/hands/{left,right}`` ``ego.HandPose``               frame with a metric pose
``/robot/joint_states`` ``ego.JointState``             frame (valid flag + names)
``/camera/image/compressed`` ``foxglove.CompressedImage``  frame (opt-in, JPEG)
======================  =============================  ==========================

Message ``log_time`` is the source-video timestamp in nanoseconds.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer

from .datatypes import AnnotatedEpisode, AnnotationFrame

MCAP_MAGIC = b"\x89MCAP0\r\n"
FRAME_TOPIC = "/annotations/frame"
REQUIRED_TOPICS = ("/episode/metadata", FRAME_TOPIC)

_TIME = {
    "type": "object", "title": "time",
    "properties": {"sec": {"type": "integer", "minimum": 0},
                   "nsec": {"type": "integer", "minimum": 0, "maximum": 999999999}},
}
_VEC3 = {"type": "object", "properties": {k: {"type": "number"} for k in "xyz"},
         "required": ["x", "y", "z"]}
_QUAT = {"type": "object", "properties": {k: {"type": "number"} for k in "xyzw"},
         "required": ["x", "y", "z", "w"]}
_NUM_ARRAY = {"type": "array", "items": {"type": "number"}}

SCHEMAS: Dict[str, dict] = {
    "foxglove.FrameTransform": {
        "title": "foxglove.FrameTransform", "type": "object",
        "properties": {
            "timestamp": _TIME, "parent_frame_id": {"type": "string"},
            "child_frame_id": {"type": "string"},
            "translation": {"title": "foxglove.Vector3", **_VEC3},
            "rotation": {"title": "foxglove.Quaternion", **_QUAT},
        },
        "required": ["timestamp", "parent_frame_id", "child_frame_id", "translation", "rotation"],
    },
    "foxglove.CompressedImage": {
        "title": "foxglove.CompressedImage", "type": "object",
        "properties": {
            "timestamp": _TIME, "frame_id": {"type": "string"},
            "data": {"type": "string", "contentEncoding": "base64"},
            "format": {"type": "string"},
        },
        "required": ["timestamp", "frame_id", "data", "format"],
    },
    "ego.EpisodeMetadata": {
        "title": "ego.EpisodeMetadata", "type": "object",
        "properties": {
            "episode_id": {"type": "string"}, "task_description": {"type": "string"},
            "num_frames": {"type": "integer"}, "duration_seconds": {"type": "number"},
            "effective_fps": {"type": ["number", "null"]}, "target_robot": {"type": "string"},
            "robot_joint_names": {"type": "array", "items": {"type": "string"}},
            "geometry": {"type": "object"}, "retargeting_gate": {"type": ["object", "null"]},
            "segments": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["episode_id", "task_description", "num_frames"],
    },
    "ego.FrameAnnotation": {
        "title": "ego.FrameAnnotation", "type": "object",
        "properties": {
            "frame_idx": {"type": "integer"}, "timestamp": {"type": "number"},
            "action_segment": {"type": "string"}, "frame_description": {"type": "string"},
            "left_contact": {"type": ["object", "null"]}, "right_contact": {"type": ["object", "null"]},
            "left_grasp": {"type": ["object", "null"]}, "right_grasp": {"type": ["object", "null"]},
            "action": {"type": ["object", "null"]},
            "objects": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["frame_idx", "timestamp"],
    },
    "ego.HandPose": {
        "title": "ego.HandPose", "type": "object",
        "properties": {
            "timestamp": _TIME, "frame_idx": {"type": "integer"},
            "frame_id": {"type": "string", "description": "camera optical frame"},
            "hand": {"type": "string"}, "is_interpolated": {"type": "boolean"},
            "wrist_position_m": _NUM_ARRAY,
            "wrist_rotation": {"type": "array", "items": _NUM_ARRAY},
            "keypoints_m": {"type": "array", "items": _NUM_ARRAY},
            "aperture_m": {"type": "number"}, "reprojection_error_px": {"type": "number"},
            "intrinsics_source": {"type": "string"},
        },
        "required": ["timestamp", "frame_idx", "hand", "wrist_position_m", "keypoints_m"],
    },
    "ego.JointState": {
        "title": "ego.JointState", "type": "object",
        "properties": {
            "timestamp": _TIME, "frame_idx": {"type": "integer"},
            "name": {"type": "array", "items": {"type": "string"}},
            "position": {**_NUM_ARRAY, "description": "joint positions; empty when no data"},
            "gripper_opening_m": {"type": ["number", "null"]},
            "valid": {"type": "boolean", "description": "robot_reachable for this frame"},
            "ee_pos_error_m": {"type": ["number", "null"]},
        },
        "required": ["timestamp", "frame_idx", "name", "position", "valid"],
    },
}


def _stamp(ts: float) -> Dict[str, int]:
    ns = int(round(float(ts) * 1e9))
    return {"sec": ns // 1_000_000_000, "nsec": ns % 1_000_000_000}


def quaternion_xyzw(R: np.ndarray) -> Dict[str, float]:
    """Unit quaternion of a rotation matrix (child-to-parent), Foxglove x,y,z,w order."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])}


def _transform(ts: float, parent: str, child: str, R: np.ndarray, t: np.ndarray) -> dict:
    t = np.asarray(t, dtype=np.float64)
    return {
        "timestamp": _stamp(ts), "parent_frame_id": parent, "child_frame_id": child,
        "translation": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
        "rotation": quaternion_xyzw(R),
    }


def _finite_array(a: Any) -> Optional[np.ndarray]:
    arr = np.asarray(a, dtype=np.float64)
    return arr if np.all(np.isfinite(arr)) else None


def _json(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False).encode("utf-8")


def export_episode_to_mcap(
    episode: AnnotatedEpisode,
    output_path: Path | str,
    *,
    include_images: bool = False,
    image_search_dirs: Sequence[Path | str] = (),
    chunk_size: int = 1 << 20,
) -> Path:
    """Write ``episode`` to a chunked, indexed MCAP file; atomically replaces ``output_path``."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(out_file.parent), prefix=f".{out_file.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            writer = Writer(f, chunk_size=chunk_size, compression=CompressionType.ZSTD)
            writer.start(profile="", library="ego_annotate_mcap_exporter")
            sid = {name: writer.register_schema(
                name=name, encoding="jsonschema", data=json.dumps(schema).encode("utf-8"))
                for name, schema in SCHEMAS.items() if include_images or name != "foxglove.CompressedImage"}

            def chan(topic: str, schema: str, **meta: str) -> int:
                return writer.register_channel(
                    topic=topic, message_encoding="json", schema_id=sid[schema], metadata=meta)

            c_meta = chan("/episode/metadata", "ego.EpisodeMetadata")
            c_frame = chan(FRAME_TOPIC, "ego.FrameAnnotation")
            c_tf = chan("/tf", "foxglove.FrameTransform")
            c_hand = {s: chan(f"/hands/{s}", "ego.HandPose", frame="camera") for s in ("left", "right")}
            c_joint = chan("/robot/joint_states", "ego.JointState", target_robot=str(episode.target_robot))
            c_img = chan("/camera/image/compressed", "foxglove.CompressedImage") if include_images else None

            gate = (episode.retargeting or {}).get("report", {})
            gate = gate.get("gate") if isinstance(gate, dict) else None
            t0 = int(round(float(episode.frames[0].timestamp) * 1e9)) if episode.frames else 0
            writer.add_message(c_meta, t0, _json({
                "episode_id": episode.episode_id, "task_description": episode.task_description,
                "num_frames": len(episode.frames), "duration_seconds": float(episode.duration_seconds),
                "effective_fps": episode.effective_fps, "target_robot": episode.target_robot,
                "robot_joint_names": list(episode.robot_joint_names),
                "geometry": episode.geometry or {}, "retargeting_gate": gate,
                "segments": [s.to_dict() for s in episode.segments],
            }), t0)
            writer.add_metadata("episode", {
                "episode_id": str(episode.episode_id), "task_description": str(episode.task_description),
                "target_robot": str(episode.target_robot)})

            search = [Path(d) for d in image_search_dirs]
            for seq, fr in enumerate(episode.frames):
                _write_frame(writer, fr, seq, c_frame, c_tf, c_hand, c_joint, c_img, episode, search)
            writer.finish()
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_file)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return out_file


def _write_frame(writer, fr: AnnotationFrame, seq: int, c_frame, c_tf, c_hand, c_joint, c_img,
                 episode: AnnotatedEpisode, search: List[Path]) -> None:
    ns = int(round(float(fr.timestamp) * 1e9))
    stamp = _stamp(fr.timestamp)

    def put(channel: int, payload: dict) -> None:
        writer.add_message(channel, ns, _json(payload), ns, seq)

    put(c_frame, {
        "frame_idx": fr.frame_idx, "timestamp": float(fr.timestamp),
        "action_segment": fr.action_segment, "frame_description": fr.frame_description,
        "left_contact": fr.left_contact.to_dict() if fr.left_contact else None,
        "right_contact": fr.right_contact.to_dict() if fr.right_contact else None,
        "left_grasp": fr.left_grasp.to_dict() if fr.left_grasp else None,
        "right_grasp": fr.right_grasp.to_dict() if fr.right_grasp else None,
        "action": fr.action.to_dict() if fr.action else None,
        "objects": [o.to_dict() for o in fr.objects],
    })

    T = _finite_array(fr.T_world_cam) if fr.T_world_cam is not None else None
    if T is not None and T.shape == (4, 4):
        put(c_tf, _transform(fr.timestamp, "world", "camera", T[:3, :3], T[:3, 3]))
    for side in ("left", "right"):
        pose = getattr(fr, f"{side}_hand_metric")
        if pose is None:
            continue
        p = _finite_array(pose.wrist_position_m)
        R = _finite_array(pose.wrist_rotation)
        kp = _finite_array(pose.keypoints_m)
        if p is None or R is None or kp is None:
            continue
        put(c_tf, _transform(fr.timestamp, "camera", f"{side}_wrist", R, p))
        put(c_hand[side], {
            "timestamp": stamp, "frame_idx": fr.frame_idx, "frame_id": "camera", "hand": side,
            "is_interpolated": bool(pose.is_interpolated),
            "wrist_position_m": p.tolist(), "wrist_rotation": R.tolist(), "keypoints_m": kp.tolist(),
            "aperture_m": float(pose.aperture_m()),
            "reprojection_error_px": float(pose.reprojection_error_px),
            "intrinsics_source": pose.intrinsics_source,
        })

    q = fr.robot_joint_angles
    finite_q = q is not None and np.all(np.isfinite(np.asarray(q, dtype=np.float64)))
    gripper = fr.robot_gripper_opening_m
    put(c_joint, {
        "timestamp": stamp, "frame_idx": fr.frame_idx, "name": list(episode.robot_joint_names),
        "position": [float(x) for x in q] if finite_q else [],
        "gripper_opening_m": float(gripper) if gripper is not None and np.isfinite(gripper) else None,
        "valid": bool(fr.robot_reachable is True and finite_q),
        "ee_pos_error_m": (float(fr.robot_ee_pos_error_m)
                           if fr.robot_ee_pos_error_m is not None and np.isfinite(fr.robot_ee_pos_error_m)
                           else None),
    })

    if c_img is not None:
        import cv2
        from .lerobot_exporter import resolve_image_path
        path = resolve_image_path(fr.image_path, search)
        img = cv2.imread(str(path)) if path else None
        if img is None:
            raise FileNotFoundError(f"cannot read frame image for MCAP: {fr.image_path}")
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError(f"JPEG encoding failed for {fr.image_path}")
        put(c_img, {"timestamp": stamp, "frame_id": "camera",
                    "data": base64.b64encode(buf.tobytes()).decode("ascii"), "format": "jpeg"})


def verify_mcap_file(path: Path | str) -> Dict[str, Any]:
    """Read an MCAP with the official reader, including its summary section.

    Raises ``ValueError`` for a missing/truncated/corrupt file or one without a
    summary. Returns per-topic message counts counted from the data section and
    cross-checked against the summary statistics.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size < 2 * len(MCAP_MAGIC):
        raise ValueError(f"MCAP file missing or truncated: {p}")
    with open(p, "rb") as f:
        head = f.read(len(MCAP_MAGIC))
        f.seek(-len(MCAP_MAGIC), os.SEEK_END)
        tail = f.read(len(MCAP_MAGIC))
    if head != MCAP_MAGIC or tail != MCAP_MAGIC:
        raise ValueError(f"Invalid MCAP magic bytes in {p}")
    try:
        with open(p, "rb") as f:
            reader = make_reader(f, validate_crcs=True)
            summary = reader.get_summary()
            if summary is None or summary.statistics is None:
                raise ValueError(f"MCAP has no summary/statistics section: {p}")
            topics: Dict[str, int] = {}
            schemas: Dict[str, str] = {}
            last_ns = -1
            monotonic = True
            for schema, channel, message in reader.iter_messages(log_time_order=False):
                topics[channel.topic] = topics.get(channel.topic, 0) + 1
                schemas[channel.topic] = schema.name if schema else ""
                if message.log_time < last_ns:
                    monotonic = False
                last_ns = message.log_time
            by_summary = {summary.channels[cid].topic: n
                          for cid, n in summary.statistics.channel_message_counts.items()}
            if by_summary != topics:
                raise ValueError(f"MCAP summary counts {by_summary} != data counts {topics}")
            return {
                "valid_magic": True, "size_bytes": p.stat().st_size,
                "topics": topics, "schemas": schemas,
                "total_messages": sum(topics.values()),
                "frames": topics.get(FRAME_TOPIC, 0),
                "has_summary": True, "chunk_count": len(summary.chunk_indexes),
                "message_index_ok": len(summary.chunk_indexes) > 0,
                "log_time_monotonic": monotonic,
            }
    except ValueError:
        raise
    except Exception as exc:  # reader errors are heterogeneous
        raise ValueError(f"MCAP file is not readable ({type(exc).__name__}): {exc}") from exc
