"""Zero-dependency spec-compliant MCAP v0 exporter for EgoAnnotate episodes.

Writes standard MCAP v0 files (\x89MCAP0\r\n magic bytes) readable by Foxglove Studio
and mcap CLI tools, separating embodiment-agnostic /human/* topics from /robot/* topics.
"""
from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict

from .datatypes import AnnotatedEpisode

MCAP_MAGIC = b"\x89MCAP0\r\n"

# MCAP v0 Record Opcodes
OP_HEADER = 0x01
OP_FOOTER = 0x02
OP_SCHEMA = 0x03
OP_CHANNEL = 0x04
OP_MESSAGE = 0x05
OP_DATA_END = 0x0F


def _mcap_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _mcap_bytes(b: bytes) -> bytes:
    return struct.pack("<I", len(b)) + b


def _mcap_map(m: Dict[str, str]) -> bytes:
    body = b"".join(_mcap_str(k) + _mcap_str(v) for k, v in m.items())
    return struct.pack("<I", len(body)) + body


def _mcap_record(opcode: int, payload: bytes) -> bytes:
    return struct.pack("<BQ", opcode, len(payload)) + payload


def export_episode_to_mcap(episode: AnnotatedEpisode, output_path: Path | str) -> Path:
    """Export an AnnotatedEpisode to a spec-compliant MCAP v0 file atomically."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=str(out_file.parent), prefix=f".{out_file.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(MCAP_MAGIC)

            # 1. Header record (profile="ego_annotate_v1", library="ego_annotate_mcap_writer")
            f.write(_mcap_record(OP_HEADER, _mcap_str("ego_annotate_v1") + _mcap_str("ego_annotate_mcap_writer")))

            # 2. Schemas (1: HumanFrame, 2: RobotFrame, 3: EpisodeMetadata)
            schemas = [
                (1, "ego_annotate.HumanFrame", "jsonschema", json.dumps({"type": "object"}).encode("utf-8")),
                (2, "ego_annotate.RobotFrame", "jsonschema", json.dumps({"type": "object"}).encode("utf-8")),
                (3, "ego_annotate.EpisodeMetadata", "jsonschema", json.dumps({"type": "object"}).encode("utf-8")),
            ]
            for sid, sname, enc, sdata in schemas:
                payload = struct.pack("<H", sid) + _mcap_str(sname) + _mcap_str(enc) + _mcap_bytes(sdata)
                f.write(_mcap_record(OP_SCHEMA, payload))

            # 3. Channels (1: /human/frame, 2: /robot/frame, 3: /episode/metadata)
            channels = [
                (1, 1, "/human/frame", "json", {"frame": "camera_metric_approx"}),
                (2, 2, "/robot/frame", "json", {"target_robot": str(episode.target_robot)}),
                (3, 3, "/episode/metadata", "json", {"episode_id": str(episode.episode_id)}),
            ]
            for cid, sid, topic, msg_enc, meta in channels:
                payload = (
                    struct.pack("<HH", cid, sid)
                    + _mcap_str(topic)
                    + _mcap_str(msg_enc)
                    + _mcap_map(meta)
                )
                f.write(_mcap_record(OP_CHANNEL, payload))

            # 4. Episode Metadata Message (seq=0, t=0)
            ep_meta_bytes = json.dumps({
                "episode_id": episode.episode_id,
                "task_description": episode.task_description,
                "num_frames": episode.num_frames,
                "duration_seconds": episode.duration_seconds,
                "target_robot": episode.target_robot,
                "camera_motion_flag": bool(getattr(episode, "camera_motion_flag", False)),
                "camera_motion_px_per_sec": float(getattr(episode, "camera_motion_px_per_sec", 0.0)),
                "segments": [s.to_dict() for s in episode.segments],
            }).encode("utf-8")
            f.write(_mcap_record(OP_MESSAGE, struct.pack("<HIQQ", 3, 0, 0, 0) + ep_meta_bytes))

            # 5. Per-frame Human and Robot Messages
            for seq_idx, fr in enumerate(episode.frames):
                ts_ns = int(round(float(fr.timestamp) * 1e9))
                human_payload = json.dumps({
                    "frame_idx": fr.frame_idx,
                    "timestamp": fr.timestamp,
                    "left_hand": fr.left_hand.to_dict() if fr.left_hand else None,
                    "right_hand": fr.right_hand.to_dict() if fr.right_hand else None,
                    "left_contact": fr.left_contact.to_dict() if fr.left_contact else None,
                    "right_contact": fr.right_contact.to_dict() if fr.right_contact else None,
                    "left_grasp": fr.left_grasp.to_dict() if fr.left_grasp else None,
                    "right_grasp": fr.right_grasp.to_dict() if fr.right_grasp else None,
                    "action": fr.action.to_dict() if fr.action else None,
                    "action_segment": fr.action_segment,
                    "frame_description": fr.frame_description,
                }).encode("utf-8")
                f.write(_mcap_record(OP_MESSAGE, struct.pack("<HIQQ", 1, seq_idx, ts_ns, ts_ns) + human_payload))

                robot_payload = json.dumps({
                    "frame_idx": fr.frame_idx,
                    "timestamp": fr.timestamp,
                    "target_robot": episode.target_robot,
                    "robot_joint_angles": getattr(fr, "robot_joint_angles", None),
                    "robot_gripper_opening_m": getattr(fr, "robot_gripper_opening_m", None),
                    "robot_reachable": bool(getattr(fr, "robot_reachable", False)),
                }).encode("utf-8")
                f.write(_mcap_record(OP_MESSAGE, struct.pack("<HIQQ", 2, seq_idx, ts_ns, ts_ns) + robot_payload))

            # 6. DataEnd + Footer + Magic
            f.write(_mcap_record(OP_DATA_END, struct.pack("<I", 0)))
            f.write(_mcap_record(OP_FOOTER, struct.pack("<QQI", 0, 0, 0)))
            f.write(MCAP_MAGIC)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_file)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
    return out_file


def verify_mcap_file(path: Path | str) -> Dict[str, Any]:
    """Verify MCAP v0 header/footer magic and count messages per channel."""
    p = Path(path)
    if not p.exists() or p.stat().st_size < 32:
        raise ValueError(f"MCAP file missing or truncated: {p}")
    data = p.read_bytes()
    if not data.startswith(MCAP_MAGIC) or not data.endswith(MCAP_MAGIC):
        raise ValueError(f"Invalid MCAP magic bytes in {p}")

    offset = len(MCAP_MAGIC)
    end = len(data) - len(MCAP_MAGIC)
    msg_counts: Dict[int, int] = {}
    channels: Dict[int, str] = {}

    while offset < end:
        op, rec_len = struct.unpack_from("<BQ", data, offset)
        offset += 9
        rec = data[offset : offset + rec_len]
        offset += rec_len
        if op == OP_CHANNEL:
            cid, _ = struct.unpack_from("<HH", rec, 0)
            t_len = struct.unpack_from("<I", rec, 4)[0]
            topic = rec[8 : 8 + t_len].decode("utf-8")
            channels[cid] = topic
        elif op == OP_MESSAGE:
            cid = struct.unpack_from("<H", rec, 0)[0]
            msg_counts[cid] = msg_counts.get(cid, 0) + 1

    return {
        "valid_magic": True,
        "size_bytes": len(data),
        "topics": {channels.get(cid, str(cid)): cnt for cid, cnt in msg_counts.items()},
        "total_messages": sum(msg_counts.values()),
    }
