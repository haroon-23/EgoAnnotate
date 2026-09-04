"""Dataclass definitions for the ego_annotate_vla_pipeline.

All annotation data structures used across the pipeline — from per-frame
hand landmarks and object detections through to full annotated episodes
ready for VLA model training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Primitive / per-component dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HandLandmarks:
    """21-point hand landmark data from MediaPipe.

    Attributes:
        x: X-coordinates for all 21 landmarks, normalised to [0, 1].
        y: Y-coordinates for all 21 landmarks, normalised to [0, 1].
        z: Z-coordinates (depth) for all 21 landmarks.
        confidence: Overall detection confidence in [0, 1].
        handedness: ``"Left"`` or ``"Right"``.
    """

    x: np.ndarray  # shape (21,)
    y: np.ndarray  # shape (21,)
    z: np.ndarray  # shape (21,)
    confidence: float
    handedness: str
    is_interpolated: bool = False

    # -- helper constants ---------------------------------------------------
    # MediaPipe hand landmark indices for each fingertip.
    _FINGERTIP_INDICES: list[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 20],
        init=False,
        repr=False,
    )

    def to_array(self) -> np.ndarray:
        """Stack x/y/z into a single ``(21, 3)`` array.

        Returns:
            np.ndarray of shape ``(21, 3)`` with columns [x, y, z].
        """
        return np.stack([self.x, self.y, self.z], axis=-1)  # (21, 3)

    def fingertip_positions(self) -> np.ndarray:
        """Return positions of the five fingertip landmarks.

        Landmark indices: thumb=4, index=8, middle=12, ring=16, pinky=20.

        Returns:
            np.ndarray of shape ``(5, 3)`` — one row per fingertip.
        """
        arr = self.to_array()  # (21, 3)
        return arr[self._FINGERTIP_INDICES]  # (5, 3)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "z": self.z.tolist(),
            "confidence": self.confidence,
            "handedness": self.handedness,
            "is_interpolated": self.is_interpolated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HandLandmarks:
        """Reconstruct from a dictionary produced by :meth:`to_dict`."""
        return cls(
            x=np.asarray(data["x"], dtype=np.float32),
            y=np.asarray(data["y"], dtype=np.float32),
            z=np.asarray(data["z"], dtype=np.float32),
            confidence=float(data["confidence"]),
            handedness=str(data["handedness"]),
            is_interpolated=bool(data.get("is_interpolated", False)),
        )


@dataclass
class ObjectAnnotation:
    """A single detected object in a frame.

    Attributes:
        name: Semantic label (e.g. ``"mug"``, ``"screwdriver"``).
        location_description: Natural-language spatial description
            (e.g. ``"on the table to the left"``).
        touched: Whether the object is currently in contact with a hand.
        bbox: Optional bounding box as ``[x_min, y_min, x_max, y_max]``
            in normalised coordinates.
        state: Object state, e.g. ``"idle"``, ``"grasped"``, ``"moving"``.
    """

    name: str
    location_description: str
    touched: bool
    bbox: Optional[np.ndarray] = None  # shape (4,)
    state: str = "idle"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "location_description": self.location_description,
            "touched": self.touched,
            "bbox": self.bbox.tolist() if self.bbox is not None else None,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObjectAnnotation:
        """Reconstruct from a dictionary produced by :meth:`to_dict`."""
        bbox = (
            np.asarray(data["bbox"], dtype=np.float32)
            if data.get("bbox") is not None
            else None
        )
        return cls(
            name=str(data["name"]),
            location_description=str(data["location_description"]),
            touched=bool(data["touched"]),
            bbox=bbox,
            state=str(data.get("state", "idle")),
        )


@dataclass
class ContactState:
    """Per-hand contact state between individual fingers and an object.

    Attributes:
        fingers: Boolean array of shape ``(5,)`` — one flag per finger
            (thumb, index, middle, ring, pinky).
        object_name: Name of the object in contact, or ``None``.
        in_contact: ``True`` if *any* finger is touching an object.
        confidence: Confidence score [0, 1] based on proximity margin and
            temporal consistency over the last N frames. MediaPipe z is relative
            depth, not metric 3D — this is a heuristic to filter false positives
            when a hand passes in front of an object without touching.
    """

    fingers: np.ndarray  # shape (5,), dtype bool
    object_name: Optional[str]
    in_contact: bool
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "fingers": self.fingers.tolist(),
            "object_name": self.object_name,
            "in_contact": self.in_contact,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContactState:
        """Reconstruct from a dictionary produced by :meth:`to_dict`."""
        return cls(
            fingers=np.asarray(data["fingers"], dtype=bool),
            object_name=data.get("object_name"),
            in_contact=bool(data["in_contact"]),
            confidence=float(data.get("confidence", 0.5)),
        )


@dataclass
class GraspType:
    """Classification of the current grasp.

    Attributes:
        type: Grasp taxonomy label — e.g. ``"power"``, ``"precision"``,
            ``"lateral"``, ``"hook"``, ``"spherical"``, ``"none"``.
        confidence: Classification confidence in [0, 1].
        thumb_index_distance: Euclidean distance between thumb tip and
            index fingertip (normalised coordinates).
        num_curled_fingers: Count of fingers whose tip is closer to the
            wrist than its MCP joint (0–5).
    """

    type: str
    confidence: float
    thumb_index_distance: float
    num_curled_fingers: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "type": self.type,
            "confidence": self.confidence,
            "thumb_index_distance": self.thumb_index_distance,
            "num_curled_fingers": self.num_curled_fingers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraspType:
        """Reconstruct from a dictionary produced by :meth:`to_dict`."""
        return cls(
            type=str(data["type"]),
            confidence=float(data["confidence"]),
            thumb_index_distance=float(data["thumb_index_distance"]),
            num_curled_fingers=int(data["num_curled_fingers"]),
        )


@dataclass
class RobotAgnosticAction:
    """Robot-agnostic action representation derived from hand motion.

    Designed to be re-targeted to any embodiment (humanoid, bimanual arm,
    single gripper, etc.) via a downstream retargeting layer.

    Attributes:
        wrist_delta: Frame-to-frame wrist displacement ``[dx, dy, dz]``
            in normalised coordinates.
        finger_angles: Joint angles for 5 fingers × 3 joints = 15 values,
            in radians.
        gripper_openness: Scalar in [0, 1] — 0 = fully closed, 1 = fully open.
        hand_orientation: Optional rotation as a quaternion ``[qw, qx, qy, qz]``
            or ``None`` if unavailable.
    """

    wrist_delta: np.ndarray       # shape (3,)
    finger_angles: np.ndarray     # shape (15,)
    gripper_openness: float
    hand_orientation: Optional[np.ndarray] = None  # shape (4,)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "wrist_delta": self.wrist_delta.tolist(),
            "finger_angles": self.finger_angles.tolist(),
            "gripper_openness": self.gripper_openness,
            "hand_orientation": (
                self.hand_orientation.tolist()
                if self.hand_orientation is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RobotAgnosticAction:
        """Reconstruct from a dictionary produced by :meth:`to_dict`."""
        orientation = (
            np.asarray(data["hand_orientation"], dtype=np.float32)
            if data.get("hand_orientation") is not None
            else None
        )
        return cls(
            wrist_delta=np.asarray(data["wrist_delta"], dtype=np.float32),
            finger_angles=np.asarray(data["finger_angles"], dtype=np.float32),
            gripper_openness=float(data["gripper_openness"]),
            hand_orientation=orientation,
        )


# ---------------------------------------------------------------------------
# Per-frame composite
# ---------------------------------------------------------------------------


@dataclass
class AnnotationFrame:
    """All annotations for a single video frame.

    This is the central data structure that every pipeline stage reads from
    and writes to.  Downstream stages progressively fill in the ``Optional``
    fields.

    Attributes:
        frame_idx: Zero-based frame index in the source video.
        timestamp: Timestamp in seconds from the start of the video.
        image_path: Filesystem path to the extracted frame image.
        left_hand: Landmarks for the left hand, if detected.
        right_hand: Landmarks for the right hand, if detected.
        objects: All objects detected in this frame.
        left_contact: Contact state for the left hand.
        right_contact: Contact state for the right hand.
        left_grasp: Grasp classification for the left hand.
        right_grasp: Grasp classification for the right hand.
        action: Robot-agnostic action computed for this frame.
        frame_description: Free-text description (filled by language generator).
        action_segment: Current action segment label (filled by segmenter).
        metadata: Arbitrary key-value metadata for extensibility.
    """

    frame_idx: int
    timestamp: float
    image_path: str
    left_hand: Optional[HandLandmarks] = None
    right_hand: Optional[HandLandmarks] = None
    objects: List[ObjectAnnotation] = field(default_factory=list)
    left_contact: Optional[ContactState] = None
    right_contact: Optional[ContactState] = None
    left_grasp: Optional[GraspType] = None
    right_grasp: Optional[GraspType] = None
    action: Optional[RobotAgnosticAction] = None
    frame_description: str = ""
    action_segment: str = "idle"
    robot_joint_angles: Optional[List[float]] = None
    robot_gripper_opening_m: Optional[float] = None
    robot_gripper_method: Optional[str] = None
    robot_reachable: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "left_hand": self.left_hand.to_dict() if self.left_hand else None,
            "right_hand": self.right_hand.to_dict() if self.right_hand else None,
            "objects": [obj.to_dict() for obj in self.objects],
            "left_contact": (
                self.left_contact.to_dict() if self.left_contact else None
            ),
            "right_contact": (
                self.right_contact.to_dict() if self.right_contact else None
            ),
            "left_grasp": (
                self.left_grasp.to_dict() if self.left_grasp else None
            ),
            "right_grasp": (
                self.right_grasp.to_dict() if self.right_grasp else None
            ),
            "action": self.action.to_dict() if self.action else None,
            "frame_description": self.frame_description,
            "action_segment": self.action_segment,
            "robot_joint_angles": self.robot_joint_angles,
            "robot_gripper_opening_m": self.robot_gripper_opening_m,
            "robot_gripper_method": self.robot_gripper_method,
            "robot_reachable": self.robot_reachable,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnnotationFrame:
        """Reconstruct from a dictionary produced by :meth:`to_dict`."""
        return cls(
            frame_idx=int(data["frame_idx"]),
            timestamp=float(data["timestamp"]),
            image_path=str(data["image_path"]),
            left_hand=(
                HandLandmarks.from_dict(data["left_hand"])
                if data.get("left_hand")
                else None
            ),
            right_hand=(
                HandLandmarks.from_dict(data["right_hand"])
                if data.get("right_hand")
                else None
            ),
            objects=[
                ObjectAnnotation.from_dict(obj)
                for obj in data.get("objects", [])
            ],
            left_contact=(
                ContactState.from_dict(data["left_contact"])
                if isinstance(data.get("left_contact"), dict)
                else (ContactState(fingers=np.zeros(5, dtype=bool), object_name=data.get("left_contact_object"), in_contact=bool(data["left_contact"])) if isinstance(data.get("left_contact"), bool) else None)
            ),
            right_contact=(
                ContactState.from_dict(data["right_contact"])
                if isinstance(data.get("right_contact"), dict)
                else (ContactState(fingers=np.zeros(5, dtype=bool), object_name=data.get("right_contact_object"), in_contact=bool(data["right_contact"])) if isinstance(data.get("right_contact"), bool) else None)
            ),
            left_grasp=(
                GraspType.from_dict(data["left_grasp"])
                if isinstance(data.get("left_grasp"), dict)
                else (GraspType(type=str(data["left_grasp"]), confidence=1.0, thumb_index_distance=0.0, num_curled_fingers=0) if isinstance(data.get("left_grasp"), str) else None)
            ),
            right_grasp=(
                GraspType.from_dict(data["right_grasp"])
                if isinstance(data.get("right_grasp"), dict)
                else (GraspType(type=str(data["right_grasp"]), confidence=1.0, thumb_index_distance=0.0, num_curled_fingers=0) if isinstance(data.get("right_grasp"), str) else None)
            ),
            action=(
                RobotAgnosticAction.from_dict(data["action"])
                if data.get("action")
                else None
            ),
            frame_description=str(data.get("frame_description", "")),
            action_segment=str(data.get("action_segment", "idle")),
            robot_joint_angles=data.get("robot_joint_angles"),
            robot_gripper_opening_m=(
                float(data["robot_gripper_opening_m"])
                if data.get("robot_gripper_opening_m") is not None
                else None
            ),
            robot_gripper_method=data.get("robot_gripper_method"),
            robot_reachable=(
                bool(data["robot_reachable"])
                if data.get("robot_reachable") is not None
                else None
            ),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Temporal segment
# ---------------------------------------------------------------------------


@dataclass
class CandidateSegment:
    """A candidate temporal segment derived from signal transitions (before VLM labeling).
    
    Internal representation used by SignalSegmenter. Not exported to final datasets.
    
    Attributes:
        start_frame: Inclusive start frame index.
        end_frame: Exclusive end frame index.
        start_time: Segment start in seconds.
        end_time: Segment end in seconds.
        transition_type: What triggered this segment boundary:
            "contact_on", "contact_off", "grasp_change", "idle", "start", "end"
        contact_state: "contact" or "no_contact" for this segment.
        grasp_type: Grasp type string for this segment, or "none".
        object_name: Primary object name, or None.
    """
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    transition_type: str
    contact_state: str
    grasp_type: str
    object_name: Optional[str] = None


@dataclass
class ActionSegment:
    """A temporally contiguous action segment within a video.

    Attributes:
        name: Action label (e.g. ``"pick_up"``, ``"pour"``, ``"place"``).
        start_time: Segment start in seconds.
        end_time: Segment end in seconds.
        object_name: Primary object involved in the action.
        hand_used: ``"left"``, ``"right"``, or ``"both"``.
        description: Optional natural-language description of the action.
    """

    name: str
    start_time: float
    end_time: float
    object_name: str
    hand_used: str
    description: str = ""

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "object_name": self.object_name,
            "hand_used": self.hand_used,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActionSegment:
        """Reconstruct from a dictionary produced by :meth:`to_dict`."""
        return cls(
            name=str(data["name"]),
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            object_name=str(data["object_name"]),
            hand_used=str(data["hand_used"]),
            description=str(data.get("description", "")),
        )


# ---------------------------------------------------------------------------
# Top-level episode container
# ---------------------------------------------------------------------------


@dataclass
class AnnotatedEpisode:
    """Complete annotated episode — the final pipeline output.

    Groups all per-frame annotations and temporal segments for a single
    video, ready for export to VLA training formats.

    Attributes:
        episode_id: Unique identifier for this episode.
        video_path: Path to the source video file.
        task_description: High-level description of the task being performed.
        frames: Ordered list of per-frame annotations.
        segments: Temporal action segments spanning the episode.
        num_frames: Total number of annotated frames.
        duration_seconds: Video duration in seconds.
        target_robot: Target embodiment for action retargeting.
    """

    episode_id: str
    video_path: str
    task_description: str
    frames: List[AnnotationFrame]
    segments: List[ActionSegment]
    num_frames: int
    duration_seconds: float
    target_robot: str = "humanoid_generic"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "episode_id": self.episode_id,
            "video_path": self.video_path,
            "task_description": self.task_description,
            "frames": [f.to_dict() for f in self.frames],
            "segments": [s.to_dict() for s in self.segments],
            "num_frames": self.num_frames,
            "duration_seconds": self.duration_seconds,
            "target_robot": self.target_robot,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnnotatedEpisode:
        """Reconstruct from a dictionary produced by :meth:`to_dict`."""
        return cls(
            episode_id=str(data["episode_id"]),
            video_path=str(data["video_path"]),
            task_description=str(data["task_description"]),
            frames=[
                AnnotationFrame.from_dict(f) for f in data.get("frames", [])
            ],
            segments=[
                ActionSegment.from_dict(s) for s in data.get("segments", [])
            ],
            num_frames=int(data["num_frames"]),
            duration_seconds=float(data["duration_seconds"]),
            target_robot=str(data.get("target_robot", "humanoid_generic")),
        )

    def save_json(self, path: str | Path) -> None:
        """Write the episode to a JSON file.

        Args:
            path: Destination file path. Parent directories are created
                automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str | Path) -> AnnotatedEpisode:
        """Load an episode from a JSON file.

        Args:
            path: Source file path.

        Returns:
            Reconstructed :class:`AnnotatedEpisode`.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)
