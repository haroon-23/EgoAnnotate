"""RLDS-compatible HDF5 exporter for annotated egocentric video episodes."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List

import cv2
import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Grasp mapping: power_wrap=0, precision_pinch=1, lateral_pinch=2, hook=3, open=4, unknown=5
GRASP_MAPPING = {
    "power_wrap": 0,
    "precision_pinch": 1,
    "lateral_pinch": 2,
    "hook": 3,
    "open": 4,
    "unknown": 5,
}


def rot6d_from_normal(normal_vec: np.ndarray) -> np.ndarray:
    """Convert a 3D normal vector to rotation matrix using Gram-Schmidt.

    Then flatten the first two columns to get a rot6d representation.
    Handles near-zero normal by falling back to identity rotation.
    """
    norm = np.linalg.norm(normal_vec)
    if norm < 1e-6:
        # Fallback to identity rotation matrix
        # [1, 0, 0] for X, [0, 1, 0] for Y
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # Orthonormal basis calculation (Gram-Schmidt)
    z = normal_vec / norm

    # Choose a stable reference vector that is not parallel to z
    if np.abs(z[0]) < 0.9:
        v_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        v_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Project v_ref onto the plane perpendicular to z
    x = v_ref - np.dot(v_ref, z) * z
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    x = x / x_norm

    y = np.cross(z, x)

    # Columns of rotation matrix are [x, y, z]
    # Return first two columns flattened: [x_x, x_y, x_z, y_x, y_y, y_z]
    rot6d = np.concatenate([x, y]).astype(np.float32)
    return rot6d


def parse_hand_keypoints(
    frame: Dict[str, Any]
) -> tuple[np.ndarray | None, bool]:
    """Extract keypoints for the dominant hand (right if present, else left).

    Returns:
        A tuple of (keypoints_array (21, 3) or None, is_right_hand).
    """
    if frame.get("right_hand_present", False):
        kp = np.array(frame["right_hand_keypoints"], dtype=np.float32).reshape(21, 3)
        return kp, True
    elif frame.get("left_hand_present", False):
        kp = np.array(frame["left_hand_keypoints"], dtype=np.float32).reshape(21, 3)
        return kp, False
    return None, False


def export_to_rlds(episode_id: str, output_dir: str) -> Path:
    """Read annotations for an episode and export them in RLDS HDF5 format."""
    output_path = Path(output_dir)
    episode_dir = output_path / episode_id

    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")

    frame_annot_path = episode_dir / "frame_annotations.json"
    metadata_path = episode_dir / "metadata.json"

    if not frame_annot_path.exists():
        raise FileNotFoundError(f"Frame annotations JSON not found: {frame_annot_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_path}")

    with open(frame_annot_path, "r") as f:
        frames = json.load(f)
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    num_frames = len(frames)
    if num_frames == 0:
        raise ValueError(f"No frames found in annotations for episode {episode_id}")

    # 1. Initialize arrays
    images = np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
    wrist_translations = np.zeros((num_frames, 3), dtype=np.float32)
    wrist_rotations = np.zeros((num_frames, 6), dtype=np.float32)
    hand_poses = np.zeros((num_frames, 15), dtype=np.float32)
    proprioceptions = np.zeros((num_frames, 8), dtype=np.float32)
    language_instructions = []
    actions = np.zeros((num_frames, 24), dtype=np.float32)

    is_first = np.zeros(num_frames, dtype=bool)
    is_last = np.zeros(num_frames, dtype=bool)
    is_terminal = np.zeros(num_frames, dtype=bool)

    is_first[0] = True
    is_last[-1] = True
    is_terminal[-1] = True

    # 2. Process frames
    project_root = Path(__file__).resolve().parent.parent

    for idx, frame in enumerate(frames):
        # Image Loading & Resizing (BGR -> RGB)
        img_rel_path = frame["image_path"]
        img_path = project_root / img_rel_path
        if not img_path.exists():
            # Try absolute path or fallback to current working directory
            img_path = Path(img_rel_path)
            if not img_path.exists():
                img_path = Path(os.getcwd()) / img_rel_path

        if img_path.exists():
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if img_rgb.shape[:2] != (224, 224):
                    img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
                images[idx] = img_rgb
            else:
                logger.warning("Could not read image: %s", img_path)
        else:
            logger.warning("Image path does not exist: %s", img_path)

        # Dominant Hand Extraction
        kp, is_right = parse_hand_keypoints(frame)

        # 2.1 wrist_translation
        if kp is not None:
            # keypoint 0 is the wrist
            wrist_normalized = kp[0]
            # Convert [0,1] to metric space by multiplying by [0.5, 0.5, 0.5]
            wrist_translations[idx] = wrist_normalized * 0.5
        else:
            wrist_translations[idx] = np.zeros(3, dtype=np.float32)

        # 2.2 wrist_rotation
        if kp is not None:
            p0 = kp[0]
            p5 = kp[5]
            p17 = kp[17]

            v_index = p5 - p0
            v_pinky = p17 - p0
            normal = np.cross(v_index, v_pinky)
            wrist_rotations[idx] = rot6d_from_normal(normal)
        else:
            # Fallback to identity rotation
            wrist_rotations[idx] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # 2.3 hand_pose (proxy using finger angles)
        finger_angles = np.array(frame.get("action_finger_angles", [0.0] * 15), dtype=np.float32)
        hand_poses[idx] = finger_angles

        # 2.4 proprioception
        # [wrist_x, wrist_y, wrist_z, gripper_openness, left_contact_bool, right_contact_bool, left_grasp_onehot_idx, right_grasp_onehot_idx]
        gripper_open = float(frame.get("action_gripper_openness", 1.0))
        left_contact = 1.0 if frame.get("left_contact", False) else 0.0
        right_contact = 1.0 if frame.get("right_contact", False) else 0.0

        left_grasp = frame.get("left_grasp_type")
        left_grasp_idx = GRASP_MAPPING.get(left_grasp, 5) if left_grasp is not None else 5

        right_grasp = frame.get("right_grasp_type")
        right_grasp_idx = GRASP_MAPPING.get(right_grasp, 5) if right_grasp is not None else 5

        proprioceptions[idx] = np.array(
            [
                wrist_translations[idx][0],
                wrist_translations[idx][1],
                wrist_translations[idx][2],
                gripper_open,
                left_contact,
                right_contact,
                float(left_grasp_idx),
                float(right_grasp_idx),
            ],
            dtype=np.float32,
        )

        # 2.5 language instruction
        language_instructions.append(frame.get("language_instruction", ""))

        # 2.6 action: [wrist_delta(3), wrist_rot_delta(6), finger_angles(15)]
        wrist_delta = np.array(frame.get("action_wrist_delta", [0.0] * 3), dtype=np.float32)
        if idx == 0:
            wrist_rot_delta = np.zeros(6, dtype=np.float32)
        else:
            wrist_rot_delta = wrist_rotations[idx] - wrist_rotations[idx - 1]

        actions[idx] = np.concatenate([wrist_delta, wrist_rot_delta, finger_angles]).astype(np.float32)

    # 3. Create HDF5 dataset
    rlds_file_path = episode_dir / "episode_rlds.hdf5"
    logger.info("Saving RLDS dataset to: %s", rlds_file_path)

    with h5py.File(rlds_file_path, "w") as f:
        # Create groups
        ep_group = f.create_group(episode_id)
        steps_group = ep_group.create_group("steps")
        obs_group = steps_group.create_group("observation")
        meta_group = ep_group.create_group("metadata")

        # Save step observation datasets
        obs_group.create_dataset("image", data=images, dtype=np.uint8)
        obs_group.create_dataset("wrist_translation", data=wrist_translations, dtype=np.float32)
        obs_group.create_dataset("wrist_rotation", data=wrist_rotations, dtype=np.float32)
        
        hand_pose_ds = obs_group.create_dataset("hand_pose", data=hand_poses, dtype=np.float32)
        hand_pose_ds.attrs["hand_pose_note"] = "proxy using finger angles, labs should fit MANO PCA"

        obs_group.create_dataset("proprioception", data=proprioceptions, dtype=np.float32)

        # String arrays in HDF5
        utf8_type = h5py.string_dtype(encoding="utf-8")
        lang_ds = obs_group.create_dataset("language_instruction", (num_frames,), dtype=utf8_type)
        for i, text in enumerate(language_instructions):
            lang_ds[i] = text

        # Save step datasets
        steps_group.create_dataset("action", data=actions, dtype=np.float32)
        steps_group.create_dataset("is_first", data=is_first, dtype=bool)
        steps_group.create_dataset("is_last", data=is_last, dtype=bool)
        steps_group.create_dataset("is_terminal", data=is_terminal, dtype=bool)

        # Save episode metadata
        meta_group.create_dataset("task_description", data=meta.get("task_description", ""), dtype=utf8_type)
        meta_group.create_dataset("num_frames", data=int(meta.get("num_frames", num_frames)), dtype=np.int32)
        meta_group.create_dataset("duration_seconds", data=float(meta.get("duration_seconds", 0.0)), dtype=np.float32)

    return rlds_file_path


def export_all_episodes(output_dir: str = "data/output") -> List[Path]:
    """Scan output directory and export all annotated episodes to RLDS format."""
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning("Output directory %s does not exist. Nothing to export.", output_dir)
        return []

    exported_paths = []
    # Identify episode directories (any folder that contains frame_annotations.json)
    for path in output_path.iterdir():
        if path.is_dir() and (path / "frame_annotations.json").exists():
            try:
                rlds_path = export_to_rlds(path.name, str(output_path))
                exported_paths.append(rlds_path)
                print(f"Successfully exported RLDS for episode: {path.name} -> {rlds_path.name}")
            except Exception as e:
                logger.error("Failed to export episode %s: %s", path.name, e, exc_info=True)
                print(f"Error exporting RLDS for episode {path.name}: {e}")

    return exported_paths
