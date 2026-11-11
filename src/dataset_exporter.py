from __future__ import annotations
"""Dataset export to JSON and Parquet formats.

Implements DatasetExporter to serialize VLA training frames, action segments,
overall episode metadata, grasp distributions, compile debug HUD videos,
and generate client-facing overlay videos with delivery validation.
"""


import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from .datatypes import AnnotatedEpisode
from .visualizer import EgoVisualizer, VizConfig, render_annotated_video
from .rlds_exporter import export_to_rlds
from .lerobot_exporter import export_to_lerobot


logger = logging.getLogger(__name__)


@dataclass
class ExporterConfig:
    """Configuration for the DatasetExporter."""
    output_dir: str
    format: str = "json"                 # "json" or "parquet"
    include_image_bytes: bool = False
    save_viz_video: bool = False          # Internal 224x224 debug HUD → debug_hud_preview.mp4
    save_overlay_video: bool = True      # Client-facing native-res → overlay_annotated.mp4
    lerobot_export_mode: str = "human"   # "human" (24D) or "robot" (8D joint+gripper)
    consent_manifest_path: Optional[str] = None  # Genuine DPDP consent file path
    internal_test_no_consent: bool = False  # Mark internal test footage without writing fake consent


class DatasetExporter:
    """Handles serialization and export of annotated egocentric VLA datasets."""

    def __init__(self, config: ExporterConfig):
        """Initialise the DatasetExporter."""
        self.config = config
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def export_episode(self, episode: AnnotatedEpisode) -> Path:
        """Export an AnnotatedEpisode to the configured output format and directory."""
        from .privacy import validate_real_consent_manifest
        episode_dir = self.output_path / episode.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting episode %s to: %s", episode.episode_id, episode_dir)

        # Enforce Real DPDP Consent Manifest (fail-loud: export blocked if missing)
        consent_out = episode_dir / "consent_manifest.json"
        capture_meta = episode_dir / "capture_meta.json"
        if getattr(self.config, "consent_manifest_path", None) and Path(self.config.consent_manifest_path).exists():
            shutil.copy2(self.config.consent_manifest_path, consent_out)

        internal_test = getattr(self.config, "internal_test_no_consent", False)

        if internal_test:
            # Writes consent_status="UNSET_INTERNAL_TEST" and stamps NOT_FOR_DISTRIBUTION
            # Never write acknowledgment, collector or date values.
            consent_doc = {
                "consent_status": "UNSET_INTERNAL_TEST",
                "distribution_restriction": "NOT_FOR_DISTRIBUTION",
            }
            consent_out.write_text(json.dumps(consent_doc, indent=2) + "\n", encoding="utf-8")
        elif consent_out.exists():
            validate_real_consent_manifest(consent_out)
        elif capture_meta.exists():
            validate_real_consent_manifest(capture_meta)
        else:
            raise RuntimeError(
                f"EXPORT BLOCKED for episode '{episode.episode_id}': missing real consent manifest "
                f"(requires real consent_manifest.json or capture_meta.json with consent block; synthetic generation is forbidden)."
            )
        # 1. Export frame annotations (JSON or Parquet) and canonical episode JSON
        self._export_frames(episode, episode_dir)
        episode.save_json(episode_dir / "annotations.json")

        # 2. Export episode metadata
        self._export_metadata(episode, episode_dir)

        # 3. Export action segments
        self._export_segments(episode, episode_dir)

        # 4. Save debug HUD video if requested (224×224 internal preview)
        # Removed 224x224 debug_hud_preview.mp4 per v1.3 spec: overlay_annotated.mp4 is the canonical video.

        # 5. Save client-facing overlay video (native resolution, all annotations)
        if self.config.save_overlay_video:
            self._create_overlay_video(episode, episode_dir)

        # 6. Create summary metrics file
        self._create_summary(episode, episode_dir)
        try:
            from .mcap_exporter import export_episode_to_mcap
            export_episode_to_mcap(episode, episode_dir / "episode.mcap")
        except Exception as e:
            logger.error("Failed to export episode.mcap: %s", e)
            raise

        # 7. Export RLDS HDF5 — fail-loud
        self._export_rlds(episode, episode_dir)

        # 8. Export LeRobot v3.0 dataset — fail-loud
        self._export_lerobot(episode, episode_dir)

        # 9. Validate delivery artifacts across all 4 required deliverables
        # side_by_side.mp4 is only expected when retargeting was run
        has_robot_data = any(
            f.robot_joint_angles is not None for f in episode.frames
        )
        side_by_side_path = episode_dir / "side_by_side.mp4" if has_robot_data else None
        if self.config.save_overlay_video:
            self.validate_delivery(episode_dir, episode.video_path, side_by_side_path=side_by_side_path)

        return episode_dir

    def _export_frames(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Compile and serialize the detailed frame-by-frame annotations."""
        frame_data = []

        for frame in episode.frames:
            # Flatten Left Hand keypoints to shape (63,)
            left_kp = [0.0] * 63
            if frame.left_hand is not None:
                left_kp = frame.left_hand.to_array().flatten().tolist()

            # Flatten Right Hand keypoints to shape (63,)
            right_kp = [0.0] * 63
            if frame.right_hand is not None:
                right_kp = frame.right_hand.to_array().flatten().tolist()

            # Contact states
            left_contact_val = frame.left_contact.in_contact if frame.left_contact else False
            left_contact_obj = frame.left_contact.object_name if frame.left_contact else None

            right_contact_val = frame.right_contact.in_contact if frame.right_contact else False
            right_contact_obj = frame.right_contact.object_name if frame.right_contact else None

            # Grasp types
            left_grasp_val = frame.left_grasp.type if frame.left_grasp else None
            right_grasp_val = frame.right_grasp.type if frame.right_grasp else None

            # Actions
            action_wrist = [0.0, 0.0, 0.0]
            action_angles = [0.0] * 15
            action_gripper = 1.0

            if frame.action is not None:
                action_wrist = frame.action.wrist_delta.tolist()
                action_angles = frame.action.finger_angles.tolist()
                action_gripper = float(frame.action.gripper_openness)

            # Language instruction (frame_description if filled, fallback to task_description)
            lang_inst = frame.frame_description if frame.frame_description else episode.task_description

            row = {
                "frame_idx": frame.frame_idx,
                "timestamp": frame.timestamp,
                "image_path": frame.image_path,
                "left_hand_present": frame.left_hand is not None,
                "right_hand_present": frame.right_hand is not None,
                "left_hand_interpolated": frame.left_hand.is_interpolated if frame.left_hand is not None else False,
                "right_hand_interpolated": frame.right_hand.is_interpolated if frame.right_hand is not None else False,
                "left_hand_keypoints": left_kp,
                "right_hand_keypoints": right_kp,
                "left_contact": left_contact_val,
                "left_contact_object": left_contact_obj,
                "right_contact": right_contact_val,
                "right_contact_object": right_contact_obj,
                "left_grasp_type": left_grasp_val,
                "right_grasp_type": right_grasp_val,
                "action_wrist_delta": action_wrist,
                "action_finger_angles": action_angles,
                "action_gripper_openness": action_gripper,
                "language_instruction": lang_inst,
                "action_segment": frame.action_segment,
                "robot_joint_angles": frame.robot_joint_angles,
                "robot_gripper_opening_m": frame.robot_gripper_opening_m,
                "robot_gripper_method": frame.robot_gripper_method,
                "robot_reachable": frame.robot_reachable,
                "metadata": frame.metadata,
                "process_reward": (frame.metadata or {}).get("process_reward", 0.0) if frame.metadata else 0.0,
            }
            frame_data.append(row)

        # Export in chosen format
        if self.config.format.lower() == "parquet":
            try:
                import pandas as pd
                pq_data = []
                for r in frame_data:
                    rc = dict(r)
                    if isinstance(rc.get("metadata"), dict):
                        rc["metadata"] = json.dumps(rc["metadata"])
                    pq_data.append(rc)
                df = pd.DataFrame(pq_data)
                df.to_parquet(episode_dir / "frame_annotations.parquet", index=False)
                logger.info(
                    "Exported frame annotations to Parquet "
                    "(additive — JSON still written below)."
                )
            except Exception as e:
                logger.warning("Parquet export failed (%s). Falling back to JSON frames export.", e)

        # Default JSON export
        with open(episode_dir / "frame_annotations.json", "w") as f:
            json.dump(frame_data, f, indent=2)
        logger.info("Exported frame annotations to JSON.")

    def _export_metadata(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Export high-level metadata info for the episode."""
        from .perception.backends import CameraIntrinsicsProfile
        dev_profile = CameraIntrinsicsProfile.load("wide_fisheye_actioncam")

        meta = {
            "episode_id": episode.episode_id,
            "video_path": episode.video_path,
            "task_description": episode.task_description,
            "num_frames": episode.num_frames,
            "duration_seconds": episode.duration_seconds,
            "target_robot": episode.target_robot,
            "physics_verified": bool(episode.physics_verified),
            "physics_report_path": episode.physics_report_path,
            "camera_profile": dev_profile.to_dict(),
        }
        if getattr(self.config, "internal_test_no_consent", False):
            meta["consent_status"] = "UNSET_INTERNAL_TEST"
            meta["distribution_restriction"] = "NOT_FOR_DISTRIBUTION"
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Record camera profile in capture_meta.json per episode
        capture_meta_path = episode_dir / "capture_meta.json"
        cap_meta = {}
        if capture_meta_path.exists():
            try:
                cap_meta = json.loads(capture_meta_path.read_text(encoding="utf-8"))
            except Exception:
                cap_meta = {}
        cap_meta["camera_profile"] = dev_profile.to_dict()
        cap_meta["device_id"] = dev_profile.device_id
        cap_meta["calibrated"] = dev_profile.calibrated
        capture_meta_path.write_text(json.dumps(cap_meta, indent=2) + "\n", encoding="utf-8")

    def persist_physics_verification(
        self, episode: AnnotatedEpisode, episode_dir: Path
    ) -> None:
        """Persist Stage-9b physics-verification labels set after export.

        The verifier runs after :meth:`export_episode` (the HDF5 must exist
        first), so its labels are written back into ``metadata.json`` and the
        RLDS HDF5 root attributes here. Never raises — a persistence failure
        is logged loudly but must not lose the episode or its report.
        """
        episode_dir = Path(episode_dir)
        meta_path = episode_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception as exc:
                logger.warning(
                    "[Exporter] Could not re-read metadata.json (%s); "
                    "physics labels kept in memory only.", exc)
                meta = None
            if meta is not None:
                meta["physics_verified"] = bool(episode.physics_verified)
                meta["physics_report_path"] = episode.physics_report_path
                meta_path.write_text(json.dumps(meta, indent=2))
        try:
            import h5py
        except ImportError:
            logger.warning(
                "[Exporter] h5py missing; physics labels not written to HDF5.")
            return
        rlds_path = episode_dir / "episode_hdf5.h5"
        if not rlds_path.exists():
            rlds_path = episode_dir / "episode_hdf5.h5"
        if not rlds_path.exists():
            logger.warning(
                "[Exporter] HDF5 missing; physics labels not written to HDF5.")
            return
        try:
            with h5py.File(str(rlds_path), "r+") as hf:
                hf.attrs["physics_verified"] = bool(episode.physics_verified)
                if episode.physics_report_path:
                    hf.attrs["physics_report_path"] = str(episode.physics_report_path)
        except Exception as exc:  # noqa: BLE001 - persistence must not kill the pipeline
            logger.warning("[Exporter] Could not write physics attrs to HDF5 (%s).", exc)

    def _export_segments(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Export temporal action segments list."""
        segments_data = [s.to_dict() for s in episode.segments]
        with open(episode_dir / "action_segments.json", "w") as f:
            json.dump(segments_data, f, indent=2)

    def _export_json(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Export all primary JSON deliverables (frames, metadata, segments)."""
        self._export_frames(episode, episode_dir)
        self._export_metadata(episode, episode_dir)
        self._export_segments(episode, episode_dir)

    def _create_viz_video(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Render and save a debug HUD video (224×224, internal-only) in the output directory."""
        try:
            visualizer = EgoVisualizer(VizConfig())
            visualizer.render_episode(episode, episode_dir / "debug_hud_preview.mp4")
        except Exception as e:
            logger.error("Failed to render debug HUD video during export: %s", e)

    def _create_overlay_video(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Render and save client-facing overlay video at native resolution.

        Uses atomic write: renders to .tmp first, renames on success.
        Fails loudly if overlay generation fails when enabled.
        """
        overlay_path = episode_dir / "overlay_annotated.mp4"
        try:
            render_annotated_video(
                episode.video_path,
                episode,
                overlay_path,
                atomic=True,
            )
            print(f"[Exporter] Client overlay saved: {overlay_path}")
        except Exception as e:
            # Fail-loud: do not silently skip overlay generation
            raise RuntimeError(
                f"Failed to render client-facing overlay video '{overlay_path}': {e}. "
                f"Pipeline cannot silently deliver without the overlay. "
                f"Set 'save_overlay_video: false' in config to disable."
            ) from e

    def _export_rlds(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Export RLDS HDF5 — fail-loud if export fails."""
        try:
            rlds_path = export_to_rlds(
                episode_id=episode.episode_id,
                output_dir=str(self.output_path),
            )
            print(f"[Exporter] RLDS HDF5 saved: {rlds_path}")
        except Exception as e:
            raise RuntimeError(
                f"RLDS HDF5 export FAILED for episode '{episode.episode_id}': {e}. "
                f"Pipeline cannot deliver without episode_hdf5.h5."
            ) from e

    def _export_lerobot(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Export LeRobot v3.0 dataset — fail-loud if export fails."""
        try:
            lerobot_path = export_to_lerobot(
                episode_id=episode.episode_id,
                output_dir=str(self.output_path),
                export_mode=self.config.lerobot_export_mode,
                internal_test_no_consent=getattr(self.config, "internal_test_no_consent", False),
            )
            print(f"[Exporter] LeRobot v3.0 dataset saved: {lerobot_path}")
        except Exception as e:
            raise RuntimeError(
                f"LeRobot v3.0 export FAILED for episode '{episode.episode_id}': {e}. "
                f"Pipeline cannot deliver without lerobot_v3/."
            ) from e

    def validate_delivery(
        self,
        episode_dir: Path,
        source_video_path: str,
        side_by_side_path: Optional[Path] = None,
    ) -> None:
        """Validate ALL required delivery artifacts against strict criteria.

        Checks:
        1. overlay_annotated.mp4 — exists, >0 bytes, resolution matches source, probe-able
        2. side_by_side.mp4     — exists, >0 bytes, H.264 codec, probe-able (if retargeting ran)
        3. episode_hdf5.h5    — exists, opens via h5py, contains required groups
        4. lerobot_v3/          — directory exists, meta/info.json present, parquet file present

        Raises RuntimeError with diagnostic detail if ANY check fails.
        """
        failures = []

        # --- 1. overlay_annotated.mp4 ---
        overlay_path = episode_dir / "overlay_annotated.mp4"
        if not overlay_path.exists():
            failures.append("overlay_annotated.mp4: MISSING")
        elif overlay_path.stat().st_size == 0:
            failures.append("overlay_annotated.mp4: ZERO BYTES")
        else:
            source_info = self._ffprobe_video_info(source_video_path)
            overlay_info = self._ffprobe_video_info(str(overlay_path))
            if overlay_info is None:
                failures.append("overlay_annotated.mp4: ffprobe FAILED (unplayable file)")
            elif source_info is not None:
                if (overlay_info["width"] != source_info["width"]
                        or overlay_info["height"] != source_info["height"]):
                    failures.append(
                        f"overlay_annotated.mp4: RESOLUTION MISMATCH — "
                        f"overlay={overlay_info['width']}x{overlay_info['height']}, "
                        f"source={source_info['width']}x{source_info['height']} "
                        f"(debug HUD substituted for real overlay?)"
                    )
                else:
                    logger.info(
                        "[Delivery] overlay_annotated.mp4 OK: %dx%d, %s, %.1f MB",
                        overlay_info["width"], overlay_info["height"],
                        overlay_info["codec"], overlay_path.stat().st_size / (1024 * 1024),
                    )
                    print(
                        f"[Delivery] ✓ overlay_annotated.mp4  "
                        f"{overlay_info['width']}x{overlay_info['height']} "
                        f"codec={overlay_info['codec']} "
                        f"{overlay_path.stat().st_size / (1024*1024):.1f} MB"
                    )

        if side_by_side_path is not None and Path(side_by_side_path).exists():
            src_info = self._ffprobe_video_info(source_video_path)
            sbs_info = self._ffprobe_video_info(str(side_by_side_path))
            if src_info and sbs_info:
                if abs(src_info["duration"] - sbs_info["duration"]) > 0.2:
                    raise RuntimeError(
                        f"DURATION MISMATCH: side_by_side {sbs_info['duration']:.2f}s "
                        f"vs source {src_info['duration']:.2f}s")

        frame_json_path = episode_dir / "frame_annotations.json"
        if frame_json_path.exists():
            try:
                with open(frame_json_path, "r") as f:
                    fa_frames = json.load(f)
                openings = [
                    f["robot_gripper_opening_m"]
                    for f in fa_frames
                    if isinstance(f, dict) and f.get("robot_gripper_opening_m") is not None
                ]
                if openings:
                    max_opening = max(openings)
                    episode_has_open_grasp = any(
                        isinstance(f, dict) and (f.get("left_grasp_type") == "open" or f.get("right_grasp_type") == "open")
                        for f in fa_frames
                    )
                    if max_opening < 0.06 and episode_has_open_grasp:
                        raise RuntimeError(
                            f"Gripper channel suspect: max opening {max_opening:.4f} m "
                            f"despite open-hand frames (expected >= 0.06)")
            except (json.JSONDecodeError, OSError):
                pass

        # --- 2. side_by_side.mp4 (only when retargeting was run) ---
        if side_by_side_path is not None:
            sbs = side_by_side_path
            if not sbs.exists():
                failures.append("side_by_side.mp4: MISSING (retargeting ran but no proof video written)")
            elif sbs.stat().st_size == 0:
                failures.append("side_by_side.mp4: ZERO BYTES")
            else:
                sbs_info = self._ffprobe_video_info(str(sbs))
                if sbs_info is None:
                    failures.append("side_by_side.mp4: ffprobe FAILED (unplayable file)")
                elif sbs_info["codec"] not in ("h264", "avc1"):
                    failures.append(
                        f"side_by_side.mp4: WRONG CODEC '{sbs_info['codec']}' — "
                        f"expected h264/avc1. Legacy mp4v will not play in browsers/QuickTime."
                    )
                else:
                    logger.info(
                        "[Delivery] side_by_side.mp4 OK: %dx%d, %s, %.1f MB",
                        sbs_info["width"], sbs_info["height"],
                        sbs_info["codec"], sbs.stat().st_size / (1024 * 1024),
                    )
                    print(
                        f"[Delivery] ✓ side_by_side.mp4       "
                        f"{sbs_info['width']}x{sbs_info['height']} "
                        f"codec={sbs_info['codec']} "
                        f"{sbs.stat().st_size / (1024*1024):.1f} MB"
                    )

        # --- 3. episode_hdf5.h5 / episode_hdf5.h5 ---
        rlds_path = episode_dir / "episode_hdf5.h5"
        if not rlds_path.exists():
            rlds_path = episode_dir / "episode_hdf5.h5"
        if not rlds_path.exists():
            failures.append("episode_hdf5.h5 (episode_hdf5.h5): MISSING")
        elif rlds_path.stat().st_size == 0:
            failures.append(f"{rlds_path.name}: ZERO BYTES")
        else:
            try:
                import h5py
                with h5py.File(rlds_path, "r") as hf:
                    # Must have at least one episode group with steps/action and steps/observation
                    top_keys = list(hf.keys())
                    if not top_keys:
                        failures.append("episode_hdf5.h5: no top-level groups (empty HDF5)")
                    else:
                        ep_grp = hf[top_keys[0]]
                        missing_paths = []
                        for req in ["steps/action", "steps/observation/language_instruction"]:
                            if req not in ep_grp:
                                missing_paths.append(req)
                        if missing_paths:
                            failures.append(
                                f"episode_hdf5.h5: missing required datasets: {missing_paths}"
                            )
                        else:
                            n_steps = ep_grp["steps/action"].shape[0]
                            print(
                                f"[Delivery] ✓ episode_hdf5.h5      "
                                f"{n_steps} steps, "
                                f"{rlds_path.stat().st_size / (1024*1024):.1f} MB"
                            )
            except Exception as e:
                failures.append(f"episode_hdf5.h5: h5py open FAILED — {e}")

        # --- 4. lerobot_v3/ ---
        lerobot_dir = episode_dir / "lerobot_v3"
        if not lerobot_dir.is_dir():
            failures.append("lerobot_v3/: directory MISSING")
        else:
            info_json = lerobot_dir / "meta" / "info.json"
            if not info_json.exists():
                failures.append("lerobot_v3/meta/info.json: MISSING")
            else:
                # Check at least one parquet chunk exists
                parquet_files = list((lerobot_dir / "data").rglob("*.parquet"))
                if not parquet_files:
                    failures.append("lerobot_v3/data/: no parquet files found")
                else:
                    try:
                        import pandas as pd
                        df = pd.read_parquet(parquet_files[0])
                        n_rows = len(df)
                        print(
                            f"[Delivery] ✓ lerobot_v3/             "
                            f"{n_rows} rows, parquet OK, info.json present"
                        )
                    except Exception as e:
                        failures.append(f"lerobot_v3/data parquet: read FAILED — {e}")

        # --- Final verdict ---
        if failures:
            report = "\n".join(f"  FAIL: {f}" for f in failures)
            raise RuntimeError(
                f"Delivery validation FAILED — {len(failures)} artifact(s) missing or invalid:\n"
                f"{report}\n"
                f"Fix all failures before delivering. Silent partial deliveries are not acceptable."
            )

        print("[Delivery] ✓ All deliverables validated successfully.")
        logger.info("Delivery validation PASSED: all 4 artifacts validated.")

    @staticmethod
    def _ffprobe_video_info(path: str) -> Optional[Dict[str, Any]]:
        """Extract video properties using ffprobe.

        Returns dict with keys: width, height, frame_count, duration_sec, codec.
        Returns None if ffprobe fails.
        """
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_frames,codec_name,duration",
                "-show_entries", "format=duration",
                "-of", "json",
                str(path),
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
            )
            if result.returncode != 0:
                logger.warning("ffprobe failed for %s: %s", path, result.stderr.decode())
                return None

            data = json.loads(result.stdout.decode())
            streams = data.get("streams", [])
            if not isinstance(streams, list) or not streams:
                logger.warning("ffprobe found no video streams in %s", path)
                return None
            stream = streams[0]
            fmt = data.get("format", {})

            # nb_frames may be "N/A" for some codecs — fall back to 0
            nb_frames_raw = stream.get("nb_frames", "0")
            try:
                nb_frames = int(nb_frames_raw)
            except (ValueError, TypeError):
                nb_frames = 0

            # Duration: prefer stream duration, fall back to format duration
            dur_raw = stream.get("duration") or fmt.get("duration", "0")
            try:
                duration = float(dur_raw)
            except (ValueError, TypeError):
                duration = 0.0

            return {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "frame_count": nb_frames,
                "duration": duration,
                "duration_sec": duration,
                "codec": stream.get("codec_name", "unknown"),
            }
        except Exception as e:
            logger.warning("ffprobe error for %s: %s", path, e)
            return None

    def _create_summary(self, episode: AnnotatedEpisode, episode_dir: Path) -> None:
        """Compile and serialize general statistics summary for the episode."""
        left_grasps: Dict[str, int] = {}
        right_grasps: Dict[str, int] = {}
        left_contact_count = 0
        right_contact_count = 0

        for frame in episode.frames:
            if frame.left_grasp:
                left_grasps[frame.left_grasp.type] = left_grasps.get(frame.left_grasp.type, 0) + 1
            if frame.right_grasp:
                right_grasps[frame.right_grasp.type] = right_grasps.get(frame.right_grasp.type, 0) + 1

            if frame.left_contact and frame.left_contact.in_contact:
                left_contact_count += 1
            if frame.right_contact and frame.right_contact.in_contact:
                right_contact_count += 1

        total_frames = len(episode.frames)
        detected = [f.left_hand is not None or f.right_hand is not None for f in episode.frames]
        active_mask = [False] * total_frames
        for i in range(total_frames):
            if detected[i]:
                for k in range(max(0, i - 10), min(total_frames, i + 11)):
                    active_mask[k] = True
        active_frames = sum(1 for a in active_mask if a)
        lost_active = sum(1 for i, f in enumerate(episode.frames) if active_mask[i] and (f.left_hand is None or f.right_hand is None))
        default_loss_active = 100.0 * lost_active / max(active_frames, 1) if active_frames > 0 else 0.0

        summary = {
            "left_grasp_distribution": left_grasps,
            "right_grasp_distribution": right_grasps,
            "left_contact_frames": left_contact_count,
            "right_contact_frames": right_contact_count,
            "total_frames": total_frames,
            "ratio_left_contact": left_contact_count / total_frames if total_frames > 0 else 0.0,
            "ratio_right_contact": right_contact_count / total_frames if total_frames > 0 else 0.0,
            "tracking_loss_pct": getattr(episode, "tracking_loss_pct", 100.0 * sum(1 for f in episode.frames if f.left_hand is None or f.right_hand is None) / max(total_frames, 1)),
            "tracking_loss_pct_active": getattr(episode, "tracking_loss_pct_active", default_loss_active),
            "tracking_rescued_pct": getattr(episode, "tracking_rescued_pct", 0.0),
            "tracking_interpolated_pct": getattr(episode, "tracking_interpolated_pct", 100.0 * sum(1 for f in episode.frames if (f.left_hand and f.left_hand.is_interpolated) or (f.right_hand and f.right_hand.is_interpolated)) / max(total_frames, 1)),
            "camera_motion_flag": bool(getattr(episode, "camera_motion_flag", False)),
            "camera_motion_px_per_sec": float(getattr(episode, "camera_motion_px_per_sec", 0.0)),
        }

        try:
            from .flora_engine import compute_flora_rewards_and_rubric
            summary["flora_evaluation"] = compute_flora_rewards_and_rubric(episode)
        except Exception:
            pass

        from .atomic_io import atomic_write_json
        atomic_write_json(episode_dir / "summary.json", summary)
