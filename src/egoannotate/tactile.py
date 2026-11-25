"""Honest RGB-derived tactile/contact proxies + time-synced real hardware tactile sensor ingestion."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from src.egoannotate.config import TactileConfig
from src.egoannotate.schemas import FrameAnnotation, HandState, TactileHandChannel


def _derive_hand_tactile_proxy(
    hand: HandState,
    prev_hand: HandState | None,
    frame: FrameAnnotation,
    side: Literal["left", "right"],
    dt_s: float,
    cfg: TactileConfig,
) -> TactileHandChannel:
    """Compute physically-grounded RGB tactile PROXIES (never fabricated physical Newtons)."""
    if not hand.detected:
        return TactileHandChannel(
            source="derived_rgb_proxy",
            contact_prob=0.0,
            contact_zones_6d=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            normal_force_proxy_01=0.0,
            slip_proxy_01=0.0,
            impact_proxy_01=0.0,
            compliance_prior="unknown",
            uncertainty=1.0,
        )

    curls = np.asarray(hand.finger_curls_5d, dtype=np.float64)
    tips = (
        np.asarray(hand.fingertips_world_m, dtype=np.float64)
        if len(hand.fingertips_world_m) == 5
        else np.zeros((5, 3))
    )
    wrist = np.asarray(hand.wrist_pos_world_m, dtype=np.float64)

    min_obj_dist = 1.0
    held_obj_vel: np.ndarray | None = None
    in_hand_obj = False
    tip_dists = np.full(5, 1.0, dtype=np.float64)

    for obj in frame.objects:
        obj_p = np.asarray(obj.pos_world_m, dtype=np.float64)
        d_wrist = float(np.linalg.norm(wrist - obj_p))
        d_tips = (
            np.linalg.norm(tips - obj_p[None, :], axis=1)
            if len(hand.fingertips_world_m) == 5
            else np.full(5, d_wrist)
        )
        if float(np.min(d_tips)) < min_obj_dist:
            min_obj_dist = float(np.min(d_tips))
            tip_dists = d_tips
            held_obj_vel = np.asarray(obj.vel_world_mps, dtype=np.float64)
        if obj.in_hand in (side, "both"):
            in_hand_obj = True

    prox_score = float(np.clip(1.0 - min_obj_dist / 0.14, 0.0, 1.0))
    aperture_closed = float(np.clip(1.0 - hand.gripper_aperture_m / 0.12, 0.0, 1.0))
    mean_curl = float(np.mean(curls))

    contact_prob = float(
        np.clip(
            (0.45 if in_hand_obj else 0.0) + 0.35 * prox_score + 0.25 * aperture_closed + 0.20 * mean_curl,
            0.0,
            0.96,
        )
    )
    if not in_hand_obj and min_obj_dist > 0.14 and mean_curl < 0.45:
        contact_prob *= 0.25

    zones = []
    for i in range(5):
        z_prox = float(np.clip(1.0 - tip_dists[i] / 0.13, 0.0, 1.0))
        zones.append(round(float(np.clip(contact_prob * (0.55 * float(curls[i]) + 0.45 * z_prox), 0.0, 0.98)), 4))
    palm_contact = round(float(np.clip(contact_prob * (0.7 * mean_curl + 0.3 * prox_score), 0.0, 0.95)), 4)
    zones.append(palm_contact)

    aperture_rate = 0.0
    accel_mag = 0.0
    if prev_hand is not None and prev_hand.detected and dt_s > 1e-4:
        aperture_rate = max(0.0, (prev_hand.gripper_aperture_m - hand.gripper_aperture_m) / dt_s)
        dv = np.asarray(hand.wrist_vel_world_mps, dtype=np.float64) - np.asarray(
            prev_hand.wrist_vel_world_mps, dtype=np.float64
        )
        accel_mag = float(np.linalg.norm(dv) / dt_s)

    force_proxy = float(
        np.clip(
            contact_prob * (0.55 * mean_curl + 0.30 * aperture_closed + 0.15 * min(1.0, aperture_rate / 0.25)),
            0.0,
            1.0,
        )
    )

    slip_proxy = 0.0
    if contact_prob > 0.35 and held_obj_vel is not None:
        rel_spd = float(np.linalg.norm(np.asarray(hand.wrist_vel_world_mps, dtype=np.float64) - held_obj_vel))
        slip_proxy = float(np.clip(rel_spd / max(cfg.slip_velocity_threshold_mps * 3.0, 1e-4), 0.0, 1.0))

    impact_proxy = 0.0
    if contact_prob > 0.25 and accel_mag > cfg.impact_accel_threshold_mps2 * 0.5:
        impact_proxy = float(np.clip(accel_mag / max(cfg.impact_accel_threshold_mps2 * 2.5, 1e-4), 0.0, 1.0))

    compliance: Literal["rigid", "semi_rigid", "deformable", "unknown"] = "unknown"
    if contact_prob >= 0.45:
        if aperture_rate > 0.08:
            compliance = "deformable"
        elif aperture_rate > 0.02:
            compliance = "semi_rigid"
        else:
            compliance = "rigid"

    uncertainty = float(np.clip(0.35 + 0.45 * (1.0 - hand.confidence * hand.visibility), 0.35, 0.95))

    return TactileHandChannel(
        source="derived_rgb_proxy",
        contact_prob=round(contact_prob, 4),
        contact_zones_6d=zones,
        normal_force_proxy_01=round(force_proxy, 4),
        slip_proxy_01=round(slip_proxy, 4),
        impact_proxy_01=round(impact_proxy, 4),
        compliance_prior=compliance,
        uncertainty=round(uncertainty, 4),
    )


def populate_rgb_tactile_proxies(frames: list[FrameAnnotation], cfg: TactileConfig) -> None:
    """Populate synchronized derived RGB tactile proxies across all frames 0..N-1."""
    if not cfg.enable_rgb_proxies or not frames:
        return
    for i, fr in enumerate(frames):
        prev = frames[i - 1] if i > 0 else None
        dt_s = (fr.timestamp_s - prev.timestamp_s) if prev is not None else (1.0 / 30.0)
        fr.tactile_left = _derive_hand_tactile_proxy(
            fr.left_hand,
            prev.left_hand if prev else None,
            fr,
            "left",
            dt_s,
            cfg,
        )
        fr.tactile_right = _derive_hand_tactile_proxy(
            fr.right_hand,
            prev.right_hand if prev else None,
            fr,
            "right",
            dt_s,
            cfg,
        )


def ingest_external_tactile_stream(
    frames: list[FrameAnnotation],
    sensor_path: Path,
) -> dict[str, int | str]:
    """Ingest real customer-supplied tactile glove / FT sensor data (CSV or HDF5) time-synced to frames."""
    path = Path(sensor_path)
    if not path.exists():
        raise FileNotFoundError(f"External tactile file not found: {path}")

    streams: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if path.suffix.lower() == ".csv":
        by_hand: dict[str, list[tuple[float, list[float]]]] = {"left": [], "right": []}
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = float(row["timestamp_s"])
                hand = str(row.get("hand", "right")).strip().lower()
                if hand not in by_hand:
                    continue
                vals = [float(row.get(f"z{k}", 0.0)) for k in range(6)]
                by_hand[hand].append((ts, vals))
        for hand, items in by_hand.items():
            if items:
                items.sort(key=lambda x: x[0])
                streams[hand] = (
                    np.array([it[0] for it in items], dtype=np.float64),
                    np.array([it[1] for it in items], dtype=np.float64),
                )
    elif path.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(path, "r") as hf:
            ts_arr = np.asarray(hf["timestamps_s"][:], dtype=np.float64)
            for hand in ("left", "right"):
                key = f"{hand}_forces"
                if key in hf:
                    streams[hand] = (ts_arr, np.asarray(hf[key][:], dtype=np.float64))
    else:
        raise ValueError(f"Unsupported external tactile format: {path.suffix} (use .csv or .h5/.hdf5)")

    synced_frames = 0
    for fr in frames:
        for hand in ("left", "right"):
            if hand not in streams:
                continue
            ts_arr, vals_arr = streams[hand]
            idx = int(np.argmin(np.abs(ts_arr - fr.timestamp_s)))
            if abs(float(ts_arr[idx]) - fr.timestamp_s) > 0.25:
                continue
            raw_v = vals_arr[idx]
            peak = float(np.max(np.abs(raw_v))) if raw_v.size else 0.0
            norm_v = np.clip(np.abs(raw_v) / max(10.0, peak), 0.0, 1.0)
            contact_prob = float(np.clip(peak / 0.5, 0.0, 1.0))
            channel = TactileHandChannel(
                source="hardware_sensor",
                contact_prob=round(contact_prob, 4),
                contact_zones_6d=[round(float(x), 4) for x in norm_v[:6]],
                normal_force_proxy_01=round(float(np.mean(norm_v[:6])), 4),
                slip_proxy_01=0.0,
                impact_proxy_01=0.0,
                compliance_prior="unknown",
                uncertainty=0.05,
                raw_sensor_values=[round(float(x), 5) for x in raw_v.tolist()],
            )
            if hand == "left":
                fr.tactile_left = channel
            else:
                fr.tactile_right = channel
            synced_frames += 1

    return {"source": "hardware_sensor", "file": str(path), "synced_hand_frames": synced_frames}
