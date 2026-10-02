#!/usr/bin/env python3
"""Phase E hero clip: side-by-side human | MuJoCo verification replay.

Renders the best weld-grasp trial of a physics-verified episode and writes a
side-by-side H.264 clip (human video | robot verification replay) with a
verification HUD.

Usage:
    python scripts/hero_clip.py --hdf5 <episode_rlds.hdf5> \\
        --report <physics_verification.json> --video <source.mp4> \\
        --urdf <robot.urdf> --out hero_clip.mp4

REFUSES loudly (exit 2) unless the verification report passed every check —
a hero clip is only meaningful for a verified episode.

Renderer fallback chain (loud at each step):
  1. ``mujoco.renderer.Renderer`` — offscreen MuJoCo replay of the weld trial
     (object rigidly welded to the EE, exactly as the verifier ran it).
  2. PyBullet ``render_robot_frame_pybullet`` (needs the pybullet package).
  3. ``render_robot_skeleton_2d`` (always available; labelled SKELETON FALLBACK).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import cv2
import numpy as np


def refuse(msg: str) -> NoReturn:
    print(f"[hero_clip] REFUSES: {msg}", file=sys.stderr)
    sys.exit(2)


def load_passed_report(report_path: str) -> dict:
    p = Path(report_path)
    if not p.exists():
        refuse(f"report not found: {report_path}")
    try:
        report = json.loads(p.read_text())
    except Exception as exc:
        refuse(f"could not parse report {report_path}: {exc}")
    if not report.get("passed", False):
        failed = [c["name"] for c in report.get("checks", []) if not c.get("passed")]
        refuse(
            "episode is NOT physics-verified "
            f"({len(failed)} failed check(s): {failed[:5]}). "
            "A hero clip is only rendered for a fully verified episode."
        )
    windows = report.get("windows", [])
    if not windows:
        refuse("report passed but contains no grasp windows.")
    return report


def read_episode_arrays(hdf5_path: str):
    """Read robot fields with the exporter-written keys (legacy fallback)."""
    import h5py
    p = Path(hdf5_path)
    if not p.exists():
        refuse(f"HDF5 not found: {hdf5_path}")
    with h5py.File(str(p), "r") as f:
        keys = list(f.keys())
        if not keys:
            refuse(f"HDF5 has no episode groups: {hdf5_path}")
        ep = keys[0]
        obs = f[ep]["steps"]["observation"]
        q = np.asarray(obs["robot_joint_angles"], dtype=float)
        g = np.asarray(obs["robot_gripper_opening_m"], dtype=float)
        steps = f[ep]["steps"]
        if "robot_reachable" in obs:
            r = np.asarray(obs["robot_reachable"], dtype=bool)
        elif "robot_reachable" in steps:
            r = np.asarray(steps["robot_reachable"], dtype=bool)
        else:
            refuse(f"{hdf5_path}: no robot_reachable in steps/observation or steps")
    return q, g, r


def render_mujoco_trial(urdf_path, q, g, window, frame_indices):
    """Replay the weld trial in MuJoCo, capturing offscreen renders."""
    from src.retargeting.episode_verifier import (
        build_verification_scene,
        replay_frame,
        activate_weld,
        deactivate_weld,
    )
    try:
        import mujoco as mj
    except ImportError:
        raise ImportError("mujoco not installed — cannot do the MuJoCo replay")
    try:
        from mujoco import renderer as mj_renderer
    except ImportError as exc:
        raise RuntimeError(f"mujoco.renderer unavailable: {exc}") from exc

    model, info = build_verification_scene(urdf_path)
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    data.eq_active[info.weld_eq_id] = 0
    qa = info.obj_qpos_adr
    data.qpos[qa:qa + 3] = np.asarray(window["object_spawn_pos"], float)
    data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
    mj.mj_forward(model, data)

    i0, gf = int(window["start_idx"]), int(window["grasp_frame_idx"])
    for i in range(i0, gf):
        replay_frame(mj, model, data, info, q[i], float(g[i, 0]))
    replay_frame(mj, model, data, info, q[gf], float(g[gf, 0]))
    mj.mj_forward(model, data)
    activate_weld(mj, model, data, info)

    renderer = mj_renderer.Renderer(model, height=480, width=640)
    frames = []
    try:
        for i in frame_indices:
            replay_frame(mj, model, data, info, q[i], float(g[i, 0]))
            renderer.update_scene(data)
            rgb = renderer.render()
            frames.append(cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
    finally:
        deactivate_weld(data, info)
        renderer.close()
    print(f"[hero_clip] MuJoCo replay rendered ({len(frames)} frames, weld trial).")
    return frames


def render_pybullet_frames(urdf_path, q, g, frame_indices):
    """Fallback 2: PyBullet CPU rendering of the joint trajectory."""
    import pybullet as pb
    import pybullet_data
    from src.retargeting.urdf_loader import URDFLoader
    from scripts.validate_retargeting import (
        render_robot_frame_pybullet,
        render_robot_skeleton_2d,
    )

    kin = URDFLoader().load(urdf_path=urdf_path)
    client_id = pb.connect(pb.DIRECT)
    try:
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        robot_id = pb.loadURDF(kin.urdf_path, basePosition=[0, 0, 0],
                               useFixedBase=True, physicsClientId=client_id)
        test = render_robot_frame_pybullet(
            pb, client_id, robot_id, kin, q[frame_indices[0]],
            float(g[frame_indices[0], 0]))
        if test is None:
            raise RuntimeError("pybullet test render returned None")
        frames = []
        for i in frame_indices:
            img = render_robot_frame_pybullet(
                pb, client_id, robot_id, kin, q[i], float(g[i, 0]))
            frames.append(img if img is not None
                          else render_robot_skeleton_2d(q[i], kin))
    finally:
        pb.disconnect(client_id)
    print(f"[hero_clip] PyBullet fallback rendered ({len(frames)} frames).")
    return frames


def render_skeleton_frames(q, frame_indices):
    """Fallback 3: 2D skeleton (always available)."""
    from scripts.validate_retargeting import render_robot_skeleton_2d
    kin = SimpleNamespace(robot_name="ROBOT")
    frames = [render_robot_skeleton_2d(q[i], kin) for i in frame_indices]
    print(f"[hero_clip] 2D skeleton fallback rendered ({len(frames)} frames).")
    return frames


def add_verification_banner(comp: np.ndarray, text: str) -> np.ndarray:
    overlay = comp.copy()
    cv2.rectangle(overlay, (0, 0), (comp.shape[1], 34), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, comp, 0.45, 0, comp)
    cv2.putText(comp, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0, 255, 0), 2, cv2.LINE_AA)
    return comp


def encode_h264(tmp_path: Path, output_path: Path) -> None:
    """Re-encode mp4v -> H.264 (libx264 + yuv420p + faststart), pipeline parity."""
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "fast",
        str(output_path),
    ]
    try:
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg H.264 re-encode failed (returncode={result.returncode}):\n"
                f"{result.stderr.decode()}")
        tmp_path.unlink(missing_ok=True)
        print(f"[hero_clip] H.264 clip: {output_path}")
    except Exception as e:
        tmp_path.rename(output_path)
        print(f"[hero_clip] WARNING: H.264 re-encode failed ({e}). "
              f"Kept mp4v at {output_path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", required=True, help="Exported episode_rlds.hdf5")
    ap.add_argument("--report", required=True, help="physics_verification.json")
    ap.add_argument("--video", required=True, help="Source human video")
    ap.add_argument("--urdf", required=True, help="Robot URDF (for the replay scene)")
    ap.add_argument("--out", default="hero_clip.mp4", help="Output clip path")
    ap.add_argument("--fps", type=float, default=30.0, help="Output FPS")
    a = ap.parse_args()

    # Gate FIRST, before any heavy import — the refusal path stays hermetic.
    report = load_passed_report(a.report)
    windows = report["windows"]
    best = max(windows, key=lambda w: float(w.get("peak_lift_m", 0.0)))
    i0, i1 = int(best["start_idx"]), int(best["end_idx"])
    gf = int(best["grasp_frame_idx"])
    print(f"[hero_clip] best window [{i0},{i1}) grasp_frame={gf} "
          f"lift={best.get('peak_lift_m', 0):.4f} m weld_held={best.get('weld_held')}")

    q, g, r = read_episode_arrays(a.hdf5)
    n = len(q)
    frame_indices = list(range(gf, min(i1 + 15, n)))
    if not frame_indices:
        refuse("no trial frames to render.")

    # Renderer fallback chain.
    robot_frames = None
    renderer_name = "mujoco"
    try:
        robot_frames = render_mujoco_trial(a.urdf, q, g, best, frame_indices)
    except Exception as e:
        print(f"[hero_clip] MuJoCo renderer unavailable ({e}); trying PyBullet.",
              file=sys.stderr)
        renderer_name = "pybullet"
        try:
            robot_frames = render_pybullet_frames(a.urdf, q, g, frame_indices)
        except Exception as e2:
            print(f"[hero_clip] PyBullet unavailable ({e2}); using 2D skeleton.",
                  file=sys.stderr)
            renderer_name = "skeleton-2d"
            robot_frames = render_skeleton_frames(q, frame_indices)

    from scripts.validate_retargeting import compose_side_by_side
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        refuse(f"could not open video: {a.video}")
    src_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    banner = (f"PHYSICS VERIFIED  |  lift {best.get('peak_lift_m', 0):.3f} m  |  "
              f"weld held  |  window [{i0},{i1})  |  {renderer_name}")
    out_path = Path(a.out)
    tmp_path = out_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    try:
        for k, i in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(i, max(src_n - 1, 0)))
            ret, human_img = cap.read()
            if not ret or human_img is None:
                human_img = np.zeros((480, 640, 3), np.uint8)
            comp = compose_side_by_side(
                human_img, robot_frames[k], i, i / a.fps,
                bool(r[i]), float(g[i, 0]), "hero_clip_weld",
                robot_name="MUJOCO-VERIFY" if renderer_name == "mujoco" else "ROBOT",
            )
            comp = add_verification_banner(comp, banner)
            if writer is None:
                h, w = comp.shape[:2]
                writer = cv2.VideoWriter(str(tmp_path), fourcc, a.fps, (w, h))
            writer.write(comp)
    finally:
        if writer is not None:
            writer.release()
        cap.release()
    encode_h264(tmp_path, out_path)


if __name__ == "__main__":
    main()
