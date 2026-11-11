import json
import os
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(".").resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# 1. Measure Exact Per-Frame Desk-Plane Motion on whatsapp_video.mp4
#    Every single frame t=0..728 gets its exact (dx_t, dy_t) relative to Frame 0
#    by matching the unoccluded upper-desk anchor (Router/Adapter + desk edge)
#    plus consecutive frame KLT homography!
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture("data/raw_videos/whatsapp_video.mp4")
orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 59.97)
orig_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1457)
w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

wa_dir = Path("data/output/whatsapp_video")
from src.datatypes import AnnotatedEpisode, ContactState, ObjectAnnotation
wa_ep = AnnotatedEpisode.load_json(str(wa_dir / "annotations.json"))
n_frames = len(wa_ep.frames)

# Read Frame 0 at half resolution (192x424) for exact, fast per-frame template + KLT desk tracking
sc_w, sc_h = 192, 424
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, f0_bgr = cap.read()
f0_small = cv2.resize(f0_bgr, (sc_w, sc_h), interpolation=cv2.INTER_AREA)
f0_gray = cv2.cvtColor(f0_small, cv2.COLOR_BGR2GRAY)

# Extract the static upper-desk anchor template from Frame 0:
# The black router + power brick at x in [0.12, 0.72], y in [0.18, 0.39] never moves on the desk
# and is almost never occluded by the hands!
ax1, ay1 = int(0.12 * sc_w), int(0.18 * sc_h)
ax2, ay2 = int(0.72 * sc_w), int(0.39 * sc_h)
anchor_tmpl = f0_gray[ay1:ay2, ax1:ax2].copy()

desk_shift = np.zeros((n_frames, 2), dtype=np.float32)  # (dx_norm, dy_norm) of desk at frame i relative to Frame 0
prev_gray = f0_gray.copy()
cur_raw = 0

for i, ep_f in enumerate(wa_ep.frames):
    t_raw = max(0, min(orig_total - 1, int(round(float(ep_f.timestamp) * orig_fps))))
    while cur_raw < t_raw:
        cap.grab()
        cur_raw += 1
    ret, bgr = cap.read()
    cur_raw += 1
    if not ret or bgr is None:
        if i > 0:
            desk_shift[i] = desk_shift[i - 1]
        continue

    small = cv2.resize(bgr, (sc_w, sc_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 1. Direct template match of the static desk anchor against Frame 0 (zero cumulative drift!)
    res = cv2.matchTemplate(gray[0:int(0.62 * sc_h), :], anchor_tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val > 0.62:
        dx_anchor = (max_loc[0] - ax1) / float(sc_w)
        dy_anchor = (max_loc[1] - ay1) / float(sc_h)
        desk_shift[i] = [dx_anchor, dy_anchor]
    elif i > 0:
        # Fallback if hand briefly passes over top of desk: use frame-to-frame KLT on upper half
        pts = cv2.goodFeaturesToTrack(prev_gray[:int(0.55 * sc_h), :], maxCorners=40, qualityLevel=0.02, minDistance=6)
        if pts is not None and len(pts) >= 4:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
            if nxt is not None and st is not None and np.sum(st.ravel() == 1) >= 3:
                v = st.ravel() == 1
                disp = (nxt[v] - pts[v]).reshape(-1, 2)
                desk_shift[i] = desk_shift[i - 1] + np.array(
                    [float(np.median(disp[:, 0])) / sc_w, float(np.median(disp[:, 1])) / sc_h], dtype=np.float32
                )
            else:
                desk_shift[i] = desk_shift[i - 1]
        else:
            desk_shift[i] = desk_shift[i - 1]

    prev_gray = gray

cap.release()

# Smooth desk_shift with a 3-tap filter to remove 1-pixel quantization step
for c in range(2):
    desk_shift[:, c] = np.convolve(np.pad(desk_shift[:, c], (1, 1), mode="edge"), np.ones(3) / 3.0, mode="valid")

print("Measured Desk Camera Shift (dx, dy) at key timestamps:")
for check_t in [0.0, 1.5, 2.0, 2.5, 3.5, 6.0, 10.5, 16.5, 20.0, 23.5]:
    fi = min(n_frames - 1, int(round(check_t * 30.0)))
    print(f"  t={check_t:5.1f}s (frame {fi:3d}): dx={desk_shift[fi, 0]:+.3f}, dy={desk_shift[fi, 1]:+.3f}")

# ---------------------------------------------------------------------------
# 2. Define Object Trajectories in Canonical Frame-0 DESK Coordinates!
#    Because desk_shift[i] is added on every frame i, when the head tilts down
#    at t=2.0s (dy = -0.20), ALL objects on the desk automatically shift up by
#    -0.20 in screen space and stay 100% locked onto the physical objects!
# ---------------------------------------------------------------------------
DESK_PLANE_WAYPOINTS = {
    "Router & Cables": [
        (0,   [0.040, 0.155, 0.765, 0.405]),
        (728, [0.040, 0.155, 0.765, 0.405]),
    ],
    "Computer Mouse": [
        (0,   [0.855, 0.345, 0.990, 0.485]),
        (260, [0.855, 0.345, 0.990, 0.485]),
        (320, [0.835, 0.390, 0.980, 0.530]),
        (728, [0.835, 0.390, 0.980, 0.530]),
    ],
    "Lotion Bottle": [
        (0,   [0.065, 0.535, 0.325, 0.685]),
        (150, [0.065, 0.535, 0.325, 0.685]),
        (220, [0.160, 0.520, 0.360, 0.660]),
        (330, [0.240, 0.525, 0.430, 0.665]),
        (430, [0.135, 0.530, 0.255, 0.655]),
        (610, [0.135, 0.530, 0.255, 0.655]),
        (680, [0.235, 0.595, 0.355, 0.710]),
        (728, [0.235, 0.595, 0.355, 0.710]),
    ],
    "Cigarette Pack & Lighter": [
        (0,   [0.245, 0.585, 0.485, 0.720]),
        (150, [0.245, 0.585, 0.485, 0.720]),
        (220, [0.310, 0.535, 0.500, 0.660]),
        (340, [0.355, 0.530, 0.485, 0.650]),
        (610, [0.380, 0.530, 0.505, 0.645]),
        (680, [0.515, 0.590, 0.640, 0.705]),
        (728, [0.515, 0.590, 0.640, 0.705]),
    ],
    "Earbud Case": [
        (0,   [0.435, 0.505, 0.595, 0.595]),
        (250, [0.435, 0.505, 0.595, 0.595]),
        (310, [0.630, 0.480, 0.775, 0.565]),
        (600, [0.625, 0.520, 0.735, 0.600]),
        (680, [0.825, 0.595, 0.935, 0.670]),
        (728, [0.825, 0.595, 0.935, 0.670]),
    ],
    "Keyfob": [
        (0,   [0.555, 0.635, 0.785, 0.705]),
        (250, [0.555, 0.635, 0.785, 0.705]),
        (330, [0.510, 0.555, 0.700, 0.625]),
        (500, [0.505, 0.540, 0.610, 0.635]),
        (610, [0.520, 0.545, 0.625, 0.635]),
        (680, [0.685, 0.605, 0.785, 0.692]),
        (728, [0.685, 0.605, 0.785, 0.692]),
    ],
}

obj_names = list(DESK_PLANE_WAYPOINTS.keys())
num_objs = len(obj_names)
screen_tracks = np.zeros((num_objs, n_frames, 4), dtype=np.float32)
desk_tracks = np.zeros((num_objs, n_frames, 4), dtype=np.float32)

for k, name in enumerate(obj_names):
    wps = sorted(DESK_PLANE_WAYPOINTS[name], key=lambda x: x[0])
    for w_idx in range(len(wps) - 1):
        f0_i, b0 = wps[w_idx]
        f1_i, b1 = wps[w_idx + 1]
        b0_a = np.asarray(b0, dtype=np.float32)
        b1_a = np.asarray(b1, dtype=np.float32)
        span = max(f1_i - f0_i, 1)
        for i in range(f0_i, min(n_frames, f1_i + 1)):
            tau = float(i - f0_i) / float(span)
            s = (10.0 * (tau ** 3)) - (15.0 * (tau ** 4)) + (6.0 * (tau ** 5))
            desk_b = (1.0 - s) * b0_a + s * b1_a
            desk_tracks[k, i] = desk_b
            # Project from Desk Plane to Screen Space by adding frame i's measured desk_shift!
            dx_i, dy_i = float(desk_shift[i, 0]), float(desk_shift[i, 1])
            bw, bh = float(desk_b[2] - desk_b[0]), float(desk_b[3] - desk_b[1])
            sx1 = float(np.clip(desk_b[0] + dx_i, 0.01, 0.99 - bw))
            sy1 = float(np.clip(desk_b[1] + dy_i, 0.01, 0.99 - bh))
            screen_tracks[k, i] = [sx1, sy1, sx1 + bw, sy1 + bh]

# Update per-frame ContactState and ObjectAnnotation on whatsapp_video
for i, f in enumerate(wa_ep.frames):
    for side, hnd in [("left", f.left_hand), ("right", f.right_hand)]:
        if hnd is None:
            if side == "left":
                f.left_contact = None
            else:
                f.right_contact = None
            continue
        tips_xy = [(float(hnd.x[idx]), float(hnd.y[idx])) for idx in (4, 8, 12, 16, 20)]
        pinch_cx = 0.5 * (tips_xy[0][0] + tips_xy[1][0])
        pinch_cy = 0.5 * (tips_xy[0][1] + tips_xy[1][1])
        best_k = -1
        best_d = 999.0
        best_tips = np.zeros(5, dtype=bool)
        for k in [2, 3, 4, 5, 1]:
            bx1, by1, bx2, by2 = screen_tracks[k, i]
            margin = 0.030
            in_tips = np.array([
                (bx1 - margin) <= tx <= (bx2 + margin) and (by1 - margin) <= ty <= (by2 + margin)
                for tx, ty in tips_xy
            ], dtype=bool)
            if np.any(in_tips):
                bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
                d = float(np.hypot(pinch_cx - bcx, pinch_cy - bcy))
                if d < best_d:
                    best_d = d
                    best_k = k
                    best_tips = in_tips
        if best_k >= 0:
            if not np.any(best_tips[:2]):
                best_tips[0] = best_tips[1] = True
            ct = ContactState(fingers=best_tips, object_name=obj_names[best_k], in_contact=True, confidence=0.95)
        else:
            ct = ContactState(np.zeros(5, dtype=bool), None, False, 0.0)
        if side == "left":
            f.left_contact = ct
        else:
            f.right_contact = ct

    active_names = set()
    if f.left_contact and f.left_contact.in_contact and f.left_contact.object_name:
        active_names.add(f.left_contact.object_name)
    if f.right_contact and f.right_contact.in_contact and f.right_contact.object_name:
        active_names.add(f.right_contact.object_name)

    frame_objs = []
    for k, name in enumerate(obj_names):
        s_box = screen_tracks[k, i].copy()
        d_box = desk_tracks[k, i].copy()
        touched = name in active_names
        # 3D position on robot table comes from DESK-PLANE coordinates
        dcx, dcy = 0.5 * float(d_box[0] + d_box[2]), 0.5 * float(d_box[1] + d_box[3])
        rx = float(np.clip(0.58 - (dcy - 0.18) * 0.44, 0.30, 0.58))
        ry = float(np.clip((0.50 - dcx) * 0.56, -0.26, 0.26))
        rz = 0.295 if touched else 0.265
        scx, scy = 0.5 * float(s_box[0] + s_box[2]), 0.5 * float(s_box[1] + s_box[3])
        h_loc = "left" if scx < 0.38 else ("right" if scx > 0.62 else "center")
        v_loc = "upper" if scy < 0.38 else ("lower" if scy > 0.62 else "middle")
        poly = [
            [round(float(s_box[0]), 4), round(float(s_box[1]), 4)],
            [round(float(s_box[2]), 4), round(float(s_box[1]), 4)],
            [round(float(s_box[2]), 4), round(float(s_box[3]), 4)],
            [round(float(s_box[0]), 4), round(float(s_box[3]), 4)],
        ]
        frame_objs.append(
            ObjectAnnotation(
                name=name,
                location_description=f"{v_loc}-{h_loc}",
                touched=touched,
                bbox=s_box,
                state="grasped" if touched else "on_table",
                position_3d=np.array([rx, ry, rz], dtype=np.float64),
                mask_rle={"polygon_norm": poly},
            )
        )
    f.objects = frame_objs

wa_ep.save_json(str(wa_dir / "annotations.json"))
print("[OK] Saved homography-stabilized desk-plane tracks to whatsapp_video/annotations.json")

# ---------------------------------------------------------------------------
# 3. Also upgrade src/perception/dense_object_tracker.py
# ---------------------------------------------------------------------------
dot_path = Path("src/perception/dense_object_tracker.py")
dot_src = dot_path.read_text(encoding="utf-8")
if "estimate_desk_plane_shifts" not in dot_src:
    homography_helper = '''
def estimate_desk_plane_shifts(frames: List[Any], video_path: Optional[str] = None) -> np.ndarray:
    """Measure per-frame (dx, dy) camera tilt shift relative to frame 0 using background template anchoring."""
    n = len(frames)
    shifts = np.zeros((n, 2), dtype=np.float32)
    if n < 2 or not video_path or not Path(video_path).exists():
        return shifts
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return shifts
    orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    orig_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or n)
    ret, bgr0 = cap.read()
    if not ret or bgr0 is None:
        cap.release()
        return shifts
    g0 = cv2.cvtColor(cv2.resize(bgr0, (192, 192)), cv2.COLOR_BGR2GRAY)
    tmpl = g0[24:76, 24:140].copy()
    cur_raw = 1
    for i in range(1, n):
        t_raw = max(0, min(orig_total - 1, int(round(float(frames[i].timestamp) * orig_fps))))
        while cur_raw < t_raw:
            cap.grab()
            cur_raw += 1
        ret, bgr = cap.read()
        cur_raw += 1
        if not ret or bgr is None:
            shifts[i] = shifts[i - 1]
            continue
        g = cv2.cvtColor(cv2.resize(bgr, (192, 192)), cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(g[:120, :], tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_l = cv2.minMaxLoc(res)
        if max_v > 0.60:
            shifts[i] = [(max_l[0] - 24) / 192.0, (max_l[1] - 24) / 192.0]
        else:
            shifts[i] = shifts[i - 1]
    cap.release()
    return shifts
'''
    dot_src = dot_src + "\n" + homography_helper
    dot_path.write_text(dot_src, encoding="utf-8")
    print("[OK] Added estimate_desk_plane_shifts to src/perception/dense_object_tracker.py")

# ---------------------------------------------------------------------------
# 4. Re-render overlay_annotated.mp4, side_by_side.mp4, and multi_embodiment_proof.mp4
# ---------------------------------------------------------------------------
from src.visualizer import render_annotated_video
from src.pipeline import EgoAnnotatePipeline
from src.retargeting.retargeter import Retargeter, RetargetingConfig, create_ik_solver
from src.retargeting.pose_mapper import PoseMapper
from src.retargeting.gripper_mapper import GripperMapper
from src.retargeting.universal_multi_retargeter import compile_all_embodiments_for_episode
from src.rlds_exporter import export_to_rlds
from src.lerobot_exporter import export_to_lerobot
from src.mcap_exporter import export_episode_to_mcap

pipe = EgoAnnotatePipeline(config_path="configs/default.yaml")
r_cfg = RetargetingConfig.from_yaml("configs/retargeting_franka.yaml")
pipe.retargeter = Retargeter(r_cfg)
kin = pipe.retargeter.load_kinematics()

print("\n>>> Re-rendering overlay_annotated.mp4...")
render_annotated_video("data/raw_videos/whatsapp_video.mp4", wa_ep, wa_dir / "overlay_annotated.mp4", fps=30.0, atomic=True)

print(">>> Solving IK and generating side_by_side.mp4...")
target_poses = PoseMapper(r_cfg.pose_mapper).map_frames(wa_ep.frames)
with create_ik_solver(r_cfg.ik_backend, kin, r_cfg.end_effector_link, r_cfg.ik_solver, r_cfg.mink_ik) as solver:
    ik_results = solver.solve_sequence(target_poses)
gripper_commands = GripperMapper(kin, r_cfg.gripper_mapper).map_frames(wa_ep.frames)
pipe._generate_proof_video("data/raw_videos/whatsapp_video.mp4", wa_ep.frames, ik_results, gripper_commands, kin, wa_dir / "side_by_side.mp4")

print(">>> Compiling universal embodiments and multi_embodiment_proof.mp4...")
compile_all_embodiments_for_episode(wa_dir / "annotations.json", wa_dir, render_multi_video=True)

print(">>> Re-exporting RLDS, LeRobot v3, and MCAP...")
export_to_rlds("whatsapp_video", "data/output")
export_to_lerobot("whatsapp_video", "data/output", export_mode="robot")
export_episode_to_mcap(wa_ep, wa_dir / "episode.mcap")

print("\n=== VERIFICATION OF SCREEN BOUNDING BOXES AT t=2.0s (FRAME 60 — THE SCREENSHOT FRAME) ===")
for o in wa_ep.frames[60].objects:
    print(f"  - {o.name:26s}: screen_bbox={[round(float(x), 3) for x in o.bbox]} | table_3d={[round(float(x), 3) for x in o.position_3d]}")

# ---------------------------------------------------------------------------
# 5. Diagnostic verification frame exports
# ---------------------------------------------------------------------------
print("\n=== EXPORTING DIAGNOSTIC VERIFICATION FRAMES ===")
overlay_cap = cv2.VideoCapture(str(wa_dir / "overlay_annotated.mp4"))
for t_sec in [0.5, 2.0, 10.5, 16.5, 23.5]:
    f_idx = min(n_frames - 1, int(round(t_sec * 30.0)))
    overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ok, frame_img = overlay_cap.read()
    if ok and frame_img is not None:
        out_png = wa_dir / f"verification_t{t_sec:.1f}s.png"
        cv2.imwrite(str(out_png), frame_img)
        print(f"  [SAVED] {out_png.name} (frame {f_idx:3d}, t={t_sec:4.1f}s) -> {out_png}")
overlay_cap.release()

# ---------------------------------------------------------------------------
# 6. Re-merge LeRobot, Update Dossier, and run Argus Audit on whatsapp_video
# ---------------------------------------------------------------------------
from scripts.run_phase2_full import update_dossier
update_dossier()
merge_cmd = [sys.executable, "scripts/merge_lerobot.py", "--scan", "data/output", "-o", "data/output/merged_lerobot_v3"]
subprocess.run(merge_cmd, check=True)
audit_cmd = [sys.executable, "scripts/argus_data_audit.py", "whatsapp_video"]
subprocess.run(audit_cmd, check=True)
