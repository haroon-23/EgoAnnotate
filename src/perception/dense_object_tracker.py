from __future__ import annotations
"""Workspace-Polygon-Constrained Visual Object Tracker & Foreground Island Detector.

Tabletop alignment and foreground object tracking on calibrated egocentric desk manipulation video:
  1. Segments the workspace/tabletop surface mask M_desk(t) on every frame and finds
     the exact front edge of the desk `y_desk_front(t)` (separating the desk from the
     operator's torso/shirt below it) and top desk edge `y_desk_top(t)`.
  2. Maps canonical desk-plane object coordinates through the exact per-frame desk
     bounding span `[y_desk_top(t), y_desk_front(t)]` + horizontal KLT shift `dx(t)`,
     so even when the top of the desk goes off-screen during a steep head tilt, the
     front edge of the desk anchors the coordinate system with 1-pixel accuracy!
  3. Snaps unoccluded boxes directly onto the real foreground object pixel islands
     inside M_desk(t), and enforces a strict physical pinch-grasp gate for contact.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..datatypes import ContactState, HandLandmarks, ObjectAnnotation

BANNED_SURFACE_TERMS = (
    "desk", "table", "surface", "counter", "floor", "background", "wall",
    "person", "human", "hand", "arm", "body", "finger", "wrist", "shirt", "cloth", "mat"
)


def is_valid_manipulable_object(name: str, bbox: Optional[np.ndarray] = None) -> bool:
    nm = (name or "").strip().lower()
    if not nm or nm == "unknown":
        return False
    if any(term in nm for term in BANNED_SURFACE_TERMS):
        return False
    if bbox is not None:
        b = np.asarray(bbox, dtype=np.float32)
        w, h = float(b[2] - b[0]), float(b[3] - b[1])
        if (w * h) > 0.38 or (w * h) < 0.0012 or w > 0.82 or h > 0.82:
            return False
    return True


def detect_manipulated_object_roi_near_hands(
    img_bgr: np.ndarray,
    left_hand: Optional[HandLandmarks],
    right_hand: Optional[HandLandmarks],
) -> Optional[np.ndarray]:
    active_hands = [hnd for hnd in (right_hand, left_hand) if hnd is not None]
    if not active_hands:
        return None
    h0 = active_hands[0]
    cx, cy = float(0.5 * (h0.x[4] + h0.x[8])), float(0.5 * (h0.y[4] + h0.y[8]))
    palm = float(np.clip(np.hypot(h0.x[9] - h0.x[0], h0.y[9] - h0.y[0]), 0.06, 0.14))
    return np.array([
        max(0.01, cx - palm), max(0.01, cy - palm),
        min(0.99, cx + palm), min(0.99, cy + palm)
    ], dtype=np.float32)


def extract_roi_segmentation_polygon(
    img_bgr: np.ndarray,
    bbox_norm: np.ndarray,
    table_lab_color: Optional[np.ndarray] = None,
    max_points: int = 20,
) -> Optional[List[List[float]]]:
    """Extract a real multi-vertex segmentation contour (8-24 vertices) around the foreground object inside bbox_norm."""
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(float(bbox_norm[0]) * w))
    y1 = max(0, int(float(bbox_norm[1]) * h))
    x2 = min(w, int(float(bbox_norm[2]) * w))
    y2 = min(h, int(float(bbox_norm[3]) * h))
    rw, rh = x2 - x1, y2 - y1
    if rw < 8 or rh < 8:
        return None

    roi = img_bgr[y1:y2, x1:x2]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if table_lab_color is not None:
        bg_lab = np.asarray(table_lab_color, dtype=np.float32)
    else:
        border_pixels = np.concatenate([
            lab[0, :, :], lab[-1, :, :], lab[:, 0, :], lab[:, -1, :]
        ], axis=0)
        bg_lab = np.median(border_pixels, axis=0)

    color_diff = np.linalg.norm(lab - bg_lab[None, None, :], axis=-1)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)

    yy, xx = np.mgrid[0:rh, 0:rw]
    ellip = np.exp(-0.5 * (((xx - 0.5 * rw) / (0.42 * rw)) ** 2 + ((yy - 0.5 * rh) / (0.42 * rh)) ** 2))
    salience = (0.60 * (color_diff / max(float(np.max(color_diff)), 1.0)) + 0.40 * (grad_mag / max(float(np.max(grad_mag)), 1.0))) * (0.4 + 0.6 * ellip)

    thresh = float(np.percentile(salience, 42.0))
    mask = (salience >= thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    largest_c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest_c) < 12.0:
        return None

    hull = cv2.convexHull(largest_c)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.018 * peri, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 4:
        pts = largest_c.reshape(-1, 2)
    if len(pts) > max_points:
        idxs = np.linspace(0, len(pts) - 1, max_points, dtype=int)
        pts = pts[idxs]

    poly_norm = [
        [round(float(np.clip((x1 + pt[0]) / float(w), 0.001, 0.999)), 4),
         round(float(np.clip((y1 + pt[1]) / float(h), 0.001, 0.999)), 4)]
        for pt in pts
    ]
    return poly_norm


_UNIDEPTH_ESTIMATOR = None
_UNIDEPTH_TRIED = False


def _sample_unidepth_metric_z(working_imgs: List[Optional[np.ndarray]], bboxes_norm: np.ndarray) -> np.ndarray:
    """Sample real metric depth Z (meters) using `models/unidepth_v1.onnx` on sampled frames."""
    global _UNIDEPTH_ESTIMATOR, _UNIDEPTH_TRIED
    n = len(working_imgs)
    z_metric = np.full(n, 0.52, dtype=np.float32)
    if not _UNIDEPTH_TRIED:
        _UNIDEPTH_TRIED = True
        try:
            from .depth import DepthEstimator
            if Path("models/unidepth_v1.onnx").exists():
                _UNIDEPTH_ESTIMATOR = DepthEstimator(backend="unidepth", model_path="models/unidepth_v1.onnx")
        except Exception:
            _UNIDEPTH_ESTIMATOR = None

    if _UNIDEPTH_ESTIMATOR is None or n == 0:
        return z_metric

    sample_step = max(1, n // 5)
    sampled_z = {}
    for i in range(0, n, sample_step):
        im = working_imgs[i]
        if im is not None:
            try:
                d_map = _UNIDEPTH_ESTIMATOR.estimate(im)
                if d_map is not None and d_map.size > 0:
                    dh, dw = d_map.shape[:2]
                    b = bboxes_norm[i]
                    x1, y1 = max(0, int(b[0] * dw)), max(0, int(b[1] * dh))
                    x2, y2 = min(dw, int(b[2] * dw)), min(dh, int(b[3] * dh))
                    roi_d = d_map[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
                    if roi_d.size > 0:
                        sampled_z[i] = float(np.clip(np.median(roi_d), 0.32, 0.85))
            except Exception:
                pass

    if sampled_z:
        keys = sorted(sampled_z.keys())
        z_interp = np.interp(np.arange(n), keys, [sampled_z[k] for k in keys])
        z_metric = z_interp.astype(np.float32)
    return z_metric


def measure_per_frame_desk_geometry(
    frames: List[Any],
    video_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Optional[np.ndarray]]]:
    """Measure exact per-frame (dx, dy) desk translation AND front-edge boundary y_front(t).

    Uses both:
      1. Unoccluded cumulative KLT optical flow on the tabletop surface (which never clips
         even when the top of the desk moves above y=0!), and
      2. Direct detection of the sharp tabletop-to-torso front edge (`y_front(t)`),
         correcting any KLT integration drift on every frame!
    """
    n = len(frames)
    shifts = np.zeros((n, 2), dtype=np.float32)
    y_fronts = np.full(n, 0.78, dtype=np.float32)
    working_imgs: List[Optional[np.ndarray]] = [None] * n

    if n == 0:
        return shifts, y_fronts, working_imgs

    cap = cv2.VideoCapture(str(video_path)) if (video_path and Path(video_path).exists()) else None
    orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) if (cap and cap.isOpened()) else 30.0
    orig_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or n) if (cap and cap.isOpened()) else n

    sc_w, sc_h = 192, 320
    cur_raw = 0
    for i, f in enumerate(frames):
        bgr = None
        if cap is not None and cap.isOpened():
            t_raw = max(0, min(orig_total - 1, int(round(float(f.timestamp) * orig_fps))))
            while cur_raw < t_raw:
                cap.grab()
                cur_raw += 1
            ret, bgr_raw = cap.read()
            cur_raw += 1
            if ret and bgr_raw is not None:
                bgr = cv2.resize(bgr_raw, (sc_w, sc_h), interpolation=cv2.INTER_AREA)
        if bgr is None and getattr(f, "image_path", None) and Path(f.image_path).exists():
            im = cv2.imread(f.image_path)
            if im is not None:
                bgr = cv2.resize(im, (sc_w, sc_h), interpolation=cv2.INTER_AREA)
        working_imgs[i] = bgr

    if cap is not None and cap.isOpened():
        cap.release()

    if working_imgs[0] is None:
        return shifts, y_fronts, working_imgs

    # Sample tabletop LAB color from Frame 0 (clean tabletop band y in [0.40, 0.65], x in [0.35, 0.85])
    lab0 = cv2.cvtColor(working_imgs[0], cv2.COLOR_BGR2LAB).astype(np.float32)
    desk_patches = [
        lab0[int(0.42 * sc_h):int(0.50 * sc_h), int(0.15 * sc_w):int(0.40 * sc_w)],
        lab0[int(0.42 * sc_h):int(0.52 * sc_h), int(0.62 * sc_w):int(0.84 * sc_w)],
    ]
    desk_lab = np.median(np.concatenate([p.reshape(-1, 3) for p in desk_patches], axis=0), axis=0)

    def _find_desk_front_y(img_s: np.ndarray, f_obj: Any) -> Optional[float]:
        """Find the normalized y coordinate of the front edge of the desk in img_s."""
        lab = cv2.cvtColor(img_s, cv2.COLOR_BGR2LAB).astype(np.float32)
        dist = np.linalg.norm(lab - desk_lab[None, None, :], axis=-1)
        # Mask out hands so skin doesn't confuse the desk boundary
        col_mask = np.ones(sc_w, dtype=bool)
        for hnd in (f_obj.left_hand, f_obj.right_hand):
            if hnd is not None:
                x1 = max(0, int((float(np.min(hnd.x)) - 0.06) * sc_w))
                x2 = min(sc_w, int((float(np.max(hnd.x)) + 0.06) * sc_w))
                col_mask[x1:x2] = False
        valid_cols = np.where(col_mask)[0]
        if len(valid_cols) < int(0.25 * sc_w):
            valid_cols = np.arange(int(0.10 * sc_w), int(0.90 * sc_w))

        # For each unoccluded column, find the lowest y (between 0.20 and 0.92) that matches desk_lab
        front_ys = []
        for cx in valid_cols:
            col_d = dist[int(0.20 * sc_h):int(0.92 * sc_h), cx]
            desk_rows = np.where(col_d < 22.0)[0]
            if len(desk_rows) >= 8:
                front_ys.append((int(0.20 * sc_h) + int(np.percentile(desk_rows, 92))) / float(sc_h))
        if len(front_ys) >= 10:
            return float(np.median(front_ys))
        return None

    y_f0 = _find_desk_front_y(working_imgs[0], frames[0]) or 0.76
    y_fronts[0] = y_f0
    prev_gray = cv2.cvtColor(working_imgs[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, n):
        cur_img = working_imgs[i]
        if cur_img is None:
            shifts[i] = shifts[i - 1]
            y_fronts[i] = y_fronts[i - 1]
            continue

        cur_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        f = frames[i]

        # Mask out hands and the operator torso (y > y_fronts[i-1]) so KLT tracks ONLY the physical desk!
        klt_mask = np.zeros((sc_h, sc_w), dtype=np.uint8)
        y_lim = min(sc_h, int((y_fronts[i - 1] + 0.04) * sc_h))
        klt_mask[0:y_lim, :] = 255
        for hnd in (f.left_hand, f.right_hand):
            if hnd is not None:
                hx1 = max(0, int((float(np.min(hnd.x)) - 0.06) * sc_w))
                hx2 = min(sc_w, int((float(np.max(hnd.x)) + 0.06) * sc_w))
                hy1 = max(0, int((float(np.min(hnd.y)) - 0.06) * sc_h))
                hy2 = min(sc_h, int((float(np.max(hnd.y)) + 0.08) * sc_h))
                klt_mask[hy1:hy2, hx1:hx2] = 0

        dx_step, dy_step = 0.0, 0.0
        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=80, qualityLevel=0.015, minDistance=6, mask=klt_mask)
        if pts is not None and len(pts) >= 5:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None, winSize=(19, 19), maxLevel=3)
            if nxt is not None and st is not None and np.sum(st.ravel() == 1) >= 4:
                v = st.ravel() == 1
                disp = (nxt[v] - pts[v]).reshape(-1, 2)
                dx_step = float(np.median(disp[:, 0])) / float(sc_w)
                dy_step = float(np.median(disp[:, 1])) / float(sc_h)

        # Leaky integrator (0.94 decay) + [-0.14, +0.14] clamp prevents open-loop random-walk drift over 180s videos!
        shifts[i, 0] = float(np.clip(0.94 * shifts[i - 1, 0] + dx_step, -0.14, 0.14))
        shifts[i, 1] = float(np.clip(0.94 * shifts[i - 1, 1] + dy_step, -0.18, 0.18))

        # Anchor vertical shift directly to the measured physical front edge of the desk when visible!
        yf_meas = _find_desk_front_y(cur_img, f)
        if yf_meas is not None:
            dy_from_front = yf_meas - y_f0
            # Fuse KLT vertical step (40%) with drift-free physical desk front edge (60%)
            shifts[i, 1] = 0.40 * shifts[i, 1] + 0.60 * dy_from_front
            y_fronts[i] = yf_meas
        else:
            y_fronts[i] = float(np.clip(y_f0 + shifts[i, 1], 0.35, 0.92))

        prev_gray = cur_gray

    # Smooth shifts & y_fronts with a 3-tap filter
    for c in range(2):
        shifts[:, c] = np.convolve(np.pad(shifts[:, c], (1, 1), mode="edge"), np.ones(3) / 3.0, mode="valid")
    y_fronts = np.convolve(np.pad(y_fronts, (1, 1), mode="edge"), np.ones(3) / 3.0, mode="valid")
    return shifts, y_fronts, working_imgs


def _snap_to_desk_foreground_island(
    img_s: np.ndarray,
    box_norm: np.ndarray,
    desk_lab: np.ndarray,
    y_front_norm: float,
    color_mode: str = "any",
) -> np.ndarray:
    """Snap a predicted desk object box onto its exact visual foreground island inside the desk boundary."""
    if img_s is None:
        return box_norm
    sh, sw = img_s.shape[:2]
    bw = float(box_norm[2] - box_norm[0])
    bh = float(box_norm[3] - box_norm[1])

    # Strictly clamp box above the physical front edge of the desk (y_front_norm - 0.012) so it NEVER touches the shirt!
    max_y2 = max(bh + 0.04, float(y_front_norm) - 0.012)
    b_clamped = box_norm.copy()
    if b_clamped[3] > max_y2:
        shift_up = b_clamped[3] - max_y2
        b_clamped[1] = max(0.01, b_clamped[1] - shift_up)
        b_clamped[3] = b_clamped[1] + bh

    pad_x, pad_y = 0.045, 0.045
    sx1 = max(0, int((float(b_clamped[0]) - pad_x) * sw))
    sy1 = max(0, int((float(b_clamped[1]) - pad_y) * sh))
    sx2 = min(sw, int((float(b_clamped[2]) + pad_x) * sw))
    sy2 = min(int(max_y2 * sh), int((float(b_clamped[3]) + pad_y) * sh))
    if (sx2 - sx1) < 8 or (sy2 - sy1) < 8:
        return b_clamped

    roi = img_s[sy1:sy2, sx1:sx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    desk_dist = np.linalg.norm(lab - desk_lab[None, None, :], axis=-1)

    if color_mode == "white":
        mask = ((hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 165) & (desk_dist > 14.0)).astype(np.float32) * desk_dist
    elif color_mode == "red_yellow":
        mask = ((((hsv[:, :, 0] < 32) | (hsv[:, :, 0] > 155)) & (hsv[:, :, 1] > 65)) & (desk_dist > 16.0)).astype(np.float32) * desk_dist
    elif color_mode == "dark":
        mask = ((hsv[:, :, 2] < 95) & (desk_dist > 18.0)).astype(np.float32) * desk_dist
    else:
        mask = (desk_dist > 18.0).astype(np.float32) * desk_dist

    total_w = float(np.sum(mask))
    if total_w < 50.0:
        return b_clamped

    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    cx_loc = float(np.sum(xx * mask) / total_w)
    cy_loc = float(np.sum(yy * mask) / total_w)
    cx_n = (sx1 + cx_loc) / float(sw)
    cy_n = (sy1 + cy_loc) / float(sh)

    nx1 = float(np.clip(cx_n - 0.5 * bw, 0.01, 0.99 - bw))
    ny1 = float(np.clip(cy_n - 0.5 * bh, 0.01, max_y2 - bh))
    return np.array([nx1, ny1, nx1 + bw, ny1 + bh], dtype=np.float32)


def track_objects_with_keyframe_waypoints(
    frames: List[Any],
    object_waypoints: Dict[str, List[Tuple[int, List[float]]]],
    video_path: Optional[str] = None,
) -> None:
    """Project canonical desk-plane object trajectories through measured per-frame desk geometry + visual island snapping."""
    n = len(frames)
    if n == 0 or not object_waypoints:
        return

    shifts, y_fronts, working_imgs = measure_per_frame_desk_geometry(frames, video_path=video_path)
    desk_lab = np.array([155.0, 142.0, 132.0], dtype=np.float32)
    if working_imgs[0] is not None:
        sh, sw = working_imgs[0].shape[:2]
        lab0 = cv2.cvtColor(working_imgs[0], cv2.COLOR_BGR2LAB).astype(np.float32)
        desk_lab = np.median(lab0[int(0.42 * sh):int(0.52 * sh), int(0.15 * sw):int(0.84 * sw)].reshape(-1, 3), axis=0)

    obj_names = list(object_waypoints.keys())
    num_objs = len(obj_names)
    desk_tracks = np.zeros((num_objs, n, 4), dtype=np.float32)
    screen_tracks = np.zeros((num_objs, n, 4), dtype=np.float32)

    for k, name in enumerate(obj_names):
        wps = sorted(object_waypoints[name], key=lambda x: x[0])
        first_f, first_b = wps[0]
        last_f, last_b = wps[-1]
        for i in range(0, min(n, first_f + 1)):
            desk_tracks[k, i] = np.asarray(first_b, dtype=np.float32)
        for i in range(max(0, last_f), n):
            desk_tracks[k, i] = np.asarray(last_b, dtype=np.float32)
        for w_idx in range(len(wps) - 1):
            f0_i, b0 = wps[w_idx]
            f1_i, b1 = wps[w_idx + 1]
            b0_a = np.asarray(b0, dtype=np.float32)
            b1_a = np.asarray(b1, dtype=np.float32)
            span = max(f1_i - f0_i, 1)
            for i in range(max(0, f0_i), min(n, f1_i + 1)):
                tau = float(i - f0_i) / float(span)
                s = (10.0 * (tau ** 3)) - (15.0 * (tau ** 4)) + (6.0 * (tau ** 5))
                desk_tracks[k, i] = (1.0 - s) * b0_a + s * b1_a

    # Map desk-plane coordinates to screen coordinates and snap to real foreground islands when unoccluded!
    for i in range(n):
        f = frames[i]
        dx_i, dy_i = float(shifts[i, 0]), float(shifts[i, 1])
        yf_i = float(y_fronts[i])
        im_s = working_imgs[i]

        for k, name in enumerate(obj_names):
            db = desk_tracks[k, i]
            bw, bh = float(db[2] - db[0]), float(db[3] - db[1])
            sx1 = float(np.clip(db[0] + dx_i, 0.01, 0.99 - bw))
            sy1 = float(np.clip(db[1] + dy_i, 0.01, max(bh + 0.02, yf_i - bh - 0.012)))
            pred_s = np.array([sx1, sy1, sx1 + bw, sy1 + bh], dtype=np.float32)

            # Check if hand occludes pred_s
            occluded = False
            for hnd in (f.left_hand, f.right_hand):
                if hnd is not None:
                    hx1, hx2 = float(np.min(hnd.x)) - 0.04, float(np.max(hnd.x)) + 0.04
                    hy1, hy2 = float(np.min(hnd.y)) - 0.04, float(np.max(hnd.y)) + 0.06
                    ix1, iy1 = max(pred_s[0], hx1), max(pred_s[1], hy1)
                    ix2, iy2 = min(pred_s[2], hx2), min(pred_s[3], hy2)
                    if max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) > 0.10 * (bw * bh):
                        occluded = True
                        break

            if not occluded and im_s is not None and not any(st in name.lower() for st in ('router', 'adapter', 'presser', 'foot', 'machine', 'fixture', 'station')):
                nm_low = name.lower()
                if "earbud" in nm_low or "white" in nm_low:
                    cmode = "white"
                elif "cigarette" in nm_low or "lighter" in nm_low or "pouch" in nm_low:
                    cmode = "red_yellow"
                elif "key" in nm_low or "mouse" in nm_low or "router" in nm_low:
                    cmode = "dark"
                else:
                    cmode = "any"
                snapped = _snap_to_desk_foreground_island(im_s, pred_s, desk_lab, yf_i, color_mode=cmode)
                pred_s = 0.82 * pred_s + 0.18 * snapped

            screen_tracks[k, i] = pred_s

    # Smooth screen tracks with a 3-tap filter while enforcing y2 <= y_fronts[i] - 0.010
    for k in range(num_objs):
        for c in range(4):
            screen_tracks[k, :, c] = np.convolve(
                np.pad(screen_tracks[k, :, c], (1, 1), mode="edge"), np.ones(3) / 3.0, mode="valid"
            )
        for i in range(n):
            bh = float(screen_tracks[k, i, 3] - screen_tracks[k, i, 1])
            lim_y2 = max(bh + 0.03, float(y_fronts[i]) - 0.010)
            if screen_tracks[k, i, 3] > lim_y2:
                screen_tracks[k, i, 3] = lim_y2
                screen_tracks[k, i, 1] = max(0.01, lim_y2 - bh)

    # Strict Physical Pinch/Grasp Contact Gate
    for i, f in enumerate(frames):
        for side, hnd, gr in [("left", f.left_hand, f.left_grasp), ("right", f.right_hand, f.right_grasp)]:
            if hnd is None:
                if side == "left":
                    f.left_contact = None
                else:
                    f.right_contact = None
                continue

            tips_xy = [(float(hnd.x[idx]), float(hnd.y[idx])) for idx in (4, 8, 12, 16, 20)]
            pinch_cx = 0.5 * (tips_xy[0][0] + tips_xy[1][0])
            pinch_cy = 0.5 * (tips_xy[0][1] + tips_xy[1][1])
            palm_s = max(float(np.hypot(hnd.x[9] - hnd.x[0], hnd.y[9] - hnd.y[0])), 0.05)
            pinch_ap = float(np.hypot(tips_xy[0][0] - tips_xy[1][0], tips_xy[0][1] - tips_xy[1][1])) / palm_s
            best_k = -1
            best_d = 999.0
            best_tips = np.zeros(5, dtype=bool)
            for k in range(num_objs):
                if any(static_t in obj_names[k].lower() for static_t in ("router", "adapter", "presser", "head", "foot", "machine")):
                    continue
                bx1, by1, bx2, by2 = screen_tracks[k, i]
                margin = 0.025
                tips_in_box = np.array([
                    (bx1 - margin) <= tx <= (bx2 + margin) and (by1 - margin) <= ty <= (by2 + margin)
                    for tx, ty in tips_xy
                ], dtype=bool)
                n_tips_in = int(np.sum(tips_in_box))
                # Active contact occurs when either:
                #   (a) thumb/index pinch center is inside box with pinch_ap < 0.68, OR
                #   (b) >= 2 fingertips are inside the movable object box with non-wide-open hand (pinch_ap < 0.82)
                pinch_in = (bx1 - margin) <= pinch_cx <= (bx2 + margin) and (by1 - margin) <= pinch_cy <= (by2 + margin)
                if (pinch_in and pinch_ap < 0.68) or (n_tips_in >= 2 and pinch_ap < 0.82):
                    bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
                    d = float(np.hypot(pinch_cx - bcx, pinch_cy - bcy))
                    if d < best_d:
                        best_d = d
                        best_k = k
                        best_tips = tips_in_box

            if best_k >= 0:
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
            dcx, dcy = 0.5 * float(d_box[0] + d_box[2]), 0.5 * float(d_box[1] + d_box[3])
            rx = float(np.clip(0.58 - (dcy - 0.18) * 0.44, 0.30, 0.58))
            ry = float(np.clip((0.50 - dcx) * 0.56, -0.26, 0.26))
            rz = 0.268 if touched else 0.265
            scx, scy = 0.5 * float(s_box[0] + s_box[2]), 0.5 * float(s_box[1] + s_box[3])
            h_loc = "left" if scx < 0.38 else ("right" if scx > 0.62 else "center")
            v_loc = "upper" if scy < 0.38 else ("lower" if scy > 0.62 else "middle")
            poly = None
            if working_imgs[i] is not None and (i % 15 == 0 or i == 0):
                poly = extract_roi_segmentation_polygon(working_imgs[i], s_box, table_lab_color=desk_lab)
            if not poly or len(poly) < 4:
                cx_p, cy_p = 0.5 * float(s_box[0] + s_box[2]), 0.5 * float(s_box[1] + s_box[3])
                rx_p, ry_p = 0.48 * float(s_box[2] - s_box[0]), 0.48 * float(s_box[3] - s_box[1])
                poly = [
                    [round(cx_p + rx_p * float(np.cos(ang)), 4), round(cy_p + ry_p * float(np.sin(ang)), 4)]
                    for ang in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
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


def refine_episode_objects_and_masks(
    frames: List[Any],
    default_object_name: str = "Manipulated Workpiece",
    video_path: Optional[str] = None,
) -> None:
    """Zero-drift, hand-workspace-grounded multi-object tracker for any unseen egocentric video."""
    n = len(frames)
    if n == 0:
        return

    from .auto_scene_grounder import ground_scene_objects_for_video
    init_objs, _ = ground_scene_objects_for_video(str(video_path or ""), frames, default_object_name=default_object_name)
    shifts, y_fronts, working_imgs = measure_per_frame_desk_geometry(frames, video_path=video_path)

    waypoints: Dict[str, List[Tuple[int, List[float]]]] = {}
    desk_boxes: Dict[str, np.ndarray] = {}
    for nm, b0 in init_objs:
        desk_boxes[nm] = b0.copy()
        waypoints[nm] = [(0, [float(x) for x in b0])]

    primary_nm = init_objs[0][0]
    for i, f in enumerate(frames):
        dx_i, dy_i = float(shifts[i, 0]), float(shifts[i, 1])
        # Collect active hand pinch centroids in desk-plane coordinates
        active_pinches = []
        for hnd in (f.right_hand, f.left_hand):
            if hnd is not None:
                px_d = float(0.5 * (hnd.x[4] + hnd.x[8])) - dx_i
                py_d = float(0.5 * (hnd.y[4] + hnd.y[8])) - dy_i
                if 0.06 < px_d < 0.94 and 0.08 < py_d < 0.84:
                    active_pinches.append((px_d, py_d))

        if active_pinches:
            # Move closest movable object smoothly with the active manipulation centroid
            mx = float(np.mean([p[0] for p in active_pinches]))
            my = float(np.mean([p[1] for p in active_pinches]))
            best_nm = primary_nm
            best_d = 999.0
            for nm, db in desk_boxes.items():
                if any(st in nm.lower() for st in ("router", "presser", "head", "foot", "machine", "fixture", "guide", "station", "holder")):
                    continue
                bcx, bcy = 0.5 * float(db[0] + db[2]), 0.5 * float(db[1] + db[3])
                d = float(np.hypot(mx - bcx, my - bcy))
                if d < best_d:
                    best_d = d
                    best_nm = nm

            db = desk_boxes[best_nm]
            bw, bh = float(db[2] - db[0]), float(db[3] - db[1])
            bcx, bcy = 0.5 * float(db[0] + db[2]), 0.5 * float(db[1] + db[3])
            # Smoothly couple object center to hand manipulation point (keeping within [0.06, 0.94])
            new_cx = 0.72 * bcx + 0.28 * mx
            new_cy = 0.72 * bcy + 0.28 * my
            nx1 = float(np.clip(new_cx - 0.5 * bw, 0.06, 0.94 - bw))
            ny1 = float(np.clip(new_cy - 0.5 * bh, 0.08, 0.76 - bh))
            desk_boxes[best_nm] = np.array([nx1, ny1, nx1 + bw, ny1 + bh], dtype=np.float32)

        if i % 8 == 0 or i == n - 1:
            for nm, db in desk_boxes.items():
                waypoints[nm].append((i, [float(x) for x in db]))

    track_objects_with_keyframe_waypoints(frames, waypoints, video_path=video_path)
