"""Camera-stabilized visual appearance & multi-object tabletop tracker.

Eliminates box drift and false hand-dragging:
  1. Tracks global tabletop camera motion between consecutive frames via RANSAC affine flow.
  2. Anchors each object with its visual appearance template & LAB color signature from
     unoccluded tabletop frames, re-localizing each object via template matching when visible
     and holding its table-stabilized position when occluded by hands.
  3. Extracts tight foreground contours only around true non-table object pixels.
"""
from __future__ import annotations

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
        if (w * h) > 0.32 or (w * h) < 0.0015 or w > 0.78 or h > 0.75:
            return False
    return True


def detect_manipulated_object_roi_near_hands(
    img_bgr: np.ndarray,
    left_hand: Optional[HandLandmarks],
    right_hand: Optional[HandLandmarks],
) -> Optional[np.ndarray]:
    """Backwards-compatible helper: locate object ROI near active hands."""
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
    max_points: int = 28,
) -> Optional[List[List[float]]]:
    """Extract a tight object contour inside bbox_norm by subtracting the tabletop LAB color."""
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(float(bbox_norm[0]) * w))
    y1 = max(0, int(float(bbox_norm[1]) * h))
    x2 = min(w, int(float(bbox_norm[2]) * w))
    y2 = min(h, int(float(bbox_norm[3]) * h))
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None

    roi = img_bgr[y1:y2, x1:x2]
    scale = min(1.0, 72.0 / float(max(bw, bh)))
    sw, sh = max(10, int(round(bw * scale))), max(10, int(round(bh * scale)))
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

    if table_lab_color is None:
        border = np.concatenate([lab[0, :, :], lab[-1, :, :], lab[:, 0, :], lab[:, -1, :]], axis=0)
        bg_med = np.median(border, axis=0)
    else:
        bg_med = table_lab_color

    dist = np.linalg.norm(lab - bg_med[None, None, :], axis=-1)
    # Require genuine color/luminance contrast from the flat desk surface
    if float(np.percentile(dist, 85)) < 14.0:
        return None

    mask = (dist > max(16.0, float(np.percentile(dist, 45)))).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / float(sw * sh)
    # Reject tiny noise or full-box rectangular border artifacts
    if area_ratio < 0.08 or area_ratio > 0.90:
        return None

    eps = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True).reshape(-1, 2)
    if len(approx) < 4:
        return None
    if len(approx) > max_points:
        approx = approx[np.linspace(0, len(approx) - 1, max_points, dtype=int)]

    poly_norm: List[List[float]] = []
    for px_s, py_s in approx:
        nx = float(np.clip((x1 + (px_s / scale)) / float(w), 0.0, 1.0))
        ny = float(np.clip((y1 + (py_s / scale)) / float(h), 0.0, 1.0))
        poly_norm.append([round(nx, 4), round(ny, 4)])
    return poly_norm


def _hand_occludes_box(box: np.ndarray, left_hand: Optional[HandLandmarks], right_hand: Optional[HandLandmarks]) -> bool:
    for hnd in (left_hand, right_hand):
        if hnd is None:
            continue
        hx1, hx2 = float(np.min(hnd.x)) - 0.03, float(np.max(hnd.x)) + 0.03
        hy1, hy2 = float(np.min(hnd.y)) - 0.03, float(np.max(hnd.y)) + 0.03
        ix1, iy1 = max(float(box[0]), hx1), max(float(box[1]), hy1)
        ix2, iy2 = min(float(box[2]), hx2), min(float(box[3]), hy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        b_area = max(float((box[2] - box[0]) * (box[3] - box[1])), 1e-5)
        if inter / b_area > 0.20:
            return True
    return False


def discover_tabletop_objects_from_unoccluded_frames(
    frames: List[Any],
    video_path: Optional[str] = None,
    default_object_name: str = "tabletop_object",
) -> List[Dict[str, Any]]:
    catalog: Dict[str, np.ndarray] = {}
    if frames:
        for o in getattr(frames[0], "objects", []) or []:
            if o.bbox is not None and is_valid_manipulable_object(o.name, o.bbox):
                clean_name = o.name.split("(")[0].strip()
                catalog[clean_name] = np.asarray(o.bbox, dtype=np.float32)

    if catalog:
        return [{"name": k, "init_bbox": v} for k, v in catalog.items()]

    # Discover distinct foreground objects on the clearest frame
    clear_img = None
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        ret, frm = cap.read()
        if ret:
            clear_img = frm
        cap.release()
    if clear_img is None and frames and getattr(frames[0], "image_path", None) and Path(frames[0].image_path).exists():
        clear_img = cv2.imread(frames[0].image_path)

    discovered: List[Dict[str, Any]] = []
    if clear_img is not None:
        h, w = clear_img.shape[:2]
        lab = cv2.cvtColor(clear_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        roi_table = lab[int(h * 0.25):int(h * 0.70), int(w * 0.15):int(w * 0.85)]
        table_med = np.median(roi_table.reshape(-1, 3), axis=0)
        diff = np.linalg.norm(lab - table_med[None, None, :], axis=-1)
        diff[:int(h * 0.14), :] = 0
        diff[int(h * 0.76):, :] = 0
        binary = (diff > 24.0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area_norm = (bw * bh) / float(w * h)
            if 0.003 <= area_norm <= 0.22 and bw < 0.70 * w and bh < 0.55 * h:
                blobs.append((area_norm, np.array([x / w, y / h, (x + bw) / w, (y + bh) / h], dtype=np.float32)))
        blobs.sort(key=lambda t: t[0], reverse=True)
        for idx_b, (_, b_norm) in enumerate(blobs[:5]):
            lbl = default_object_name if idx_b == 0 else f"{default_object_name}_{idx_b+1}"
            discovered.append({"name": lbl, "init_bbox": b_norm})

    if not discovered:
        for f in frames:
            if f.right_hand is not None or f.left_hand is not None:
                b = detect_manipulated_object_roi_near_hands(np.zeros((224, 224, 3), np.uint8), f.left_hand, f.right_hand)
                if b is not None:
                    discovered.append({"name": default_object_name, "init_bbox": b})
                    break
    if not discovered:
        discovered.append({"name": default_object_name, "init_bbox": np.array([0.40, 0.42, 0.60, 0.62], dtype=np.float32)})
    return discovered


def refine_episode_objects_and_masks(
    frames: List[Any],
    default_object_name: str = "manipulated_object",
    video_path: Optional[str] = None,
) -> None:
    """Track tabletop objects using camera motion compensation + visual template re-localization."""
    n = len(frames)
    if n == 0:
        return

    objs_meta = discover_tabletop_objects_from_unoccluded_frames(
        frames, video_path=video_path, default_object_name=default_object_name
    )
    num_objs = len(objs_meta)

    # Read working-resolution (224x224) frames for fast template + optical flow tracking
    imgs_small: List[Optional[np.ndarray]] = []
    grays_small: List[Optional[np.ndarray]] = []
    for f in frames:
        if getattr(f, "image_path", None) and Path(f.image_path).exists():
            im = cv2.imread(f.image_path)
            if im is not None:
                im_s = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
                imgs_small.append(im_s)
                grays_small.append(cv2.cvtColor(im_s, cv2.COLOR_BGR2GRAY))
                continue
        imgs_small.append(None)
        grays_small.append(None)

    # Estimate tabletop LAB background color from frame 0
    table_lab = None
    if imgs_small[0] is not None:
        lab0 = cv2.cvtColor(imgs_small[0], cv2.COLOR_BGR2LAB).astype(np.float32)
        table_lab = np.median(lab0[60:150, 30:190].reshape(-1, 3), axis=0)

    # Extract initial visual templates from frame 0 for each object
    templates: List[Optional[np.ndarray]] = [None] * num_objs
    if imgs_small[0] is not None:
        for k, om in enumerate(objs_meta):
            b = om["init_bbox"]
            x1, y1 = max(0, int(b[0] * 224)), max(0, int(b[1] * 224))
            x2, y2 = min(224, int(b[2] * 224)), min(224, int(b[3] * 224))
            if (x2 - x1) >= 8 and (y2 - y1) >= 8:
                templates[k] = imgs_small[0][y1:y2, x1:x2].copy()

    tracks = np.zeros((num_objs, n, 4), dtype=np.float32)
    for k, om in enumerate(objs_meta):
        tracks[k, 0, :] = om["init_bbox"]

    # Track across frames using camera motion + visual template matching
    for i in range(1, n):
        tracks[:, i, :] = tracks[:, i - 1, :].copy()
        g_prev, g_cur = grays_small[i - 1], grays_small[i]
        im_cur = imgs_small[i]
        f = frames[i]

        # 1. Global tabletop camera translation between frame i-1 and i
        cam_dx, cam_dy = 0.0, 0.0
        if g_prev is not None and g_cur is not None:
            mask = np.full((224, 224), 255, dtype=np.uint8)
            for hnd in (f.left_hand, f.right_hand):
                if hnd is not None:
                    hx1 = max(0, int((float(np.min(hnd.x)) - 0.06) * 224))
                    hx2 = min(224, int((float(np.max(hnd.x)) + 0.06) * 224))
                    hy1 = max(0, int((float(np.min(hnd.y)) - 0.06) * 224))
                    hy2 = min(224, int((float(np.max(hnd.y)) + 0.06) * 224))
                    mask[hy1:hy2, hx1:hx2] = 0
            pts = cv2.goodFeaturesToTrack(g_prev, maxCorners=50, qualityLevel=0.02, minDistance=8, mask=mask)
            if pts is not None and len(pts) >= 6:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(g_prev, g_cur, pts, None)
                if nxt is not None and st is not None and np.sum(st.ravel() == 1) >= 4:
                    v = st.ravel() == 1
                    disp = (nxt[v] - pts[v]).reshape(-1, 2)
                    cam_dx = float(np.median(disp[:, 0])) / 224.0
                    cam_dy = float(np.median(disp[:, 1])) / 224.0

        # Shift all object boxes by the measured tabletop camera motion
        for k in range(num_objs):
            b = tracks[k, i]
            bw, bh = b[2] - b[0], b[3] - b[1]
            nx1 = float(np.clip(b[0] + cam_dx, 0.0, 1.0 - bw))
            ny1 = float(np.clip(b[1] + cam_dy, 0.0, 1.0 - bh))
            tracks[k, i] = [nx1, ny1, nx1 + bw, ny1 + bh]

            # 2. If object k is NOT occluded by a hand, refine its position via visual template matching!
            if im_cur is not None and templates[k] is not None and not _hand_occludes_box(tracks[k, i], f.left_hand, f.right_hand):
                tmpl = templates[k]
                th, tw = tmpl.shape[:2]
                # Search window around current box (+/- 28 pixels in 224x224 space)
                cx_px = int(0.5 * (tracks[k, i, 0] + tracks[k, i, 2]) * 224)
                cy_px = int(0.5 * (tracks[k, i, 1] + tracks[k, i, 3]) * 224)
                pad = 26
                sx1 = max(0, cx_px - tw // 2 - pad)
                sy1 = max(0, cy_px - th // 2 - pad)
                sx2 = min(224, cx_px + tw // 2 + pad)
                sy2 = min(224, cy_px + th // 2 + pad)
                if (sx2 - sx1) > tw and (sy2 - sy1) > th:
                    search_roi = im_cur[sy1:sy2, sx1:sx2]
                    res = cv2.matchTemplate(search_roi, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > 0.52:
                        matched_x1 = (sx1 + max_loc[0]) / 224.0
                        matched_y1 = (sy1 + max_loc[1]) / 224.0
                        # Blend smoothly (40% template match, 60% camera-stabilized prediction)
                        ux1 = 0.60 * tracks[k, i, 0] + 0.40 * matched_x1
                        uy1 = 0.60 * tracks[k, i, 1] + 0.40 * matched_y1
                        tracks[k, i] = [ux1, uy1, ux1 + bw, uy1 + bh]

    # Zero-phase temporal smoothing of tracks
    for k in range(num_objs):
        for c in range(4):
            tracks[k, :, c] = np.convolve(
                np.pad(tracks[k, :, c], (3, 3), mode="edge"), np.ones(7) / 7.0, mode="valid"
            )

    # Determine per-frame hand contact against the visually tracked object boxes
    for i, f in enumerate(frames):
        for side, hnd, gr in [
            ("left", f.left_hand, f.left_grasp),
            ("right", f.right_hand, f.right_grasp),
        ]:
            if hnd is None:
                if side == "left":
                    f.left_contact = None
                else:
                    f.right_contact = None
                continue

            # Fingertip coordinates (thumb 4, index 8, middle 12)
            tips_xy = [(float(hnd.x[idx]), float(hnd.y[idx])) for idx in (4, 8, 12, 16, 20)]
            best_k = -1
            best_overlap_tips = np.zeros(5, dtype=bool)
            best_dist = 999.0

            for k in range(num_objs):
                bx1, by1, bx2, by2 = tracks[k, i]
                margin = 0.025
                in_tips = np.array([
                    (bx1 - margin) <= tx <= (bx2 + margin) and (by1 - margin) <= ty <= (by2 + margin)
                    for tx, ty in tips_xy
                ], dtype=bool)
                if np.any(in_tips):
                    bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
                    d = float(np.hypot(tips_xy[1][0] - bcx, tips_xy[1][1] - bcy))
                    if d < best_dist:
                        best_dist = d
                        best_k = k
                        best_overlap_tips = in_tips

            if best_k >= 0:
                ct_state = ContactState(
                    fingers=best_overlap_tips,
                    object_name=objs_meta[best_k]["name"],
                    in_contact=True,
                    confidence=0.90,
                )
            else:
                ct_state = ContactState(np.zeros(5, dtype=bool), None, False, 0.0)

            if side == "left":
                f.left_contact = ct_state
            else:
                f.right_contact = ct_state

        # Build ObjectAnnotation list for frame i
        im_s = imgs_small[i]
        active_names = set()
        if f.left_contact and f.left_contact.in_contact and f.left_contact.object_name:
            active_names.add(f.left_contact.object_name)
        if f.right_contact and f.right_contact.in_contact and f.right_contact.object_name:
            active_names.add(f.right_contact.object_name)

        frame_objs: List[ObjectAnnotation] = []
        for k, om in enumerate(objs_meta):
            name = om["name"]
            bbox = tracks[k, i].copy()
            touched = name in active_names
            poly = extract_roi_segmentation_polygon(im_s, bbox, table_lab_color=table_lab) if im_s is not None else None
            cx, cy = 0.5 * float(bbox[0] + bbox[2]), 0.5 * float(bbox[1] + bbox[3])
            rx = float(np.clip(0.56 - (cy - 0.20) * 0.42, 0.32, 0.58))
            ry = float(np.clip((0.50 - cx) * 0.52, -0.24, 0.24))
            rz = 0.33 if touched else 0.278
            h_loc = "left" if cx < 0.38 else ("right" if cx > 0.62 else "center")
            v_loc = "upper" if cy < 0.38 else ("lower" if cy > 0.62 else "middle")
            frame_objs.append(
                ObjectAnnotation(
                    name=name,
                    location_description=f"{v_loc}-{h_loc}",
                    touched=touched,
                    bbox=bbox,
                    state="grasped" if touched else "on_table",
                    position_3d=np.array([rx, ry, rz], dtype=np.float64),
                    mask_rle={"polygon_norm": poly} if poly else None,
                )
            )
        f.objects = frame_objs
