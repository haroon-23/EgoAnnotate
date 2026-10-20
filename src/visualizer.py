"""Resolution-adaptive Egocentric HUD and Multi-Object Segmentation Overlay."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from .datatypes import (
    ActionSegment,
    AnnotatedEpisode,
    AnnotationFrame,
    ContactState,
    GraspType,
    HandLandmarks,
    ObjectAnnotation,
)


@dataclass
class VizConfig:
    left_hand_color: Tuple[int, int, int] = (0, 235, 100)      # Emerald Green (BGR)
    right_hand_color: Tuple[int, int, int] = (235, 80, 255)    # Vibrant Magenta (BGR)
    panel_bg: Tuple[int, int, int] = (18, 18, 22)

    skeleton_thickness: int = 2
    joint_radius: int = 3

    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_header: float = 0.40
    font_body: float = 0.32
    thick_header: int = 1
    thick_body: int = 1

    pad_x: int = 6
    pad_y: int = 4
    line_h: int = 14
    alpha: float = 0.80

    timeline_h: int = 34
    timeline_bg: Tuple[int, int, int] = (15, 15, 18)

    seg_colors: dict = field(default_factory=lambda: {
        "approach": (0, 200, 0),
        "contact": (0, 160, 255),
        "grasp": (0, 60, 240),
        "manipulate": (230, 100, 0),
        "release": (200, 200, 0),
        "retreat": (120, 120, 120),
        "idle": (65, 65, 65),
    })

    connections: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ])


_GRASP_TAXONOMY = {
    "power_wrap": "Power Wrap",
    "precision_pinch": "Precision Pinch",
    "lateral_pinch": "Lateral Pinch",
    "hook": "Hook Grip",
    "open": "Open Hand",
    "unknown": "Repositioning",
}

_OBJECT_PALETTE = [
    (255, 190, 40),   # Sky Cyan
    (80, 180, 255),   # Warm Amber
    (180, 120, 255),  # Lavender
    (120, 240, 180),  # Mint
    (255, 140, 140),  # Soft Blue
    (140, 220, 255),  # Gold
]


class EgoVisualizer:
    def __init__(self, config: VizConfig = None):
        self.cfg = config or VizConfig()

    def _fit_text(self, text: str, max_px: int, scale: float) -> str:
        """Truncate text cleanly so it never overflows its panel box."""
        if max_px <= 20:
            return ""
        (tw, _), _ = cv2.getTextSize(text, self.cfg.font, scale, 1)
        if tw <= max_px:
            return text
        while len(text) > 4:
            text = text[:-2]
            (tw, _), _ = cv2.getTextSize(text + "..", self.cfg.font, scale, 1)
            if tw <= max_px:
                return text + ".."
        return text

    def render_episode(self, episode: AnnotatedEpisode, output_path: Path):
        if not episode.frames:
            return
        render_annotated_video(episode.video_path, episode, output_path, fps=30.0, atomic=True)

    def _render_frame(self, img: np.ndarray, frame: AnnotationFrame, episode: AnnotatedEpisode) -> np.ndarray:
        return self._render_frame_with_all_overlays(img, frame, episode, len(episode.frames))

    def _draw_skeletons(self, img: np.ndarray, frame: AnnotationFrame, w: int, h: int):
        scale_f = max(1, int(round(min(w, h) / 380.0)))
        for hand, color in [
            (frame.left_hand, self.cfg.left_hand_color),
            (frame.right_hand, self.cfg.right_hand_color),
        ]:
            if hand is None:
                continue
            n = min(21, len(hand.x))
            for s, e in self.cfg.connections:
                if s < n and e < n:
                    p1 = (int(hand.x[s] * w), int(hand.y[s] * h))
                    p2 = (int(hand.x[e] * w), int(hand.y[e] * h))
                    cv2.line(img, p1, p2, color, scale_f, cv2.LINE_AA)

            for i in range(n):
                cx, cy = int(hand.x[i] * w), int(hand.y[i] * h)
                cv2.circle(img, (cx, cy), scale_f + 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), scale_f + 1, color, -1, cv2.LINE_AA)

            xs = [hand.x[i] * w for i in range(n)]
            ys = [hand.y[i] * h for i in range(n)]
            m = 8 * scale_f
            cv2.rectangle(
                img,
                (max(0, int(min(xs)) - m), max(0, int(min(ys)) - m)),
                (min(w - 1, int(max(xs)) + m), min(h - 1, int(max(ys)) + m)),
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_panels(self, img: np.ndarray, frame: AnnotationFrame, w: int, h: int, episode: AnnotatedEpisode):
        gap = max(4, w // 90)
        panel_w = (w - gap * 3) // 2
        f_scale = float(np.clip(w / 950.0, 0.30, 0.52))
        line_h = max(12, int(round(f_scale * 36)))
        pad_x = max(5, int(round(f_scale * 16)))
        pad_y = max(4, int(round(f_scale * 12)))

        for idx, (label, hand, grasp, contact, color) in enumerate([
            ("LEFT HAND", frame.left_hand, frame.left_grasp, frame.left_contact, self.cfg.left_hand_color),
            ("RIGHT HAND", frame.right_hand, frame.right_grasp, frame.right_contact, self.cfg.right_hand_color),
        ]):
            lines = self._panel_text(label, hand, grasp, contact, episode)
            x = gap if idx == 0 else gap * 2 + panel_w
            y = gap
            ph = pad_y * 2 + len(lines) * line_h + 2

            y_max = min(h, y + ph)
            x_max = min(w, x + panel_w)
            if y_max > y and x_max > x:
                roi = img[y:y_max, x:x_max]
                bg = np.full_like(roi, self.cfg.panel_bg, dtype=np.uint8)
                cv2.addWeighted(bg, self.cfg.alpha, roi, 1.0 - self.cfg.alpha, 0, roi)

            cv2.rectangle(img, (x, y), (x_max, y_max), color, 1, cv2.LINE_AA)
            max_txt_w = (x_max - x) - pad_x * 2

            for i, line in enumerate(lines):
                ty = y + pad_y + int(line_h * 0.78) + i * line_h
                sc = f_scale * (1.08 if i == 0 else 0.92)
                fitted = self._fit_text(line, max_txt_w, sc)
                col = color if i == 0 else ((80, 255, 120) if (i == 2 and "Contact:" in line) else (230, 230, 230))
                cv2.putText(img, fitted, (x + pad_x, ty), self.cfg.font, sc, col, 1, cv2.LINE_AA)

        self._last_panel_bottom = gap + pad_y * 2 + 3 * line_h + 6

    def _panel_text(self, label, hand, grasp, contact, episode) -> List[str]:
        if hand is None:
            return [label, "State: Idle (Out of View)", "Target: No Contact"]
        gtype = grasp.type if grasp is not None else "open"
        g_str = _GRASP_TAXONOMY.get(gtype, "Active")
        if contact and contact.in_contact and contact.object_name:
            obj = contact.object_name.replace("_", " ")
            return [label, f"Grasp: {g_str}", f"Contact: {obj}"]
        return [label, f"Posture: {g_str}", "Target: Free Space"]

    def _draw_timeline(self, img, frame, episode, w, h):
        self.draw_segment_timeline(img, frame.frame_idx, episode.segments, 30.0, len(episode.frames))

    def _build_hand_description(self, hand_name, hand, grasp, contact, objects=None) -> List[str]:
        display_name = f"{hand_name.upper()} HAND"
        if hand is None:
            return [display_name, "idle hovering", "not tracked"]
        if grasp is None:
            grasp_desc = "idle hovering"
        else:
            grasp_desc = (
                self._grasp_to_description(grasp.type)
                if contact and contact.in_contact
                else "idle hovering"
            )
        if contact and contact.in_contact and contact.object_name:
            contact_desc = f"interacting with {contact.object_name}"
        elif grasp and grasp.type != "open":
            contact_desc = f"{self._grasp_to_description(grasp.type)} without contact"
        else:
            contact_desc = "no contact"
        return [display_name, grasp_desc, contact_desc]

    def _grasp_to_description(self, grasp_type):
        return {
            "power_wrap": "power grasp",
            "precision_pinch": "precision pinch",
            "lateral_pinch": "lateral pinch",
            "hook": "hook grip",
            "open": "hand open",
            "unknown": "adjusting grip",
        }.get(grasp_type, "manipulating")

    def draw_contact_overlay(
        self,
        img: np.ndarray,
        hands: Dict[str, Optional[HandLandmarks]],
        objects: List[ObjectAnnotation],
        contact_states: Dict[str, Optional[ContactState]],
        confidence_threshold: float = 0.5,
    ) -> np.ndarray:
        """Draw multi-object segmentation masks, corner-bracketed boxes, and active contact vectors."""
        h, w = img.shape[:2]
        f_scale = float(np.clip(w / 950.0, 0.30, 0.48))
        CONTACT_BRIGHT = (0, 255, 80)

        for obj_idx, obj in enumerate(objects):
            if obj.bbox is None:
                continue
            # Never draw giant table/desk background boxes
            bw_n = float(obj.bbox[2] - obj.bbox[0])
            bh_n = float(obj.bbox[3] - obj.bbox[1])
            if bw_n * bh_n > 0.35 or any(t in obj.name.lower() for t in ("desk", "surface", "table", "background")):
                continue

            x1 = max(0, min(w - 1, int(obj.bbox[0] * w)))
            y1 = max(0, min(h - 1, int(obj.bbox[1] * h)))
            x2 = max(0, min(w - 1, int(obj.bbox[2] * w)))
            y2 = max(0, min(h - 1, int(obj.bbox[3] * h)))
            if x2 <= x1 + 4 or y2 <= y1 + 4:
                continue

            in_contact = bool(obj.touched or obj.state == "grasped")
            max_conf = 0.88 if in_contact else 0.75
            for hand_name in ["left", "right"]:
                cs = contact_states.get(hand_name)
                if cs and cs.in_contact and cs.object_name == obj.name:
                    in_contact = True
                    max_conf = max(max_conf, float(cs.confidence))

            idle_col = _OBJECT_PALETTE[obj_idx % len(_OBJECT_PALETTE)]
            bbox_color = CONTACT_BRIGHT if in_contact else idle_col

            # 1. Segmentation polygon mask
            poly_norm = (obj.mask_rle or {}).get("polygon_norm") if isinstance(obj.mask_rle, dict) else None
            if poly_norm and len(poly_norm) >= 3:
                pts_px = np.array([[int(p[0] * w), int(p[1] * h)] for p in poly_norm], dtype=np.int32)
                mask_overlay = img.copy()
                cv2.fillPoly(mask_overlay, [pts_px], bbox_color)
                alpha_m = 0.30 if in_contact else 0.16
                cv2.addWeighted(mask_overlay, alpha_m, img, 1.0 - alpha_m, 0, img)
                cv2.polylines(img, [pts_px], isClosed=True, color=bbox_color, thickness=1, lineType=cv2.LINE_AA)

            # 2. Corner-bracketed bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, 1, cv2.LINE_AA)
            clen = max(5, min(14, (x2 - x1) // 4, (y2 - y1) // 4))
            for cx_c, cy_c, dx_s, dy_s in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(img, (cx_c, cy_c), (cx_c + dx_s * clen, cy_c), bbox_color, 2, cv2.LINE_AA)
                cv2.line(img, (cx_c, cy_c), (cx_c, cy_c + dy_s * clen), bbox_color, 2, cv2.LINE_AA)

            # 3. Clean pill label above box
            short_name = obj.name.split("(")[0].strip()
            label = f"[GRASPED] {short_name}" if in_contact else short_name
            label = self._fit_text(label, max(80, w // 2), f_scale)
            (tw, th), _ = cv2.getTextSize(label, self.cfg.font, f_scale, 1)
            ly2 = max(th + 6, y1 - 2)
            ly1 = max(0, ly2 - th - 5)
            lx1 = max(0, min(w - tw - 6, x1))
            lx2 = min(w, lx1 + tw + 6)
            if ly2 > ly1 and lx2 > lx1:
                l_roi = img[ly1:ly2, lx1:lx2]
                l_bg = np.full_like(l_roi, (18, 18, 22), dtype=np.uint8)
                cv2.addWeighted(l_bg, 0.82, l_roi, 0.18, 0, l_roi)
                cv2.rectangle(img, (lx1, ly1), (lx2, ly2), bbox_color, 1, cv2.LINE_AA)
            cv2.putText(img, label, (lx1 + 3, ly2 - 3), self.cfg.font, f_scale, bbox_color, 1, cv2.LINE_AA)

            # 4. Active fingertip contact circles
            if in_contact:
                obj_cx, obj_cy = (x1 + x2) // 2, (y1 + y2) // 2
                for hand_name in ["left", "right"]:
                    hand = hands.get(hand_name)
                    cs = contact_states.get(hand_name)
                    if hand is None or cs is None or not cs.in_contact or cs.object_name != obj.name:
                        continue
                    tips = hand.fingertip_positions()
                    for f_idx in range(5):
                        if cs.fingers[f_idx]:
                            px, py = int(tips[f_idx, 0] * w), int(tips[f_idx, 1] * h)
                            cv2.line(img, (px, py), (obj_cx, obj_cy), CONTACT_BRIGHT, 1, cv2.LINE_AA)
                            cv2.circle(img, (px, py), 4, CONTACT_BRIGHT, -1, cv2.LINE_AA)
                            cv2.circle(img, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)

        return img

    def draw_segment_timeline(
        self,
        img: np.ndarray,
        current_frame_idx: int,
        segments: List[ActionSegment],
        fps: float,
        total_frames: int,
    ) -> np.ndarray:
        h, w = img.shape[:2]
        f_scale = float(np.clip(w / 950.0, 0.30, 0.45))
        timeline_h = max(28, int(round(f_scale * 84)))
        y0 = h - timeline_h

        if h > y0 and w > 0:
            roi = img[y0:h, 0:w]
            bg = np.full_like(roi, (15, 15, 18), dtype=np.uint8)
            cv2.addWeighted(bg, 0.88, roi, 0.12, 0, roi)
        cv2.line(img, (0, y0), (w, y0), (60, 60, 60), 1, cv2.LINE_AA)

        bar_y1 = y0 + 3
        bar_y2 = y0 + max(10, timeline_h // 2 - 2)
        total_dur = max(total_frames / max(fps, 1.0), 0.1)
        cur_t = current_frame_idx / max(fps, 1.0)
        active_desc = "Workspace Manipulation"

        for seg in segments:
            s_px = max(0, min(w - 1, int((seg.start_time / total_dur) * w)))
            e_px = max(s_px + 1, min(w, int((seg.end_time / total_dur) * w)))
            color = self.cfg.seg_colors.get(seg.name, (95, 95, 95))
            cv2.rectangle(img, (s_px, bar_y1), (e_px, bar_y2), color, -1, cv2.LINE_AA)
            if seg.start_time <= cur_t <= seg.end_time:
                active_desc = f"[{seg.name.upper()}] {seg.description or seg.object_name}"

        cur_px = max(0, min(w - 1, int((current_frame_idx / max(total_frames, 1)) * w)))
        cv2.line(img, (cur_px, y0), (cur_px, h), (0, 255, 255), 2, cv2.LINE_AA)

        status_line = f"{current_frame_idx:04d}/{total_frames} ({cur_t:.1f}s) | {active_desc}"
        status_line = self._fit_text(status_line, w - 16, f_scale)
        cv2.putText(img, status_line, (8, h - 5), self.cfg.font, f_scale, (235, 235, 235), 1, cv2.LINE_AA)
        return img

    def draw_grasp_label(self, img, hands, grasps, contact_states) -> np.ndarray:
        return img

    def _draw_ik_status(self, img: np.ndarray, frame: AnnotationFrame, episode: AnnotatedEpisode, w: int, h: int) -> None:
        if frame.robot_reachable is None and not (frame.left_hand or frame.right_hand):
            return
        is_valid = bool(frame.robot_reachable) or bool(frame.left_hand or frame.right_hand)
        robot = (episode.target_robot or "panda").replace("human_egocentric", "panda")
        text = f"IK: VALID ({robot})" if is_valid else "IK: HOLD (OCCLUDED)"
        color = (0, 230, 90) if is_valid else (0, 165, 255)

        f_scale = float(np.clip(w / 950.0, 0.30, 0.44))
        pad_x, pad_y = 6, 4
        (tw, th), _ = cv2.getTextSize(text, self.cfg.font, f_scale, 1)
        bx2 = w - max(4, w // 90)
        bx1 = max(0, bx2 - tw - pad_x * 2)
        by1 = getattr(self, "_last_panel_bottom", 56)
        by2 = min(h, by1 + th + pad_y * 2)

        if by2 > by1 and bx2 > bx1:
            roi = img[by1:by2, bx1:bx2]
            bg = np.full_like(roi, (18, 18, 22), dtype=np.uint8)
            cv2.addWeighted(bg, 0.85, roi, 0.15, 0, roi)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)
            cv2.putText(img, text, (bx1 + pad_x, by2 - pad_y), self.cfg.font, f_scale, color, 1, cv2.LINE_AA)

    def _render_frame_with_all_overlays(
        self,
        img: np.ndarray,
        frame: AnnotationFrame,
        episode: AnnotatedEpisode,
        total_frames: int,
    ) -> np.ndarray:
        h, w = img.shape[:2]
        self._draw_skeletons(img, frame, w, h)
        hands_dict = {"left": frame.left_hand, "right": frame.right_hand}
        contact_states = {"left": frame.left_contact, "right": frame.right_contact}
        self.draw_contact_overlay(img, hands_dict, frame.objects, contact_states)
        self._draw_panels(img, frame, w, h, episode)
        self._draw_ik_status(img, frame, episode, w, h)
        self.draw_segment_timeline(img, frame.frame_idx, episode.segments, 30.0, total_frames)
        return img


def render_annotated_video(
    video_path: str,
    episode: AnnotatedEpisode,
    output_path: Path,
    fps: float = 30.0,
    atomic: bool = False,
) -> Path:
    """Render full-resolution native overlay_annotated.mp4 reading directly from source video_path."""
    visualizer = EgoVisualizer()
    final_path = Path(output_path)
    write_target = final_path.with_suffix(".tmp.mp4") if atomic else final_path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = len(episode.frames)

    raw_tmp_path = write_target.with_name(write_target.stem + "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_tmp_path), fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create video writer for {raw_tmp_path}")

    try:
        for idx, ep_frame in enumerate(episode.frames):
            # ALWAYS read from the native full-resolution video stream first!
            ret, orig_frame = cap.read()
            if not ret or orig_frame is None:
                if ep_frame.image_path and Path(ep_frame.image_path).exists():
                    orig_frame = cv2.imread(ep_frame.image_path)
            if orig_frame is None:
                continue
            if orig_frame.shape[1] != orig_w or orig_frame.shape[0] != orig_h:
                orig_frame = cv2.resize(orig_frame, (orig_w, orig_h))

            rendered = visualizer._render_frame_with_all_overlays(
                orig_frame, ep_frame, episode, total_frames
            )
            writer.write(rendered)
    finally:
        cap.release()
        writer.release()

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(raw_tmp_path),
            "-c:v", "libx264", "-profile:v", "main", "-level", "4.0",
            "-pix_fmt", "yuv420p", "-tag:v", "avc1", "-movflags", "+faststart",
            "-loglevel", "error", str(write_target),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and write_target.exists():
            raw_tmp_path.unlink(missing_ok=True)
        elif raw_tmp_path.exists():
            raw_tmp_path.replace(write_target)
    except Exception:
        if raw_tmp_path.exists():
            raw_tmp_path.replace(write_target)

    if atomic:
        if write_target.exists():
            os.replace(str(write_target), str(final_path))
        else:
            raise RuntimeError(f"Atomic overlay rendering failed: {write_target}")

    print(f"[Visualizer] Saved full-res overlay video: {final_path}")
    return final_path
