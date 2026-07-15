"""Egocentric HUD visualization overlay for annotated episodes."""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

from .datatypes import AnnotatedEpisode, AnnotationFrame, HandLandmarks


@dataclass
class VizConfig:
    left_hand_color: Tuple[int, int, int] = (0, 255, 0)      # Green
    right_hand_color: Tuple[int, int, int] = (255, 0, 255)    # Magenta
    panel_bg: Tuple[int, int, int] = (20, 20, 20)

    skeleton_thickness: int = 1
    joint_radius: int = 2

    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_header: float = 0.42
    font_body: float = 0.34
    thick_header: int = 1
    thick_body: int = 1

    pad_x: int = 8
    pad_y: int = 5
    line_h: int = 16
    alpha: float = 0.78

    timeline_h: int = 26
    timeline_bg: Tuple[int, int, int] = (15, 15, 15)

    seg_colors: dict = field(default_factory=lambda: {
        "approach": (0, 200, 0), "contact": (0, 160, 255),
        "grasp": (0, 0, 230), "manipulate": (230, 0, 0),
        "release": (230, 230, 0), "retreat": (120, 120, 120), "idle": (70, 70, 70),
    })

    connections: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17),
    ])


# ── Manipulation grasp taxonomy ──────────────────────────────────
_GRASP_TAXONOMY = {
    "power_wrap":      "grasping wrapping",
    "precision_pinch": "grasping pinching",
    "lateral_pinch":   "lateral pinching",
    "hook":            "hook gripping",
    "open":            "hand open",
    "unknown":         "adjusting grip",
}

_GRASP_DETAIL = {
    "power_wrap":      "fingers curled around object",
    "precision_pinch": "fingers extended",
    "lateral_pinch":   "thumb pressing laterally",
    "hook":            "fingers curled in hook",
    "open":            "fingers extended",
    "unknown":         "fingers repositioning",
}


class EgoVisualizer:
    def __init__(self, config: VizConfig = None):
        self.cfg = config or VizConfig()

    def render_episode(self, episode: AnnotatedEpisode, output_path: Path):
        if not episode.frames:
            return
        first = cv2.imread(episode.frames[0].image_path)
        if first is None:
            return

        h, w = first.shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        if not writer.isOpened():
            return

        for frame in tqdm(episode.frames, desc="Rendering", leave=False):
            img = cv2.imread(frame.image_path)
            if img is None:
                continue
            writer.write(self._render_frame(img.copy(), frame, episode))

        writer.release()
        print(f"[Visualizer] Saved: {output_path}")

    def _render_frame(self, img: np.ndarray, frame: AnnotationFrame,
                      episode: AnnotatedEpisode) -> np.ndarray:
        h, w = img.shape[:2]
        self._draw_skeletons(img, frame, w, h)
        self._draw_panels(img, frame, w, h, episode)
        self._draw_timeline(img, frame, episode, w, h)
        return img

    # ── Skeleton overlay ──────────────────────────────────────────────
    def _draw_skeletons(self, img, frame, w, h):
        for hand, color in [(frame.left_hand, self.cfg.left_hand_color),
                            (frame.right_hand, self.cfg.right_hand_color)]:
            if hand is None:
                continue
            n = min(21, len(hand.x))

            # Connections
            for s, e in self.cfg.connections:
                if s < n and e < n:
                    p1 = (int(hand.x[s] * w), int(hand.y[s] * h))
                    p2 = (int(hand.x[e] * w), int(hand.y[e] * h))
                    cv2.line(img, p1, p2, color, self.cfg.skeleton_thickness, cv2.LINE_AA)

            # Joints — small white ring + colored fill
            for i in range(n):
                cx, cy = int(hand.x[i] * w), int(hand.y[i] * h)
                cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 2, color, -1, cv2.LINE_AA)

            # Bounding box
            xs = [hand.x[i] * w for i in range(n)]
            ys = [hand.y[i] * h for i in range(n)]
            m = 12
            cv2.rectangle(img,
                          (max(0, int(min(xs)) - m), max(0, int(min(ys)) - m)),
                          (min(w, int(max(xs)) + m), min(h, int(max(ys)) + m)),
                          color, 1, cv2.LINE_AA)

    # ── HUD panels (always both visible) ──────────────────────────────
    def _draw_panels(self, img, frame, w, h, episode):
        gap = 4
        panel_w = (w - gap * 3) // 2

        for idx, (label, hand, grasp, contact, color) in enumerate([
            ("LEFT",  frame.left_hand, frame.left_grasp,
             frame.left_contact,  self.cfg.left_hand_color),
            ("RIGHT", frame.right_hand, frame.right_grasp,
             frame.right_contact, self.cfg.right_hand_color),
        ]):
            lines = self._panel_text(label, hand, grasp, contact, episode)

            x = gap if idx == 0 else gap * 2 + panel_w
            y = gap
            ph = self.cfg.pad_y * 2 + len(lines) * self.cfg.line_h + 2

            # Background
            overlay = img.copy()
            cv2.rectangle(overlay, (x, y), (x + panel_w, y + ph),
                          self.cfg.panel_bg, -1)
            cv2.addWeighted(overlay, self.cfg.alpha, img,
                            1 - self.cfg.alpha, 0, img)

            # Border
            cv2.rectangle(img, (x, y), (x + panel_w, y + ph),
                          color, 1, cv2.LINE_AA)

            # Text
            for i, line in enumerate(lines):
                ty = y + self.cfg.pad_y + 12 + i * self.cfg.line_h
                if i == 0:
                    cv2.putText(img, line, (x + self.cfg.pad_x, ty),
                                self.cfg.font, self.cfg.font_header,
                                color, self.cfg.thick_header, cv2.LINE_AA)
                else:
                    cv2.putText(img, line, (x + self.cfg.pad_x, ty),
                                self.cfg.font, self.cfg.font_body,
                                (230, 230, 230), self.cfg.thick_body,
                                cv2.LINE_AA)



    def _panel_text(self, label, hand, grasp, contact, episode) -> List[str]:
        """Build 2-3 line HUD description per hand."""

        # ── Hand not tracked ──
        if hand is None:
            return [
                label,
                "idle hovering; hand resting near",
                "workspace without contact",
            ]

        # ── Grasp taxonomy line ──
        if grasp is not None:
            gtype = grasp.type
            action = _GRASP_TAXONOMY.get(gtype, "manipulating")
            detail = _GRASP_DETAIL.get(gtype, "hand active")
            grasp_line = f"{action}; {detail}"
        else:
            grasp_line = "hand open; fingers extended"

        # ── Contact / spatial context line ──
        if contact and contact.in_contact and contact.object_name:
            obj = contact.object_name.replace("_", " ")
            contact_line = f"to grip the {obj} ({obj})"
        else:
            # Not in contact — describe spatially
            contact_line = "without contact"

        return [label, grasp_line, contact_line]

    # ── Timeline bar ──────────────────────────────────────────────────
    def _draw_timeline(self, img, frame, episode, w, h):
        y0 = h - self.cfg.timeline_h

        overlay = img.copy()
        cv2.rectangle(overlay, (0, y0), (w, h), self.cfg.timeline_bg, -1)
        cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
        cv2.line(img, (0, y0), (w, y0), (40, 40, 40), 1, cv2.LINE_AA)

        # Current segment
        seg = None
        for s in episode.segments:
            if s.start_time <= frame.timestamp <= s.end_time:
                seg = s
                break

        if seg:
            desc = seg.description if seg.description else episode.task_description
            txt = f"{seg.start_time:05.2f}s-{seg.end_time:05.2f}s  {desc}"
            c = self.cfg.seg_colors.get(seg.name, (100, 100, 100))
            cv2.rectangle(img, (0, y0), (4, h), c, -1)
        else:
            txt = f"{frame.timestamp:.2f}s  {episode.task_description}"

        max_chars = int(w / 5.8)
        if len(txt) > max_chars:
            txt = txt[:max_chars - 3] + "..."

        cv2.putText(img, txt, (10, h - 8), self.cfg.font, self.cfg.font_body,
                    (255, 255, 255), self.cfg.thick_body, cv2.LINE_AA)

        if episode.duration_seconds > 0:
            p = min(1.0, frame.timestamp / episode.duration_seconds)
            bx = int(p * w)
            cv2.line(img, (bx, y0 + 2), (bx, h - 2), (0, 220, 0), 2,
                     cv2.LINE_AA)

    # ── Backwards-compatible test helper ──────────────────────────────
    def _build_hand_description(self, hand_name, hand, grasp, contact,
                                objects=None) -> List[str]:
        display_name = f"{hand_name.upper()} HAND"
        if hand is None:
            return [display_name, "idle hovering", "not tracked"]

        if grasp is None:
            grasp_desc = "idle hovering"
        else:
            grasp_desc = (self._grasp_to_description(grasp.type)
                          if contact and contact.in_contact
                          else "idle hovering")

        if contact and contact.in_contact and contact.object_name:
            contact_desc = f"interacting with {contact.object_name}"
        elif grasp and grasp.type != "open":
            contact_desc = (f"{self._grasp_to_description(grasp.type)} "
                            "without contact")
        else:
            contact_desc = "no contact"

        return [display_name, grasp_desc, contact_desc]

    def _grasp_to_description(self, grasp_type):
        return {"power_wrap": "power grasp", "precision_pinch": "precision pinch",
                "lateral_pinch": "lateral pinch", "hook": "hook grip",
                "open": "hand open", "unknown": "adjusting grip",
                }.get(grasp_type, "manipulating")
