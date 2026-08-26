"""Egocentric HUD visualization overlay for annotated episodes."""
import cv2
import numpy as np
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from tqdm import tqdm

from .datatypes import AnnotatedEpisode, AnnotationFrame, HandLandmarks, ContactState, ObjectAnnotation, GraspType, ActionSegment


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
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(str(output_path), fourcc, 30, (w, h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, 30, (w, h))
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

    # ── Contact overlay with bboxes and fingertips ───────────────────────
    def draw_contact_overlay(
        self,
        img: np.ndarray,
        hands: Dict[str, Optional[HandLandmarks]],
        objects: List[ObjectAnnotation],
        contact_states: Dict[str, Optional[ContactState]],
        confidence_threshold: float = 0.5,
    ) -> np.ndarray:
        """Draw object bboxes and contact fingertips on the frame.
        
        Args:
            img: BGR image to draw on (modified in place)
            hands: Dict with "left" and "right" HandLandmarks
            objects: List of ObjectAnnotation with bbox
            contact_states: Dict with "left" and "right" ContactState
            confidence_threshold: Min confidence for bright contact visualization
            
        Returns:
            Modified image with overlay
        """
        h, w = img.shape[:2]
        
        # Colors for contact visualization
        CONTACT_BRIGHT = (0, 255, 0)      # Bright green
        CONTACT_DIM = (100, 100, 100)     # Gray
        BBOX_THICKNESS = 2
        FINGERTIP_RADIUS = 5
        
        # Draw each object bbox
        for obj in objects:
            if obj.bbox is None:
                continue
            
            x_min, y_min, x_max, y_max = obj.bbox
            # Convert normalized to pixel coords
            x1 = int(x_min * w)
            y1 = int(y_min * h)
            x2 = int(x_max * w)
            y2 = int(y_max * h)
            
            # Check if this object is in contact (either hand)
            in_contact = False
            max_conf = 0.0
            for hand_name in ["left", "right"]:
                cs = contact_states.get(hand_name)
                if cs and cs.in_contact and cs.object_name == obj.name:
                    in_contact = True
                    max_conf = max(max_conf, cs.confidence)
            
            # Choose color based on contact and confidence
            if in_contact and max_conf >= confidence_threshold:
                bbox_color = CONTACT_BRIGHT
                tip_color = CONTACT_BRIGHT
            else:
                bbox_color = CONTACT_DIM
                tip_color = CONTACT_DIM
            
            # Draw bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, BBOX_THICKNESS, cv2.LINE_AA)
            
            # Draw object label
            label = f"{obj.name}"
            if in_contact:
                label += f" ({max_conf:.2f})"
            cv2.putText(img, label, (x1, max(0, y1 - 5)), self.cfg.font, 0.4,
                        bbox_color, 1, cv2.LINE_AA)
            
            # Draw fingertips that are in contact with this object
            for hand_name in ["left", "right"]:
                hand = hands.get(hand_name)
                cs = contact_states.get(hand_name)
                
                if hand is None or cs is None:
                    continue
                
                if cs.in_contact and cs.object_name == obj.name:
                    tips = hand.fingertip_positions()  # (5, 3)
                    for f_idx in range(5):
                        if cs.fingers[f_idx]:
                            px = int(tips[f_idx, 0] * w)
                            py = int(tips[f_idx, 1] * h)
                            cv2.circle(img, (px, py), FINGERTIP_RADIUS, tip_color, -1, cv2.LINE_AA)
                            cv2.circle(img, (px, py), FINGERTIP_RADIUS + 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        return img

    # ── Detailed segment timeline ───────────────────────────────────────
    def draw_segment_timeline(
        self,
        img: np.ndarray,
        current_frame_idx: int,
        segments: List[ActionSegment],
        fps: float,
        total_frames: int,
    ) -> np.ndarray:
        """Draw detailed segment timeline at bottom of frame.
        
        Shows all segments as colored blocks with a playhead.
        """
        h, w = img.shape[:2]
        timeline_h = 40  # taller for detailed view
        y0 = h - timeline_h
        
        # Background
        overlay = img.copy()
        cv2.rectangle(overlay, (0, y0), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
        cv2.line(img, (0, y0), (w, y0), (60, 60, 60), 1, cv2.LINE_AA)
        
        if not segments:
            return img
        
        # Draw each segment as a colored block
        for seg in segments:
            # Convert segment times to pixel positions
            start_px = int((seg.start_time / (total_frames / fps)) * w) if total_frames > 0 else 0
            end_px = int((seg.end_time / (total_frames / fps)) * w) if total_frames > 0 else w
            start_px = max(0, min(w - 1, start_px))
            end_px = max(0, min(w, end_px))
            
            if end_px <= start_px:
                continue
            
            color = self.cfg.seg_colors.get(seg.name, (100, 100, 100))
            cv2.rectangle(img, (start_px, y0 + 2), (end_px, h - 2), color, -1, cv2.LINE_AA)
            
            # Segment label
            label = seg.name
            text_size = cv2.getTextSize(label, self.cfg.font, 0.35, 1)[0]
            text_x = start_px + 2
            text_y = y0 + 14
            if text_x + text_size[0] < end_px:
                cv2.putText(img, label, (text_x, text_y), self.cfg.font, 0.35,
                            (255, 255, 255), 1, cv2.LINE_AA)
        
        # Playhead (current frame)
        current_px = int((current_frame_idx / total_frames) * w) if total_frames > 0 else 0
        current_px = max(0, min(w - 1, current_px))
        cv2.line(img, (current_px, y0), (current_px, h), (0, 255, 255), 2, cv2.LINE_AA)
        
        # Time markers
        cv2.putText(img, f"{current_frame_idx}/{total_frames}  {current_frame_idx/fps:.2f}s",
                    (10, h - 6), self.cfg.font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        return img

    # ── Grasp label in top-left ─────────────────────────────────────────
    def draw_grasp_label(
        self,
        img: np.ndarray,
        hands: Dict[str, Optional[HandLandmarks]],
        grasps: Dict[str, Optional[GraspType]],
        contact_states: Dict[str, Optional[ContactState]],
    ) -> np.ndarray:
        """Draw current grasp type and confidence for each hand in top-left."""
        h, w = img.shape[:2]
        
        for idx, hand_name in enumerate(["left", "right"]):
            hand = hands.get(hand_name)
            grasp = grasps.get(hand_name)
            contact = contact_states.get(hand_name)
            
            if hand is None:
                continue
            
            # Determine label
            if grasp is not None:
                gtype = grasp.type
                if contact and contact.in_contact:
                    label = f"{hand_name.upper()}: {gtype} ({contact.confidence:.2f})"
                else:
                    label = f"{hand_name.upper()}: {gtype} (hover)"
            else:
                label = f"{hand_name.upper()}: no detection"
            
            # Position: top-left, stacked
            y = 25 + idx * 22
            x = 10
            
            # Background
            overlay = img.copy()
            cv2.rectangle(overlay, (x - 4, y - 18), (x + 280, y + 4), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            
            # Color based on contact
            if contact and contact.in_contact:
                color = (0, 255, 0)  # Green
            else:
                color = (200, 200, 200)  # Gray
            
            cv2.putText(img, label, (x, y), self.cfg.font, 0.5, color, 1, cv2.LINE_AA)
        
        return img

    def _render_frame_with_all_overlays(
        self,
        img: np.ndarray,
        frame: AnnotationFrame,
        episode: AnnotatedEpisode,
        total_frames: int,
    ) -> np.ndarray:
        """Extended render frame with all new overlays."""
        h, w = img.shape[:2]
        
        # 1. Draw skeletons (existing)
        self._draw_skeletons(img, frame, w, h)
        
        # 2. Draw contact overlay (NEW)
        hands_dict = {"left": frame.left_hand, "right": frame.right_hand}
        contact_states = {"left": frame.left_contact, "right": frame.right_contact}
        grasps_dict = {"left": frame.left_grasp, "right": frame.right_grasp}
        
        self.draw_contact_overlay(img, hands_dict, frame.objects, contact_states)
        
        # 3. Draw grasp labels (NEW)
        self.draw_grasp_label(img, hands_dict, grasps_dict, contact_states)
        
        # 4. Draw panels (existing HUD)
        self._draw_panels(img, frame, w, h, episode)
        
        # 5. Draw detailed segment timeline (NEW)
        self.draw_segment_timeline(img, frame.frame_idx, episode.segments, 30.0, total_frames)
        
        return img


def render_annotated_video(
    video_path: str,
    episode: AnnotatedEpisode,
    output_path: Path,
    fps: float = 30.0,
) -> Path:
    """Create a burned-in overlay video from original video + pipeline episode.
    
    Args:
        video_path: Path to original video file
        episode: AnnotatedEpisode from pipeline
        output_path: Where to save the overlay MP4
        fps: Output video FPS
        
    Returns:
        Path to created video
    """
    visualizer = EgoVisualizer()
    
    # Open original video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use temporary file for initial OpenCV output
    raw_tmp_path = output_path.with_name(output_path.stem + "_raw.mp4")

    # Create writer using mp4v fourcc
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(raw_tmp_path), fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {raw_tmp_path}")

    # Map episode frames to original video frames
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    sample_every = max(1, int(round(orig_fps / fps)))

    pbar = tqdm(total=len(episode.frames), desc="Rendering overlay", leave=False)

    try:
        for ep_frame in episode.frames:
            orig_frame = None
            if ep_frame.image_path and Path(ep_frame.image_path).exists():
                orig_frame = cv2.imread(ep_frame.image_path)

            if orig_frame is None and cap.isOpened():
                ret, frame_cap = cap.read()
                if ret:
                    orig_frame = frame_cap

            if orig_frame is None:
                pbar.update(1)
                continue

            if orig_frame.shape[1] != orig_w or orig_frame.shape[0] != orig_h:
                orig_frame = cv2.resize(orig_frame, (orig_w, orig_h))

            rendered = visualizer._render_frame_with_all_overlays(
                orig_frame, ep_frame, episode, total_frames
            )

            writer.write(rendered)
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()

    # Convert to native H.264 (avc1 / yuv420p / faststart) for 100% macOS QuickTime compatibility
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(raw_tmp_path),
            "-c:v", "libx264", "-profile:v", "main", "-level", "4.0",
            "-pix_fmt", "yuv420p", "-tag:v", "avc1", "-movflags", "+faststart",
            "-loglevel", "error", str(output_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and output_path.exists():
            if raw_tmp_path.exists():
                raw_tmp_path.unlink()
        else:
            # Fallback if ffmpeg fails: move raw file to output_path
            if raw_tmp_path.exists():
                raw_tmp_path.replace(output_path)
    except Exception:
        if raw_tmp_path.exists():
            raw_tmp_path.replace(output_path)

    print(f"[Visualizer] Saved overlay video: {output_path}")
    return output_path
