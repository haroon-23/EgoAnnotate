"""Structured JSON logging, HTML QA report, offline interactive 3D viewer, and full-video QA overlay."""

from __future__ import annotations

import html
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from src.egoannotate.schemas import EpisodeIR


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, separators=(",", ":"))


def configure_json_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("egoannotate")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonLogFormatter())
        logger.addHandler(handler)
    return logger


def generate_html_qa_report(ir: EpisodeIR, out_path: Path) -> Path:
    """Generate a self-contained offline HTML QA report covering quality, segments, grasps, retargeting & privacy."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dims = ir.quality.dimensions.model_dump()
    dim_rows = "\n".join(
        f"<tr><td>{html.escape(k)}</td><td>{v:.3f}</td>"
        f"<td><div class='bar'><div class='fill' style='width:{round(v * 100)}%'></div></div></td></tr>"
        for k, v in dims.items()
    )

    ret_rows = "\n".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(str(m.get('embodiment_class', '')))}</td>"
        f"<td>{m.get('dof', 0)}</td><td>{m.get('mean_fk_pos_error_mm', 0.0)} mm</td>"
        f"<td>{m.get('joint_limit_compliance_pct', 0.0)}%</td>"
        f"<td><b>{m.get('overall_feasibility', 0.0):.3f}</b></td></tr>"
        for name, m in ir.retargeting_summary.items()
    )

    seg_rows = "\n".join(
        f"<tr><td>{s.segment_id}</td><td>{s.start_s:.2f}s - {s.end_s:.2f}s</td>"
        f"<td><b>{html.escape(s.verb)}</b></td><td>{html.escape(s.acting_hand)}</td>"
        f"<td>{html.escape(s.grasp_type)}</td><td>{html.escape(s.bimanual_coordination)}</td>"
        f"<td>{s.geometry_agreement_score:.2f}</td><td>{html.escape(s.semantic_description)}</td></tr>"
        for s in ir.segments[:100]
    )

    verdict_cls = ir.quality.verdict.lower()
    tactile_src = ir.frames[0].tactile_right.source if ir.frames else "derived_rgb_proxy"
    priv_ok = bool(ir.privacy_audit.get("privacy_pass", False))
    priv_faces = ir.privacy_audit.get("verified_unblurred_faces", "N/A")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>EgoAnnotate v3 QA Report — {html.escape(ir.episode_id)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #111317; color: #e6e8ec; margin: 24px; line-height: 1.5; }}
  h1, h2 {{ color: #f5f7fa; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 16px 0; }}
  .card {{ background: #1a1e24; border: 1px solid #2a303b; border-radius: 8px; padding: 14px; }}
  .label {{ font-size: 12px; color: #9aa4b2; text-transform: uppercase; letter-spacing: 0.05em; }}
  .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  .pass {{ color: #4ade80; }} .warn {{ color: #fbbf24; }} .quarantine {{ color: #f87171; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid #2a303b; padding: 8px 10px; text-align: left; }}
  th {{ color: #9aa4b2; background: #161a20; }}
  .bar {{ width: 140px; height: 8px; background: #262c36; border-radius: 4px; overflow: hidden; }}
  .fill {{ height: 100%; background: #60a5fa; }}
</style>
</head>
<body>
  <h1>EgoAnnotate v3 — Episode QA Report</h1>
  <p><b>Episode:</b> {html.escape(ir.episode_id)} &nbsp;|&nbsp; <b>Profile:</b> {html.escape(ir.profile_used)} ({html.escape(ir.device_used)}) &nbsp;|&nbsp; <b>Created:</b> {html.escape(ir.created_utc)}</p>
  <p><b>Task Goal:</b> {html.escape(ir.language.task_goal)}</p>

  <div class="grid">
    <div class="card"><div class="label">Robot Readiness</div><div class="value {verdict_cls}">{ir.quality.robot_readiness_score:.3f} ({ir.quality.verdict})</div></div>
    <div class="card"><div class="label">Full Video Coverage</div><div class="value">{ir.video.processed_frames} / {ir.video.total_frames} ({ir.video.duration_s:.1f}s)</div></div>
    <div class="card"><div class="label">Privacy Gate</div><div class="value {'pass' if priv_ok else 'quarantine'}">{'PASS' if priv_ok else 'CHECK'} ({priv_faces} faces)</div></div>
    <div class="card"><div class="label">Scale Provenance</div><div class="value" style="font-size:15px">{html.escape(ir.scale_provenance.scale_source)} (&plusmn;{ir.scale_provenance.scale_uncertainty_pct:.0f}%)</div></div>
    <div class="card"><div class="label">Tactile Channel</div><div class="value" style="font-size:15px">{html.escape(tactile_src)}</div></div>
    <div class="card"><div class="label">Behavior Events</div><div class="value">{len(ir.segments)} segs / {len(ir.grasp_events)} grasps</div></div>
  </div>

  <h2>10-Dimension Quality Breakdown</h2>
  <table><thead><tr><th>Dimension</th><th>Score</th><th>Visual</th></tr></thead><tbody>{dim_rows}</tbody></table>

  <h2>Multi-Embodiment Retargeting Feasibility</h2>
  <table><thead><tr><th>Embodiment</th><th>Class</th><th>DoF</th><th>Mean FK Err</th><th>Joint Limits</th><th>Feasibility</th></tr></thead><tbody>{ret_rows}</tbody></table>

  <h2>Atomic Action Segments (showing up to 100 of {len(ir.segments)})</h2>
  <table><thead><tr><th>#</th><th>Window</th><th>Verb</th><th>Hand</th><th>Feix Grasp</th><th>Coordination</th><th>Geo Agree</th><th>Description</th></tr></thead><tbody>{seg_rows}</tbody></table>
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def export_interactive_3d_viewer_html(ir: EpisodeIR, out_path: Path, max_samples: int = 900) -> Path:
    """Export a zero-CDN, 100% offline rotatable 3D HTML5 Canvas viewer of world camera + hand trajectories."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(ir.frames)
    stride = max(1, n // max_samples)
    cam_pts = []
    left_pts = []
    right_pts = []
    for i in range(0, n, stride):
        f = ir.frames[i]
        cam_pts.append([round(c, 3) for c in f.camera_pose.pos_world_m])
        if f.left_hand.detected:
            left_pts.append([round(c, 3) for c in f.left_hand.wrist_pos_world_m])
        if f.right_hand.detected:
            right_pts.append([round(c, 3) for c in f.right_hand.wrist_pos_world_m])

    sparse = [[round(c, 3) for c in p] for p in ir.scene_graph.point_cloud_sample_world_m[:350]]
    hotspots = [
        [round(c, 3) for c in h["pos_world_m"]]
        for h in ir.scene_graph.affordance_hotspots_world_m[:80]
        if "pos_world_m" in h
    ]

    scene_data = json.dumps(
        {
            "episode_id": ir.episode_id,
            "total_frames": n,
            "cam": cam_pts,
            "left": left_pts,
            "right": right_pts,
            "sparse": sparse,
            "hotspots": hotspots,
        },
        separators=(",", ":"),
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>EgoAnnotate 3D World Viewer — {html.escape(ir.episode_id)}</title>
<style>
  body {{ margin: 0; background: #0d1015; color: #e5e7eb; font-family: system-ui, sans-serif; overflow: hidden; }}
  #hud {{ position: fixed; top: 12px; left: 12px; background: rgba(18,22,28,0.88); padding: 10px 14px; border-radius: 8px; border: 1px solid #2b323d; font-size: 13px; }}
  canvas {{ display: block; width: 100vw; height: 100vh; cursor: grab; }}
  .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }}
</style>
</head>
<body>
<div id="hud">
  <b>EgoAnnotate v3 World Frame (Z-Up)</b> — {html.escape(ir.episode_id)} ({n} frames)<br/>
  <span class="dot" style="background:#60a5fa"></span>Camera VO &nbsp;
  <span class="dot" style="background:#34d399"></span>Left Hand &nbsp;
  <span class="dot" style="background:#f59e0b"></span>Right Hand &nbsp;
  <span class="dot" style="background:#f43f5e"></span>Grasp Affordances<br/>
  <small>Drag to orbit &bull; Scroll to zoom</small>
</div>
<canvas id="c"></canvas>
<script>
const DATA = {scene_data};
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let yaw = 0.7, pitch = 0.45, zoom = 260, dragging = false, lx = 0, ly = 0;

const allPts = [...DATA.cam, ...DATA.left, ...DATA.right];
let cx = 0, cy = 0, cz = 0;
if (allPts.length) {{
  for (const p of allPts) {{ cx += p[0]; cy += p[1]; cz += p[2]; }}
  cx /= allPts.length; cy /= allPts.length; cz /= allPts.length;
}}

function project(p) {{
  const x = p[0] - cx, y = p[1] - cy, z = p[2] - cz;
  const cyaw = Math.cos(yaw), syaw = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const rx = cyaw * x - syaw * y;
  const ry = syaw * x + cyaw * y;
  const rz = sp * ry + cp * z;
  const depth = cp * ry - sp * z + 4.0;
  const s = zoom / Math.max(depth, 0.3);
  return [canvas.width * 0.5 + rx * s, canvas.height * 0.5 - rz * s];
}}

function drawPolyline(pts, color, lw) {{
  if (pts.length < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
  const p0 = project(pts[0]); ctx.moveTo(p0[0], p0[1]);
  for (let i = 1; i < pts.length; i++) {{ const pi = project(pts[i]); ctx.lineTo(pi[0], pi[1]); }}
  ctx.stroke();
}}

function drawPoints(pts, color, r) {{
  ctx.fillStyle = color;
  for (const p of pts) {{ const xy = project(p); ctx.fillRect(xy[0]-r*0.5, xy[1]-r*0.5, r, r); }}
}}

function render() {{
  canvas.width = window.innerWidth; canvas.height = window.innerHeight;
  ctx.fillStyle = '#0d1015'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawPoints(DATA.sparse, '#4b5563', 2);
  drawPolyline(DATA.cam, '#60a5fa', 2.0);
  drawPolyline(DATA.left, '#34d399', 1.6);
  drawPolyline(DATA.right, '#f59e0b', 1.6);
  drawPoints(DATA.hotspots, '#f43f5e', 5);
}}

canvas.addEventListener('mousedown', e => {{ dragging = true; lx = e.clientX; ly = e.clientY; }});
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', e => {{
  if (!dragging) return;
  yaw += (e.clientX - lx) * 0.008; pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - ly) * 0.008));
  lx = e.clientX; ly = e.clientY; render();
}});
canvas.addEventListener('wheel', e => {{ zoom = Math.max(60, Math.min(1200, zoom * (e.deltaY > 0 ? 0.92 : 1.08))); render(); }});
window.addEventListener('resize', render);
render();
</script>
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def render_full_qa_overlay_video(
    ir: EpisodeIR,
    sanitized_video: Path,
    out_video: Path,
    crf: int = 28,
) -> dict[str, Any]:
    """Stream ALL frames 0..N-1 of the sanitized video and burn in skeletons, bboxes, verbs & tactile HUD."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg required to render QA overlay video.")

    cap = cv2.VideoCapture(str(sanitized_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for QA overlay: {sanitized_video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(ir.frames)
    seg_by_frame: list[str] = ["idle"] * n_frames
    grasp_by_frame: list[str] = ["none"] * n_frames
    for s in ir.segments:
        for fi in range(max(0, s.start_frame), min(n_frames, s.end_frame + 1)):
            seg_by_frame[fi] = f"{s.verb} ({s.acting_hand})"
            grasp_by_frame[fi] = s.grasp_type

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out_video),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    written = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            if written < n_frames:
                fr = ir.frames[written]
                for hand, col in ((fr.left_hand, (80, 220, 120)), (fr.right_hand, (40, 170, 255))):
                    if hand.detected and len(hand.landmarks_2d_px) == 21:
                        pts = [(round(p[0]), round(p[1])) for p in hand.landmarks_2d_px]
                        for a, b in HAND_BONES:
                            cv2.line(bgr, pts[a], pts[b], col, 2, cv2.LINE_AA)
                        for px, py in pts:
                            cv2.circle(bgr, (px, py), 3, col, -1)
                for obj in fr.objects:
                    x0, y0, x1, y1 = (round(v) for v in obj.bbox_xyxy)
                    cv2.rectangle(bgr, (x0, y0), (x1, y1), (250, 120, 180), 2)
                    cv2.putText(
                        bgr,
                        f"{obj.label} [{obj.in_hand}]",
                        (x0, max(18, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (250, 120, 180),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.rectangle(bgr, (0, 0), (w, 42), (16, 20, 26), -1)
                hud_txt = (
                    f"F:{written:04d}/{n_frames} {fr.timestamp_s:6.2f}s | "
                    f"ACT:{seg_by_frame[written]} | GRASP:{grasp_by_frame[written]} | "
                    f"TAC L:{fr.tactile_left.contact_prob:.2f} R:{fr.tactile_right.contact_prob:.2f} | "
                    f"VO:{'OK' if fr.camera_pose.tracking_ok else 'LOST'}"
                )
                cv2.putText(bgr, hud_txt, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 240, 245), 1, cv2.LINE_AA)
            proc.stdin.write(bgr.tobytes())
            written += 1
    finally:
        cap.release()
        proc.stdin.close()
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"ffmpeg failed rendering QA overlay ({rc})")
    return {"qa_overlay_mp4": str(out_video), "frames_rendered": written, "size_bytes": out_video.stat().st_size}
