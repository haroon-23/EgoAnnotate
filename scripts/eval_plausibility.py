"""eval_plausibility.py — Automated plausibility and consistency checks for pipeline outputs.

PROXY METRICS — NOT VALIDATED AGAINST HUMAN GROUND TRUTH.
These checks catch implausible or internally inconsistent outputs, but do NOT
confirm real-world detection accuracy against any human-labeled reference.

Usage:
    python scripts/eval_plausibility.py --output-dir data/output/<episode_id>
                                         --video data/raw_videos/<video>.mp4
    python scripts/eval_plausibility.py --output-dir data/output/test_10s \
                                         --video data/raw_videos/test_10s.mp4

Optional:
    --no-vlm       Skip Task 2 Gemini cross-check (saves API quota)
    --vlm-frames N Number of frames to sample for VLM check (default: 10)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCLAIMER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROXY METRICS — NOT VALIDATED AGAINST HUMAN GROUND TRUTH                  ║
║  These checks catch implausible/inconsistent outputs but do NOT confirm     ║
║  real-world accuracy.  All flagged issues are HEURISTIC signals, not        ║
║  confirmed errors.  Human review is required for true accuracy assessment.  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

BODY_TERMS = {"person", "human", "arm", "body", "hand", "torso", "face", "head"}
MIN_CONTACT_FRAMES = 3        # < 3 consecutive frames = likely jitter
BBOX_AREA_THRESH = 0.70       # bbox covering >70% of frame = suspect
MARGINAL_CONFIDENCE = 0.3     # contact confidence below this = marginal
MIN_SEGMENT_SEC = 0.2         # segments shorter than this = likely noise
LARGE_BBOX_JUMP = 0.15        # normalized bbox-centre jump threshold per frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _section(title: str):
    print(f"\n{'═'*78}")
    print(f"  {title}")
    print(f"{'═'*78}")


def _flag(msg: str):
    print(f"  ⚠ FLAG  {msg}")


def _ok(msg: str):
    print(f"  ✓ OK    {msg}")


def _info(msg: str):
    print(f"  ·       {msg}")


def _pct(n, total) -> str:
    if total == 0:
        return "0.0% (0/0)"
    return f"{100*n/total:.1f}% ({n}/{total})"


# ---------------------------------------------------------------------------
# Task 1a — Contact physical plausibility
# ---------------------------------------------------------------------------

def check_contact_plausibility(frames: List[dict], fps: float) -> List[str]:
    """Returns list of flagged issue strings."""
    issues = []
    total = len(frames)
    video_dur_frames = total

    for side in ("left", "right"):
        key = f"{side}_contact"

        # Build run-length encoded contact events
        events = []  # list of (start_idx, length)
        in_run = False
        run_start = 0
        for i, f in enumerate(frames):
            val = bool(f.get(key, False))
            if val and not in_run:
                in_run = True
                run_start = i
            elif not val and in_run:
                events.append((run_start, i - run_start))
                in_run = False
        if in_run:
            events.append((run_start, total - run_start))

        # 1. Short events (< MIN_CONTACT_FRAMES)
        jitter_events = [e for e in events if e[1] < MIN_CONTACT_FRAMES]
        if jitter_events:
            issues.append(
                f"[{side.upper()} CONTACT] {len(jitter_events)} event(s) < {MIN_CONTACT_FRAMES} frames "
                f"({MIN_CONTACT_FRAMES/fps:.2f}s): likely jitter/noise. "
                f"Durations: {sorted([e[1] for e in jitter_events])}"
            )

        # 2. Events spanning the full video
        frozen_events = [e for e in events if e[1] >= video_dur_frames]
        if frozen_events:
            issues.append(
                f"[{side.upper()} CONTACT] Contact event spans ENTIRE VIDEO "
                f"({video_dur_frames} frames = {video_dur_frames/fps:.1f}s): "
                f"likely stuck/frozen detection."
            )

        # 3. Contact event duration distribution
        if events:
            durations = [e[1] for e in events]
            dur_sec = [d / fps for d in durations]
            _info(
                f"{side.upper()} contact event durations (frames): "
                f"min={min(durations)}, max={max(durations)}, "
                f"median={float(np.median(durations)):.1f}, "
                f"mean={float(np.mean(durations)):.1f}  "
                f"[sec: min={min(dur_sec):.2f}, max={max(dur_sec):.2f}, "
                f"median={float(np.median(dur_sec)):.2f}]"
            )
            all_one_frame = sum(1 for d in durations if d == 1)
            if all_one_frame / max(len(durations), 1) > 0.5:
                issues.append(
                    f"[{side.upper()} CONTACT] >50% of events are exactly 1 frame long "
                    f"({all_one_frame}/{len(durations)}): distribution strongly suggests "
                    f"detector instability, not real contact signal."
                )
        else:
            _info(f"{side.upper()} contact: no contact events detected.")

        # 4. Marginal confidence: contact_confidence not exported — proxy via
        #    frames where in_contact=True but object_name is None (proxy for low-conf fallback)
        obj_key = f"{side}_contact_object"
        no_obj_contact = sum(
            1 for f in frames
            if bool(f.get(key, False)) and f.get(obj_key) is None
        )
        total_contact = sum(1 for f in frames if bool(f.get(key, False)))
        if total_contact > 0 and no_obj_contact > 0:
            pct = 100 * no_obj_contact / total_contact
            msg = (
                f"[{side.upper()} CONTACT] {_pct(no_obj_contact, total_contact)} of in-contact frames "
                f"have no identified object (in_contact=True, object=None). "
                f"These are fallback detections and are inherently low-confidence."
            )
            if pct > 30:
                issues.append(msg)
            else:
                _info(msg)

    return issues


# ---------------------------------------------------------------------------
# Task 1b — Object detection plausibility
# ---------------------------------------------------------------------------

def check_object_plausibility(frames: List[dict]) -> List[str]:
    """
    Note: frame_annotations.json does not export per-object bbox fields or
    confidence scores directly — those are held in the in-memory pipeline and
    used by contact_detector but not serialised to JSON today.

    We derive proxies from what IS available:
      - Whether any contact_object was identified per frame
      - language_instruction for object mention consistency
    """
    issues = []
    total = len(frames)

    # % frames with NO detected object on either side
    no_obj_frames = sum(
        1 for f in frames
        if not f.get("left_contact_object") and not f.get("right_contact_object")
    )
    _info(
        f"Frames with zero identified objects (both sides null): "
        f"{_pct(no_obj_frames, total)}"
    )
    if no_obj_frames / max(total, 1) > 0.5:
        issues.append(
            f"[OBJECT DETECTION] {_pct(no_obj_frames, total)} of frames have NO identified "
            f"contact object on either hand. This is a high null-detection rate and may indicate "
            f"the object candidate list did not match video content."
        )

    # Note about bboxes not being in the JSON
    _info(
        "BBOX area and jitter checks: per-object bbox data is not currently written "
        "to frame_annotations.json. The pipeline uses bboxes internally for contact "
        "detection but does not serialise them. These checks run on live pipeline output "
        "only (see Task 2 VLM check for cross-validation of spatial claims)."
    )
    _info(
        "ACTION ITEM: To enable bbox plausibility checks, add 'detected_objects' with "
        "bbox fields to dataset_exporter._export_frames() output."
    )

    return issues


# ---------------------------------------------------------------------------
# Task 1c — Segment/action plausibility
# ---------------------------------------------------------------------------

def check_segment_plausibility(segments: List[dict], metadata: dict) -> List[str]:
    issues = []
    total_dur = metadata.get("duration_seconds", 0)

    if not segments:
        issues.append("[SEGMENTS] No action segments found. Total segmentation failure.")
        return issues

    durations = [s.get("end_time", 0) - s.get("start_time", 0) for s in segments]
    labels = [s.get("name", "") for s in segments]
    descriptions = [s.get("description", "") for s in segments]

    _info(
        f"Segment count: {len(segments)}, "
        f"durations (sec): min={min(durations):.2f}, max={max(durations):.2f}, "
        f"median={float(np.median(durations)):.2f}, mean={float(np.mean(durations)):.2f}"
    )

    # Short segments
    short = [d for d in durations if d < MIN_SEGMENT_SEC]
    if short:
        issues.append(
            f"[SEGMENTS] {len(short)} segment(s) shorter than {MIN_SEGMENT_SEC}s (noise threshold): "
            f"{[f'{d:.3f}s' for d in sorted(short)]}"
        )

    # Single segment spanning the entire video
    full_span = [s for s, d in zip(segments, durations) if d >= total_dur * 0.98]
    if full_span and len(segments) == 1:
        issues.append(
            f"[SEGMENTS] Single segment spanning the full video ({durations[0]:.1f}s = "
            f"{100*durations[0]/max(total_dur,1):.0f}% of {total_dur:.1f}s). "
            f"Segmentation has produced NO meaningful boundaries — total failure."
        )

    # Repeated labels
    unique_labels = set(labels)
    unique_descs = set(descriptions)
    if len(unique_labels) == 1:
        issues.append(
            f"[SEGMENTS] ALL {len(segments)} segments have the SAME label: "
            f"'{labels[0]}'. The labeler produced no differentiation across the video."
        )
    elif len(unique_descs) == 1 and len(segments) > 1:
        issues.append(
            f"[SEGMENTS] ALL {len(segments)} segments have the SAME description: "
            f"'{descriptions[0]}'. Labels vary ({unique_labels}) but descriptions are identical."
        )
    else:
        _info(f"Unique labels: {sorted(unique_labels)}")
        _info(f"Unique descriptions: {sorted(unique_descs)}")

    return issues


# ---------------------------------------------------------------------------
# Task 1d — Caption plausibility
# ---------------------------------------------------------------------------

def check_caption_plausibility(frames: List[dict]) -> List[str]:
    issues = []
    total = len(frames)

    # Collect unique instructions and their frame counts
    instr_counts: Dict[str, int] = {}
    for f in frames:
        instr = (f.get("language_instruction") or "").strip()
        instr_counts[instr] = instr_counts.get(instr, 0) + 1

    _info(f"Unique language instructions: {len(instr_counts)}")
    for instr, cnt in sorted(instr_counts.items(), key=lambda x: -x[1]):
        _info(f"  '{instr}' → {_pct(cnt, total)} of frames")

    # 1. Body-part-as-object tripwire
    flagged_body = []
    for instr in instr_counts:
        instr_lower = instr.lower()
        # Look for patterns like "manipulate the <body_term>" or "gripping the <body_term>"
        # Simple check: any body term appears as the apparent object (not as part of "hand touches")
        words = instr_lower.replace(",", " ").replace(".", " ").split()
        for i, w in enumerate(words):
            if w in BODY_TERMS:
                # Check if the word is being used as the manipulated object
                # (preceded by 'the', 'a', 'an', or action verb context)
                if i > 0 and words[i-1] in {"the", "a", "an", "this"}:
                    flagged_body.append(instr)
                    break
    if flagged_body:
        issues.append(
            f"[CAPTION] {len(flagged_body)} instruction(s) describe a body part as the "
            f"OBJECT being manipulated — this is the exact 'gripping the person' failure mode:\n"
            + "\n".join(f"    → '{i}'" for i in flagged_body)
        )

    # 2. Caption-object mismatch: mentioned object not in detected object set
    # Build per-frame object set from contact_object fields
    mismatch_frames = 0
    mismatch_examples = []
    for f in frames[:]:  # scan all, but cap examples
        instr = (f.get("language_instruction") or "").strip().lower()
        detected_objs = set(filter(None, [
            (f.get("left_contact_object") or "").lower(),
            (f.get("right_contact_object") or "").lower(),
        ]))
        # Skip frames with no detected objects (can't compare)
        if not detected_objs or not instr:
            continue
        # Check if ANY detected object name appears anywhere in the instruction
        if not any(obj in instr for obj in detected_objs if obj):
            mismatch_frames += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append(
                    f"frame {f['frame_idx']}: instr='{instr}', "
                    f"detected={detected_objs}"
                )

    frames_with_objs = sum(
        1 for f in frames
        if f.get("left_contact_object") or f.get("right_contact_object")
    )
    if mismatch_frames > 0:
        pct = 100 * mismatch_frames / max(frames_with_objs, 1)
        msg = (
            f"[CAPTION] {mismatch_frames} frame(s) ({pct:.1f}% of frames with objects) "
            f"have instructions that don't mention any detected object. "
            f"Caption and detection data may have disconnected.\n"
            + "\n".join(f"    → {ex}" for ex in mismatch_examples)
        )
        if pct > 20:
            issues.append(msg)
        else:
            _info(msg)

    return issues


# ---------------------------------------------------------------------------
# Task 2 — Lightweight VLM cross-check
# ---------------------------------------------------------------------------

def vlm_cross_check(
    video_path: str,
    frames: List[dict],
    n_frames: int = 10,
) -> Tuple[int, int, List[str]]:
    """
    Sample n_frames evenly, ask Gemini independently: "Is a hand touching or
    manipulating an object in this image? If yes, which object? Answer in one
    short sentence."

    Returns (agreements, disagreements, side_by_side_lines)
    """
    try:
        import google.generativeai as genai
        from PIL import Image
    except ImportError:
        print("  [VLM] google-generativeai or Pillow not installed — skipping.")
        return 0, 0, []

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Try loading from .env
        for env_path in [
            Path(__file__).resolve().parent.parent / ".env",
            Path(".env"),
        ]:
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            api_key = line.split("=", 1)[1].strip("'\"")
                            if api_key:
                                os.environ["GEMINI_API_KEY"] = api_key
                                break
            if api_key:
                break

    if not api_key:
        print("  [VLM] GEMINI_API_KEY not found — skipping VLM cross-check.")
        return 0, 0, []

    genai.configure(api_key=api_key, transport='rest')
    model_name = "models/gemini-flash-lite-latest"
    model = genai.GenerativeModel(model_name)

    total = len(frames)
    sample_indices = list(np.linspace(0, total - 1, min(n_frames, total), dtype=int))

    # Open video for frame extraction
    cap = cv2.VideoCapture(str(video_path))

    PROMPT = (
        "Is a hand touching or manipulating an object in this image? "
        "If yes, which object? Answer in one short sentence."
    )

    agreements = 0
    disagreements = 0
    side_by_side = []

    for sample_rank, frame_idx in enumerate(sample_indices):
        f = frames[frame_idx]
        pipeline_contact = bool(f.get("left_contact") or f.get("right_contact"))
        pipeline_object = (
            f.get("right_contact_object")
            or f.get("left_contact_object")
            or "none"
        )
        pipeline_instr = f.get("language_instruction", "")

        print(f"  [VLM] ({sample_rank+1}/{len(sample_indices)}) Processing frame {frame_idx}...")

        # Read frame from video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = cap.read()
        if not ret:
            # Try loading from image_path as fallback
            img_path = f.get("image_path", "")
            if img_path and Path(img_path).exists():
                bgr = cv2.imread(img_path)
            if bgr is None:
                print(f"  [VLM] Could not read frame {frame_idx}, skipping.")
                continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        if pil_img.width > 1024 or pil_img.height > 1024:
            pil_img.thumbnail((1024, 1024))

        print(f"  [VLM] Sending frame {frame_idx} ({pil_img.width}x{pil_img.height}) to Gemini...")
        try:
            response = model.generate_content(
                [PROMPT, pil_img],
                generation_config={"temperature": 0.1},
                request_options={"timeout": 10.0},
            )
            vlm_answer = response.text.strip()
            print(f"  [VLM] Gemini response: {vlm_answer}")
        except Exception as e:
            print(f"  [VLM] Frame {frame_idx}: API error: {e}")
            time.sleep(10)
            continue

        # Simple agreement heuristic:
        # Agreement if VLM and pipeline both say contact, or both say no contact
        vlm_lower = vlm_answer.lower()
        vlm_says_contact = not any(
            neg in vlm_lower
            for neg in ["no hand", "no contact", "not touching", "no object", "cannot see", "can't see", "no visible"]
        )

        agree = (vlm_says_contact == pipeline_contact)
        print(f"  [VLM] Agreement check: VLM contact={vlm_says_contact}, Pipeline contact={pipeline_contact}. Match={agree}")
        if agree:
            agreements += 1
        else:
            disagreements += 1
            side_by_side.append(
                f"  Frame {frame_idx} (t={f.get('timestamp', 0):.2f}s):\n"
                f"    Pipeline : contact={pipeline_contact}, object='{pipeline_object}', "
                f"instr='{pipeline_instr}'\n"
                f"    Gemini   : {vlm_answer}"
            )

        print(f"  [VLM] Sleeping 10s to respect rate limits...")
        time.sleep(10)  # respect rate limits

    cap.release()
    return agreements, disagreements, side_by_side


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(
    output_dir: Path,
    video_path: Optional[Path],
    run_vlm: bool,
    vlm_frames: int,
):
    print(DISCLAIMER)

    episode_id = output_dir.name
    print(f"Episode:    {episode_id}")
    print(f"Output dir: {output_dir}")
    if video_path:
        print(f"Video:      {video_path}")
    print()

    # Load data
    frames_path = output_dir / "frame_annotations.json"
    segments_path = output_dir / "action_segments.json"
    metadata_path = output_dir / "metadata.json"

    if not frames_path.exists():
        print(f"ERROR: {frames_path} not found. Run the pipeline first.")
        sys.exit(1)

    frames = _load_json(frames_path)
    segments = _load_json(segments_path) if segments_path.exists() else []
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}

    fps = metadata.get("num_frames", len(frames)) / max(metadata.get("duration_seconds", 1), 0.001)
    total_dur = metadata.get("duration_seconds", len(frames) / fps)

    _info(f"Total frames: {len(frames)}, FPS: {fps:.2f}, Duration: {total_dur:.2f}s")

    all_issues = []

    # ── TASK 1a: Contact plausibility ──────────────────────────────────────
    _section("TASK 1a — Contact Physical Plausibility")
    issues = check_contact_plausibility(frames, fps)
    for iss in issues:
        _flag(iss)
    if not issues:
        _ok("No contact plausibility issues detected.")
    all_issues.extend(issues)

    # ── TASK 1b: Object detection plausibility ─────────────────────────────
    _section("TASK 1b — Object Detection Plausibility")
    issues = check_object_plausibility(frames)
    for iss in issues:
        _flag(iss)
    if not issues:
        _ok("No object detection plausibility issues detected.")
    all_issues.extend(issues)

    # ── TASK 1c: Segment plausibility ──────────────────────────────────────
    _section("TASK 1c — Segment / Action Plausibility")
    issues = check_segment_plausibility(segments, metadata)
    for iss in issues:
        _flag(iss)
    if not issues:
        _ok("No segment plausibility issues detected.")
    all_issues.extend(issues)

    # ── TASK 1d: Caption plausibility ──────────────────────────────────────
    _section("TASK 1d — Caption Plausibility")
    issues = check_caption_plausibility(frames)
    for iss in issues:
        _flag(iss)
    if not issues:
        _ok("No caption plausibility issues detected.")
    all_issues.extend(issues)

    # ── TASK 2: VLM cross-check ────────────────────────────────────────────
    if run_vlm and video_path and video_path.exists():
        _section(f"TASK 2 — VLM Cross-check (Gemini, {vlm_frames} sampled frames)")
        print(f"  Querying Gemini on {vlm_frames} evenly-spaced frames...")
        print(f"  (10s delay between frames to respect API rate limits)")
        agreements, disagreements, side_by_side = vlm_cross_check(
            str(video_path), frames, vlm_frames
        )
        checked = agreements + disagreements
        if checked > 0:
            _info(f"Agreement rate: {_pct(agreements, checked)}")
            _info(f"Disagreements:  {_pct(disagreements, checked)}")
            if disagreements > 0:
                _flag(
                    f"{disagreements} frame(s) where Gemini's independent answer "
                    f"disagrees with pipeline output:"
                )
                for line in side_by_side:
                    print(line)
            else:
                _ok("Gemini agreed with pipeline on all sampled frames.")
            if disagreements / max(checked, 1) > 0.3:
                all_issues.append(
                    f"[VLM CHECK] {_pct(disagreements, checked)} disagreement rate "
                    f"between Gemini cross-check and pipeline output (>30% threshold)."
                )
        else:
            _info("No frames checked (API unavailable or all reads failed).")
    elif run_vlm and (not video_path or not video_path.exists()):
        _section("TASK 2 — VLM Cross-check")
        print("  Skipped: video file not found. Pass --video to enable.")
    else:
        _section("TASK 2 — VLM Cross-check")
        print("  Skipped (--no-vlm flag set).")

    # ── Summary ────────────────────────────────────────────────────────────
    _section("SUMMARY")
    print(DISCLAIMER)
    if not all_issues:
        print("  No plausibility issues flagged.")
    else:
        print(f"  {len(all_issues)} issue(s) flagged:\n")
        for i, iss in enumerate(all_issues, 1):
            print(f"  [{i}] {iss}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Automated plausibility checks for pipeline output (no ground truth needed)."
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Path to episode output directory (e.g. data/output/test_10s)"
    )
    parser.add_argument(
        "--video", default=None,
        help="Path to the raw video file (required for VLM cross-check and bbox checks)"
    )
    parser.add_argument(
        "--no-vlm", action="store_true",
        help="Skip Task 2 Gemini VLM cross-check"
    )
    parser.add_argument(
        "--vlm-frames", type=int, default=10,
        help="Number of evenly-spaced frames to send to Gemini for cross-check (default: 10)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    video_path = Path(args.video) if args.video else None

    run_eval(
        output_dir=output_dir,
        video_path=video_path,
        run_vlm=not args.no_vlm,
        vlm_frames=args.vlm_frames,
    )


if __name__ == "__main__":
    main()
