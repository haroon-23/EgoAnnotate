#!/usr/bin/env python3
"""Gate 10: Privacy & DPDP Compliance (Audio stripped, 0 unblurred faces, Consent Manifest, VLM call-path blur)."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import List
import cv2
import mediapipe as mp
import numpy as np

REQUIRED_CONSENT_FIELDS = [
    "consent_id",
    "collector_id",
    "collection_date",
    "purpose",
    "license",
    "retention_policy",
    "dpdp_lawful_basis",
    "subject_acknowledged",
]


def _has_audio_stream(mp4_path: Path) -> bool:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(mp4_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return bool(res.stdout.strip())


def _detect_unblurred_faces(mp4_path: Path, sample_count: int = 25, max_laplacian_var: float = 120.0) -> int:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    idxs = np.linspace(0, max(0, total - 1), min(total, sample_count), dtype=int)

    face_det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    unblurred_count = 0

    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, bgr = cap.read()
        if not ret or bgr is None:
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = face_det.process(rgb)
        if res.detections:
            for det in res.detections:
                bb = det.location_data.relative_bounding_box
                x1 = max(0, int(bb.xmin * w))
                y1 = max(0, int(bb.ymin * h))
                x2 = min(w, int((bb.xmin + bb.width) * w))
                y2 = min(h, int((bb.ymin + bb.height) * h))
                roi = bgr[y1:y2, x1:x2]
                if roi.size > 100:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    if lap_var > max_laplacian_var:
                        unblurred_count += 1
    cap.release()
    face_det.close()
    return unblurred_count


def _check_vlm_call_path_blurred() -> List[str]:
    """Static check of src/ to verify VLM client calls receive blurred frames."""
    issues: List[str] = []
    vlm_files = [Path("src/vlm_client.py"), Path("src/perception/privacy.py"), Path("src/privacy.py")]
    has_privacy_module = any(p.exists() for p in vlm_files[1:])
    if not has_privacy_module:
        issues.append("Missing dedicated privacy/face-blur module (`src/privacy.py` or `src/perception/privacy.py`)")

    vlm_client_p = Path("src/vlm_client.py")
    if vlm_client_p.exists():
        txt = vlm_client_p.read_text(encoding="utf-8", errors="ignore")
        if "blur" not in txt.lower() and " redact" not in txt.lower() and "privacy" not in txt.lower():
            issues.append("`src/vlm_client.py` does not reference face blurring/redaction before sending frames to VLM")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 10: Privacy, Audio Stripping, & DPDP Consent Audit")
    parser.add_argument("episode_dir", type=str)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print("=== G10: PRIVACY, FACE BLUR, AUDIO STRIPPING & DPDP CONSENT GATE ===")
    ep_dir = Path(args.episode_dir)
    failures: List[str] = []

    # 1. Check all exported MP4s for audio streams and unblurred faces
    mp4s = sorted(list(ep_dir.glob("*.mp4")) + list((ep_dir / "lerobot_v3" / "videos").rglob("*.mp4")))
    if not mp4s:
        failures.append(f"No exported MP4 videos found under {ep_dir}")

    for m in mp4s:
        has_aud = _has_audio_stream(m)
        unblurred = _detect_unblurred_faces(m)
        print(f"  Video `{m.relative_to(ep_dir)}`: has_audio={has_aud}, unblurred_faces_detected={unblurred}")
        if has_aud:
            failures.append(f"Exported video `{m.name}` still contains an audio stream (must strip with ffmpeg -an)")
        if unblurred > 0:
            failures.append(f"Exported video `{m.name}` contains {unblurred} unblurred face detection(s)")

    # 2. Check per-episode consent manifest JSON
    consent_candidates = [ep_dir / "consent_manifest.json", ep_dir / "capture_meta.json"]
    consent_path = next((p for p in consent_candidates if p.exists()), None)
    if consent_path is None:
        failures.append(f"Missing `consent_manifest.json` (or `capture_meta.json`) in {ep_dir}")
    else:
        c_doc = json.loads(consent_path.read_text(encoding="utf-8"))
        c_block = c_doc.get("consent", c_doc)
        missing_f = [k for k in REQUIRED_CONSENT_FIELDS if k not in c_block]
        unset_f = [
            k for k in REQUIRED_CONSENT_FIELDS
            if k in c_block and (c_block[k] in (None, "", "UNSET", "TODO", False))
        ]
        print(f"  Consent manifest `{consent_path.name}`: missing={missing_f}, unset={unset_f}")
        if missing_f:
            failures.append(f"Consent manifest missing required DPDP fields: {missing_f}")
        if unset_f:
            failures.append(f"Consent manifest has UNSET/invalid DPDP fields: {unset_f}")

    # 3. Static check of VLM call path
    vlm_issues = _check_vlm_call_path_blurred()
    for iss in vlm_issues:
        failures.append(iss)

    passed = len(failures) == 0
    for f_msg in failures:
        print(f"  [FAIL] {f_msg}")
    print(f"G10 VERDICT: {'PASS' if passed else 'FAIL'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gate": "g10_privacy",
            "status": "PASS" if passed else "FAIL",
            "videos_checked": [str(m) for m in mp4s],
            "failures": failures,
        }, indent=2) + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
