#!/usr/bin/env python3
"""Evaluate privacy face redaction using a second independent detector.

Compares face hits on raw source video frames vs unblurred face hits on
exported training video frames (lerobot_v3/videos/.../file-000.mp4) using OpenCV's
independent Haar Cascade face detector.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np


def evaluate_episode(raw_video_path: Path, exported_video_path: Path, sample_count: int = 50) -> dict:
    if not raw_video_path.exists():
        raise FileNotFoundError(f"Raw video not found: {raw_video_path}")
    if not exported_video_path.exists():
        raise FileNotFoundError(f"Exported video not found: {exported_video_path}")

    # Primary detector: MediaPipe Face Detection (Full range)
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)

    # Independent second detector: OpenCV Haar Cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    haar_face = cv2.CascadeClassifier(cascade_path)

    # 1. Fast sequential scan of SOURCE frames
    cap_src = cv2.VideoCapture(str(raw_video_path))
    src_total = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    step_src = max(1, src_total // sample_count)

    src_mp_hits = 0
    src_haar_hits = 0
    sampled_frames = 0

    frame_idx = 0
    while True:
        ret, frame = cap_src.read()
        if not ret or frame is None:
            break
        if frame_idx % step_src == 0:
            sampled_frames += 1
            # Resize detection frame to max 960px for speed
            h, w = frame.shape[:2]
            scale = 960.0 / max(w, h) if max(w, h) > 960 else 1.0
            if scale < 1.0:
                det_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                det_frame = frame

            rgb = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
            res = mp_face.process(rgb)
            if res and res.detections:
                src_mp_hits += len(res.detections)

            gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
            faces = haar_face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                src_haar_hits += len(faces)

            if sampled_frames >= sample_count:
                break
        frame_idx += 1
    cap_src.release()

    # 2. Fast sequential scan of EXPORTED frames with independent second detector
    cap_exp = cv2.VideoCapture(str(exported_video_path))
    exp_total = int(cap_exp.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    step_exp = max(1, exp_total // sample_count)

    exp_haar_hits = 0
    exp_unblurred_faces = 0
    exp_sampled = 0
    frame_idx = 0

    while True:
        ret, frame = cap_exp.read()
        if not ret or frame is None:
            break
        if frame_idx % step_exp == 0:
            exp_sampled += 1
            h, w = frame.shape[:2]
            scale = 960.0 / max(w, h) if max(w, h) > 960 else 1.0
            if scale < 1.0:
                det_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                det_frame = frame
            gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
            faces = haar_face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))
            for (x, y, fw, fh) in faces:
                roi = gray[y:y+fh, x:x+fw]
                lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
                exp_haar_hits += 1
                if lap_var > 100.0:  # unblurred sharp face threshold
                    exp_unblurred_faces += 1
            if exp_sampled >= sample_count:
                break
        frame_idx += 1
    cap_exp.release()
    mp_face.close()

    return {
        "episode": raw_video_path.stem,
        "sampled_frames": sampled_frames,
        "source_mediapipe_face_hits": src_mp_hits,
        "source_haar_face_hits": src_haar_hits,
        "exported_haar_detections": exp_haar_hits,
        "exported_unblurred_face_hits": exp_unblurred_faces,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate dual-detector face privacy")
    parser.add_argument("--episodes", nargs="+", default=["client_video", "my_video"])
    parser.add_argument("--sample-count", type=int, default=50)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("PRIVACY EVALUATION: SOURCE VS EXPORTED FACE DETECTION (DUAL DETECTOR)", flush=True)
    print("=" * 70, flush=True)

    for ep_name in args.episodes:
        raw_p = Path(f"data/raw_videos/{ep_name}.mp4")
        v_candidates = list((Path("data/output") / ep_name / "lerobot_v3" / "videos").rglob("*.mp4"))
        if not v_candidates:
            print(f"[-] No exported lerobot video found for {ep_name}", flush=True)
            continue
        exp_p = v_candidates[0]
        res = evaluate_episode(raw_p, exp_p, sample_count=args.sample_count)
        print(f"\nEpisode: {ep_name}", flush=True)
        print(f"  Source frames sampled:          {res['sampled_frames']}", flush=True)
        print(f"  Source face hits (MediaPipe):   {res['source_mediapipe_face_hits']}", flush=True)
        print(f"  Source face hits (Haar/Indep):  {res['source_haar_face_hits']}", flush=True)
        print(f"  Exported unblurred hits (Haar): {res['exported_unblurred_face_hits']}", flush=True)
        print(f"  Privacy status:                 {'PROTECTED (0 unblurred faces)' if res['exported_unblurred_face_hits'] == 0 else 'UNPROTECTED'}", flush=True)


if __name__ == "__main__":
    main()
