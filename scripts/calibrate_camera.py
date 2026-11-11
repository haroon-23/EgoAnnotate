from __future__ import annotations
#!/usr/bin/env python3
"""OpenCV chessboard camera intrinsics & lens distortion calibration utility.

Computes camera matrix K (fx, fy, cx, cy), distortion coefficients (k1, k2, p1, p2, k3),
and RMS reprojection error (px) from a calibration video or directory of chessboard images,
writing a device profile YAML to `configs/devices/<device_id>.yaml`.
"""

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import yaml


def extract_calibration_frames(source: Path, max_frames: int = 40) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    if source.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for p in sorted(source.glob(ext)):
                img = cv2.imread(str(p))
                if img is not None:
                    frames.append(img)
    elif source.is_file():
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        step = max(1, total // max_frames)
        idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret or bgr is None:
                break
            if idx % step == 0:
                frames.append(bgr)
            idx += 1
        cap.release()
    else:
        raise FileNotFoundError(f"Calibration source not found: {source}")
    return frames


def calibrate_from_frames(
    frames: List[np.ndarray],
    board_cols: int,
    board_rows: int,
    square_size_m: float,
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int], int]:
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * float(square_size_m)

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    img_size: Tuple[int, int] = (0, 0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for bgr in frames:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        img_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(
            gray,
            (board_cols, board_rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_sub)

    if len(objpoints) < 5:
        raise RuntimeError(
            f"Only detected chessboard in {len(objpoints)}/{len(frames)} frames "
            f"(minimum 5 required for stable calibration)."
        )

    rms_px, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist, float(rms_px), img_size, len(objpoints)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics via OpenCV chessboard")
    parser.add_argument("--source", type=str, required=True, help="Video file or image directory of chessboard")
    parser.add_argument("--device-id", type=str, required=True, help="Device identifier (e.g. iphone14_pro_wide)")
    parser.add_argument("--cols", type=int, default=9, help="Inner chessboard corners horizontally")
    parser.add_argument("--rows", type=int, default=6, help="Inner chessboard corners vertically")
    parser.add_argument("--square-mm", type=float, default=25.0, help="Chessboard square size in millimeters")
    parser.add_argument("--out-dir", type=str, default="configs/devices", help="Output directory for device YAML")
    args = parser.parse_args()

    frames = extract_calibration_frames(Path(args.source))
    mtx, dist, rms_px, (w, h), n_used = calibrate_from_frames(
        frames, args.cols, args.rows, args.square_mm / 1000.0
    )

    fx = float(mtx[0, 0])
    fy = float(mtx[1, 1])
    cx = float(mtx[0, 2])
    cy = float(mtx[1, 2])
    dist_flat = [round(float(x), 8) for x in dist.reshape(-1)[:5]]

    profile = {
        "device_id": args.device_id,
        "calibrated": True,
        "scale_source": "calibrated_chessboard",
        "calibration_date_utc": datetime.now(timezone.utc).isoformat(),
        "image_width": int(w),
        "image_height": int(h),
        "frames_used": int(n_used),
        "reprojection_rms_px": round(rms_px, 4),
        "intrinsics": {
            "fx": round(fx, 4),
            "fy": round(fy, 4),
            "cx": round(cx, 4),
            "cy": round(cy, 4),
            "dist_coeffs": dist_flat,
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.device_id}.yaml"
    with open(out_path, "w", encoding="utf-8") as f_out:
        yaml.safe_dump(profile, f_out, sort_keys=False)

    print(json.dumps(profile, indent=2))
    print(f"Saved calibrated device profile to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
