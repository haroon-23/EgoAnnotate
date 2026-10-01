"""Manual smoke test for the LocateAnything-3B detector backend (Mac only, NOT part of CI).

Usage:
    python scripts/check_locate_anything.py --weights models/locate-anything-3b [--image path.jpg] [--label "a red mug"]

Exits 0 with a SKIP message when the weights directory is absent (no
auto-download, by design). When weights are present, runs one real
detection query and prints the boxes + latency.

NOTE: the weights are NVIDIA non-commercial, research-only — eval/research
use, not the commercial default.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from PIL import Image

from src.perception.detector import Detector2DConfig, create_detector_2d


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Smoke-test the LocateAnything-3B detector backend."
    )
    ap.add_argument("--weights", default="models/locate-anything-3b",
                    help="Local LocateAnything-3B weights directory (manual download).")
    ap.add_argument("--image", default=None,
                    help="Optional test image (default: synthetic).")
    ap.add_argument("--label", default="a red mug",
                    help="Text query to ground.")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.is_dir():
        print(f"[check_locate_anything] SKIP: weights dir '{weights}' not present.")
        print("  Download manually from https://huggingface.co/nvidia/LocateAnything-3B")
        return 0

    det = create_detector_2d(Detector2DConfig(
        backend="locate_anything", locate_anything_weights=str(weights)))
    if det is None:
        print("[check_locate_anything] SKIP: backend unavailable "
              "(transformers missing? pip install 'transformers==4.57.1').")
        return 0

    if args.image:
        img_rgb = np.asarray(Image.open(args.image).convert("RGB"))
    else:
        img_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        img_rgb[60:180, 100:220] = (200, 60, 40)  # fake "object"
    img_bgr = img_rgb[:, :, ::-1].copy()

    print(f"[check_locate_anything] backend={det.backend_name} weights={weights} "
          f"label={args.label!r} — running query (slow CPU mode)...")
    t0 = time.time()
    dets = det.detect(img_bgr, [args.label])
    dt = time.time() - t0
    print(f"[check_locate_anything] latency: {dt:.1f}s, {len(dets)} detection(s)")
    for d in dets:
        print(f"  {d.label} box={np.round(d.bbox_xyxy_norm, 3).tolist()} score={d.score}")
    print("[check_locate_anything] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
