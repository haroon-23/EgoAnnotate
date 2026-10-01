"""Manual smoke test for the local SmolVLM backend (Mac only, NOT part of CI).

Usage:
    python scripts/check_local_vlm.py --weights models/smolvlm-500m-instruct [--image path.jpg]

Exits 0 with a SKIP message when the weights directory is absent (no
auto-download, by design). When weights are present, runs one real
inference and prints the text + latency.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image

from src.vlm_backend import VLMBackendConfig, VLMRequest, create_vlm_backend


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-test the local SmolVLM VLM backend.")
    ap.add_argument("--weights", default="models/smolvlm-500m-instruct",
                    help="Local SmolVLM weights directory (manual download).")
    ap.add_argument("--image", default=None, help="Optional test image (default: synthetic).")
    ap.add_argument("--prompt", default='Describe what you see in one short sentence. Reply with ONLY the sentence.',
                    help="Prompt to send.")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.is_dir():
        print(f"[check_local_vlm] SKIP: weights dir '{weights}' not present.")
        print("  Download manually from https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct")
        return 0

    cfg = VLMBackendConfig(backend="local", local_weights=str(weights))
    backend = create_vlm_backend(cfg)
    if backend is None:
        print("[check_local_vlm] SKIP: local backend unavailable "
              "(transformers missing?). pip install transformers torch --index-url https://download.pytorch.org/whl/cpu")
        return 0

    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        img = Image.new("RGB", (320, 240), color=(200, 60, 40))

    req = VLMRequest(images=[img], prompt=args.prompt, max_new_tokens=64, temperature=0.0)
    print(f"[check_local_vlm] backend={backend.name} weights={weights} — running inference...")
    resp = backend.generate(req)
    print(f"[check_local_vlm] latency: {resp.latency_s:.1f}s")
    print(f"[check_local_vlm] text: {resp.text!r}")
    print("[check_local_vlm] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
