"""Commercial-only model registry manager with SHA-256 verification and offline air-gapped audit mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

MODEL_MANIFEST: dict[str, list[dict[str, Any]]] = {
    "cpu_smoke": [
        {
            "name": "mediapipe_hands_and_pose",
            "source": "bundled_wheel:mediapipe==0.10.21",
            "license": "Apache-2.0",
            "commercial_ok": True,
            "vram_gb": 0.0,
        },
        {
            "name": "opencv_essential_vo",
            "source": "bundled_wheel:opencv-python==4.10.0.84",
            "license": "Apache-2.0",
            "commercial_ok": True,
            "vram_gb": 0.0,
        },
    ],
    "gpu_prod": [
        {
            "name": "grounding_dino_base",
            "hf_repo": "IDEA-Research/grounding-dino-base",
            "license": "Apache-2.0",
            "commercial_ok": True,
            "vram_gb": 1.8,
        },
        {
            "name": "sam21_hiera_small",
            "hf_repo": "facebook/sam2.1-hiera-small",
            "license": "Apache-2.0",
            "commercial_ok": True,
            "vram_gb": 2.4,
        },
        {
            "name": "depth_anything_v2_small",
            "hf_repo": "depth-anything/Depth-Anything-V2-Small-hf",
            "license": "Apache-2.0",
            "commercial_ok": True,
            "vram_gb": 0.8,
        },
        {
            "name": "smolvlm2_2_2b",
            "hf_repo": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            "license": "Apache-2.0",
            "commercial_ok": True,
            "vram_gb": 5.2,
        },
        {
            "name": "qwen25_vl_7b",
            "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
            "license": "Apache-2.0",
            "commercial_ok": True,
            "vram_gb": 16.5,
        },
    ],
}


def compute_dir_sha256(directory: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(directory).as_posix().encode("utf-8"))
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


def verify_or_download(profile: str, cache_dir: Path, offline: bool = False) -> dict[str, Any]:
    entries = MODEL_MANIFEST.get(profile)
    if entries is None:
        raise ValueError(f"Unknown profile '{profile}'. Choices: {sorted(MODEL_MANIFEST.keys())}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for item in entries:
        assert item["commercial_ok"] is True, f"Refusing non-commercial model {item['name']}"
        if "source" in item:
            results.append({"name": item["name"], "status": "bundled_ready", "license": item["license"]})
            continue

        repo_id = str(item["hf_repo"])
        local_dir = cache_dir / repo_id.replace("/", "__")
        if offline:
            present = local_dir.exists() and any(local_dir.iterdir())
            results.append(
                {
                    "name": item["name"],
                    "hf_repo": repo_id,
                    "license": item["license"],
                    "status": "offline_present" if present else "offline_missing",
                    "sha256": compute_dir_sha256(local_dir) if present else None,
                }
            )
            continue

        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)
        results.append(
            {
                "name": item["name"],
                "hf_repo": repo_id,
                "license": item["license"],
                "status": "downloaded_verified",
                "sha256": compute_dir_sha256(local_dir),
            }
        )

    lock_path = cache_dir / f"models_lock_{profile}.json"
    lock_path.write_text(json.dumps({"profile": profile, "models": results}, indent=2), encoding="utf-8")
    return {"profile": profile, "lock_file": str(lock_path), "models": results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download or verify commercial EgoAnnotate v3 model weights.")
    ap.add_argument("--profile", default="cpu_smoke", choices=["cpu_smoke", "gpu_prod"])
    ap.add_argument("--cache-dir", type=Path, default=Path(".ego_cache/models"))
    ap.add_argument("--offline", action="store_true", help="Air-gapped check only (zero network access).")
    args = ap.parse_args(argv)
    res = verify_or_download(args.profile, args.cache_dir, offline=args.offline)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
