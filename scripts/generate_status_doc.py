from __future__ import annotations
#!/usr/bin/env python3
"""Auto-generates `docs/STATUS.md` directly from `gates_report.json` using honest R4 vocabulary."""

import argparse
import json
import sys
from pathlib import Path

CAPABILITY_MATRIX = [
    ("21-Keypoint 2D Hand Detection (MediaPipe CPU)", "REAL", "g03_action_signal", "Runs per-frame on CPU; outputs 21 normalized landmarks."),
    ("Wearer Hand Chirality & Persistent Identity", "PARTIAL", "g05_chirality", "Geometric + bipartite tracker active; requires >=60 human-labelled frames per camera mount to mark REAL."),
    ("Metric 3D Hand Pose & Camera Intrinsics", "PARTIAL", "g04_layout_consistency", "Uses pinhole back-projection with 85mm palm prior unless chessboard-calibrated device YAML is supplied."),
    ("6-DoF Camera Pose / SLAM Trajectory", "NOT VERIFIED", "g04_layout_consistency", "Baseline uses 2D planar KLT homography; 6-DoF ARKit/ARCore/SLAM sidecar ingestion supported when provided."),
    ("3D MANO Hand Reconstruction (WiLoR / HaWoR)", "EXPERIMENTAL", "g03_action_signal", "Sidecar adapter interface implemented; requires external CUDA GPU worker."),
    ("24-D Camera-Frame State/Action & Validity Mask", "REAL", "g03_action_signal", "Full 24-D state/delta-action with explicit `observation.hand_valid` mask and zero dead undeclared dims."),
    ("Franka Panda 7-DoF Kinematic Retargeting & FK Audit", "REAL", "g08_retarget_independent", "Base-placement grid search + workspace scaling + independent PyBullet FK position & orientation residual check."),
    ("MuJoCo Unwelded Dynamic Actuator Replay", "REAL", "g09_physics_replay", "Replays 7-DoF position actuators in MuJoCo 3.3.4 without artificial weld constraints."),
    ("Humanoid & Dexterous Retargeting (G1, H1, Allegro)", "EXPERIMENTAL", "g08_retarget_independent", "Kinematic mapping only; not dynamically closed-loop verified on physical humanoid hardware."),
    ("LeRobot v3 Dataset Export & PyTorch DataLoader", "REAL", "g06_lerobot_v3", "Strict float32 Parquet + ffprobe packet sync + verified via `LeRobotDataset` DataLoader."),
    ("HDF5 Episode Export (`episode_hdf5.h5`)", "REAL", "g07_hdf5_honesty", "Gzip-compressed single-layout HDF5 (<500 MB/hr); TFDS RLDS writer is NOT VERIFIED unless `tensorflow_datasets` installed."),
    ("Face Redaction, Audio Strip (`-an`), & DPDP Consent", "REAL", "g10_privacy", "MediaPipe face blur before VLM/export, FFmpeg audio removal, and `consent_manifest.json` gate."),
    ("Action Segment Captions (`caption_source`)", "PARTIAL", "g11_captions", "Tracks `vlm` vs `template` provenance and merges segments < 0.50s; free-tier Gemini rate limits require `--no-vlm` fallback or paid API key."),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docs/STATUS.md from gates_report.json")
    parser.add_argument("--report", type=str, default="gates_report.json")
    parser.add_argument("--out", type=str, default="docs/STATUS.md")
    args = parser.parse_args()

    rep_path = Path(args.report)
    if not rep_path.exists():
        print(f"Missing {rep_path}; run `gates/run_all.py` first.")
        return 1

    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    gates = rep.get("gates", {})

    lines = [
        "# EgoAnnotate Capability & Verification Status (`docs/STATUS.md`)",
        "",
        "> **Auto-Generated from `gates_report.json`** — Do not edit manually.",
        f"> - **Git Commit:** `{rep.get('git_commit', 'UNKNOWN')}`",
        f"> - **Timestamp (UTC):** `{rep.get('timestamp_utc', 'UNKNOWN')}`",
        f"> - **Evaluated Episode:** `{rep.get('episode_dir', 'UNKNOWN')}`",
        f"> - **Overall Gate Suite Status:** **{rep.get('overall_status', 'FAIL')}**",
        "",
        "## 1. Allowed Status Vocabulary (Rule R4)",
        "- **`REAL`**: Implemented, executed end-to-end on CPU, and verified by an automated read-only gate.",
        "- **`PARTIAL`**: Functional on CPU with documented physical/calibration approximations (e.g., 85 mm palm scale prior or template captions).",
        "- **`EXPERIMENTAL`**: Research/kinematic prototype (e.g., humanoid G1/H1 retargeting or GPU sidecar adapters); not certified for direct hardware deployment.",
        "- **`NOT VERIFIED`**: Code path exists but requires external hardware, paid API tier, or optional heavy dependency (`tensorflow_datasets`, CUDA GPU, physical robot).",
        "- **`NOT IMPLEMENTED`**: Not built.",
        "",
        "## 2. Capability Truth Table",
        "",
        "| Capability | Honest Status | Verifying Gate | Gate Verdict | Technical Notes |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for cap_name, default_status, gate_key, notes in CAPABILITY_MATRIX:
        g_verdict = gates.get(gate_key, {}).get("status", "NOT RUN")
        effective_status = default_status if g_verdict == "PASS" else f"FAIL ({default_status})"
        lines.append(f"| **{cap_name}** | `{effective_status}` | `{gate_key}` | **{g_verdict}** | {notes} |")

    lines.extend([
        "",
        "## 3. Raw Gate Verdicts (`gates_report.json`)",
        "",
        "| Gate Script | Verdict | Return Code |",
        "| :--- | :--- | :--- |",
    ])
    for gk, gv in gates.items():
        lines.append(f"| `gates/{gk}.py` | **{gv.get('status')}** | `{gv.get('returncode')}` |")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {out_p} (overall_status={rep.get('overall_status')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
