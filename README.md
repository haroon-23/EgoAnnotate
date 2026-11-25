# EgoAnnotate v3 — Commercial Egocentric Video-to-VLA Pipeline

EgoAnnotate v3 transforms raw, full-length egocentric videos into embodiment-agnostic, privacy-redacted, robot-ready training datasets for Vision-Language-Action (VLA) models and multi-robot manipulation policies—with **zero paid APIs**, **100% commercially permissible default licenses**, and **air-gapped on-premise / Kaggle GPU support**.

## Key Capabilities
- **100% Commercial License Firewall:** All default models and dependencies use Apache-2.0, MIT, BSD, or zlib (`LICENSES.md`, `THIRD_PARTY_NOTICES.md`). Paid cloud APIs and non-commercial weights (`MANO`/`HaMeR`, `UniDepth`, `Depth-Anything-V2-Base/Large/Giant`, `MASt3R`, `VGGT`) are programmatically blocked by default.
- **Full-Length Streaming Processing:** Bounded-RAM chunked streaming across complete videos (`0..N-1` frames, zero truncation), verified on a complete `180.0s` (`5400`-frame) real egocentric clip.
- **Perception & 6-DoF World Geometry:** MediaPipe Hands + Upper-Body Pose, manipulated-object tracking (`opencv_motion_roi` on CPU; `Grounding DINO` + `SAM 2.1` on CUDA GPU), and 6-DoF monocular VO with pure-rotation homography fallback + ORB loop-closure (`DPVO`/`DROID-SLAM` + `Depth-Anything-V2-Small` on GPU).
- **Behavior, Tactile & Grounded Semantics:** 11-verb atomic action segmentation, Feix GRASP taxonomy, bimanual coordination, honest RGB tactile proxies (`source="derived_rgb_proxy"`) + CSV/HDF5 hardware tactile ingestion (`source="hardware_sensor"`), and local open-source VLM semantics (`SmolVLM2-2.2B` / `Qwen2.5-VL-7B`, Apache-2.0).
- **Config-Only Multi-Embodiment Retargeting:** Onboard any robot strictly via YAML + URDF/MJCF with zero code changes (`franka_panda`, `dual_arm_franka`, `unitree_g1_upper`, `unitree_h1_upper`, `allegro_hand_right`, `teleop_glove`).
- **Multi-Format Exports & Privacy Gate:** Exports **LeRobot v3** (strict `float32` Parquet + `ffprobe`-synced MP4 + GR00T `modality.json`), **RLDS/OXE**, **Canonical gzip HDF5**, and **Parquet + Pydantic JSON IR**, with automated face-blur (`0` unblurred faces verified) and `-an` audio stripping.
- **Production Operations:** `egoannotate` CLI, Python API, crash-safe deduplicating SQLite batch queue, FastAPI REST worker, offline `qa_report.html`, offline rotatable `viewer_3d.html`, full-video `qa_overlay.mp4`, and CPU/CUDA Dockerfiles.

## Verified Full-Video Benchmark (180s / 5400 Frames, CPU Profile)
- **VO Tracking:** `5400/5400` (`100.0%`), `4` ORB loop closures, `10.57m` camera path
- **Quality & Privacy:** `0.950 (PASS)` readiness score, `0` unblurred faces, `0` audio streams
- **7-DoF Retargeting Feasibility:** Franka `0.973` (`6.36 mm` FK), Dual Franka `0.973` (`6.52 mm`), Unitree G1 `0.928` (`7.14 mm`), Allegro `0.866`, Teleop Glove `0.889`
- **Policy Training Smoke (`LeRobot v3`):** `5392` action-chunk samples, PyTorch loss `0.2602 -> 0.2549`

## Quickstart

    pip install -e .
    egoannotate audit-licenses
    egoannotate run /path/to/video.mp4 --out ./runs/ep01 --profile auto
    egoannotate train-smoke ./runs/ep01/lerobot_v3

## Batch Queue & REST Worker

    egoannotate batch ./incoming_videos --out ./processed --db ./jobs.db --profile auto --workers 2
    egoannotate worker --db ./jobs.db --host 0.0.0.0 --port 8080

## Documentation
- `AUDIT.md` — 15-category skeptical commercial readiness audit
- `docs/VALIDATION.md` — Full-video and edge-case validation matrix
- `docs/KAGGLE_GPU_RUNBOOK.md` — Step-by-step Kaggle GPU `gpu_prod` certification guide
- `docs/DEPLOYMENT.md` — Docker CPU/CUDA and air-gapped deployment guide
- `docs/COMMERCIAL_LICENSING.md` — Commercial distribution and license firewall details
