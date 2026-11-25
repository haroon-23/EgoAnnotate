# Production Deployment Guide (Linux + NVIDIA CUDA, Kaggle GPU & CPU Smoke)

## 1. Hardware Profiles

| Profile | Target Hardware | Stack | Typical Throughput |
| :--- | :--- | :--- | :--- |
| `cpu_smoke` | macOS / Linux CPU (dev, CI, edge QA) | MediaPipe Hands/Pose, OpenCV ROI, OpenCV 6-DoF Essential VO + ORB loop-closure, Geometric-Grounded VLM | ~12–15 fps (perception+VO) |
| `gpu_prod` | Linux + NVIDIA GPU (Kaggle T4/P100 or L4/A10G/RTX 4090/A100/H100) | MediaPipe Hands/Pose, Grounding DINO + SAM 2.1, Depth-Anything-V2-Small, DPVO/DROID-SLAM, Qwen2.5-VL-7B / SmolVLM2-2.2B | High-accuracy batch production |

## 2. Quickstart (Local Python 3.10–3.12)

    pip install -e .
    egoannotate audit-licenses
    egoannotate run /path/to/video.mp4 --out ./runs/ep01 --profile auto
    egoannotate train-smoke ./runs/ep01/lerobot_v3

## 3. Multi-Video Resumable Batch & REST Service

    egoannotate batch ./incoming_videos --out ./processed --db ./jobs.db --profile auto --workers 2
    egoannotate worker --db ./jobs.db --host 0.0.0.0 --port 8080

## 4. Docker Deployment

    docker compose up --build egoannotate-cpu
    docker compose --profile gpu up --build egoannotate-gpu

## 5. Air-Gapped / Offline Model Pre-Staging

    python scripts/download_models.py --profile gpu_prod --cache-dir /opt/ego_models
    python scripts/download_models.py --profile gpu_prod --cache-dir /opt/ego_models --offline
