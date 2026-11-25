# EgoAnnotate Commercial License Audit (November 2026)

**Policy:** Every default component shipped in EgoAnnotate must have BOTH a commercially permissible code license AND commercially permissible weights license (Apache-2.0, MIT, BSD-2/3-Clause, zlib, or CC-BY-4.0). No paid APIs, no GPL/AGPL copyleft dependencies, and no CC-BY-NC / research-only weights are enabled by default.

## 1. Default Commercial Stack Audit Table

| Component | Version / Checkpoint | Code License | Weights License | Commercial OK | Source URL | Notes |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **EgoAnnotate Core** | v3.0.0 | Proprietary / Dual-License (Owner Choice; currently MIT in repo) | N/A | **YES** | https://github.com/haroon-23/EgoAnnotate | See `docs/COMMERCIAL_LICENSING.md` for rental/lease licensing options. |
| **PyTorch** | 2.2.2 (CPU dev) / 2.4+ (CUDA prod) | BSD-3-Clause | N/A | **YES** | https://github.com/pytorch/pytorch | Core tensor runtime. |
| **OpenCV (`opencv-python`)** | 4.10.0 | Apache-2.0 | N/A | **YES** | https://github.com/opencv/opencv | Video I/O, 6-DoF VO, PnP, optical flow, privacy redaction. |
| **MediaPipe Hand & Pose Landmarker** | 0.10.21 | Apache-2.0 | Apache-2.0 | **YES** | https://github.com/google-ai-edge/mediapipe | Default 21-joint hand + 33-joint upper-body pose; no MANO dependency. |
| **Grounding DINO** | `IDEA-Research/grounding-dino-base` | Apache-2.0 | Apache-2.0 | **YES** | https://github.com/IDEA-Research/GroundingDINO | GPU production open-vocab object detector. |
| **Meta SAM 2.1** | `facebook/sam2.1-hiera-small/large` | Apache-2.0 | Apache-2.0 | **YES** | https://github.com/facebookresearch/sam2 | GPU production video object/hand mask tracker. |
| **Florence-2-base** | `microsoft/Florence-2-base` | MIT | MIT | **YES** | https://huggingface.co/microsoft/Florence-2-base | Optional lightweight open-vocab detector/captioner. |
| **Depth Anything V2 Small** | `depth-anything/Depth-Anything-V2-Small-hf` | Apache-2.0 | Apache-2.0 | **YES** | https://github.com/DepthAnything/Depth-Anything-V2 | **Small ONLY** is Apache-2.0; Base/Large/Giant are CC-BY-NC and excluded. |
| **Video Depth Anything Small** | `fastforks/video-depth-anything` (`vits`) | Apache-2.0 | Apache-2.0 | **YES** | https://huggingface.co/fastforks/video-depth-anything | Temporally consistent depth (Small variant only). |
| **DPVO / DROID-SLAM** | Latest upstream | MIT (DPVO) / BSD-3-Clause (DROID) | MIT / BSD-3-Clause | **YES** | https://github.com/princeton-vl/DPVO | GPU production 6-DoF visual SLAM backends. |
| **SmolVLM2-2.2B-Instruct** | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | Apache-2.0 | Apache-2.0 | **YES** | https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct | Compact local video-capable VLM (5.2 GB VRAM). |
| **Qwen2.5-VL-7B-Instruct / Qwen3-VL** | `Qwen/Qwen2.5-VL-7B-Instruct` | Apache-2.0 | Apache-2.0 | **YES** | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | High-accuracy local GPU VLM for hierarchical semantics. |
| **HuggingFace LeRobot** | 0.4.4 | Apache-2.0 | N/A | **YES** | https://github.com/huggingface/lerobot | LeRobotDataset v3 exporter & loader validation. |
| **MuJoCo & Mink** | MuJoCo 3.10 / Mink 0.0.5+ | Apache-2.0 | N/A | **YES** | https://github.com/google-deepmind/mujoco | Physics verification & differential IK retargeting. |
| **PyBullet** | 3.2+ | zlib | N/A | **YES** | https://github.com/bulletphysics/bullet3 | Independent FK/collision verification. |
| **SciPy / NumPy / PyArrow / h5py / Pydantic / FastAPI** | Pinned | BSD / Apache-2.0 / MIT | N/A | **YES** | PyPI | Core math, Parquet/HDF5 I/O, schemas, REST worker. |
| **FFmpeg (CLI subprocess)** | System binary (`8.x`) | LGPL-2.1+ / GPL (external CLI) | N/A | **YES** | https://ffmpeg.org | Invoked strictly as external subprocess, never linked in-process. |

## 2. Explicitly Quarantined / Excluded Non-Commercial or Copyleft Models (OFF by Default)

| Component | License Issue | Status in EgoAnnotate v3 |
| :--- | :--- | :--- |
| **MANO / HaMeR / WiLoR** | MANO hand model assets are MPI research-only (non-commercial). | **OFF by default**; blocked unless user explicitly sets `allow_non_commercial_plugins: true` with warning. |
| **Depth Anything V2 Base / Large / Giant** | Weights are `CC-BY-NC-4.0`. | **Excluded** from default configs. |
| **UniDepth / UniDepthV2** | Weights are `CC-BY-NC-4.0`. | **Removed** from default configs. |
| **MASt3R / DUSt3R / Unik3D** | `CC-BY-NC-SA-4.0`. | **Excluded** from default configs. |
| **VGGT-1B** | Meta non-commercial weights / AUP. | **Excluded** from default configs. |
| **YOLO-World (Ultralytics)** | `GPL-3.0` / paid Ultralytics Enterprise license required for proprietary commercial use. | **Excluded** from default configs. |
| **Meta SAM 3** | Custom Meta SAM License (AUP conditions). | **Excluded** from default configs; use Apache-2.0 `SAM 2.1`. |
| **Google Gemini API (`google-genai`)** | Paid proprietary cloud API. | **Removed** from default pipeline (zero paid APIs, zero external telemetry). |
