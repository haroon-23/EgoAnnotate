# EgoAnnotate v1.0.0

**Auditable, Single-Machine Egocentric Video-to-Robot Training Data Pipeline (CPU-First)**

EgoAnnotate converts raw first-person (egocentric) video of human hand manipulation into validated, multi-embodiment robot training datasets (**LeRobot v3**, **RLDS HDF5**, **MCAP v0**, **JSON**, plus native-resolution **overlay** and **side-by-side MuJoCo proof videos**).

Designed specifically for small robotics teams running on commodity hardware (verified on a 2017 Intel MacBook Air, x86_64 CPU-only, 8 GB RAM), EgoAnnotate enforces **fail-loud execution**, **atomic file writes**, **stage-level and VLM disk caching**, and **honest physical telemetry**.

---

## Architecture: 10-Stage Resumable Pipeline

1. **VideoProcessor:** Uniform frame sampling with native resolution preservation.
2. **HandTracker:** MediaPipe 21-keypoint 3D hand tracking in `VIDEO` mode with short-gap interpolation ($\le 5$ frames, flagged `is_interpolated=True`) and **persistent Left/Right hand identity** (never swaps hands mid-clip).
3. **GeminiObjectDetector + Local Grounding:** Dynamic open-vocabulary object discovery via `gemini-2.5-flash` (cached in `cache/vlm_cache.json`) paired with local 2D bounding-box detection (`OWL-ViT` / `GroundingDINO`).
4. **ContactDetector:** Geometric fingertip-to-bbox proximity state machine with depth consistency scoring (zero no-bbox contact hallucinations).
5. **GraspClassifier:** 5-class grasp taxonomy (`pinch`, `power_wrap`, `lateral`, `tripod`, `open`) with temporal majority smoothing.
6. **SignalSegmenter & SegmentLabeler:** Deterministic contact/grasp boundary detection labeled with `gemini-2.5-flash`.
7. **GeminiLanguageGenerator:** Fail-loud natural-language task & segment instruction generation.
8. **CameraMotion & ActionComputer:** Background sparse optical flow (`camera_motion_flag`, `camera_motion_px_per_sec`) + 24-D embodiment-agnostic human action primitives.
9. **Universal Embodiment Retargeter (`mink` + `MuJoCo` / `PyBullet`):** Fixed-anchor anthropometric metric scaling ($0.085\text{ m}$ palm reference, $\pm 15\%$ error margin) feeding config-driven adapters (`configs/embodiments/*.yaml`) for single arms, dual-arm humanoids, and dexterous multi-finger hands.
10. **DatasetExporter & DeliveryValidator:** Atomic exports to `episode_rlds.hdf5`, `lerobot_v3/`, `episode.mcap`, `annotations.json`, `summary.json`, `manifest.json`, `overlay_annotated.mp4`, and `side_by_side.mp4`.

---

## Quickstart (Intel macOS / Linux CPU)

```bash
git clone https://github.com/haroon-23/EgoAnnotate.git
cd EgoAnnotate
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Add your GEMINI_API_KEY
```

### 1. Process a Demonstration Video

```bash
# Run complete end-to-end pipeline with Franka Panda retargeting (default)
.venv/bin/python scripts/run.py --input data/raw_videos/whatsapp_video.mp4 --config configs/default.yaml

# Run with Unitree G1 Bimanual Humanoid
.venv/bin/python scripts/run.py --input data/raw_videos/whatsapp_video.mp4 --robot unitree_g1_upper

# Run with Allegro 16-DoF Dexterous Hand
.venv/bin/python scripts/run.py --input data/raw_videos/whatsapp_video.mp4 --robot allegro_hand_right
```

### 2. Multi-Robot Retargeting (Batch)

Retarget an already-annotated episode across all 4 robot platforms in one pass:

```bash
.venv/bin/python scripts/retarget_multi.py data/output/whatsapp_video/annotations.json \
  --embodiments configs/embodiments/franka_arm.yaml \
                configs/embodiments/unitree_g1_upper.yaml \
                configs/embodiments/unitree_h1_upper.yaml \
                configs/embodiments/allegro_hand.yaml
```

### 3. Validate Delivery Deliverables

Validate that all 5 deliverable files and native-resolution videos are present and spec-compliant:

```bash
.venv/bin/python scripts/validate_delivery.py data/output/whatsapp_video --source-video data/raw_videos/whatsapp_video.mp4
```

### 4. Run the Full Test Suite

```bash
.venv/bin/pytest tests/ -v
```

---

## Export Deliverables per Episode

Each output directory `data/output/<episode_id>/` contains:

```
data/output/<episode_id>/
├── annotations.json          # Primary JSON metadata & full frame annotations
├── summary.json              # Feasibility summary metrics & grasp distributions
├── manifest.json             # Atomic run manifest (exit code 0=ok, 1=failed, 2=partial)
├── episode_rlds.hdf5         # RLDS HDF5 dataset for OpenVLA / Octo
├── lerobot_v3/               # Hugging Face LeRobot format
│   ├── meta/info.json
│   └── data/chunk-000/episode_000000.parquet
├── episode.mcap              # Foxglove Studio & ROS2 streaming format
├── overlay_annotated.mp4     # Native-resolution video with 2D/3D tracking overlay
└── side_by_side.mp4          # Synchronized human vs simulated robot proof video
```

---

## Supported Embodiments

| Embodiment | Kind | Config | URDF |
|---|---|---|---|
| **Franka Emika Panda** | 7-DoF Arm + Parallel Gripper | `configs/embodiments/franka_arm.yaml` | `data/urdfs/panda.urdf` |
| **Unitree G1 Upper** | 29-DoF Bimanual Humanoid | `configs/embodiments/unitree_g1_upper.yaml` | `data/urdfs/embodiments/g1_clean.urdf` |
| **Unitree H1 Upper** | 19-DoF Humanoid Upper | `configs/embodiments/unitree_h1_upper.yaml` | `data/urdfs/embodiments/h1_clean.urdf` |
| **Allegro Hand** | 16-DoF Dexterous Hand | `configs/embodiments/allegro_hand.yaml` | `data/urdfs/embodiments/allegro_clean.urdf` |

---

## License

This project is licensed under the [MIT License](LICENSE). Bundled third-party URDF models retain their original open-source licenses (Apache-2.0, BSD-3-Clause, BSD-2-Clause).
