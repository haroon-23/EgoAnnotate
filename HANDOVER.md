# EgoAnnotate v1.0.0 Handover Document

## 1. System Overview & Deliverables
EgoAnnotate converts monocular egocentric videos into robotic manipulation datasets across four embodiments:
- **Franka Emika Panda** (7-DoF arm + parallel gripper)
- **Unitree G1** (29-DoF bimanual humanoid upper body)
- **Unitree H1** (19-DoF humanoid upper body)
- **Wonik Robotics Allegro Hand** (16-DoF dexterous hand)

### Output Deliverables per Episode
1. `annotations.json`: Episode-level metadata, temporal segments, camera motion, and per-frame annotations.
2. `summary.json`: Quality & reachability summary statistics.
3. `manifest.json`: Atomic run manifest with exit codes, stage runtimes, error log, and system metrics.
4. `episode_rlds.hdf5`: RLDS HDF5 dataset for OpenVLA / Octo policy training.
5. `lerobot_v3/`: Hugging Face LeRobot format (`meta/info.json` + `data/chunk-*.parquet`).
6. `episode.mcap`: Spec-compliant MCAP v0 stream separating `/human/*` and `/robot/*` channels.
7. `overlay_annotated.mp4`: Client-facing native-resolution tracking overlay.
8. `side_by_side.mp4`: Synchronized side-by-side human vs simulated robot proof video.

## 2. Core Architectural Invariants
- **Fail-Loud Execution:** Never swallow VLM or retargeting failures silently; exceptions raise loudly unless partial execution is explicitly requested.
- **Zero Hallucinated Contacts:** Contact requires active bounding-box proximity; objects with `bbox=None` never produce false 5-finger contact flags.
- **Atomic I/O:** All JSON, Parquet, HDF5, and MCAP files are written to `.tmp` staging files and atomically renamed via `os.replace` with `os.fsync`.
- **Persistent Hand Identity:** Driving hand side is locked per clip (`lock_driving_hand=True`), preventing mid-clip left/right hand swapping.
- **Anthropometric Metric Scaling:** MediaPipe z is scaled using a fixed anthropometric palm reference ($0.085\text{ m} \pm 15\%$), avoiding artificial min-max workspace stretching.

## 3. Key Commands

### Process Video
```bash
.venv/bin/python scripts/run.py --input data/raw_videos/whatsapp_video.mp4 --config configs/default.yaml
```

### Multi-Robot Retargeting
```bash
.venv/bin/python scripts/retarget_multi.py data/output/whatsapp_video/annotations.json \
  --embodiments configs/embodiments/franka_arm.yaml \
                configs/embodiments/unitree_g1_upper.yaml \
                configs/embodiments/unitree_h1_upper.yaml \
                configs/embodiments/allegro_hand.yaml
```

### Validate Delivery
```bash
.venv/bin/python scripts/validate_delivery.py data/output/whatsapp_video --source-video data/raw_videos/whatsapp_video.mp4
```

### Run Tests
```bash
.venv/bin/pytest tests/ -v
```
