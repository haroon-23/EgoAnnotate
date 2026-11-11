# EgoAnnotate: Egocentric Video to Verified Robot & VLA Training Data Pipeline

EgoAnnotate converts raw egocentric manipulation videos (`30.00 Hz`) into verified 24-D camera-frame hand state/action trajectories, 7-DoF arm kinematic trajectories, and standardized **HuggingFace LeRobot v3**, **HDF5 (`episode_hdf5.h5`)**, **Safetensors**, and **Foxglove MCAP** datasets.

## 1. Capability & Verification Truth Table

See [`docs/STATUS.md`](docs/STATUS.md) (auto-generated from `gates_report.json`) and [`docs/PHYSICS_FINDINGS.md`](docs/PHYSICS_FINDINGS.md) for full technical disclosures:

| Capability | Status | Notes |
| :--- | :--- | :--- |
| **21-Keypoint 2D Hand Detection (MediaPipe CPU)** | `REAL` | Runs per-frame on CPU; outputs 21 landmarks with persistent bipartite Left/Right tracking. |
| **24-D Camera-Frame State/Action & Validity Mask** | `REAL` | `[wrist_xyz(3), rot6(6), gripper(1), fingers_f0_f13(14)]` + `observation.hand_valid` (`[2] int64`). |
| **Metric 3D Hand Scale & Intrinsics** | `PARTIAL` | Uses pinhole back-projection with 85 mm palm prior (`is_metric: false`) unless chessboard-calibrated YAML (`scripts/calibrate_camera.py`) is provided. |
| **6-DoF Camera Pose / SLAM** | `NOT VERIFIED` | Baseline uses 2D planar KLT homography; 6-DoF SLAM/ARKit/ARCore sidecar supported when supplied. |
| **Franka Panda 7-DoF Kinematic Retargeting & FK Audit** | `REAL` | Workspace-scaled damped-least-squares IK + independent PyBullet FK position (`mm`) & orientation (`deg`) verification (`gates/g08_retarget_independent.py`). |
| **MuJoCo 3.3.4 Unwelded Dynamic Actuator Replay** | `REAL` | Verified via unwelded position actuators (`gates/g09_physics_replay.py`) with 0 self-collisions. |
| **Humanoid & Dexterous Retargeting (G1, H1, Allegro)** | `EXPERIMENTAL` | Kinematic mapping only; not dynamically closed-loop verified on physical humanoid hardware. |
| **LeRobot v3 Dataset Export** | `REAL` | Strict `float32` Parquet + `ffprobe` packet-synced MP4s + verified via `LeRobotDataset` PyTorch `DataLoader` (`gates/g06_lerobot_v3.py`). |
| **HDF5 Episode Export (`episode_hdf5.h5`)** | `REAL` | Gzip-compressed HDF5 (`< 500 MB/hr`); TFDS RLDS builder is `NOT VERIFIED` unless `tensorflow_datasets` is installed (`gates/g07_hdf5_honesty.py`). |
| **Privacy (Face Blur, Audio `-an`, DPDP Consent)** | `REAL` | MediaPipe face redaction prior to VLM/export, FFmpeg audio removal, and `consent_manifest.json` gate (`gates/g10_privacy.py`). |

## 2. Running the Read-Only 12-Gate Verification Suite

```bash
.venv/bin/python gates/run_all.py data/output/test_10s --labels-csv data/chirality_labels.csv --report-out gates_report.json
.venv/bin/python gates/run_all.py data/output/whatsapp_video --labels-csv data/chirality_labels.csv --report-out data/output/whatsapp_video/gates_report.json
```
