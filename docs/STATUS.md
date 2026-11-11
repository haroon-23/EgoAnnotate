# EgoAnnotate Capability & Verification Status (`docs/STATUS.md`)

> **Auto-Generated from `gates_report.json`** — Do not edit manually.
> - **Git Commit:** `0ca9957d4d48773483f160e9d3ffa20284d4ee87`
> - **Timestamp (UTC):** `2026-11-08T21:47:00.530998+00:00`
> - **Evaluated Episode:** `data/output/test_10s`
> - **Overall Gate Suite Status:** **PASS**

## 1. Allowed Status Vocabulary (Rule R4)
- **`REAL`**: Implemented, executed end-to-end on CPU, and verified by an automated read-only gate.
- **`PARTIAL`**: Functional on CPU with documented physical/calibration approximations (e.g., 85 mm palm scale prior or template captions).
- **`EXPERIMENTAL`**: Research/kinematic prototype (e.g., humanoid G1/H1 retargeting or GPU sidecar adapters); not certified for direct hardware deployment.
- **`NOT VERIFIED`**: Code path exists but requires external hardware, paid API tier, or optional heavy dependency (`tensorflow_datasets`, CUDA GPU, physical robot).
- **`NOT IMPLEMENTED`**: Not built.

## 2. Capability Truth Table

| Capability | Honest Status | Verifying Gate | Gate Verdict | Technical Notes |
| :--- | :--- | :--- | :--- | :--- |
| **21-Keypoint 2D Hand Detection (MediaPipe CPU)** | `REAL` | `g03_action_signal` | **PASS** | Runs per-frame on CPU; outputs 21 normalized landmarks. |
| **Wearer Hand Chirality & Persistent Identity** | `PARTIAL` | `g05_chirality` | **PASS** | Geometric + bipartite tracker active; requires >=60 human-labelled frames per camera mount to mark REAL. |
| **Metric 3D Hand Pose & Camera Intrinsics** | `PARTIAL` | `g04_layout_consistency` | **PASS** | Uses pinhole back-projection with 85mm palm prior unless chessboard-calibrated device YAML is supplied. |
| **6-DoF Camera Pose / SLAM Trajectory** | `NOT VERIFIED` | `g04_layout_consistency` | **PASS** | Baseline uses 2D planar KLT homography; 6-DoF ARKit/ARCore/SLAM sidecar ingestion supported when provided. |
| **3D MANO Hand Reconstruction (WiLoR / HaWoR)** | `EXPERIMENTAL` | `g03_action_signal` | **PASS** | Sidecar adapter interface implemented; requires external CUDA GPU worker. |
| **24-D Camera-Frame State/Action & Validity Mask** | `REAL` | `g03_action_signal` | **PASS** | Full 24-D state/delta-action with explicit `observation.hand_valid` mask and zero dead undeclared dims. |
| **Franka Panda 7-DoF Kinematic Retargeting & FK Audit** | `REAL` | `g08_retarget_independent` | **PASS** | Base-placement grid search + workspace scaling + independent PyBullet FK position & orientation residual check. |
| **MuJoCo Unwelded Dynamic Actuator Replay** | `REAL` | `g09_physics_replay` | **PASS** | Replays 7-DoF position actuators in MuJoCo 3.3.4 without artificial weld constraints. |
| **Humanoid & Dexterous Retargeting (G1, H1, Allegro)** | `EXPERIMENTAL` | `g08_retarget_independent` | **PASS** | Kinematic mapping only; not dynamically closed-loop verified on physical humanoid hardware. |
| **LeRobot v3 Dataset Export & PyTorch DataLoader** | `REAL` | `g06_lerobot_v3` | **PASS** | Strict float32 Parquet + ffprobe packet sync + verified via `LeRobotDataset` DataLoader. |
| **HDF5 Episode Export (`episode_hdf5.h5`)** | `REAL` | `g07_hdf5_honesty` | **PASS** | Gzip-compressed single-layout HDF5 (<500 MB/hr); TFDS RLDS writer is NOT VERIFIED unless `tensorflow_datasets` installed. |
| **Face Redaction, Audio Strip (`-an`), & DPDP Consent** | `REAL` | `g10_privacy` | **PASS** | MediaPipe face blur before VLM/export, FFmpeg audio removal, and `consent_manifest.json` gate. |
| **Action Segment Captions (`caption_source`)** | `PARTIAL` | `g11_captions` | **PASS** | Tracks `vlm` vs `template` provenance and merges segments < 0.50s; free-tier Gemini rate limits require `--no-vlm` fallback or paid API key. |

## 3. Raw Gate Verdicts (`gates_report.json`)

| Gate Script | Verdict | Return Code |
| :--- | :--- | :--- |
| `gates/g01_no_hardcoded_verdicts.py` | **PASS** | `0` |
| `gates/g02_negative_controls.py` | **PASS** | `0` |
| `gates/g03_action_signal.py` | **PASS** | `0` |
| `gates/g04_layout_consistency.py` | **PASS** | `0` |
| `gates/g05_chirality.py` | **PASS** | `0` |
| `gates/g06_lerobot_v3.py` | **PASS** | `0` |
| `gates/g07_hdf5_honesty.py` | **PASS** | `0` |
| `gates/g08_retarget_independent.py` | **PASS** | `0` |
| `gates/g09_physics_replay.py` | **PASS** | `0` |
| `gates/g10_privacy.py` | **PASS** | `0` |
| `gates/g11_captions.py` | **PASS** | `0` |
| `gates/g12_tests_hygiene.py` | **PASS** | `0` |
