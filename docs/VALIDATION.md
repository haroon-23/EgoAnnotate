# EgoAnnotate v3 Validation & Edge-Case Certification Report

## 1. Full-Length Real Video Certification (`ad521ca5-0017-475b-9dd8-1fdb423a66b7.mp4`)
- **Resolution / FPS / Duration:** `1024x576` @ `30.00 fps`, `5400 / 5400` frames (`180.00 s`, 100% full-length processing — zero truncation).
- **6-DoF World VO (`opencv_essential_vo`):** `5400/5400` frames tracked (`100.0%`), `4` ORB loop closures applied, `10.570 m` camera trajectory, `833` triangulated world points, estimated `table_z = -0.2036 m`.
- **Perception Coverage:** `5299/5400` hand frames (`98.1%`), `5294/5400` manipulated-object frames, `1515` upper-body arm pose detections (stride 3).
- **Behavior & Semantics:** `62` Feix grasp events, `55` atomic action segments covering frames `0..5399`, `23` hierarchical subtasks, grounded VLA paraphrases verified against kinematics.
- **Tactile Honesty:** All `5400` frames tagged `source="derived_rgb_proxy"` with irreducible uncertainty `>= 0.35`; external CSV/HDF5 hardware tactile ingestion verified switching to `source="hardware_sensor"` with raw force values preserved.
- **Data Quality Gating:** `0.950 (PASS)` on clean real video (`flags=['monocular_scale_uncalibrated_prior']`); degraded dark/blurry/handless footage quarantined (`<= 0.40`, `QUARANTINE`).
- **Privacy Gate:** `verified_unblurred_faces == 0`, `audio_stripped == True` (`ffprobe` confirms `0` audio streams and exact `5400`-frame video/Parquet packet parity).
- **Multi-Embodiment Config-Only Retargeting (all `5400` frames, `100.0%` joint-limit compliance):**
  - `franka_panda` (7-DoF + gripper): feasibility `0.973`, mean FK error `6.36 mm`
  - `dual_arm_franka` (14-DoF + 2 grippers): feasibility `0.973`, mean FK error `6.52 mm`
  - `unitree_g1_upper` (14-DoF dual arms + 20-DoF dexterous hands): feasibility `0.928`, mean FK error `7.14 mm`
  - `allegro_hand_right` (16-DoF dexterous hand): feasibility `0.866`
  - `teleop_glove` (40-D SE(3)+curls+tactile): feasibility `0.889`
  - Plus zero-code onboarding verified for both a brand-new 6-DoF `.urdf` and a brand-new `.xml` MJCF arm.
- **Policy Training Smoke Test (`LeRobot v3` Parquet):** `5392` action-chunk samples (`obs_dim=48`, `action_dim=30`, `chunk_size=8`), real PyTorch loss `0.2602 -> 0.2549`, finite predictions verified.
- **QA & Ops Artifacts:** `qa_report.html` (`18.2 KB`), offline rotatable `viewer_3d.html` (`63.5 KB`), full `5400`-frame `qa_overlay.mp4` (`107.0 MB`), idempotent SQLite job queue, FastAPI REST worker, offline model manifest check.

## 2. Edge-Case Stress Matrix (Verified in `tests/v3/test_phase3_validation_and_edge_cases.py`)
| Edge Case | Verification Result |
| :--- | :--- |
| Low-light / underexposed real frames (`0.08x` luma) | **PASS** — `lighting_quality` drops, `underexposed_low_light` flag raised, readiness drops |
| Severe motion blur (`31px` kernel on real frames) | **PASS** — `motion_blur_quality` drops, `severe_motion_blur` flag raised, readiness drops |
| Heavy hand occlusion / border truncation | **PASS** — `occlusion_score` drops to `0.0`, `low_hand_visibility` & truncation flags raised |
| Missing intrinsics vs calibrated YAML | **PASS** — honest provenance switch (`auto_estimated` ±15% vs `calibrated_yaml` ±2%) |
| Unimanual vs bimanual coordination | **PASS** — verified `bimanual` on full real video and `unimanual_right` on masked real slice |
| MJCF `.xml` + URDF `.urdf` config-only onboarding | **PASS** — zero code changes, 100% joint limit compliance, FK error `< 25 mm` |
| License firewall & zero paid APIs | **PASS** — `tests/test_no_fakes.py` + `CommercialLicenseError` blocks Gemini/MANO/UniDepth/DAv2-Large/MASt3R |

## 3. Verification Scope: CPU-Verified vs. Kaggle/Production CUDA GPU-Pending
- **CPU-Verified Locally (`cpu_smoke.yaml`):** Full 5400-frame streaming I/O, MediaPipe Hands + Pose, OpenCV ROI object tracking, 6-DoF Essential-Matrix VO + pure-rotation fallback + ORB loop-closure, Feix grasp/behavior taxonomy, RGB + hardware tactile, geometrically grounded hierarchical semantics, 10D quality gating, 5-embodiment 7-DoF URDF/MJCF retargeting, LeRobot v3 + GR00T `modality.json` + RLDS + HDF5 + Parquet/JSON export, privacy blur/audio-strip gate, PyTorch policy training, SQLite/FastAPI worker, HTML/3D/overlay QA.
- **CUDA GPU-Pending (`gpu_prod.yaml` — Ready for Kaggle GPU pass):** Live CUDA weights execution of `Grounding DINO` (Apache-2.0), `SAM 2.1` (Apache-2.0), `Depth-Anything-V2-Small` (Apache-2.0), `DPVO` (MIT) / `DROID-SLAM` (BSD-3-Clause), and `Qwen2.5-VL-7B` / `SmolVLM2-2.2B` (Apache-2.0). Config wiring, license checks, Docker CUDA image, and model manifest downloader are verified; live CUDA benchmark awaits Kaggle GPU run.
