# EgoAnnotate v3 Commercial Architecture — Master Execution Plan

**Branch:** `feat/commercial-v3`  
**Target Environment:** Customer-run Linux + NVIDIA CUDA GPUs (developed & CPU-smoke-verified on Intel MacBook Air 2017, Python 3.10.20, 2.1 GB free SSD).  
**Core Invariants:** Zero paid APIs, 100% commercially permissible default licenses (Apache-2.0 / MIT / BSD / CC-BY), zero fabricated/mocked metrics or signals, honest distinction between CPU-verified and GPU-static components.

## Phase Status

- [x] **Phase 0: Research, Baseline Audit & License Firewall**
  - Cloned & audited existing `EgoAnnotate` repo (430 existing tests, ~60 `src/` modules).
  - Identified commercial license leaks (`google-genai` paid API, `UniDepth` CC-BY-NC-4.0, `MANO` non-commercial references) and repo bloat (`audit_frames/`, ~45 ad-hoc scripts).
  - Downloaded compact real sample video (`ad521ca5-0017-475b-9dd8-1fdb423a66b7.mp4`, 25.08 MB) and derived `smoke_real_8s.mp4` (3.24 MB).
  - Documented `docs/GAP_ANALYSIS.md`, `docs/RESEARCH.md`, `docs/DECISIONS.md`, `docs/COORDINATE_CONVENTIONS.md`, `LICENSES.md`, `THIRD_PARTY_NOTICES.md`.
- [ ] **Phase 1: Core Modular Pipeline (`src/egoannotate/`)**
  - 1A: Unified typed config + plugin registry (`configs/profiles/cpu_smoke.yaml`, `configs/profiles/gpu_prod.yaml`) + strict Pydantic v2 IR schema.
  - 1B: Hands, Upper-Body Pose & Open-Vocab Manipulated-Object Tracking (MediaPipe Hands + Pose + ROI/SAM2/GroundingDINO adapters).
  - 1C: Single-Camera Monocular Geometry & World Frame (auto-intrinsics/distortion, 6-DoF Essential-Matrix/PnP VO + DPVO/DROID-SLAM GPU hook, multi-cue metric scale recovery with uncertainty, gravity alignment, chunked drift/loop-closure stitching, hand-to-world registration).
  - 1D: Behavior Understanding & Honest Tactile Channels (atomic action taxonomy, Feix/GRASP taxonomy, 2D/3D contact, bimanual coordination, object states, affordances/gaze/scene-graph, `derived_rgb_proxy` tactile estimation + `hardware_sensor` external tactile ingestion).
  - 1E: Local Free VLM Semantic Annotation (zero paid APIs: `Qwen2.5-VL`/`SmolVLM2` local adapters + geometrically grounded structured fallback, hierarchical `task -> subtask -> atomic_action` + VLA paraphrases).
  - 1F: Calibrated Multi-Dimension Data Quality Scoring & Gating.
  - 1G: Embodiment-Agnostic IR, Pluggable URDF/MJCF Retargeting (Humanoid + Dexterous Hand, Dual/Single Arm + Gripper, Standalone Dexterous Hand, Glove/Teleop), Exporters (LeRobot v3 + GR00T `modality.json`, RLDS/OXE, HDF5/Parquet), and runnable CPU Policy Training Smoke Test.
- [ ] **Phase 2: Production Engineering, Scalability, Observability & Commercial Packaging**
  - Unified CLI (`egoannotate`), Python API, SQLite-backed resumable batch queue & FastAPI REST worker.
  - Dual CPU/CUDA Dockerfiles, `docker-compose.yml`, checksum-verified model downloader (`scripts/download_models.py`) with offline mode.
  - Observability (structured JSON logs, HTML/JSON run reports, overlay QA video, 3D interactive viewer export), privacy face-blur + audio-strip gate, commercial licensing guide & customer docs.
- [ ] **Phase 3: Comprehensive Testing, Edge-Case Validation & CI**
  - `tests/test_no_fakes.py` anti-mock CI gate, real Drive clip E2E tests, edge-case suite (low light, motion blur, occlusion, missing intrinsics, short clip, 1-hand vs bimanual), schema-conformance loader tests, `.github/workflows/ci.yml`, `docs/VALIDATION.md`.
- [ ] **Phase 4: Skeptical Self-Audit (`AUDIT.md`) & Iterative Refinement**
  - Score all 15 rubric categories with real command outputs; iterate until all non-GPU-blocked categories >= 9.5/10 and honestly report any GPU-blocked limits.
- [ ] **Phase 5: Repository Cleanup, Fresh-Clone Verification & Handoff**
  - Remove dead scripts, `audit_frames/`, stray JSON dumps, paid API clients, and non-commercial defaults; verify fresh-venv quickstart and clean `git status`.
