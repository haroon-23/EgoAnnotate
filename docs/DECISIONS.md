# Architecture & Model Decisions Log (ADR)

Every technical decision below is grounded in November 2026 commercial license verification and robotics-ML utility.

## ADR-001: Dual-Profile Hardware Architecture (`cpu_smoke.yaml` vs `gpu_prod.yaml`)
- **Context:** Development runs on a 2017 Intel MacBook Air (CPU-only, low RAM/disk), while paying customers deploy on Linux + NVIDIA CUDA GPUs.
- **Decision:** Implement a unified plugin registry (`src/egoannotate/registry.py`) with auto-hardware selection (`auto` -> `gpu_prod` when `torch.cuda.is_available()`, else `cpu_smoke`).
- **Why:** Guarantees real end-to-end execution on CPU without weakening GPU production accuracy or introducing fake/stub models.

## ADR-002: Strict Commercial License Firewall (Apache-2.0 / MIT / BSD / CC-BY Only)
- **Options Considered:** Including MANO/HaMeR/UniDepth/Depth-Anything-V2-Large/MASt3R/VGGT/YOLO-World by default vs. strictly commercial-safe defaults.
- **Decision:** Enforce Apache-2.0 / MIT / BSD / zlib only for all default components. Remove paid `google-genai` (Gemini) from default workflows. Quarantine non-commercial research models behind explicit `allow_non_commercial: false` opt-in guards.
- **Why:** EgoAnnotate is built to be legally leased/licensed to robotics companies with zero licensing liability and $0 software cost.

## ADR-003: Single Gravity-Aligned Metric World Frame for Camera, Scene & Hands
- **Options Considered:** Camera-frame-only trajectories (current v2.1) vs. 6-DoF world-frame registration with multi-cue metric scale recovery.
- **Decision:** Estimate per-frame `T_world_cam` via Essential-Matrix + PnP keyframe VO (CPU) / DPVO or DROID-SLAM (GPU), align gravity using vertical/table-plane priors, recover metric scale via multi-cue fusion (anthropometric palm prior + Apache-2.0 metric depth prior when enabled) with explicit `scale_uncertainty_pct`, and register hands (`T_world_hand_left`, `T_world_hand_right`) and objects into that single world frame.
- **Why:** Eliminates head-motion corruption in manipulation policies and enables accurate multi-embodiment retargeting.

## ADR-004: Honest Derived Tactile Proxies + Hardware Sensor Ingestion
- **Options Considered:** Omitting tactile channels vs. synthesizing fake Newton readings vs. explicitly tagged derived proxies + real sensor ingestion.
- **Decision:** Emit a unified `TactileFrame` schema with mandatory `source` tag (`"derived_rgb_proxy"` vs `"hardware_sensor"`). RGB-derived channels compute contact probability, hand-zone contact map (5 fingertips + palm), normalized grasp-aperture deformation force proxy, slip/impact proxies, and uncertainty bounds. External time-synced CSV/HDF5 glove/FT streams ingest directly into the same schema.
- **Why:** Upholds zero-fabrication integrity while giving VLA models contact-rich training signals (filling the 99.2% tactile gap across LeRobot datasets).

## ADR-005: Config-Only Multi-Embodiment Retargeting (Zero Code Changes per Robot)
- **Options Considered:** Hardcoded adapter classes per robot vs. pure YAML + URDF/MJCF kinematic profile loader.
- **Decision:** Represent every embodiment class (humanoid+dexterous, single/dual arm+gripper, standalone dexterous hand, glove/teleop) purely via YAML configs in `configs/embodiments/` referencing URDF/MJCF files, joint limits, workspace bounds, end-effector frames, and finger-mapping matrices.
- **Why:** Customers can onboard proprietary robots in minutes without modifying pipeline code.

## ADR-006: Multi-Format Export (LeRobot v3 + GR00T `modality.json` + RLDS + HDF5/Parquet)
- **Decision:** Export every processed episode from the single embodiment-agnostic IR into (1) LeRobot v3 (`float32` Parquet + packet-synced MP4 + `meta/info.json` + `meta/stats.json` + `meta/tasks.jsonl` + GR00T `meta/modality.json`), (2) RLDS/Open-X-Embodiment-compatible episode records, and (3) Canonical HDF5 + JSON sidecar.
- **Why:** Directly feeds SmolVLA, openpi (π0/π0.5), GR00T N1.7, OpenVLA, Octo, Diffusion Policy, and ACT.

## ADR-007: Local Open-Source VLM + Geometric Verification Grounding
- **Decision:** Use `Qwen2.5-VL-7B-Instruct` / `SmolVLM2-2.2B-Instruct` (Apache-2.0) on GPU and deterministic geometrically-grounded structured annotation (or tiny local `SmolVLM2` when downloaded) on CPU, cross-checking every verb/hand/object against geometric contact/velocity signals.
- **Why:** Eliminates paid API costs, ensures 100% on-premise data privacy, and prevents VLM hallucinations.

## ADR-008: Scalable Local-First Orchestration (SQLite Job Queue + Content-Hashed Stage Cache + FastAPI)
- **Decision:** Implement an idempotent, resumable SQLite job queue + SHA-256 stage cache + multi-worker CLI/FastAPI service that requires zero external paid infrastructure and scales cleanly across multi-GPU Linux nodes.
