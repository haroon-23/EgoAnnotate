# EgoAnnotate — Living Project State (update in every gate commit)
Branch: build/product-v1 | Authoritative artifact: data/output/reference_v2 | Tier: STRESS_APPENDIX
VERIFIED: trajectory downsampled 1.93x to 1409 frames (max speed 1.5 rad/s); reachable 1044/1409 (74.1%);
tracking loss 27.5%; sanitizer + pre-export assertion enforce limits/velocity at export;
147 tests pass; visualizer ROI blending 42.4 FPS; GDINO keyframe cache live.
PHASE E (built, not yet run on hardware): mink opt-in IK backend (retargeting.ik_backend; QP fallback quadprog->daqp->default), weld-grasp semantics (src/retargeting/weld_grasp.py), gated EpisodeVerifier (M1 tracking gate + per-window weld trials; Stage 9b sets physics_verified label, never drops episodes), scripts/hero_clip.py (refuses unless report passed). Tier stays STRESS_APPENDIX until a Mac run with mujoco installed produces the first physics-verified episode.
PHASE C COMPLETE (v1.2-phaseC): Config-driven retargeting adapter registry (src/retargeting/embodiment_registry.py) and multi-embodiment retargeter (scripts/retarget_multi.py) supporting Franka arm, Unitree H1 upper-body humanoid, Unitree G1 upper-body humanoid, and Allegro 4-finger dexterous hand.
MULTI-EMBODIMENT MANIFEST (reference_v2, 1409 frames):
  - franka_arm: 1044/1409 reachable, mean FK residual 0.434m (nq=9, arm)
  - unitree_h1_upper: 1044/1409 reachable, mean FK residual 0.251m (nq=19, humanoid)
  - unitree_g1_upper: 1044/1409 reachable, mean FK residual 0.062m (nq=29, humanoid)
  - allegro_hand: 1042/1409 reachable, mean FK residual 0.451m (nq=16, hand)
URDF LICENSES RECORDED:
  - panda.urdf: Apache-2.0 (Franka Emika)
  - allegro.urdf: BSD (SimLab Robotics / allegro_hand_ros_v4)
  - h1.urdf: BSD-3-Clause (Unitree Robotics / isri-aist h1_description)
  - g1.urdf: BSD-3-Clause (Unitree Robotics / robot-descriptions)
PHASE D-REDO DEBTS QUEUED:
  1. Computed-torque harness tuning for MuJoCo physics replay.
  2. Grip-channel state mapping fix for finger actuators.
STALE/DO-NOT-SHIP: data/output/whatsapp_video (old export, joint6=3.7989 rad > 3.7525 limit).
OPEN: contact channel under-detects hand-occluded objects; gripper pin-rate to re-measure on reference_v2.
PHASES: A metric truth -> B perception quality (RTMW/SAM2/consensus) -> C any-embodiment
(GMR humanoid + dex-retargeting + Pinocchio cross-check) -> D proof (MuJoCo replay + ACT
smoke-train + on-device success labels) -> E scale/governance (OpenVINO/int8, SPDX audit,
Croissant, HF publish). Parked: Grounding-DINO swap done; Gemini Robotics rejected (cloud,
not a retargeter); HaMeR/Isaac GPU-walled; ORB-SLAM3/YOLO-World GPL-excluded.
DECISIONS (Phase A Close & Phase D Physics & Phase C Multi-Embodiment):
- MoGe ViT-L rejected for CPU depth extraction (27.50s/frame CPU vs <=4s budget).
- Depth-Anything-V2-Small rejected for CPU depth extraction (avg 7.40s/frame @ 512px vs <=4s budget).
- Metric Scale Strategy: A4 table-plane calibration (scripts/calibrate_scale.py) when A4 sheet is present, falling back to anthropometric ratio (K=0.75) with disclosed ±15% caveat.
- Physics Replay Proof: Evaluated trajectory downsampling (1.5 rad/s max speed) and per-window placement trials. Classified output as Tier: STRESS_APPENDIX (74% reachable, 0% physics-verified success); committing v1.1 and moving to Phase C (any-embodiment).
- Multi-Embodiment Registry: Integrated mink IK daqp solver with posture regularization and posture joint range clipping, generating 4-robot feasibility manifests with export-side sanitization.
DISCIPLINE: raw paste only; no silent fallbacks; per-joint limits at export; commit+tag+push
after every gate; update this file in the same commit.


