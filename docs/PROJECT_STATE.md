# EgoAnnotate — Living Project State (update in every gate commit)
Branch: build/product-v1 | Authoritative artifact: data/output/reference_v1
VERIFIED: exported max joint speed 2.3475 rad/s (<=2.349 budget); reachable 540/729 (74.07%);
tracking loss 27.57%; sanitizer + pre-export assertion enforce limits/velocity at export;
145 tests pass; visualizer ROI blending 42.4 FPS (post-crash fix); GDINO keyframe cache live.
STALE/DO-NOT-SHIP: data/output/whatsapp_video (old export, joint6=3.7989 rad > 3.7525 limit).
OPEN: contact channel under-detects hand-occluded objects; gripper pin-rate to re-measure on reference_v1.
PHASES: A metric truth -> B perception quality (RTMW/SAM2/consensus) -> C any-embodiment
(GMR humanoid + dex-retargeting + Pinocchio cross-check) -> D proof (MuJoCo replay + ACT
smoke-train + on-device success labels) -> E scale/governance (OpenVINO/int8, SPDX audit,
Croissant, HF publish). Parked: Grounding-DINO swap done; Gemini Robotics rejected (cloud,
not a retargeter); HaMeR/Isaac GPU-walled; ORB-SLAM3/YOLO-World GPL-excluded.
DECISIONS (Phase A Close):
- MoGe ViT-L rejected for CPU depth extraction (27.50s/frame CPU vs <=4s budget).
- Depth-Anything-V2-Small rejected for CPU depth extraction (avg 7.40s/frame @ 512px vs <=4s budget).
- Metric Scale Strategy: A4 table-plane calibration (scripts/calibrate_scale.py) when A4 sheet is present, falling back to anthropometric ratio (K=0.6) with disclosed ±15% caveat.
DISCIPLINE: raw paste only; no silent fallbacks; per-joint limits at export; commit+tag+push
after every gate; update this file in the same commit.
