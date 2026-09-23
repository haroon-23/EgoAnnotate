# EgoAnnotate — Living Project State (update in every gate commit)
Branch: build/product-v1 | Authoritative artifact: data/output/reference_v2 | Tier: STRESS_APPENDIX
VERIFIED: trajectory downsampled 1.93x to 1409 frames (max speed 1.5 rad/s); reachable 1044/1409 (74.1%);
tracking loss 27.5%; sanitizer + pre-export assertion enforce limits/velocity at export;
145 tests pass; visualizer ROI blending 42.4 FPS; GDINO keyframe cache live.
PHYSICS PROOF (Phase D): MuJoCo physics replay executed on reference_v2 (2 grasp windows, max lift 0.017m, mean tracking err 0.2218 rad). 0% physics-verified success due to dynamic infeasibility on cluttered clip; queued for hero protocol clip.
STALE/DO-NOT-SHIP: data/output/whatsapp_video (old export, joint6=3.7989 rad > 3.7525 limit).
OPEN: contact channel under-detects hand-occluded objects; gripper pin-rate to re-measure on reference_v2.
PHASES: A metric truth -> B perception quality (RTMW/SAM2/consensus) -> C any-embodiment
(GMR humanoid + dex-retargeting + Pinocchio cross-check) -> D proof (MuJoCo replay + ACT
smoke-train + on-device success labels) -> E scale/governance (OpenVINO/int8, SPDX audit,
Croissant, HF publish). Parked: Grounding-DINO swap done; Gemini Robotics rejected (cloud,
not a retargeter); HaMeR/Isaac GPU-walled; ORB-SLAM3/YOLO-World GPL-excluded.
DECISIONS (Phase A Close & Phase D Physics):
- MoGe ViT-L rejected for CPU depth extraction (27.50s/frame CPU vs <=4s budget).
- Depth-Anything-V2-Small rejected for CPU depth extraction (avg 7.40s/frame @ 512px vs <=4s budget).
- Metric Scale Strategy: A4 table-plane calibration (scripts/calibrate_scale.py) when A4 sheet is present, falling back to anthropometric ratio (K=0.75) with disclosed ±15% caveat.
- Physics Replay Proof: Evaluated trajectory downsampling (1.5 rad/s max speed) and per-window placement trials. Classified output as Tier: STRESS_APPENDIX (74% reachable, 0% physics-verified success); committing v1.1 and moving to Phase C (any-embodiment).
DISCIPLINE: raw paste only; no silent fallbacks; per-joint limits at export; commit+tag+push
after every gate; update this file in the same commit.

