# EgoAnnotate — Living Project State (update in every gate commit)
Branch: build/product-v1 | Authoritative artifact: data/output/reference_v1
VERIFIED: exported max joint speed 2.3475 rad/s (<=2.349 budget); reachable 540/729 (74.07%);
tracking loss 27.57%; sanitizer + pre-export assertion enforce limits/velocity at export;
145 tests pass; visualizer ROI blending 42.4 FPS (post-crash fix); GDINO keyframe cache live.
STALE/DO-NOT-SHIP: data/output/whatsapp_video (old export, joint6=3.7989 rad > 3.7525 limit).
OPEN: Phase A (MoGe/VO/AprilTag metric scale) Step 1 benchmark NOT yet run; verify_demo_package
per-joint limit check added 2026-09-23; contact channel under-detects hand-occluded objects;
gripper pin-rate to re-measure on reference_v1.
PHASES: A metric truth -> B perception quality (RTMW/SAM2/consensus) -> C any-embodiment
(GMR humanoid + dex-retargeting + Pinocchio cross-check) -> D proof (MuJoCo replay + ACT
smoke-train + on-device success labels) -> E scale/governance (OpenVINO/int8, SPDX audit,
Croissant, HF publish). Parked: Grounding-DINO swap done; Gemini Robotics rejected (cloud,
not a retargeter); HaMeR/Isaac GPU-walled; ORB-SLAM3/YOLO-World GPL-excluded.
DISCIPLINE: raw paste only; no silent fallbacks; per-joint limits at export; commit+tag+push
after every gate; update this file in the same commit.
