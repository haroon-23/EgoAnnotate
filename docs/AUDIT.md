# Audit of EgoAnnotate v1.2.0 (commit ce71bc5)

Three independent audits (perception, retargeting, export/orchestration) were
run against the code as published. Findings marked **run** were reproduced by
executing code; **read** means inferred from the source only. Line numbers refer
to commit `ce71bc5`.

## Verdict

v1.2.0 was not usable for a real robot. The robot trajectory was not a metric
image of the human motion, the "hardware readiness" score could not fail, and
the repository could not run from a clean clone. The v2 rebuild
(`docs/ARCHITECTURE.md`) addresses each item below; `docs/VERIFICATION.md`
records what has been measured since.

## Repository state

| Finding | Evidence |
|---|---|
| `data/` is gitignored, so robot URDFs, input videos and the "authoritative" reference outputs were never in the repository. | `git ls-files data` is empty; 5 tests failed on `data/urdfs/*.urdf`. |
| `requirements.txt` could not be installed: `lerobot==0.4.4` needs `torchvision>=0.21`, the file pins `0.17.2`. | Project setup failed with a resolver conflict. |
| The default IK backend imports `pybullet`, which is not a dependency. | 33 tests skipped with "pybullet not installed". |
| Baseline test run: 9 failed, 334 passed, 33 skipped. | `pytest tests -q` in a clean Python 3.10 environment. |
| Documentation contradicted itself: `docs/PROJECT_STATE.md` reported 74% reachable and 0% physics-verified, `EGO_AUDIT_DOSSIER.json` reported 100% readiness. | Both files at `ce71bc5`. |

## Blockers for robot use

### Retargeting

1. **The readiness score was self-fulfilling (run).**
   `src/retargeting/control_filter.py:142` set `playable_mask` to all ones;
   velocity and acceleration were scored after being clamped to 0.85 of
   invented limits (lines 266-289). Measured: 90% tracking loss scored 100.0;
   no hand tracked at all scored 100.0 "PRODUCTION_READY"; a target 1.4 m away
   scored 100.0 with 100% reach.
2. **Human motion was re-centred, scaled and clipped (run).**
   `src/retargeting/embodiment_registry.py:340-358` subtracted the clip median,
   multiplied by 0.5 (arms) or 0.35 (humanoids) and clipped to +-0.16 m around
   the robot's home pose. A +-0.2 m human sweep became +-0.1 m. Line 388
   invented a left-hand target when none existed.
3. **No camera-to-robot transform (read, partly run).**
   `src/retargeting/pose_mapper.py:348-353` assigned camera axes directly to
   robot axes around the clip median, with no extrinsics or gravity alignment
   and no head-motion compensation, then clipped to a 0.35 x 0.60 x 0.30 m box.
   MediaPipe z is wrist-relative, so depth motion was close to zero.
4. **"Reachable" had four definitions** (5 mm, 1 cm, 0.25 m and none). In
   `scripts/run_reference_pipeline.py:372-378` joints were clipped to limits
   before the limit check and `reach=True` was set without any residual check.
5. **Exported joints could exceed limits (run).** The limiter at
   `control_filter.py:225` skipped clipping for `i >= 2`; joint 5 reached
   -2.972 rad against a -2.897 limit while the audit reported 100.
6. **IK errors were swallowed.** `embodiment_registry.py:268` wrapped the solve
   in `except Exception: pass`.
7. **Physics verification was off by default and could not fail** when enabled:
   it only labelled results, swallowed exceptions, and the welded-grasp check
   measured end-effector motion rather than object motion.

### Perception

1. **Fabricated object positions and masks (run).**
   `src/perception/dense_object_tracker.py:226-241` overwrote every 3D position
   with `[(cx-0.5)*0.45, (cy-0.5)*0.35, 0.42]`, invented a `manipulated_object`
   box when nothing was detected, and wrote Otsu-threshold polygons as
   segmentation masks.
2. **Left and right hands swapped (read).** `src/hand_tracker.py:123` kept
   MediaPipe's mirrored-input convention on unmirrored video; duplicate labels
   overwrote each other (run).
3. **Contact detection merged both hands (run).** The pipeline never passed
   `hand_side`, so both hands shared one state machine; a geometric fallback
   bypassed the debounce; a pinch fallback invented an object.
4. **Actions were not physical quantities (run).**
   `src/action_computer.py:73-80,190-208` percentile-centred each episode (a
   stationary hand became -1.0), used image-fraction deltas per frame and
   clipped them.
5. **Geometry was computed on squashed frames (run).**
   `src/video_processor.py:127` resized every frame to 224 x 224; a portrait
   1080 x 1920 frame is distorted 1.78 times.
6. **Metric scale was not metric (run).** One episode-wide scale from a palm
   length constant; `pose_mapper.py:296` read `scale_m_per_unit` while the
   calibrator wrote `scale_factor_m_per_unit`, so calibration had no effect.
   `src/perception/pnp_pose.py:253-340` mixed normalised and pixel coordinates
   (depth off by 31% at 640 x 480).

### Export and orchestration

1. **The task instruction was lost in the LeRobot dataset (run).**
   `src/lerobot_exporter.py:472-474` wrote `meta/tasks.parquet` with a range
   index; LeRobot 0.4.4 reads the text from the index, so every sample's task
   was `0`.
2. **Video and annotations could be misaligned (read).** The exporter
   transcoded the whole source video while annotations were subsampled.
3. **Unreachable frames were exported as valid actions (run).**
4. **`scripts/validate_delivery.py` passed invalid deliveries (run):** black
   images, all-zero joints, a placeholder task and mismatched frame counts
   passed; it also created missing files as a side effect.
5. **Exit codes and the manifest were unreliable (read).** Standalone exports
   always exited 0 and several stages never wrote a manifest.
6. **No way to run without a Gemini key (run).** The constructor raised.
7. **The stage cache ignored configuration (read).** Keyed only by video hash
   and stage name.
8. **"RLDS" output is a custom HDF5 layout**, not readable by OpenVLA or Octo
   loaders without conversion; the MCAP file was valid but had no index or
   summary.

## What was sound

- The mink differential IK solver wrapper.
- The LeRobot v3.0 directory layout (it loaded in LeRobot 0.4.4 apart from the
  task index).
- Atomic file writes.
- The deterministic signal-based segmentation.
