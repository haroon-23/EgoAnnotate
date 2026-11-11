# EgoAnnotate v2 architecture and contracts

This document is the binding contract between pipeline stages. It replaces the
v1 design, whose geometry and readiness metrics were found to be invalid (see
`docs/AUDIT.md`).

## 1. Goal

Turn one egocentric video of human hands into robot training data whose robot
trajectory is a rigid, metric image of the human motion, with every number in
the output measured rather than assumed. Anything that cannot be measured is
labelled as an assumption and lowers the episode's tier.

Non-goals: task success in the real world is not claimed. The pipeline proves
kinematic and dynamic feasibility in simulation; hardware validation is a
separate bring-up step (`docs/HARDWARE_BRINGUP.md`).

## 2. Rules that apply everywhere

1. No fabricated data. A missing quantity is `None` and flagged, never a
   default, a median, or a clamp. Physical saturation (a gripper cannot open
   wider than its stroke) is the only allowed clamp and must be counted.
2. No scaling, re-centring or clipping of human motion. The only transform
   between human and robot space is a rigid one, recorded in the output.
3. Metrics are computed on the exported trajectory over all tracked frames,
   never only over frames that already passed a gate.
4. Fail loud. No bare `except`. A stage either succeeds, or raises, or records a
   named degradation in the manifest.
5. SI units: metres, radians, seconds. Timestamps come from the source video.

## 3. Frames and conventions

| Frame | Definition |
|---|---|
| `C` camera | OpenCV optical frame: +x right, +y down, +z forward. |
| `H` hand | Origin at wrist (landmark 0). +x toward middle MCP (9). +z out of the palm. +y = z x x. Same rule for left and right. |
| `W` world | Fixed, gravity-aligned, +z up. Source is recorded (section 5). |
| `B` robot base | Robot model root, +x forward, +y left, +z up. |

Rotations are 3x3 matrices that map child-frame vectors to the parent frame.
Quaternions, where a library needs them, are documented at the call site.

## 4. Stage 2: metric hand pose (`src/perception/metric_hand_pose.py`)

MediaPipe gives 21 image landmarks and 21 world landmarks (metres, hand-local).
`cv2.solvePnP` between the two, using the camera matrix `K` at the tracked
resolution, yields the hand pose in `C` with metric translation.

```python
def camera_matrix(cfg: dict, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (K 3x3, dist coeffs, source). source is "calibrated" when
    camera.fx/fy/cx/cy are set (given at camera.calib_size and rescaled),
    else "assumed_fov" from camera.assumed_hfov_deg."""

def lift_hand(hand: HandLandmarks, K, dist, source: str, hand_scale: float = 1.0,
              max_reproj_px: float | None = None) -> MetricHandPose | None:
    """None when world landmarks are missing, PnP fails or reprojection error
    exceeds the limit. Interpolated input yields is_interpolated=True."""
```

`hand_scale` is the ratio of the demonstrator's real hand length to MediaPipe's
canonical hand (default 1.0); depth error is proportional to errors in it and in
the focal length, which is why assumed intrinsics lower the tier.

Tracking runs on native-aspect frames (long side limited by
`pipeline.tracking_max_side`), never on squashed 224x224 frames. MediaPipe
handedness is corrected for unmirrored input.

## 5. Stage 2b: world frame (`src/perception/world_frame.py`)

```python
def estimate_world_frame(image_paths, frames, K, dist, cfg) -> WorldFrameResult
# WorldFrameResult: T_world_cam (T,4,4), valid (T,) bool, source (str),
#                   gravity_source (str), quality (dict)
```

| `world.mode` | How | Grade |
|---|---|---|
| `aruco` | Fiducial marker of known size on the table; per-frame camera pose, gravity = marker normal. | metric |
| `static_camera` | Camera assumed fixed; up vector from `world.camera_pitch_deg`. Invalid when measured camera motion exceeds `world.max_static_motion_px_per_sec`. | metric only if motion is below threshold |
| `rotation_compensated` | Head rotation estimated from background features (pure-rotation homography); translation unknown. | approximate |

## 6. Stage 9: retargeting (`src/retargeting/`)

Inputs are plain arrays, so the retargeter does not depend on perception code:

```python
@dataclass
class HandTrack:                    # one per hand, world frame
    position: np.ndarray            # (T,3) wrist, metres
    rotation: np.ndarray            # (T,3,3) hand frame H in W
    keypoints: np.ndarray           # (T,21,3) metres
    valid: np.ndarray               # (T,) bool: tracked, not interpolated

def retarget_episode(embodiment: Embodiment, timestamps: np.ndarray,
                     right: HandTrack | None, left: HandTrack | None,
                     cfg: dict | None = None) -> RetargetResult
```

`RetargetResult` carries `joint_names`, `q (T,nq)`, `valid (T,)`,
`ee_pos_error_m (T,)`, `ee_ori_error_deg (T,)`, `gripper_opening_m (T,)` where
applicable, `T_base_world (4,4)` and `report` (dict, section 8).

Steps:

1. Models come from MuJoCo Menagerie at a pinned commit
   (`scripts/fetch_robot_models.py`), or any MJCF/URDF the user points to.
   Floating bases are fixed and uncontrolled joints are locked by config.
2. Base placement: one rigid `T_base_world` per episode (yaw about gravity plus
   translation), either given or chosen automatically to maximise reachable
   frames. Both hands share it.
3. End-effector target = `T_base_world * hand pose * tcp_offset`.
4. Differential IK (mink, daqp) with position, orientation and posture tasks,
   joint position and velocity limits from the model, real `dt` from timestamps.
5. A frame is valid iff the hand was tracked (not interpolated), position error
   <= `tolerances.pos_m` (default 0.01), orientation error <=
   `tolerances.ori_deg` (default 15, or position-only mode, flagged), all joints
   inside limits without clipping, and no self-collision.
6. Grippers: thumb-index aperture in metres, mapped to the gripper stroke.
   Dexterous hands: fingertip positions relative to the palm, scaled by the
   robot-to-human hand length ratio (recorded), solved by IK on the hand model.
7. Physics replay (`src/retargeting/replay.py`): the exported joint targets
   drive the model's own position actuators in MuJoCo with gravity; measured
   joint tracking error, actuator saturation and contacts form the replay gate.

## 7. Exports

`observation.state[t]` and `action[t]` are float32 with named dimensions in
`meta/info.json`. No per-episode normalisation; `meta/stats.json` carries the
statistics for training-time normalisation.

- Robot mode: state = joint positions (+ gripper opening) at `t`, action = joint
  targets at `t+1`. Only the longest contiguous valid run is exported; trimming
  is recorded. Export is refused when the gate fails unless `--allow-ungated`.
- Human mode: per hand, wrist position in `W` (3), rotation 6D (6), thumb-index
  aperture (1) and five fingertip positions in `H` (15), plus a validity flag.
  Action = the same quantities at `t+1`.
- Video is encoded from the sampled frames, so it has exactly one image per row.

## 8. Report and tiers

`retargeting.report` holds, per embodiment: `tracked_fraction`,
`valid_fraction`, `longest_valid_run`, position and orientation error
(mean/p95/max over all tracked frames), peak joint velocity, acceleration and
jerk against limits, limit-violation and self-collision frame counts, the
replay tracking error, and `gate: {passed, reasons}`.

| Tier | Requirements |
|---|---|
| `robot_grade` | Calibrated intrinsics, metric world frame, gate passed, replay passed. |
| `approximate` | Any assumption in the geometry chain (assumed FOV, assumed pitch, rotation-only world frame). Usable for pretraining, not for direct replay on hardware. |
| `rejected` | Gate failed. |

There is no 0-100 readiness score.
