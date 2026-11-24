# EgoAnnotate Coordinate Conventions (v3 Canonical)

All geometry, hand trajectories, object poses, and retargeting inputs adhere strictly to these right-handed coordinate conventions.

## 1. Camera Optical Frame (`C`)
- Standard OpenCV / pinhole optical convention:
  - `+X_c`: Right in image plane
  - `+Y_c`: Down in image plane
  - `+Z_c`: Forward along camera optical axis (depth > 0 in front of lens)
- Intrinsic matrix `K` (`3x3`): `fx, fy, cx, cy` in pixels, plus Brown-Conrady distortion `[k1, k2, p1, p2, k3]`.
- When EXIF/calibration YAML is absent, auto-intrinsics estimates `fx = fy = 0.85 * max(W, H)` (or FOV-refined via vanishing/homography cues), `cx = W/2`, `cy = H/2`, and sets `intrinsics_source = "auto_estimated"` with explicit uncertainty.

## 2. Gravity-Aligned World Frame (`W`)
- Robotics standard ENU / Z-up right-handed frame:
  - `+X_w`: Forward (initial horizontal heading of wearer at `t = 0`, projected orthogonal to gravity)
  - `+Y_w`: Left (right-hand rule: `Z_w × X_w`)
  - `+Z_w`: Up (anti-gravity vector `+g_hat`, estimated from initial camera pitch prior / vertical scene lines / tabletop plane normal)
- **Transform from Camera Optical (`C`) to World (`W`) at `t = 0` (level camera):**
  - `x_w = z_c` (forward)
  - `y_w = -x_c` (left)
  - `z_w = -y_c` (up)
- Homogeneous transform `T_world_cam(t) ∈ SE(3)` maps a point `p_c` in camera frame to world frame: `p_w = R_wc @ p_c + t_wc`.

## 3. Hand & Wrist Frame (`H_left`, `H_right`)
- 21 MediaPipe articulated hand landmarks (indices `0..20`):
  - `0`: Wrist origin
  - `5`: Index MCP, `9`: Middle MCP, `17`: Pinky MCP
  - Fingertips: `[4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)]`
- Local hand basis at wrist `0`:
  - `+X_h`: From wrist (`0`) toward middle MCP (`9`) (finger-pointing axis)
  - `+Z_h`: Palm normal (out of palm surface, consistent handedness via cross product with index/pinky MCP span)
  - `+Y_h`: `Z_h × X_h` (right-handed orthogonal basis)
- Orientation stored as:
  - Unit quaternion `quat_xyzw` (`[qx, qy, qz, qw]`) for SE(3) & URDF/MuJoCo kinematics.
  - Continuous 6D rotation representation (`rot6d`, first two columns of `R` flattened, Zhou et al. 2019) for neural VLA / policy state & action vectors.

## 4. Metric Scale Honesty Contract
- Every world-frame trajectory carries:
  - `scale_meters_per_unit: float`
  - `scale_source: "calibrated_intrinsics_and_metric_depth" | "metric_depth_and_palm_prior" | "anthropometric_palm_prior_only"`
  - `scale_uncertainty_pct: float` (e.g., `±12%` for palm-prior-only monocular, `±5%` for calibrated multi-cue fusion)
  - `is_calibrated_metric: bool` (only `true` when real camera calibration or hardware depth/IMU is supplied).

## 5. Robot Base Frame (`B`)
- Each embodiment YAML defines `T_robot_base_to_world` (or anchor mapping from human workspace center to robot base frame), ensuring every retargeted end-effector target `T_base_ee` respects robot reachability and table collision planes (`z >= table_z_m`).
