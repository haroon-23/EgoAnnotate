# EgoAnnotate Coordinate Frame, State/Action & Dataset Conventions

## 1. Coordinate Frames & Units (`ego_annotate` Metadata Contract)
- **Reference Frame (`frame: "camera"`):** All 24-D human hand state and action vectors are expressed in the right-handed pinhole camera coordinate frame ($+X$ right, $+Y$ down, $+Z$ optical depth in meters).
- **Scale & Intrinsics (`scale_source: "assumed_palm_85mm"`, `is_metric: false`):** When chessboard calibration (`scripts/calibrate_camera.py`) is not supplied, wrist depth $Z$ is estimated from apparent 2D palm length using an $85\text{ mm}$ anthropometric palm prior and device profile `configs/devices/generic_smartphone.yaml`.
- **24-D State Vector (`observation.state`, `float32`):**
  - `0..2`: `wrist_x`, `wrist_y`, `wrist_z` (meters in camera frame)
  - `3..8`: `rot6_0..rot6_5` (continuous 6-D orthonormal wrist rotation representation)
  - `9`: `gripper` (normalized thumb-index pinch aperture in $[0, 1]$)
  - `10..23`: `f0..f13` (14 anatomical inter-phalangeal joint flexion angles in radians)
- **24-D Action Vector (`action`, `float32`):**
  - `0..23`: `delta_wrist_x..delta_f13` defined as $\mathbf{a}_t = \mathbf{s}_{t+1} - \mathbf{s}_t$ on consecutive valid frames ($\text{hand\_valid}[t] \land \text{hand\_valid}[t+1]$), and $\mathbf{0}$ across invalid transitions.
- **Validity Mask (`observation.hand_valid`, `[2] int64`):**
  - `[left_hand_valid, right_hand_valid]` in $\{0, 1\}^2$.

## 2. Source Video Resolution & Aspect Ratio Policy
- `client_video`, `my_video`, `new_video`, and `test_10s` are native $1080\text{p}$ captures ($1920\times 1080$ or $1080\times 1920$).
- `whatsapp_video*` clips are raw WhatsApp mobile captures at native WhatsApp transport resolution ($1024\times 576$ or $384\times 848$). Aspect ratio is strictly preserved across both LeRobot v3 MP4 streams and HDF5 image tensors.
