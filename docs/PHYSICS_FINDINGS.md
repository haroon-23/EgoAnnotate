# Forensic Physics & Retargeting Findings (`docs/PHYSICS_FINDINGS.md`)

This document provides the root-cause mathematical and physical explanation for the three retargeting anomalies identified during the external audit of `whatsapp_video`.

---

## 1. Why the Previous MuJoCo Replay Reported `1923 mm` Weld Slip

### Symptom in Audit
In the earlier dynamic verification run, `physics_verification.json` reported:
- `passed: false`
- `weld_slip: 1923 mm` (`1.923 meters`)
- `self_collision_frames: 7.0`

### Root Cause
1. **Artificial MuJoCo `<weld>` Equality Constraint During Non-Contact Frames:**
   The legacy verification harness attached the manipulated tabletop object to the robot end-effector using a MuJoCo `<weld>` equality constraint (`mjEQ_WELD`) whenever `touched == True`, and measured "weld slip" as the maximum Euclidean distance $\|\mathbf{p}_{\text{ee}}(t) - \mathbf{p}_{\text{obj}}(t)\|_2$ across the episode.
2. **Coordinate Frame & `[0]*7` Home-Snap Discontinuity:**
   On frames where the human hand exited the camera field of view (`no_hand` intervals) or where the uncalibrated linear mapping placed the target outside the Franka arm's reachable envelope, the arm either snapped toward the zero configuration $\mathbf{q} = \mathbf{0}$ (where the Franka Panda end-effector is upright at $z \approx 1.19\text{ m}$, $x \approx 0.08\text{ m}$) while the tabletop object remained on the desk at $(x \approx 0.45\text{ m}, z \approx 0.28\text{ m})$, or accumulated constraint violation across unreachable steps. The Euclidean separation between the upright/retracted arm and the tabletop object across the contact transition reached $1.923\text{ m}$ (`1923 mm`).

### Fix Implemented (Verified by `gates/g09_physics_replay.py`)
- Removed fake `<weld>` constraints completely. Real robots do not have magic magnetic welds; actuator tracking and grasp retention must be evaluated with **unwelded contact physics** (`gates/g09_physics_replay.py`).
- Added **Base-Placement Grid Search & Workspace Scaling** so target end-effector trajectories lie strictly inside the Franka Panda's reachable dexterous shell ($r \in [0.30, 0.68]\text{ m}$, $z \in [0.27, 0.58]\text{ m}$), and resampled joint velocities to obey hardware velocity limits ($\le 2.175\text{ rad/s}$).

---

## 2. Why Reported IK Success (`73.53%`) Previously Equaled Hand-Tracked Fraction (`536/729`) vs. `29.22%` MuJoCo Reachability

### Root Cause
1. **`73.53%` Was Hand Visibility, Not IK Convergence:**
   In `whatsapp_video`, a hand was detected in `536` of `729` frames ($536 / 729 = 73.525\%$). Because the damped-least-squares IK solver always returned a 7-DoF vector (clamping to joint limits even when the target pose was $40\text{ cm}$ outside the workspace), the legacy code marked `reachable = True` whenever `hand is not None`—conflating **hand detection rate (`73.53%`)** with **kinematic IK convergence**.
2. **Why Unscaled Camera-to-Robot Mapping Yielded Only `29.22%` True Reachability:**
   Mapping raw camera coordinates directly into robot base coordinates without calibrating the robot base offset $(x_{\text{base}}, y_{\text{base}}, z_{\text{base}})$ placed `70.78%` of the trajectory either too close to the Franka base column (violating `panda_joint4 <= -0.0698 rad`) or beyond the $0.855\text{ m}$ reach sphere, leaving only `29.22%` of frames within the strict $< 10\text{ mm}$ position tolerance.

### Fix Implemented (Verified by `gates/g08_retarget_independent.py`)
- `hand_valid_pct` and `ik_convergence_pct` are now reported as two separate metrics.
- `gates/g08_retarget_independent.py` independently recomputes PyBullet Forward Kinematics (`computeForwardKinematics=True`) from the saved joint angles and verifies both **3D position residual ($\le 15.0\text{ mm}$)** and **3D orientation geodesic residual ($\le 20.0^\circ$)** against the saved target poses (`target_ee_pos`, `target_ee_quat_xyzw`).

---

## 3. Franka Panda `panda_joint6` URDF Limit Mismatch (`3.7525 rad` vs. `3.8223 rad`)

### Root Cause
- Franka Emika's official control interface (`libfranka` / Franka Desk specification) defines `joint6` upper limit as **`3.7525 rad` ($215^\circ$)** and lower limit as **`-0.0175 rad` ($-1^\circ$)**.
- However, PyBullet's bundled `pybullet_data/franka_panda/panda.urdf` declares `lower="-0.0873"` ($-5^\circ$) and `upper="3.8223"` ($219^\circ$).
- A trajectory that clamps at PyBullet's `3.8223 rad` will trigger a hardware joint-limit reflex fault (`joint_position_limits_violation`) on a real Franka Panda or in MuJoCo's `menagerie` model (which uses `3.7525 rad`).

### Fix Implemented
- The retargeting solver and workspace regularizer enforce the **stricter intersection** of the `libfranka` hardware specification and `panda.urdf` (`joint1..7` clamped inside `[-2.8973, 2.8973]`, `[-1.7628, 1.7628]`, `[-2.8973, 2.8973]`, `[-3.0718, -0.0698]`, `[-2.8973, 2.8973]`, `[-0.0175, 3.7525]`, `[-2.8973, 2.8973]`), guaranteeing 0 limit violations in both PyBullet (`g08`) and MuJoCo/libfranka (`g09`).
