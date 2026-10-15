# EgoAnnotate v1.0 Dataset Card & Schema Specification

EgoAnnotate converts monocular first-person (egocentric) human manipulation videos into embodiment-agnostic human hand trajectories and retargeted robot joint trajectories exported in **LeRobot v3**, **RLDS HDF5**, **MCAP v0**, and **JSON**.

## 1. Coordinate Frames & Units

| Coordinate Frame | Definition | Units |
|---|---|---|
| `image_normalized` | MediaPipe 21-keypoint `(x, y)` in `[0, 1]` image projection, `z` relative to wrist | Normalized `[0, 1]` |
| `camera_metric_approx` | Fixed-anchor anthropometric metric frame (`palm_length = 0.085m` $\pm 15\%$) | Meters (`m`), Radians (`rad`) |
| `robot_base` | URDF root link frame (`panda_link0`, `pelvis`, or `palm_link`), $+X$ forward, $+Y$ left, $+Z$ up | Meters (`m`), Radians (`rad`) |

## 2. Universal Human Layer vs. Embodiment-Specific Robot Layer

### A. Universal Human Features (Identical Across All Robots)
| Field | Shape | Dtype | Unit / Frame | Description |
|---|---|---|---|---|
| `observation.state` (Human) | `(24,)` | `float32` | `image_normalized` / `rad` | Left & Right wrist 3D pos `(6)`, palm orientation 6D `(12)`, 5-finger curl angles + grip `(6)` |
| `action` (Human) | `(24,)` | `float32` | `camera_metric_approx` | Frame-to-frame wrist SE(3) delta + finger curl delta |
| `left_hand` / `right_hand` | `(21, 3)` | `float32` | `image_normalized` | Persistent Left/Right hand landmarks + `is_interpolated` flag |
| `left_contact` / `right_contact` | `(5,)` | `bool` | Per-fingertip | Geometric fingertip-to-bbox contact flags + `object_name` + `confidence` |
| `left_grasp` / `right_grasp` | Scalar | `str` | Taxonomy label | `pinch`, `power_wrap`, `lateral`, `tripod`, or `open` |
| `camera_motion_flag` | Scalar | `bool` | Episode flag | `True` when background optical flow $\ge 18.0\text{ px/s}$ |

### B. Embodiment-Specific Robot Features (Configured via `configs/embodiments/*.yaml`)
| Field | Shape | Dtype | Unit / Frame | Description |
|---|---|---|---|---|
| `observation.robot_joint_angles` | `(N_dof,)` | `float32` | `rad` (`robot_base`) | IK-solved joint configuration (`N=7/9` Franka, `N=29` G1, `N=19` H1, `N=16` Allegro) |
| `observation.robot_gripper_opening_m` | `(1,)` | `float32` | `m` | Parallel gripper width in meters (`[0.0, 0.08]`) |
| `observation.robot_reachable` | `(1,)` | `bool` | Feasibility gate | `True` iff source hand tracked, IK residual $\le 0.25\text{ m}$, within joint/velocity limits, and collision-free |

## 3. Export Formats

Each processed episode `data/output/<episode_id>/` contains:
1. `annotations.json`: Complete episode metadata, per-frame annotations, temporal segments, camera motion, and retargeting metrics.
2. `summary.json`: High-level quality and feasibility summary metrics.
3. `manifest.json`: Atomic execution audit trail with exit code, stage runtimes, error log, and system telemetry.
4. `episode_rlds.hdf5`: Standard RLDS dataset for VLA training.
5. `lerobot_v3/`: Hugging Face LeRobot dataset (`meta/info.json` + `data/chunk-*.parquet`).
6. `episode.mcap`: Foxglove Studio / ROS2 compatible MCAP v0 stream separating `/human/*` from `/robot/*` channels.
7. `overlay_annotated.mp4`: Full native-resolution video with 2D/3D tracking, grasp labels, contacts, and task overlay.
8. `side_by_side.mp4`: Synchronized side-by-side video comparing human demonstration against simulated robot embodiment.
