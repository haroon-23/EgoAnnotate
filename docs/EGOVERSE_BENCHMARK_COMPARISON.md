# EgoVerse (RSS 2026) vs. `ego_annotate_vla_pipeline` Architecture & Signal Parity

| Feature / Dimension | EgoVerse (`egoverse.ai` / arXiv:2604.07607) | `ego_annotate_vla_pipeline` (v2.1) |
| :--- | :--- | :--- |
| **Dataset Tiers** | EgoVerse-A (aligned) + EgoVerse-I (1080p @ 30Hz filtered) | Homogeneous 1080p @ 30.0 Hz (`ACCEPT`) + Automated `<720p` Quarantine (`REJECT`) |
| **Hand Pose & Provenance** | 21 keypoints/hand in camera frame | 21 keypoints/hand in camera frame + `detected`/`rescued`/`interpolated` provenance + `observation.hand_valid` + 180-label blind chirality audit (`G05`) |
| **6-DoF Camera Pose** | Visual-inertial SLAM (`T_world_cam`) | 6-DoF camera pose trajectory (`camera_pose_6dof`) + gravity-aligned tabletop frame compensation |
| **Action & Tokenization** | Human keypoint trajectories | 24-D camera-frame deltas + EgoDex 48-D + Open-X 7-D + pi0-FAST DCT (`192` tokens/frame) + VRB post-contact trajectories |
| **Multi-Embodiment IK** | Not included in dataset | 4-Embodiment IK (`franka_panda` 7-DoF, `unitree_g1_upper` 14-DoF, `unitree_h1_upper` 10-DoF, `allegro_hand_right` 16-DoF) + PyBullet FK (`G08`) |
| **Physics Verification** | Not included in dataset | Unwelded 30 Hz MuJoCo dynamic actuator replay & self-collision audit (`G09`) + 5-Layer RoboVerify certification |
| **Privacy & Compliance** | Manual review | Surgical feathered elliptical head/face redaction (`G10`, `0` unblurred faces) + `-an` audio strip + DPDP consent gate |
