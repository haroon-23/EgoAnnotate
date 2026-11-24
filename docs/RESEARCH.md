# State-of-the-Art Research Summary (November 2026)

## 1. VLA Models & Imitation-Learning Policy Requirements
Recent open-weight embodied foundation models and policy stacks converge on tightly synchronized multimodal streams (RGB video, proprioception/state, continuous/chunked actions, and hierarchical language):

1. **LeRobot Ecosystem (`lerobot >= 0.4.x`, Hugging Face, Apache-2.0):**
   - **LeRobotDataset v3.0** (released Sept 2025, standard in 2026) shifts from per-episode files to file-based chunked Parquet (`data/chunk-*/file-*.parquet`), packet-synced concatenated MP4s (`videos/<cam>/chunk-*/file-*.mp4`), and structured metadata (`meta/info.json`, `meta/stats.json`, `meta/tasks.jsonl`, `meta/episodes/`).
   - Consumed natively by **SmolVLA** (450M params, trains on a single 24 GB GPU, supports up to 3 cameras + state), **openpi** (`π0` / `π0.5` flow-matching, 50-step action chunks at up to 50 Hz), **MolmoAct 2**, **ACT**, and **Diffusion Policy**.
   - **Language extension (LeRobot 2026 conventions):** Separates persistent lanes (`plan`, `subtask`, `memory`, `motion`, `task_aug` paraphrases) from instant `events` (`vqa`, contact/grasp events).
   - **Tactile gap opportunity (Sept 2026 Hugging Face corpus audit):** Out of 261 readable top LeRobot datasets, only 7 (0.76% of frames) declare any force/tactile channel. Shipping an explicitly typed tactile channel (clearly distinguishing `derived_rgb_proxy` from `hardware_sensor`) directly addresses a major market gap.
2. **NVIDIA Isaac GR00T N1 / N1.7:**
   - Dual-system architecture (VLM System 2 + Diffusion Transformer System 1 at high frequency), pretrained across real robots, synthetic data, and 20K+ hours of human egocentric video (EgoScale).
   - Consumes **LeRobot v2/v3 + `modality.json`** mapping state/action slices to named body parts (left/right arm, dexterous fingers, head/camera).
3. **OpenVLA / OpenVLA-OFT / Octo:**
   - Pretrained on Open X-Embodiment (**RLDS** format: episode steps with `observation`, `action`, `is_first`, `is_last`, `is_terminal`, `language_instruction`). OpenVLA-OFT uses 25-step continuous action chunks.
4. **Canonical HDF5 + JSON Sidecar:**
   - Remains standard for `robomimic`, `ALOHA`/`ACT`, and custom enterprise ingestion pipelines requiring random-access compressed arrays (`gzip`) plus complete frame-accurate metadata.

## 2. Learning From Human Egocentric Video (2025–2026 Best Practices)
- **World-Frame Consistency:** Head-mounted cameras move constantly. Policies trained in raw camera coordinates suffer from head-motion coupling; registering hands and manipulated objects into a **gravity-aligned metric world frame** (`T_world_cam`, `T_world_hand`) decouples head motion from manipulation intent while allowing camera-relative projection when desired.
- **Embodiment-Agnostic Intermediate Representation (IR):** Storing 3D wrist 6-DoF poses, articulated finger joints, object 3D trajectories, grasp taxonomy/contact events, and tactile proxies in an embodiment-agnostic world frame enables retargeting one human demonstration to humanoids (G1/H1), dual/single arms (Franka/UR5/SO-100), dexterous hands (Allegro/Ability), and teleop gloves without re-running perception.
- **Metric Scale Honesty:** Monocular RGB lacks direct stereo/LiDAR scale. Principled monocular recovery fuses commercially-licensed metric depth priors with anthropometric priors (adult palm width ~85 mm ± 8 mm) and scene constraints, reporting explicit `scale_uncertainty_pct` and `scale_source`.

## 3. Commercial Open-Source Model Landscape (Verified Nov 2026)

| Sub-Task | Candidates Evaluated | License Audit (Code / Weights) | Commercial Decision |
| :--- | :--- | :--- | :--- |
| **Hand & Upper-Body Pose** | MediaPipe Hand/Pose Landmarker; HaMeR; WiLoR; EgoHand (SOMA-Hand, Oct 2026) | MediaPipe: **Apache-2.0 / Apache-2.0**; HaMeR & WiLoR: MIT code / **MANO non-commercial** weights; EgoHand: newly published Oct 2026 (no general commercial pip wheel yet). | **Winner: MediaPipe Hand + Pose Landmarker** (Apache-2.0 default, CPU+GPU). HaMeR/WiLoR isolated as optional OFF-by-default non-commercial plugin. |
| **Open-Vocab Detection** | Grounding DINO; Florence-2; OWLv2; YOLO-World | Grounding DINO: **Apache-2.0 / Apache-2.0**; Florence-2: **MIT / MIT**; OWLv2: **Apache-2.0**; YOLO-World: **GPL-3.0** (Ultralytics). | **Winner: Grounding DINO** (GPU default) + **Florence-2-base** (MIT, lightweight/CPU option). YOLO-World excluded due to GPL-3.0. |
| **Video/Object Segmentation** | Meta SAM 2.1; Meta SAM 3; OpenCV GrabCut/Optical-Flow ROI | SAM 2.1: **Apache-2.0 / Apache-2.0**; SAM 3: Custom Meta SAM License (AUP restrictions); OpenCV: **Apache-2.0**. | **Winner: SAM 2.1** (GPU default, Apache-2.0) + **OpenCV ROI/Flow Mask Tracker** (CPU smoke fallback). |
| **Monocular Depth** | Depth-Anything-V2-Small; Video-Depth-Anything-Small; Depth-Anything-V2-Base/Large; UniDepth V2 | DAv2-Small & VDA-Small: **Apache-2.0 / Apache-2.0**; DAv2-Base/Large/Giant: **CC-BY-NC-4.0**; UniDepth V2: **CC-BY-NC-4.0**. | **Winner: Depth-Anything-V2-Small / Video-Depth-Anything-Small** (Apache-2.0). UniDepth and DAv2-B/L/G strictly excluded from defaults. |
| **6-DoF Camera Pose / SLAM** | OpenCV Essential/PnP VO; DPVO; DROID-SLAM; ViPE (Apache core); MASt3R/DUSt3R; VGGT-1B | OpenCV: **Apache-2.0**; DPVO: **MIT**; DROID-SLAM: **BSD-3-Clause**; MASt3R/DUSt3R: **CC-BY-NC-SA**; VGGT-1B: **Non-commercial weights**. | **Winner: OpenCV Keyframe 6-DoF VO** (CPU default) + **DPVO / DROID-SLAM** (GPU production). MASt3R/VGGT/Unik3D strictly excluded. |
| **Local VLM Annotation** | Qwen2.5-VL-7B / Qwen3-VL; SmolVLM2-2.2B-Instruct; SmolVLM2-500M; Gemini API | Qwen2.5-VL / Qwen3-VL: **Apache-2.0 / Apache-2.0**; SmolVLM2: **Apache-2.0 / Apache-2.0**; Gemini API: Paid proprietary. | **Winner: Qwen2.5-VL / SmolVLM2-2.2B** (GPU default) + **Geometrically-Grounded Structured Annotator / SmolVLM2-256M** (CPU default). Paid Gemini removed. |
| **Retargeting & Physics** | MuJoCo + Mink + SciPy DLS IK; PyBullet | MuJoCo: **Apache-2.0**; Mink: **Apache-2.0**; SciPy: **BSD-3-Clause**; PyBullet: **zlib**. | **Winner: MuJoCo + Mink + SciPy DLS IK** driven strictly by pluggable URDF/MJCF YAML configs. |
