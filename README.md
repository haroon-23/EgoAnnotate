# EgoAnnotate — Egocentric Video Annotation for Humanoid Robot Learning

## Overview
EgoAnnotate is an end-to-end video annotation pipeline that converts raw egocentric demonstration videos into structured datasets ready for training Vision-Language-Action (VLA) models and humanoid robot policies. Designed for robotics researchers and machine learning engineers, it automatically tracks hands in 3D, detects interacted objects, determines physical contacts, classifies grasp types, segments manipulation phases, generates semantic language annotations, and exports data into standardized formats. By bridging the gap between raw human demonstrations and robot learning frameworks, EgoAnnotate significantly accelerates data curation and policy training.

> **Status Note (Tier: STRESS_APPENDIX):** 74% reachable, 0% physics-verified success due to dynamic infeasibility (max lift 0.017m); hero protocol clip required for task-completion proof.

## Features
EgoAnnotate orchestrates a comprehensive 8-stage pipeline to process egocentric video inputs:
1. **Video Processing & Frame Sampling:** Extracts and samples video frames at a target framerate (e.g., 30 FPS) and normalizes resolution.
2. **3D Hand Tracking:** Uses MediaPipe to track 21 hand landmarks in 3D space with temporal smoothing.
3. **Object Detection:** Detects objects within active manipulation areas using Gemini Vision models.
4. **Contact Detection:** Computes spatial proximity between tracked fingertips and object bounding boxes.
5. **Grasp Classification:** Classifies grasp types (precision pinch, power wrap, hook, open, or unknown) using hand skeletal geometry.
6. **Action Primitive Computation:** Calculates relative wrist displacement (translations, rotation deltas) and finger joint angles to generate robot-agnostic action spaces.
7. **Temporal Action Segmentation:** Segments videos into temporal manipulation primitives (e.g., approach, contact, grasp, manipulate, release, retreat, idle).
8. **Language Description Generation:** Generates concise, natural-language instructions describing the overall task and individual action segments using multimodal Gemini VLMs.

## Installation

To set up the environment and install all dependencies:

```bash
git clone https://github.com/your-username/egoannotate.git
cd egoannotate

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Environment Configuration

Export your Gemini API Key to enable multimodal VLM-based object detection, temporal segmentation, and language description generation:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

## Quick Start

### Process a Single Video
Run the full annotation and export pipeline on a single egocentric demonstration:
```bash
python scripts/run.py --config configs/default.yaml --input data/raw/demonstration.mp4
```

### Batch Processing
Run the pipeline in batch mode over an entire directory of demonstration videos:
```bash
python scripts/run.py --config configs/default.yaml --input data/raw/ --batch
```

---

## Export Formats

EgoAnnotate supports exporting datasets into two primary robotics data standards:

### 1. RLDS HDF5 Format
Ideal for downstream bimanual and humanoid VLA policies (e.g., OpenVLA, EgoVLA) that fit MANO models or perform action-chunking.

```bash
# Export a specific episode to HDF5
python scripts/run.py --export-rlds my_video

# Scan and export all annotated episodes
python scripts/run.py --export-rlds
```

Generates an `episode_rlds.hdf5` file containing:
- High-resolution camera frames (`observation/image`)
- 3D wrist translation and rotation (`observation/wrist_translation`, `observation/wrist_rotation`)
- 15D joint-angle pose vector (`observation/hand_pose`)
- 8D proprioception vector (`observation/proprioception`)
- Standardized actions (`action`) and flags (`is_first`, `is_last`, `is_terminal`)

### 2. LeRobot v3.0 Format
Ideal for training policies using the Hugging Face LeRobot suite, OpenPI, ACT, Diffusion Policy, or Pi0.

```bash
# Export a specific episode to LeRobot Parquet + Video
python scripts/run.py --export-lerobot my_video

# Scan and export all annotated episodes
python scripts/run.py --export-lerobot
```

Two export modes are selected via `export.mode` in `configs/default.yaml`:

| Mode | `observation.state` / `action` | Use for |
|------|-------------------------------|---------|
| `human` (default) | 24D human motion: wrist translation (3) + wrist rotation (6) + gripper (1) + hand joints (14); action = downstream pose deltas | VL training on human demonstrations |
| `robot` | 8D robot-native **absolute** joint positions `[j1..j7 (rad), gripper_m (m)]` for `export.target_embodiment`; `action[t] = state[t+1]` | ACT / diffusion-policy / VLA training. Episodes whose HDF5 lacks `observation/robot_joint_angles` (no retargeting ran) are **excluded, never zero-padded**. |

Generates a `lerobot_v3/` directory housing:
- `meta/info.json` – v3.0 metadata, feature spec, path *templates*, `splits`, fps
- `meta/stats.json` – per-feature stats (min/max/mean/std/count/q01/q99)
- `meta/episodes/chunk-000/file-000.parquet` – per-episode rows with stats + video lookups
- `data/chunk-000/file-000.parquet` – state/action/timestamp/index tables (`timestamp = frame_index / fps`)
- `videos/observation.images.ego/chunk-000/file-000.mp4` – transcoded H.264 capture video (source read from `metadata.json → video_path`; the annotated overlay video is never used as training video)

Validate any export with:
```bash
python scripts/validate_lerobot_export.py <episode_dir>/lerobot_v3
```

### 3. Merged training dataset (Phase B)
Merge per-episode `lerobot_v3/` exports into one training dataset — concatenated
frame tables (global `index` 0..M-1, per-episode `frame_index`), deduped tasks,
repacked data chunks, videos copied 1:1, and `meta/stats.json` **recomputed
pooled** over all frames. The merge refuses loudly on mixed export modes,
differing feature schemas, or mixed `robot_type`; mixed fps warns and keeps
per-episode fps.

```bash
# Standalone:
python scripts/merge_lerobot.py <ep1/lerobot_v3> [<ep2/lerobot_v3> ...] \
    -o data/output/merged_lerobot_v3
# ...or scan an output dir:  python scripts/merge_lerobot.py --scan data/output -o data/output/merged_lerobot_v3

# As a post-export hook (default off):
python scripts/run.py --export-lerobot --merge-lerobot
```

### 4. ACT CPU smoke-train (Phase B)
A vendored minimal ACT (CVAE latent-16 + 2-layer transformer + tiny CNN,
`~0.8M` params, no `lerobot` package dependency) trains on the merged dataset
on CPU. This is a **wiring proof** — real bytes → normalized tensors →
decreasing imitation loss → saved checkpoint — not a good policy.

```bash
python scripts/train_act_smoke.py --data data/output/merged_lerobot_v3 --steps 50
# writes runs/smoke/{loss_curve.json, config.json, policy.pt}
```

Success criteria (exit 1 otherwise): mean loss of the last 10 steps
< 0.9 × mean of the first 10, no NaN/Inf anywhere, and a no-grad rollout
returning `(B, k, D)` action chunks. Smoke-train defaults live in the
`training:` section of `configs/default.yaml`; merge settings in `merge:`.

---

---

## Phase C: Perception (open-source detectors, metric depth, PnP)

Phase C adds a unified, CPU-only perception layer in `src/perception/` that
gracefully degrades when heavy dependencies are missing
(`import src.perception` never raises):

| Component | Default | Opt-in |
|---|---|---|
| 2D detector (`detector_backend`) | **OWL-ViT** (`google/owlvit-base-patch32`, installed) | Real Grounding DINO (IDEA-Research, SwinT_OGC) |
| SAM 2 box-prompted masks | off (`sam2.enabled: false`) | `models/sam2.1_hiera_tiny.pt` |
| Metric depth | **UniDepth ONNX** (`models/unidepth_v1.onnx`), keyframes only | `depth.backend: "none"` (explicit no-depth) |
| Hand PnP refinement | on (pure OpenCV/numpy, no extra deps) | needs `camera:` intrinsics, else skips with a warning |

Detections are localized to 3D (`ObjectAnnotation.position_3d`, camera frame,
meters) and can carry COCO-style RLE masks (`mask_rle`); MediaPipe hand poses
are refined to metric wrist translation via EPnP + RANSAC using the 0.090 m
anthropometric wrist→MCP reference (`src/retargeting/metric_calibration.py`).
Contact/grasp semantics stay 2D — depth-informed contact is future work.

### Why IDEA-Research Grounding DINO, not MM-Grounding-DINO

The OpenMMLab variant (MM-GDINO) requires `mmcv`/`mmdet`, which ship **no usable
macOS-Intel wheels** — it is uninstallable on the 2017 Intel Mac dev machine.
The IDEA-Research release works CPU-only without mmcv. This is a deliberate,
permanent decision: do not "upgrade" to the MM variant.

### Manual weight downloads (no auto-download anywhere — by design)

```bash
mkdir -p models
# Grounding DINO SwinT_OGC (~170 MB) + config:
#   https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
#   -> models/groundingdino_swint_ogc.pth
#   GroundingDINO_SwinT_OGC.py (from the repo) -> models/GroundingDINO_SwinT_OGC.py
# UniDepth V1 ONNX:
#   https://huggingface.co/ibaiGorordo/ONNX-UniDepth-V1 -> models/unidepth_v1.onnx
# SAM 2.1 hiera-tiny:
#   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# Optional installs (base requirements.txt works without all of these):
pip install onnxruntime          # UniDepth backend (macOS x86_64 wheels exist)
pip install sam2                 # slow on old Intel CPUs; keyframes only
# Grounding DINO (no PyPI package; torch must be pre-installed):
pip install --no-build-isolation -e git+https://github.com/IDEA-Research/GroundingDINO.git
```

Set `perception.detector_backend: "grounding_dino"`, `sam2.enabled: true`, or
`depth.backend: "unidepth"` in `configs/default.yaml` to enable each backend.
Missing weights/deps produce a loud warning naming the exact URL — never a
silent wrong result.

## Phase D: VLM dual backend (Gemini + local SmolVLM)

Phase D adds a `VLMBackend` abstraction in `src/vlm_backend.py` (mirroring the
Phase-C `Detector2D` pattern): `VLMRequest`/`VLMResponse` dataclasses, a
`VLMBackend` protocol (`name`, `is_available()`, `generate()`), and
`create_vlm_backend(cfg)` which returns a working backend or `None` — never
raises.

| Backend | Default for | Notes |
|---|---|---|
| **`gemini`** (`gemini-3.8-flash`) | everything (production) | Native video understanding via Files-API upload for the video stages |
| **`local`** (SmolVLM-500M-Instruct, transformers, CPU-only) | opt-in via `vlm.backend` or per-stage `vlm_backend` | Offline/dev; ~2 GB RAM; zero new deps (transformers + torch already required) |

All four VLM stages (`GeminiObjectDetector`, `GeminiActionSegmenter`,
`SegmentLabeler`, `GeminiLanguageGenerator` — names kept as thin aliases)
accept an injected backend; the pipeline builds it from the `vlm:` section of
`configs/default.yaml`. Per-stage `vlm_backend: "local"` overrides the global
default — allowed for `object_detection` and `segment_labeling`; the video
stages (`action_segmenter`, `language_generator`) keep the `gemini` default
because they rely on native video understanding. Under `local`, a video is
replaced by `vlm.video_frames` (default 8) evenly-sampled stills with
timestamp-prefixed prompts (`Frame 3 of 8, t≈2.1s`).

Because a 500M local VLM will not obey JSON schemas reliably, Phase D adds a
shared output-contract layer: strict JSON extraction → one repair re-prompt
("Reply with ONLY the JSON…") → each stage's existing regex fallbacks. Parsing
stays in the stages; the backend only returns raw text.

### Manual weight download (no auto-download anywhere — by design)

```bash
mkdir -p models
# SmolVLM-500M-Instruct (~1 GB):
#   https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct -> models/smolvlm-500m-instruct/
```

Then set `vlm.backend: "local"` (or per-stage `vlm_backend: "local"`) in
`configs/default.yaml`. Missing weights / missing `transformers` fail loud
with the URL above. A manual smoke check (Mac only, skipped when the weights
dir is absent): `python scripts/check_local_vlm.py --weights models/smolvlm-500m-instruct`.

Expect tens of seconds per call on the 2017 Intel CPU — `local` is the
offline/dev backend; `gemini` stays the production default. `vlm.timeout_s`
(default 300) fails loud instead of hanging.

## Compatibility Guide

| Downstream Target | Recommended Export Format | Contents |
|-------------------|---------------------------|----------|
| **LeRobot / OpenPI / ACT / Pi0** | LeRobot v3.0 (`--export-lerobot`, `export.mode` in `configs/default.yaml`) | Parquet tables + MP4 shards; `human` 24D mode or `robot` 8D absolute-joint mode. |
| **OpenVLA / EgoVLA** | RLDS HDF5 (`--export-rlds`) | Flat HDF5 structure containing raw matrix arrays. |
| **TensorFlow Datasets** | RLDS HDF5 (`--export-rlds`) | Intermediate format easily converted into TFRecords. |

---

## Project Structure

```
egoannotate/
├── configs/
│   └── default.yaml         # Configuration file for pipeline stages
├── src/
│   ├── __init__.py
│   ├── datatypes.py         # Standardized annotation data structures
│   ├── video_processor.py   # Frame extraction and sampling utilities
│   ├── hand_tracker.py      # MediaPipe 3D hand tracking module
│   ├── object_detector.py   # Gemini VLM object detection stage
│   ├── contact_detector.py  # Fingertip proximity contact logic
│   ├── grasp_classifier.py  # Geometric grasp classification
│   ├── action_computer.py   # Robot-agnostic action mapping
│   ├── action_segmenter.py  # Gemini temporal segmentation stage
│   ├── language_generator.py# Gemini semantic language model
│   ├── dataset_exporter.py  # JSON, Parquet, and HDF5 serializers
│   ├── visualizer.py        # HUD telemetry overlay visualizer
│   └── pipeline.py          # Pipeline stage orchestrator
├── scripts/
│   ├── run.py               # CLI entry point to run pipeline and exports
│   ├── inspect_datasets.py  # Debug tool to print HDF5/Parquet contents
│   └── install_lerobot.sh   # Environment installer bash script
├── tests/
│   └── test_*.py            # Suite of automated unit tests
├── requirements.txt         # Core dependencies
└── README.md                # Documentation
```

## License
MIT
