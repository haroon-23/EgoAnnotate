# EgoAnnotate — Egocentric Video Annotation for Humanoid Robot Learning

## Overview
EgoAnnotate is an end-to-end video annotation pipeline that converts raw egocentric demonstration videos into structured datasets ready for training Vision-Language-Action (VLA) models and humanoid robot policies. Designed for robotics researchers and machine learning engineers, it automatically tracks hands in 3D, detects interacted objects, determines physical contacts, classifies grasp types, segments manipulation phases, generates semantic language annotations, and exports data into standardized formats. By bridging the gap between raw human demonstrations and robot learning frameworks, EgoAnnotate significantly accelerates data curation and policy training.

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

### 2. LeRobot v2.1 Format
Ideal for training policies using the Hugging Face LeRobot suite, OpenPI, ACT, Diffusion Policy, or Pi0.

```bash
# Export a specific episode to LeRobot Parquet + Video
python scripts/run.py --export-lerobot my_video

# Scan and export all annotated episodes
python scripts/run.py --export-lerobot
```

Generates a `lerobot_v2/` directory housing:
- `meta/info.json` – Metadata description and features mapping
- `data/chunk-000/episode_000000.parquet` – Structured state, action, and timestamp tables
- `videos/chunk-000/observation.image/episode_000000.mp4` – Encoded video stream with relative references in the Parquet tables

---

## Compatibility Guide

| Downstream Target | Recommended Export Format | Contents |
|-------------------|---------------------------|----------|
| **LeRobot / OpenPI / ACT / Pi0** | LeRobot v2.1 (`--export-lerobot`) | Compact Parquet tables with relative paths to MP4 shards. |
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
