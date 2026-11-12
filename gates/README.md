# EgoAnnotate Read-Only Verification Gates (`gates/`)

**HARD RULE (R1):** Everything in `gates/` is strictly **READ-ONLY** for coding agents. Never edit, delete, relax, or bypass any file in `gates/`. When a gate fails, fix the pipeline in `src/` or `scripts/`, never the gate.

## 1. Human Chirality Ground-Truth Labelling (`tools/label_frames.py`)

Before running Gate 05 (`g05_chirality.py`), label at least 60 frames across your raw clips:

```bash
.venv/bin/python tools/label_frames.py \
  --video data/raw_videos/whatsapp_video.mp4 \
  --annotations data/output/whatsapp_video/annotations.json \
  --out data/chirality_labels.csv \
  --num-samples 60
```

## 2. Running Individual Gates
```bash
.venv/bin/python gates/g01_no_hardcoded_verdicts.py
.venv/bin/python gates/g02_negative_controls.py
.venv/bin/python gates/g03_action_signal.py data/output/whatsapp_video
.venv/bin/python gates/g04_layout_consistency.py data/output/whatsapp_video
.venv/bin/python gates/g05_chirality.py data/output/whatsapp_video --labels-csv data/chirality_labels.csv
.venv/bin/python gates/g06_lerobot_v3.py data/output/whatsapp_video
.venv/bin/python gates/g07_hdf5_honesty.py data/output/whatsapp_video
.venv/bin/python gates/g08_retarget_independent.py data/output/whatsapp_video
.venv/bin/python gates/g09_physics_replay.py data/output/whatsapp_video
.venv/bin/python gates/g10_privacy.py data/output/whatsapp_video
.venv/bin/python gates/g11_captions.py data/output/whatsapp_video
.venv/bin/python gates/g12_tests_hygiene.py
```

## 3. Running the Full Gate Suite (gates/run_all.py)
```bash
.venv/bin/python gates/run_all.py data/output/whatsapp_video \
  --labels-csv data/chirality_labels.csv \
  --report-out gates_report.json
```
