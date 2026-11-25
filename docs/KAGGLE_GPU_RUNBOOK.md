# Kaggle GPU Certification Runbook (`gpu_prod.yaml`)

Use this runbook on a Kaggle GPU notebook (T4 x2 / P100 / L4) to run the full `gpu_prod` CUDA stack (`Grounding DINO` + `Depth-Anything-V2-Small` + `SmolVLM2-2.2B` / `Qwen2.5-VL-7B`) across your complete 180-second (`5400`-frame) egocentric video.

## 1. Environment Setup on Kaggle

    !git clone -b feat/commercial-v3 https://github.com/haroon-23/EgoAnnotate.git /kaggle/working/EgoAnnotate
    %cd /kaggle/working/EgoAnnotate
    !pip install -q -e ".[gpu,dev]" gdown

## 2. Download Commercial-Only GPU Weights & Verify Checksums

    !python scripts/download_models.py --profile gpu_prod --cache-dir /kaggle/working/ego_models

## 3. Fetch Full Real Sample Video & Run Full-Length `gpu_prod` Pipeline

    !mkdir -p data/sample_raw
    !gdown --id 1n5nOCr-APW2Ho7WY8ylHGk6E77YqGDbw -O data/sample_raw/ad521ca5-0017-475b-9dd8-1fdb423a66b7.mp4
    !egoannotate run data/sample_raw/ad521ca5-0017-475b-9dd8-1fdb423a66b7.mp4 \
        --out /kaggle/working/v3_gpu_run \
        --profile gpu_prod \
        --episode-id kaggle_gpu_full_180s

## 4. Verify Policy Training & Inspect QA Artifacts

    !egoannotate train-smoke /kaggle/working/v3_gpu_run/lerobot_v3 --epochs 12
    !egoannotate audit-licenses

## 5. Expected Outputs in `/kaggle/working/v3_gpu_run/`
- `lerobot_v3/` (5400-frame `float32` Parquet + privacy-sanitized MP4 + `meta/info.json`, `stats.json`, `tasks.jsonl`, `modality.json`)
- `rlds/` (Open X-Embodiment compatible `.npz` + `dataset_info.json`)
- `episode_canonical.h5` (gzip HDF5 with world streams + all 6 retargeted embodiments)
- `qa_report.html`, `viewer_3d.html`, `qa_overlay.mp4`, `run_report.json`
