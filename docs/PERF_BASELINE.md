# EgoAnnotate Measured Performance & Unit-Economics Baseline

Auto-generated from `benchmark_report.json` (`scripts/benchmark.py`) on `data/raw_videos/test_10s.mp4` (`300` frames @ `30.0 Hz`):

## 1. Stage Wall-Clock Latency & Memory
- **Stage 1 (Video Decode + Privacy Face Redaction):** `17.729 s`
- **Stage 2 (24-D Camera-Frame State/Action Computation):** `0.302 s`
- **Stage 3 (Kinematic Retargeting & Independent PyBullet FK Audit):** `3.736 s`
- **Stage 4 (LeRobot v3 Parquet + Gzipped HDF5 Export):** `1.592 s`
- **Total CPU Wall-Clock Time:** `23.359 s` (`2.336 CPU-s / video-s`)
- **Peak RSS Memory:** `292.64 MB`

## 2. Cloud Unit Economics per Verified Output Hour
- **CPU Worker (`$0.16/hr`):** `$0.3738 / verified video hour`
- **GPU Worker (`$1.20/hr`):** `$0.6228 / verified video hour`
