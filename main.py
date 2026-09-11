#!/usr/bin/env python3
"""CLI entry point for main.py"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add project root directory to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline import EgoAnnotatePipeline


def main():
    parser = argparse.ArgumentParser(description="Run EgoAnnotate VLA Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument("--config", default="configs/retargeting_franka.yaml", help="Path to config YAML")

    args = parser.parse_args()

    out_path = Path(args.output)
    episode_id = out_path.name
    output_parent = out_path.parent

    # Always initialize EgoAnnotatePipeline with configs/default.yaml so 30 FPS pipeline defaults load
    pipeline = EgoAnnotatePipeline(config_path="configs/default.yaml")
    pipeline.video_processor.target_fps = 30.0

    # Override retargeter with args.config (e.g. configs/retargeting_franka.yaml)
    ret_path = args.config if os.path.exists(args.config) else "configs/retargeting_franka.yaml"
    from src.retargeting.retargeter import Retargeter, RetargetingConfig
    pipeline.retargeter = Retargeter(RetargetingConfig.from_yaml(ret_path))
    pipeline.enable_retargeting = True

    pipeline.dataset_exporter.output_path = Path(output_parent)
    pipeline.dataset_exporter.config.output_dir = str(output_parent)

    # Clean existing cached frames directory for this video to ensure full 30 FPS frame extraction
    frames_dir = Path("data/frames") / episode_id
    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    pipeline.process_video(args.video, episode_id=episode_id)


if __name__ == "__main__":
    main()
