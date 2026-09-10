#!/usr/bin/env python3
"""
Run the egocentric annotation pipeline on whatsapp_video.mp4 (Franka Panda config)
and execute strict delivery manifest verification (verify_demo_package.py).
"""
import sys
import os
import json
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import EgoAnnotatePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_whatsapp_video")

def main():
    video_path = PROJECT_ROOT / "data/raw_videos/whatsapp_video.mp4"
    config_path = PROJECT_ROOT / "configs/default.yaml"
    
    if not video_path.exists():
        logger.error("Video file does not exist: %s", video_path)
        sys.exit(1)
        
    logger.info("Initializing EgoAnnotatePipeline with config: %s", config_path)
    pipeline = EgoAnnotatePipeline(str(config_path))
    
    logger.info("Starting processing for video: %s", video_path)
    episode = pipeline.process_video(str(video_path))
    output_dir = pipeline.dataset_exporter.output_path / episode.episode_id
    logger.info("Pipeline run finished. Output directory: %s", output_dir)
    
    # Run verify_demo_package.py
    verify_script = PROJECT_ROOT / "verify_demo_package.py"
    logger.info("Running strict verification script: %s on %s", verify_script, output_dir)
    
    import subprocess
    cmd = [sys.executable, str(verify_script), str(output_dir)]
    res = subprocess.run(cmd, text=True, capture_output=False)
    
    if res.returncode != 0:
        logger.error("Strict delivery verification FAILED with exit code %d", res.returncode)
        sys.exit(res.returncode)
    else:
        logger.info("ALL CHECKS PASSED SUCCESSFULLY FOR whatsapp_video.mp4!")

if __name__ == "__main__":
    main()
