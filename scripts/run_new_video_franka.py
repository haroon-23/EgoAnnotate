#!/usr/bin/env python3
"""
Run the egocentric pipeline on new_video.mp4 with the Franka Panda config
and capture the guardrail validation output explicitly.

Usage: .venv/bin/python scripts/run_new_video_franka.py
"""
import logging
import sys
import os

# Configure logging to stdout so we capture everything
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.pipeline import EgoAnnotatePipeline

print("=" * 70)
print("PIPELINE RUN: new_video.mp4 | Config: configs/default.yaml (Franka)")
print("=" * 70)

pipeline = EgoAnnotatePipeline("configs/default.yaml")
episode = pipeline.process_video(
    "data/raw_videos/new_video.mp4",
    episode_id="new_video",
)

print("\n" + "=" * 70)
print("POST-RUN VERIFICATION")
print("=" * 70)

import json
from pathlib import Path
import subprocess

episode_dir = Path("data/output/new_video")

# 1. metadata.json
meta_path = episode_dir / "metadata.json"
with open(meta_path) as f:
    meta = json.load(f)
print(f"\n[CHECK] metadata.json → target_robot = '{meta['target_robot']}'")
assert meta["target_robot"] == "panda", f"FAIL: expected 'panda', got '{meta['target_robot']}'"
print("[CHECK] PASS: target_robot = panda")

# 2. side_by_side.mp4
sbs = episode_dir / "side_by_side.mp4"
result = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0",
     "-show_entries", "stream=width,height,nb_frames,codec_name,duration",
     "-show_entries", "format=duration", "-of", "json", str(sbs)],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
)
sbs_info = json.loads(result.stdout)
sbs_stream = sbs_info.get("streams", [{}])[0]
sbs_codec = sbs_stream.get("codec_name", "UNKNOWN")
print(f"\n[CHECK] side_by_side.mp4 ffprobe:")
print(f"        codec      = {sbs_codec}")
print(f"        resolution = {sbs_stream.get('width')}x{sbs_stream.get('height')}")
print(f"        nb_frames  = {sbs_stream.get('nb_frames')}")
print(f"        duration   = {sbs_stream.get('duration')}s")
print(f"        file_size  = {sbs.stat().st_size / (1024*1024):.1f} MB")
assert sbs_codec in ("h264", "avc1"), f"FAIL: side_by_side codec={sbs_codec}, expected h264"
print("[CHECK] PASS: side_by_side.mp4 codec = h264")

# 3. episode_rlds.hdf5
import h5py
rlds = episode_dir / "episode_rlds.hdf5"
assert rlds.exists(), f"FAIL: episode_rlds.hdf5 MISSING at {rlds}"
with h5py.File(rlds, "r") as hf:
    top_keys = list(hf.keys())
    ep_grp = hf[top_keys[0]]
    action_shape = ep_grp["steps/action"].shape
    lang = ep_grp["steps/observation/language_instruction"]
    has_robot_joints = "steps/observation/robot_joint_angles" in ep_grp
print(f"\n[CHECK] episode_rlds.hdf5:")
print(f"        top-level keys    = {top_keys}")
print(f"        steps/action shape= {action_shape}")
print(f"        language_inst     = present ({lang.shape[0]} entries)")
print(f"        robot_joint_angles= {'present' if has_robot_joints else 'ABSENT (no retargeting data in HDF5)'}")
print(f"        file_size         = {rlds.stat().st_size / (1024*1024):.1f} MB")
print("[CHECK] PASS: episode_rlds.hdf5 opens cleanly, required datasets present")

# 4. lerobot_v3/
import pandas as pd
lr_dir = episode_dir / "lerobot_v3"
assert lr_dir.is_dir(), f"FAIL: lerobot_v3/ directory MISSING at {lr_dir}"
info_json = lr_dir / "meta" / "info.json"
assert info_json.exists(), f"FAIL: lerobot_v3/meta/info.json MISSING"
with open(info_json) as f:
    lr_info = json.load(f)
parquet_files = sorted((lr_dir / "data").rglob("*.parquet"))
assert parquet_files, f"FAIL: no parquet files found in lerobot_v3/data/"
df = pd.read_parquet(parquet_files[0])
print(f"\n[CHECK] lerobot_v3/:")
print(f"        codebase_version  = {lr_info.get('codebase_version')}")
print(f"        total_frames      = {lr_info.get('total_frames')}")
print(f"        parquet rows      = {len(df)}")
print(f"        parquet columns   = {list(df.columns)}")
print(f"        info.json         = present")
print("[CHECK] PASS: lerobot_v3/ dataset loads cleanly via pandas")

print("\n" + "=" * 70)
print("ALL 4 DELIVERABLE CHECKS PASSED")
print("=" * 70)
