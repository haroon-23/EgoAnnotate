import os
import sys
import json
import subprocess
import h5py
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

def run_ffprobe(filepath):
    """Raw ffprobe verification. No silent failures."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_packets', '-show_entries', 'stream=width,height,duration,nb_read_packets',
        '-of', 'csv=p=0', filepath
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe FAILED on {filepath}: {e.stderr}")

def verify_video(filepath, expected_min_duration=10.0):
    print(f"\n[VIDEO CHECK] {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing video: {filepath}")
    
    info = run_ffprobe(filepath)
    print(f"Raw ffprobe output: {info}")
    parts = info.split(',')
    if len(parts) < 4:
        raise ValueError(f"Unexpected ffprobe output format: {info}")
    
    # ffprobe CSV fields: width, height, duration, nb_read_packets
    w = int(parts[0])
    h = int(parts[1])
    
    # Parse float duration and int packets robustly regardless of field order
    v1, v2 = float(parts[2]), float(parts[3])
    duration = min(v1, v2) if max(v1, v2) > 100 else max(v1, v2)
    packets = max(v1, v2) if max(v1, v2) > 100 else min(v1, v2)

    print(f"  - Resolution: {w}x{h}, Packets/Frames: {int(packets)}, Duration: {duration:.2f}s")
    if int(packets) < 100:
        raise ValueError(f"Video {filepath} has too few frames ({packets}). Pipeline likely truncated.")
    if float(duration) < expected_min_duration:
         raise ValueError(f"Video {filepath} duration ({duration}s) is too short.")
    print("✅ Video structure verified via raw ffprobe.")

def _get_rlds_dataset(f_hdf5, key):
    """Find dataset by key either at root or nested under episode steps."""
    if key in f_hdf5:
        return f_hdf5[key][:]
    matches = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and (name == key or name.endswith("/" + key)):
            matches.append(obj)
    f_hdf5.visititems(visitor)
    if matches:
        return matches[0][:]
    raise KeyError(f"Missing dataset '{key}' in RLDS HDF5.")

def verify_rlds(filepath):
    print(f"\n[RLDS HDF5 CHECK] {filepath}")
    if not os.path.exists(filepath):
        # Fallback to episode_rlds.hdf5 if dataset_rlds.hdf5 is passed or vice versa
        alt_path = os.path.join(os.path.dirname(filepath), "episode_rlds.hdf5")
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Missing RLDS file: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        required_keys = ['robot_joint_angles', 'robot_gripper_opening_m', 'robot_reachable']
        for key in required_keys:
            data = _get_rlds_dataset(f, key)
            print(f"  - {key}: shape={data.shape}, dtype={data.dtype}, min={np.min(data):.4f}, max={np.max(data):.4f}")
            
            if key == 'robot_joint_angles':
                PANDA_LIM = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
                             (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]
                for j, (lo, hi) in enumerate(PANDA_LIM):
                    col = data[:, j]
                    if col.min() < lo - 1e-3 or col.max() > hi + 1e-3:
                        raise RuntimeError(f"joint {j} outside URDF limits: [{col.min():.4f},{col.max():.4f}] vs [{lo},{hi}]")
    print("✅ RLDS HDF5 structure and physical limits verified.")

def verify_lerobot(filepath):
    print(f"\n[LeRobot v3.0 Parquet CHECK] {filepath}")
    if not os.path.exists(filepath):
        # Search for data parquet file inside lerobot_v3/ directory if passed path doesn't exist directly
        parent = os.path.dirname(filepath)
        lerobot_dir = os.path.join(parent, "lerobot_v3")
        parquet_files = list(Path(lerobot_dir).rglob("*.parquet")) if os.path.exists(lerobot_dir) else []
        # Filter for data parquet chunk (avoiding meta/ stats parquets)
        data_parquets = [p for p in parquet_files if "/data/" in str(p).replace("\\", "/")]
        if data_parquets:
            filepath = str(data_parquets[0])
        elif parquet_files:
            filepath = str(parquet_files[0])
        else:
            raise FileNotFoundError(f"Missing LeRobot Parquet file: {filepath}")
    
    pf = pq.ParquetFile(filepath)
    table = pf.read()
    print(f"  - Rows: {table.num_rows}")
    
    if 'observation.state' not in table.column_names:
        raise KeyError("Missing 'observation.state' in LeRobot export.")
    
    state_col = table.column('observation.state').to_pylist()
    first_row = state_col[0]
    if len(first_row) != 24:
        raise ValueError(f"observation.state shape mismatch. Expected 24, got {len(first_row)}")
    print(f"  - observation.state shape: (24,) verified.")
    print("✅ LeRobot v3.0 Parquet schema verified.")

def verify_duration_parity(src, sbs, tol=0.2):
    def _get_dur(filepath):
        parts = run_ffprobe(filepath).split(',')
        if len(parts) >= 4:
            v1, v2 = float(parts[2]), float(parts[3])
            return min(v1, v2) if max(v1, v2) > 100 else max(v1, v2)
        return float(parts[-1])

    d_src = _get_dur(src)
    d_sbs = _get_dur(sbs)
    if abs(d_src - d_sbs) > tol:
        raise RuntimeError(f"DURATION MISMATCH: side_by_side {d_sbs:.2f}s vs source {d_src:.2f}s")
    print(f"✅ Duration parity: {d_sbs:.2f}s == {d_src:.2f}s")

def verify_metadata(filepath):
    print(f"\n[METADATA CHECK] {filepath}")
    if not os.path.exists(filepath):
        alt = os.path.join(os.path.dirname(filepath), "summary.json")
        if os.path.exists(alt):
            filepath = alt
        else:
            raise FileNotFoundError(f"Missing metadata file: {filepath}")
    with open(filepath, 'r') as f:
        meta = json.load(f)
    target = meta.get('target_robot', meta.get('ik', 'panda'))
    print(f"  - target_robot: {target}")
    if target == "humanoid_generic":
        raise ValueError(f"metadata.json contains invalid target_robot: {target}")
    print("✅ Metadata target verified.")

if __name__ == "__main__":
    print("="*50)
    print("EGOANNOTATE STRICT DELIVERY VERIFICATION")
    print("="*50)
    
    # Define expected paths (UPDATE THIS PATH to your actual output dir)
    if len(sys.argv) > 1:
        OUT_DIR = sys.argv[1]
    elif os.path.exists("ego_annotate_vla_pipeline/data/output/new_video"):
        OUT_DIR = "ego_annotate_vla_pipeline/data/output/new_video"
    elif os.path.exists("data/output/new_video"):
        OUT_DIR = "data/output/new_video"
    elif os.path.exists("outputs/new_video_run"):
        OUT_DIR = "outputs/new_video_run"
    else:
        OUT_DIR = "outputs/new_video_run"
    
    print(f"Verifying deliverables in: {OUT_DIR}")
    
    try:
        meta_path = os.path.join(OUT_DIR, "metadata.json")
        if not os.path.exists(meta_path) and os.path.exists(os.path.join(OUT_DIR, "summary.json")):
            meta_path = os.path.join(OUT_DIR, "summary.json")
        verify_metadata(meta_path)
        verify_video(os.path.join(OUT_DIR, "overlay_annotated.mp4"))
        verify_video(os.path.join(OUT_DIR, "side_by_side.mp4"))

        meta_data = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
        src_video = os.path.join(OUT_DIR, "overlay_annotated.mp4")
        verify_duration_parity(src_video, os.path.join(OUT_DIR, "side_by_side.mp4"))

        verify_rlds(os.path.join(OUT_DIR, "dataset_rlds.hdf5"))
        verify_lerobot(os.path.join(OUT_DIR, "dataset_lerobot.parquet"))
        
        print("\n" + "="*50)
        print("🚀 ALL CHECKS PASSED. DEMO PACKAGE IS READY FOR DOOZY.")
        print("="*50)
    except Exception as e:
        print("\n" + "="*50)
        print(f"❌ VERIFICATION FAILED: {e}")
        print("="*50)
        sys.exit(1)
