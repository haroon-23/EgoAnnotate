import os
import h5py
import json
import pandas as pd

# Define paths
h5_path = "data/output/my_video/episode_rlds.hdf5"
lerobot_meta_path = "data/output/my_video/lerobot_v2/meta/info.json"
lerobot_parquet_path = "data/output/my_video/lerobot_v2/data/chunk-000/episode_000000.parquet"

print("=" * 60)
print("1. VERIFYING RLDS HDF5 DATASET")
print("=" * 60)
if os.path.exists(h5_path):
    with h5py.File(h5_path, "r") as f:
        print(f"HDF5 File: {h5_path}")
        print(f"Episodes: {list(f.keys())}")
        
        # Access first episode
        episode_key = list(f.keys())[0]
        episode_group = f[episode_key]
        print(f"\nEpisode '{episode_key}' structure:")
        
        # Recursive print helper
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name} (shape={obj.shape}, dtype={obj.dtype})")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
                for key, val in obj.attrs.items():
                    print(f"    Attr: {key} = {val}")

        episode_group.visititems(print_structure)
        
        # Print sample data from first frame
        print("\nFirst Frame Sample Data:")
        steps = episode_group['steps']
        print(f"  image shape: {steps['observation/image'].shape}")
        print(f"  wrist_translation: {steps['observation/wrist_translation'][0]}")
        print(f"  wrist_rotation (rot6d): {steps['observation/wrist_rotation'][0]}")
        print(f"  hand_pose (finger angles): {steps['observation/hand_pose'][0]}")
        print(f"  proprioception: {steps['observation/proprioception'][0]}")
        print(f"  action: {steps['action'][0]}")
else:
    print(f"Error: HDF5 file not found at {h5_path}")

print("\n" + "=" * 60)
print("2. VERIFYING LEROBOT V2.1 DATASET")
print("=" * 60)

# Check info.json
if os.path.exists(lerobot_meta_path):
    print(f"LeRobot Meta File: {lerobot_meta_path}")
    with open(lerobot_meta_path, "r") as f:
        info_data = json.load(f)
    print(f"  Codebase Version: {info_data.get('codebase_version')}")
    print(f"  Robot Type: {info_data.get('robot_type')}")
    print(f"  FPS: {info_data.get('fps')}")
    print(f"  Total Episodes: {info_data.get('total_episodes')}")
    print(f"  Total Frames: {info_data.get('total_frames')}")
    print("\n  Feature Modal Shape & Types:")
    for feat_name, feat_info in info_data.get("features", {}).items():
        print(f"    - {feat_name}: dtype={feat_info.get('dtype')}, shape={feat_info.get('shape')}")
else:
    print(f"Error: LeRobot info.json not found at {lerobot_meta_path}")

# Check parquet file
if os.path.exists(lerobot_parquet_path):
    print(f"\nLeRobot Parquet File: {lerobot_parquet_path}")
    df = pd.read_parquet(lerobot_parquet_path)
    print(f"  Total Rows (Frames): {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print("\n  First Frame Data (Excluding Image):")
    for col in df.columns:
        if col not in ['observation.image', 'video_path']:
            val = df[col].iloc[0]
            # Format vectors/lists nicely
            if isinstance(val, (list, tuple)) or hasattr(val, '__iter__'):
                val_str = f"shape={len(val)}: {list(val)[:6]}..."
            else:
                val_str = str(val)
            print(f"    {col}: {val_str}")
else:
    print(f"Error: Parquet file not found at {lerobot_parquet_path}")

print("=" * 60)
