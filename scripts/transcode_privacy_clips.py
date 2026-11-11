import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.privacy import blur_video_mp4


CLIPS = ["client_video", "my_video"]

for ep in CLIPS:
    src_p = Path(f"data/raw_videos/{ep}.mp4")
    dst_p = Path(f"data/output/{ep}/lerobot_v3/videos/observation.images.ego/chunk-000/file-000.mp4")
    print(f"\n[Transcoding] Applying multi-scale privacy redaction to {ep}...")
    stats = blur_video_mp4(src_p, dst_p, fps=30.0)
    print(f"  Processed {stats['total_frames_processed']} frames")
    print(f"  Face instances detected: {stats['face_instances_detected']}")
    print(f"  Frames with active head blur: {stats['frames_with_active_head_blur']}")
    print(f"  Mean blurred area fraction: {stats['mean_blurred_area_fraction']:.4f}")
    print(f"  Exported redacted video to: {dst_p}")
