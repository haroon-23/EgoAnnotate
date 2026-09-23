#!/usr/bin/env python3
"""Regenerate overlay for test_10s and measure timing."""
import time, json, sys, os, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(str(Path(__file__).resolve().parent))

from src.datatypes import (AnnotatedEpisode, AnnotationFrame, HandLandmarks, ContactState,
    GraspType, ActionSegment)
from src.visualizer import render_annotated_video
from src.dataset_exporter import DatasetExporter


def rebuild_episode(ep_dir, video):
    with open(ep_dir / 'frame_annotations.json') as f:
        frames_data = json.load(f)
    with open(ep_dir / 'metadata.json') as f:
        meta = json.load(f)
    segs_data = []
    if (ep_dir / 'action_segments.json').exists():
        with open(ep_dir / 'action_segments.json') as f:
            segs_data = json.load(f)

    segments = [ActionSegment(name=s.get('name','unknown'), start_time=float(s.get('start_time',0)),
        end_time=float(s.get('end_time',0)), object_name=s.get('object_name',''),
        hand_used=s.get('hand_used','right'), description=s.get('description',''))
        for s in segs_data]

    frames = []
    for fd in frames_data:
        left_hand = right_hand = left_contact = right_contact = left_grasp = right_grasp = None
        if fd.get('left_hand_present', False):
            kp = np.array(fd['left_hand_keypoints'], dtype=np.float32).reshape(21,3)
            left_hand = HandLandmarks(x=kp[:,0], y=kp[:,1], z=kp[:,2], confidence=0.9, handedness='Left',
                is_interpolated=fd.get('left_hand_interpolated', False))
        if fd.get('right_hand_present', False):
            kp = np.array(fd['right_hand_keypoints'], dtype=np.float32).reshape(21,3)
            right_hand = HandLandmarks(x=kp[:,0], y=kp[:,1], z=kp[:,2], confidence=0.9, handedness='Right',
                is_interpolated=fd.get('right_hand_interpolated', False))
        if fd.get('left_contact') is not None:
            left_contact = ContactState(fingers=np.zeros(5,dtype=bool), object_name=fd.get('left_contact_object'), in_contact=bool(fd['left_contact']))
        if fd.get('right_contact') is not None:
            right_contact = ContactState(fingers=np.zeros(5,dtype=bool), object_name=fd.get('right_contact_object'), in_contact=bool(fd['right_contact']))
        if fd.get('left_grasp_type'):
            left_grasp = GraspType(type=fd['left_grasp_type'], confidence=0.8, thumb_index_distance=0.1, num_curled_fingers=0)
        if fd.get('right_grasp_type'):
            right_grasp = GraspType(type=fd['right_grasp_type'], confidence=0.8, thumb_index_distance=0.1, num_curled_fingers=0)
        frames.append(AnnotationFrame(frame_idx=fd['frame_idx'], timestamp=fd['timestamp'],
            image_path=fd.get('image_path',''), left_hand=left_hand, right_hand=right_hand,
            left_contact=left_contact, right_contact=right_contact, left_grasp=left_grasp,
            right_grasp=right_grasp, frame_description=fd.get('language_instruction'),
            action_segment=fd.get('action_segment'), robot_joint_angles=fd.get('robot_joint_angles'),
            robot_gripper_opening_m=fd.get('robot_gripper_opening_m'),
            robot_gripper_method=fd.get('robot_gripper_method'), robot_reachable=fd.get('robot_reachable')))

    return AnnotatedEpisode(episode_id=meta.get('episode_id', ep_dir.name), video_path=video,
        task_description=meta.get('task_description',''), frames=frames, segments=segments,
        num_frames=len(frames), duration_seconds=float(meta.get('duration_seconds',0)),
        target_robot=meta.get('target_robot','human_egocentric'))


def regen(name, video):
    ep_dir = Path('data/output') / name
    if not ep_dir.exists() or not (ep_dir / 'frame_annotations.json').exists():
        print(f"SKIP {name}: no annotations")
        return
    if not Path(video).exists():
        print(f"SKIP {name}: source video {video} not found")
        return

    episode = rebuild_episode(ep_dir, video)
    print(f"\n{'='*60}")
    print(f"Regenerating overlay for: {name}")
    print(f"Frames: {len(episode.frames)}, Duration: {episode.duration_seconds:.1f}s")
    print(f"{'='*60}")

    overlay = ep_dir / 'overlay_annotated.mp4'
    start = time.time()
    render_annotated_video(video, episode, overlay, atomic=True)
    elapsed = time.time() - start

    info = DatasetExporter._ffprobe_video_info(str(overlay))
    if info:
        print(f"Overlay: {info['width']}x{info['height']}, {info['frame_count']} frames, {info['codec']}")
    fsize = overlay.stat().st_size / (1024*1024)
    print(f"File size: {fsize:.1f} MB")
    print(f"[TIMING] {name}: {elapsed:.1f}s for {len(episode.frames)} frames = {len(episode.frames)/elapsed:.2f} fps")
    return elapsed, len(episode.frames)


if __name__ == '__main__':
    results = {}
    for name, video in [
        ("test_10s", "data/raw_videos/test_10s.mp4"),
        ("whatsapp_video", "data/raw_videos/whatsapp_video.mp4"),
        ("client_video", "data/raw_videos/client_video.mp4"),
        ("new_video", "data/raw_videos/new_video.mp4"),
    ]:
        r = regen(name, video)
        if r:
            results[name] = r

    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    for name, (elapsed, nf) in results.items():
        print(f"  {name}: {elapsed:.1f}s ({nf} frames, {nf/elapsed:.2f} fps)")
