"""Test script to verify Unitree H1 Humanoid Arm URDF retargeting on new_video.mp4."""
import json
from src.retargeting.retargeter import Retargeter, RetargetingConfig

def test_humanoid():
    cfg = RetargetingConfig.from_yaml("configs/retargeting_humanoid.yaml")
    retargeter = Retargeter(cfg)
    kin = retargeter.load_kinematics()
    
    print("=== KINEMATICS REPORT FOR HUMANOID (Unitree H1 Arm) ===")
    print(f"Robot Name     : {kin.robot_name}")
    print(f"URDF Path      : {kin.urdf_path}")
    print(f"End Effector   : {kin.end_effector_link}")
    print(f"Arm Joint Count: {len(kin.arm_joints)}")
    for j in kin.arm_joints:
        print(f"  {j.name:<30} limits: [{j.lower_limit:+.2f}, {j.upper_limit:+.2f}] rad")
        
    res = retargeter.run_from_annotations("data/output/whatsapp_video/frame_annotations.json", episode_id="whatsapp_video_humanoid")
    print("\n=== HUMANOID RETARGETING SUMMARY ===")
    print(f"Total Frames : {res.n_frames}")
    print(f"Reachable    : {res.summary['n_reachable']} / {res.n_frames} ({res.summary['pct_reachable']:.1f}%)")
    print(f"Avg Residual : {res.summary['avg_ik_residual_mm']:.2f} mm")
    print(f"Avg Solve Time: {res.summary['avg_solve_time_ms']:.2f} ms/frame")
    print(f"Self-Collision: {res.summary['n_self_collision']}")

if __name__ == "__main__":
    test_humanoid()
