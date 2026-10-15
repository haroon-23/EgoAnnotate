#!/usr/bin/env python3
"""Multi-embodiment retarget of one episode's exported wrist trajectory.
Usage: python scripts/retarget_multi.py --pack data/output/whatsapp_video \
       --configs configs/embodiments/franka_arm.yaml configs/embodiments/unitree_h1_upper.yaml \
       configs/embodiments/unitree_g1_upper.yaml configs/embodiments/allegro_hand.yaml \
       --out data/output/multi_v1"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "retargeting"))

from embodiment_registry import Adapter
from run_reference_pipeline import sanitize_trajectory   # shared feasibility law
from src.datatypes import AnnotatedEpisode
from src.retargeting.pose_mapper import PoseMapper, PoseMapperConfig


def rot6_to_quat(r6):
    if np.all(r6 == 0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    m = np.eye(3)
    m[:, 0] = r6[0:3]
    m[:, 1] = r6[3:6]
    m[:, 2] = np.cross(m[:, 0], m[:, 1])
    det = np.linalg.det(m)
    if det <= 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(m).as_quat()   # xyzw


def main():
    ap = argparse.ArgumentParser(description="Multi-embodiment retarget of one episode's exported wrist trajectory.")
    ap.add_argument("pack_pos", nargs="?", default=None, help="Episode directory or annotations.json path (positional)")
    ap.add_argument("--pack", default=None, help="Episode directory or annotations.json path")
    ap.add_argument("--configs", nargs="+", required=True, help="List of embodiment yaml configs")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--frames", type=int, default=None, help="Limit number of frames to retarget")
    a = ap.parse_args()

    pack_input = a.pack or a.pack_pos
    if not pack_input:
        ap.error("Must provide either --pack <path> or positional <pack> argument.")

    os.makedirs(a.out, exist_ok=True)

    # Resolve pack directory and annotations
    if os.path.isfile(pack_input):
        pack_dir = os.path.dirname(pack_input)
        ann_path = pack_input
    else:
        pack_dir = pack_input
        ann_path = os.path.join(pack_dir, "annotations.json")
        if not os.path.exists(ann_path):
            ann_path = os.path.join(pack_dir, "frame_annotations.json")

    r_seq = []
    l_seq = None
    hands_seq = None
    grips_seq = None
    reach = None

    if os.path.exists(ann_path):
        try:
            ep = AnnotatedEpisode.load_json(ann_path)
            frames = ep.frames
            mapper_r = PoseMapper(PoseMapperConfig(preferred_hand="right", lock_driving_hand=True))
            mapper_l = PoseMapper(PoseMapperConfig(preferred_hand="left", lock_driving_hand=True))
            t_right = mapper_r.map_frames_for_hand(frames, "right")
            t_left = mapper_l.map_frames_for_hand(frames, "left")

            r_seq = [(t.position * np.array([0.8, 0.8, 0.7]), t.quaternion) for t in t_right]
            l_seq = [(t.position * np.array([0.8, 0.8, 0.7]), t.quaternion) for t in t_left]
            hands_seq = [f.right_hand for f in frames]
            grips_seq = np.array([0.8 if (f.right_grasp and f.right_grasp.type != "open") else 0.1 for f in frames], dtype=float)
            reach = np.array([t.hand_detected for t in t_right], dtype=bool)
        except Exception as e:
            print(f"[multi] Warning loading {ann_path}: {e}; falling back to parquet")

    if not r_seq:
        import pyarrow.parquet as pq
        parquet_path = os.path.join(pack_dir, "lerobot_v3/data/chunk-000/file-000.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Neither annotations nor parquet found in pack: {pack_input}")
        tbl = pq.read_table(parquet_path)
        st = np.array(tbl.column("observation.state").to_pylist(), dtype=float)
        r_seq = [(st[i, 0:3], rot6_to_quat(st[i, 3:9])) for i in range(len(st))]
        if "robot_reachable" in tbl.column_names:
            reach = np.array(tbl.column("robot_reachable").to_pylist(), dtype=bool)
        elif "observation.robot_reachable" in tbl.column_names:
            reach = np.array(tbl.column("observation.robot_reachable").to_pylist(), dtype=bool)
        else:
            reach = np.array([not np.all(st[i, :3] == 0) for i in range(len(st))], dtype=bool)
        grips_seq = st[:, 9] if st.shape[1] > 9 else np.full(len(st), 0.5)

    n = len(r_seq) if a.frames is None else min(a.frames, len(r_seq))
    dt = 1 / 30.0
    episode_name = os.path.basename(pack_dir.rstrip("/"))
    manifest = {"episode": episode_name, "frames": n, "embodiments": []}

    for cp in a.configs:
        ad = Adapter(cp)
        name = os.path.splitext(os.path.basename(cp))[0]
        Q = np.zeros((n, ad.model.nq))
        R = np.zeros(n, bool)
        qprev = None
        first_valid = True

        for i in range(n):
            if not reach[i]:
                if qprev is not None:
                    Q[i] = qprev
                continue

            step_tgt = [r_seq[i]]
            if len(ad.tasks) >= 2 and l_seq is not None and i < len(l_seq):
                step_tgt.append(l_seq[i])

            hlm = hands_seq[i] if (hands_seq is not None and i < len(hands_seq)) else None
            grip_val = float(grips_seq[i]) if (grips_seq is not None and i < len(grips_seq)) else 0.5

            iters = 80 if first_valid else 1
            q_out, ok = None, False
            for _ in range(iters):
                q_out, ok = ad.solve(step_tgt, dt=dt, grip=grip_val, hand_landmarks=hlm)
                if first_valid and ad.last_step_info and ad.last_step_info.ik_residual_m < 0.01:
                    break

            first_valid = False
            if ok:
                if qprev is None:
                    Q[: i + 1] = q_out
                else:
                    Q[i] = q_out
                qprev = q_out
                R[i] = True
            else:
                if qprev is not None:
                    Q[i] = qprev
                R[i] = False

        lo = np.array([ad.model.jnt_range[j, 0] for j in range(ad.model.nq)])
        hi = np.array([ad.model.jnt_range[j, 1] for j in range(ad.model.nq)])
        vel_limits = ad.vel[: Q.shape[1]] if len(ad.vel) >= Q.shape[1] else np.full(Q.shape[1], 2.6)
        Q, R, _ = sanitize_trajectory(
            Q,
            R,
            ["x"] * n,
            dt,
            vel_limits=vel_limits,
            lim=list(zip(lo, hi)),
        )

        err = [
            np.linalg.norm(ad.fk(Q[i])[ad.task_links[0]["link"]] - r_seq[i][0])
            for i in range(n)
            if R[i]
        ]
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "trajectory.npz"), q=Q, reach=R)
        rep = {
            "embodiment": name,
            "kind": ad.kind,
            "nq": int(ad.model.nq),
            "reachable": int(R.sum()),
            "mean_fk_residual_m": float(np.mean(err)) if err else None,
        }
        with open(os.path.join(d, "feasibility.json"), "w") as f_feas:
            json.dump(rep, f_feas, indent=2)
        manifest["embodiments"].append(rep)
        print(f"[multi] {name}: {rep}")

    with open(os.path.join(a.out, "manifest.json"), "w") as f_man:
        json.dump(manifest, f_man, indent=2)


if __name__ == "__main__":
    main()
