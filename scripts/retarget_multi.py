#!/usr/bin/env python3
"""Multi-embodiment retarget of one episode's exported wrist trajectory.
Usage: python scripts/retarget_multi.py --pack data/output/reference_v2 \
       --configs configs/embodiments/franka_arm.yaml configs/embodiments/unitree_h1_upper.yaml \
       configs/embodiments/unitree_g1_upper.yaml configs/embodiments/allegro_hand.yaml \
       --out data/output/multi_v1 --frames 150"""
import argparse, json, os, sys
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "retargeting"))
from embodiment_registry import Adapter
from run_reference_pipeline import sanitize_trajectory   # shared feasibility law

def rot6_to_quat(r6):
    m = np.eye(3); m[:, 0] = r6[0:3]; m[:, 1] = r6[3:6]
    m[:, 2] = np.cross(m[:, 0], m[:, 1])
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(m).as_quat()   # xyzw

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pack", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--frames", type=int, default=150)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    tbl = pq.read_table(os.path.join(a.pack, "lerobot_v3/data/chunk-000/file-000.parquet"))
    st = np.array(tbl.column("observation.state").to_pylist(), dtype=float)
    reach = np.array(tbl.column("robot_reachable").to_pylist(), dtype=bool)
    n = min(a.frames, len(st)); dt = 1/30.0
    manifest = {"episode": os.path.basename(a.pack), "frames": n, "embodiments": []}
    for cp in a.configs:
        ad = Adapter(cp); name = os.path.splitext(os.path.basename(cp))[0]
        Q = np.zeros((n, ad.model.nq)); R = np.zeros(n, bool); qprev = None
        for i in range(n):
            if not reach[i]:
                if qprev is not None: Q[i] = qprev
                continue
            tgt = [(st[i, 0:3], rot6_to_quat(st[i, 3:9]))]
            q = ad.solve(tgt, dt, grip=st[i, 9])
            if qprev is None:
                qprev = q
                Q[:i+1] = q
            else:
                qprev = q
                Q[i] = q
            R[i] = True
        lo = np.array([ad.model.jnt_range[j, 0] for j in range(ad.model.nq)])
        hi = np.array([ad.model.jnt_range[j, 1] for j in range(ad.model.nq)])
        Q, R, _ = sanitize_trajectory(Q, R, ["x"]*n, dt,
                                      vel_limits=ad.vel[:Q.shape[1]] if len(ad.vel) >= Q.shape[1] else np.full(Q.shape[1], 2.6),
                                      lim=list(zip(lo, hi)))
        err = [np.linalg.norm(ad.fk(Q[i])[ad.task_links[0]["link"]] - st[i, 0:3])
               for i in range(n) if R[i]]
        d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "trajectory.npz"), q=Q, reach=R)
        rep = {"embodiment": name, "kind": ad.kind, "nq": int(ad.model.nq),
               "reachable": int(R.sum()), "mean_fk_residual_m": float(np.mean(err)) if err else None}
        json.dump(rep, open(os.path.join(d, "feasibility.json"), "w"), indent=2)
        manifest["embodiments"].append(rep); print(f"[multi] {name}: {rep}")
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
