"""g20_four_robot_independent.py

Independent recomputation of retargeting metrics for ALL robots listed in quality.json (default four:
franka_panda, unitree_g1_upper, unitree_h1_upper, allegro_hand_right) using PyBullet forward kinematics straight from the URDF,
plus a cross-robot / cross-episode test that metrics are measured, not derived from another robot's numbers.

Each embodiments/<robot>/trajectory.npz must be self-describing (see GATES_SCHEMA.md):
  q (T,n)  joint_names (n,)  valid_mask (T,)  target_ee_pos (T,3)  target_ee_quat_xyzw (T,4)
  urdf_path (str, relative to --repo or absolute)  ee_link (str)  base_pos (3,)  base_quat_xyzw (4,)  fps (float)
Recomputed and compared with quality.robots.<robot>: fk_pos_residual_mean_mm, fk_pos_residual_p95_mm, fk_ori_residual_p95_deg,
ik_convergence_pct (using quality.robots.<robot>.ik_convergence_def {pos_tol_mm, ori_tol_deg}), max_joint_vel_rad_s, max_joint_jerk_rad_s3.
Velocity/jerk statistics use only consecutive valid-frame runs (never differenced across gaps).
Also: zero joint-limit violations (from URDF limits), and max |qdot| must not exceed the URDF velocity limit.
Ratio test (needs >= 3 sibling episodes under --output-root): the p95/mean position-residual ratios between robots, and the IK
convergence gaps between robots, must VARY across episodes (coefficient of variation >= 0.02): constant ratios = derived numbers.

Usage:  python gates/g20_four_robot_independent.py <episode_dir> [--repo .] [--output-root DIR] [--json-out PATH]   Exit: 0 PASS, 1 FAIL.
Needs:  pybullet, numpy.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

GATE = "g20_four_robot_independent"
DEFAULT_ROBOTS = ["franka_panda", "unitree_g1_upper", "unitree_h1_upper", "allegro_hand_right"]
REQ_KEYS = ["q", "joint_names", "valid_mask", "target_ee_pos", "target_ee_quat_xyzw", "urdf_path", "ee_link", "base_pos", "base_quat_xyzw", "fps"]
TOL = dict(pos_abs=0.05, pos_rel=0.01, ori_abs=0.05, ori_rel=0.01, ik_abs=0.1, vel_rel=0.02, jerk_rel=0.05)
MIN_CV = 0.02
MIN_SIBLINGS = 3


class R:
    def __init__(s): s.fails, s.warns, s.info = [], [], {}
    def check(s, name, ok, detail="", warn=False):
        print(f"  [{'PASS' if ok else ('WARN' if warn else 'FAIL')}] {name}: {detail}")
        if not ok: (s.warns if warn else s.fails).append(f"{name}: {detail}")
    def finish(s, out=None):
        st = "FAIL" if s.fails else "PASS"; print(f"G20 VERDICT: {st}")
        if out: Path(out).write_text(json.dumps(dict(gate=GATE, status=st, failures=s.fails, warnings=s.warns, info=s.info), indent=1, default=str))
        return 1 if s.fails else 0


def jload(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None


def scalar(z, k):
    v = z[k]; return str(v.item() if v.shape == () else v.reshape(-1)[0])


def quat_angle_deg(a, b):
    d = np.abs(np.sum(a * b, axis=1)) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(2 * np.arccos(np.clip(d, -1.0, 1.0)))


def fk_traj(urdf, joint_names, ee_link, base_pos, base_quat, q):
    """PyBullet forward kinematics of a joint trajectory. Returns (pos (T,3), quat_xyzw (T,4), limits{name:(lo,hi)}, vlimits{name:v})."""
    import pybullet as p
    urdf = Path(urdf)
    if not urdf.exists(): raise ValueError(f"urdf not found: {urdf}")
    cid = p.connect(p.DIRECT)
    try:
        body = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=cid)
        p.resetBasePositionAndOrientation(body, [float(x) for x in base_pos], [float(x) for x in base_quat], physicsClientId=cid)
        idx, lim, vlim, link_idx = {}, {}, {}, None
        for j in range(p.getNumJoints(body, physicsClientId=cid)):
            info = p.getJointInfo(body, j, physicsClientId=cid); nm = info[1].decode()
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC): idx[nm] = j; lim[nm] = (info[8], info[9]); vlim[nm] = info[11]
            if info[12].decode() == ee_link: link_idx = j
        if link_idx is None: raise ValueError(f"ee_link '{ee_link}' not in URDF")
        miss = [n for n in joint_names if n not in idx]
        if miss: raise ValueError(f"joint_names not in URDF: {miss[:5]}")
        q = np.asarray(q, float); T = len(q); pos = np.zeros((T, 3)); quat = np.zeros((T, 4))
        for t in range(T):
            for k, n in enumerate(joint_names): p.resetJointState(body, idx[n], float(q[t, k]), physicsClientId=cid)
            ls = p.getLinkState(body, link_idx, computeForwardKinematics=True, physicsClientId=cid); pos[t], quat[t] = ls[4], ls[5]
    finally:
        p.disconnect(cid)
    return pos, quat, lim, vlim


def recompute(z, repo):
    """Independent FK of the saved joint trajectory. Returns dict of metrics (or raises ValueError)."""
    urdf = Path(scalar(z, "urdf_path")); urdf = urdf if urdf.is_absolute() else Path(repo) / urdf
    names = [str(x) for x in z["joint_names"]]; q = np.asarray(z["q"], float); valid = np.asarray(z["valid_mask"]).astype(bool)
    pos, quat, lim, vlim = fk_traj(urdf, names, scalar(z, "ee_link"), z["base_pos"], z["base_quat_xyzw"], q)
    tp = np.asarray(z["target_ee_pos"], float); tq = np.asarray(z["target_ee_quat_xyzw"], float)
    pe = np.linalg.norm(pos - tp, axis=1) * 1000.0; oe = quat_angle_deg(quat, tq); fps = float(scalar(z, "fps"))
    pv, ov = pe[valid], oe[valid]; T = len(q)
    viol = 0
    for k, n in enumerate(names):
        lo, hi = lim[n]
        if hi > lo: viol += int(np.sum((q[:, k] < lo - 1e-6) | (q[:, k] > hi + 1e-6)))
    dt = 1.0 / fps                      # derivatives use ONLY runs of consecutive valid frames (no differencing across gaps)
    d1 = np.diff(q, axis=0) / dt; m1 = valid[1:] & valid[:-1]
    d2 = np.diff(d1, axis=0) / dt; m2 = m1[1:] & m1[:-1]
    d3 = np.diff(d2, axis=0) / dt; m3 = m2[1:] & m2[:-1]
    qd = d1[m1] if m1.any() else np.zeros((1, q.shape[1])); qddd = d3[m3] if m3.any() else np.zeros((1, q.shape[1]))
    over = []
    for k, n in enumerate(names):
        if vlim[n] and vlim[n] > 0 and np.abs(qd[:, k]).max() > vlim[n] * 1.0001: over.append((n, float(np.abs(qd[:, k]).max()), float(vlim[n])))
    return dict(frames=int(T), valid_frames=int(valid.sum()), pos_mean=float(pv.mean()) if len(pv) else None, pos_p95=float(np.percentile(pv, 95)) if len(pv) else None,
                ori_p95=float(np.percentile(ov, 95)) if len(ov) else None, pos_err=pv, ori_err=ov, limit_violations=viol,
                max_vel=float(np.abs(qd).max()), max_jerk=float(np.abs(qddd).max()), vel_over=over)


def close(a, b, abs_t, rel_t):
    return a is not None and b is not None and abs(a - b) <= max(abs_t, rel_t * max(abs(a), abs(b)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episode_dir"); ap.add_argument("--repo", default="."); ap.add_argument("--output-root")
    ap.add_argument("--robots", default=",".join(DEFAULT_ROBOTS)); ap.add_argument("--json-out"); a = ap.parse_args()
    ep = Path(a.episode_dir); r = R(); robots = [x for x in a.robots.split(",") if x]
    print(f"=== G20: FOUR-ROBOT INDEPENDENT RECOMPUTATION ({ep}) ===")
    try: import pybullet  # noqa
    except ImportError: r.check("pybullet available", False, "pip install pybullet"); return r.finish(a.json_out)
    q = jload(ep / "quality.json")
    r.check("quality.json present", q is not None, "")
    if q is None: return r.finish(a.json_out)
    qr = q.get("robots") or {}
    r.check("quality.robots contains all required robots", all(x in qr for x in robots), f"missing={[x for x in robots if x not in qr]}")
    for rb in robots:
        tp = ep / "embodiments" / rb / "trajectory.npz"
        if not tp.exists(): r.check(f"{rb}: trajectory.npz", False, "missing"); continue
        z = np.load(tp, allow_pickle=False); miss = [k for k in REQ_KEYS if k not in z.files]
        r.check(f"{rb}: self-describing npz keys", not miss, f"missing {miss}" if miss else "complete")
        if miss: continue
        try: m = recompute(z, a.repo)
        except Exception as e: r.check(f"{rb}: independent FK", False, repr(e)); continue
        rep = qr.get(rb) or {}; df = rep.get("ik_convergence_def") or {}
        r.info[rb] = {k: v for k, v in m.items() if k not in ("pos_err", "ori_err")}
        for lab, mine, key, at, rt in [("fk_pos_residual_mean_mm", m["pos_mean"], "fk_pos_residual_mean_mm", TOL["pos_abs"], TOL["pos_rel"]),
                                       ("fk_pos_residual_p95_mm", m["pos_p95"], "fk_pos_residual_p95_mm", TOL["pos_abs"], TOL["pos_rel"]),
                                       ("fk_ori_residual_p95_deg", m["ori_p95"], "fk_ori_residual_p95_deg", TOL["ori_abs"], TOL["ori_rel"]),
                                       ("max_joint_vel_rad_s", m["max_vel"], "max_joint_vel_rad_s", 1e-4, TOL["vel_rel"]),
                                       ("max_joint_jerk_rad_s3", m["max_jerk"], "max_joint_jerk_rad_s3", 1e-3, TOL["jerk_rel"])]:
            rv = rep.get(key); r.check(f"{rb}: {lab} reproduced", close(mine, rv, at, rt), f"independent={None if mine is None else round(mine,4)} reported={rv}")
        if "pos_tol_mm" in df and "ori_tol_deg" in df and len(m["pos_err"]):
            conv = 100.0 * float(np.mean((m["pos_err"] <= df["pos_tol_mm"]) & (m["ori_err"] <= df["ori_tol_deg"])))
            r.check(f"{rb}: ik_convergence_pct reproduced", close(conv, rep.get("ik_convergence_pct"), TOL["ik_abs"], 0.0),
                    f"independent={conv:.3f} reported={rep.get('ik_convergence_pct')} (tol {df})")
        else: r.check(f"{rb}: ik_convergence_def present", False, f"quality.robots.{rb}.ik_convergence_def needs pos_tol_mm and ori_tol_deg")
        r.check(f"{rb}: zero joint-limit violations (URDF limits)", m["limit_violations"] == 0, f"violations={m['limit_violations']}")
        r.check(f"{rb}: joint speed within URDF velocity limits", not m["vel_over"], f"exceeded: {m['vel_over'][:3]}" if m["vel_over"] else "within limits")
    root = Path(a.output_root) if a.output_root else ep.parent
    sib = []
    for d in sorted(root.glob("*/quality.json")):
        j = jload(d)
        if j and all(x in (j.get("robots") or {}) for x in robots): sib.append(j["robots"])
    r.info["sibling_episodes"] = len(sib)
    if len(sib) < MIN_SIBLINGS:
        r.check(f"ratio-variance test ({len(sib)} sibling episodes with all robots)", False, f"needs >= {MIN_SIBLINGS}", warn=True)
    else:
        base = robots[0]
        for rb in robots[1:]:
            for lab, key in (("p95 position ratio", "fk_pos_residual_p95_mm"), ("mean position ratio", "fk_pos_residual_mean_mm")):
                ratios = np.array([s[rb][key] / s[base][key] for s in sib if s[base][key]])
                cv = float(ratios.std() / ratios.mean()) if len(ratios) and ratios.mean() else 0.0
                r.check(f"{rb}/{base} {lab} varies across episodes", cv >= MIN_CV, f"CV={cv:.4f} over {len(ratios)} episodes (min {MIN_CV}) ratios={np.round(ratios[:6],3).tolist()}")
            gaps = np.array([s[rb]["ik_convergence_pct"] - s[base]["ik_convergence_pct"] for s in sib])
            cvg = float(gaps.std() / abs(gaps.mean())) if abs(gaps.mean()) > 1e-9 else float(gaps.std())
            r.check(f"{rb}-{base} IK convergence gap varies across episodes", cvg >= MIN_CV, f"CV={cvg:.4f} gaps={np.round(gaps[:6],3).tolist()}")
        pairs = [(robots[i], robots[j]) for i in range(1, len(robots)) for j in range(i + 1, len(robots))]
        for x, y in pairs:
            gaps = np.array([s[x]["ik_convergence_pct"] - s[y]["ik_convergence_pct"] for s in sib])
            cvg = float(gaps.std() / abs(gaps.mean())) if abs(gaps.mean()) > 1e-9 else float(gaps.std())
            r.check(f"{x}-{y} IK convergence gap varies across episodes", cvg >= MIN_CV, f"CV={cvg:.4f} gaps={np.round(gaps[:6],3).tolist()}")
    return r.finish(a.json_out)


if __name__ == "__main__":
    sys.exit(main())
