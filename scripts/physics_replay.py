#!/usr/bin/env python3
"""Phase D physics proof: replay exported joints in MuJoCo with gravity,
position actuators, free object on table. Reports grasp/lift success. CPU-fast."""
import argparse, json, os, re
import mujoco as mj
import numpy as np

KP = [1000, 1000, 750, 750, 300, 150, 100]
KV = [100, 100, 75, 75, 30, 15, 10]
KF, KVF = 400.0, 20.0
TABLE_TOP = 0.25

def build_scene(urdf, cache):
    if not os.path.exists(cache):
        m0 = mj.MjModel.from_xml_path(urdf); mj.mj_saveLastXML(cache, m0)
    txt = open(cache).read()
    names = re.findall(r'joint name="([^"]+)"', txt)
    arm = [j for j in names if "finger" not in j][:7]
    fing = [j for j in names if "finger" in j][:2]
    acts = "\n".join([f'    <position joint="{j}" kp="{k}" kv="{v}"/>'
                      for j, k, v in zip(arm, KP, KV)] +
                     [f'    <position joint="{j}" kp="{KF:.0f}" kv="{KVF}"/>' for j in fing])
    body = ('\n    <body name="table" pos="0.5 0 0.125"><geom type="box" '
            'size="0.35 0.30 0.125" rgba="0.45 0.3 0.15 1"/></body>'
            '\n    <body name="obj" pos="0.5 0 0.31"><freejoint/>'
            '<geom type="cylinder" size="0.030 0.060" mass="0.12" '
            'friction="1.2 0.005 0.0001" rgba="0.8 0.1 0.1 1"/></body>\n  ')
    txt = re.sub(r"<mujoco[^>]*>", lambda m: m.group(0) + '\n  <option timestep="0.00416667"/>', txt, count=1)
    txt = txt.replace("</worldbody>", body + "</worldbody>")
    txt = txt.replace("</mujoco>", f"<actuator>\n{acts}\n</actuator>\n</mujoco>")
    scene = cache.replace(".xml", "_scene.xml"); open(scene, "w").write(txt)
    return mj.MjModel.from_xml_path(scene), arm, fing

def ids(model, arm, fing):
    a = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, j) for j in arm]
    f = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, j) for j in fing]
    ob = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "obj")
    eb = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "panda_link8")
    if eb < 0: eb = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "panda_hand")
    oj = [i for i in range(model.njnt) if model.jnt_type[i] == mj.mjtJoint.mjJNT_FREE][0]
    return a, f, ob, eb, model.jnt_qposadr[oj]

def trial(model, data, q, g, i0, i1, obj_pos, A, F, OB, EB, QA, sub):
    mj.mj_resetData(model, data)
    data.qpos[QA:QA+3] = obj_pos; data.qpos[QA+3:QA+7] = [1, 0, 0, 0]
    mj.mj_forward(model, data)
    z0 = obj_pos[2]; lift = 0.0; near = False
    for i in range(i0, i1):
        data.ctrl[:len(A)] = q[i]
        data.ctrl[len(A):len(A)+len(F)] = g[i, 0] / 2.0
        for _ in range(sub): mj.mj_step(model, data)
        d = float(np.linalg.norm(data.xpos[OB][:2] - data.xpos[EB][:2]))
        if d < 0.09: near = True
        lift = max(lift, float(data.xpos[OB][2] - z0))
    return lift, near

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True); ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    with h5py.File(a.hdf5, "r") as f:
        ep = list(f.keys())[0]
        q = f[ep]["steps"]["observation"]["robot_joint_angles"][:]
        g = f[ep]["steps"]["observation"]["robot_gripper_opening_m"][:]
        r = f[ep]["steps"]["robot_reachable"][:]
    cache_path = os.path.join(os.path.dirname(a.urdf) or ".", "panda_scene_cache.xml")
    model, arm, fing = build_scene(a.urdf, cache_path)
    data = mj.MjData(model)
    A, F, OB, EB, QA = ids(model, arm, fing)
    sub = max(1, int(round((1/30.0) / model.opt.timestep)))
    # M1: tracking calibration pass over reachable frames
    mj.mj_resetData(model, data); mj.mj_forward(model, data)
    errs = []
    for i in range(len(q)):
        if not r[i]: continue
        data.ctrl[:len(A)] = q[i]; data.ctrl[len(A):len(A)+len(F)] = g[i, 0]/2.0
        for _ in range(sub): mj.mj_step(model, data)
        errs.append(float(np.max(np.abs(data.qpos[:len(A)] - q[i]))))
    track = float(np.mean(errs)) if errs else 9.9
    print(f"[M1] mean tracking err = {track:.4f} rad (gate < 0.15)")
    # grasp windows from gripper+reach
    wins = []; i = 0
    while i < len(q):
        if r[i] and g[i, 0] < 0.03:
            j = i
            while j < len(q) and r[j] and g[j, 0] < 0.03: j += 1
            if j - i >= 6: wins.append((i, j))
            i = j
        else: i += 1
    # M2+M3: per-window trials, object placed at EE midpoint
    rows = []; succ = 0
    for (i0, i1) in wins:
        mid = (i0 + i1) // 2
        # Check arm is actually moving (not at home)
        if np.all(np.abs(q[mid]) < 0.1):
            print(f"[win {i0}-{i1}] SKIP: arm at home"); continue
        mj.mj_resetData(model, data)
        data.qpos[:len(A)] = q[mid]
        data.qpos[len(A):len(A)+len(F)] = g[mid, 0]/2*np.ones(len(F))
        mj.mj_forward(model, data)
        p = data.xpos[EB].copy()
        obj_pos = [float(p[0]), float(p[1]), TABLE_TOP + 0.06]
        i0t, i1t = max(0, i0-15), min(len(q), i1+15)
        lift, near = trial(model, data, q, g, i0t, i1t, obj_pos, A, F, OB, EB, QA, sub)
        ok = bool(lift >= 0.02 and near)
        succ += ok
        rows.append({"window": [int(i0), int(i1)], "obj_at": [round(v, 3) for v in obj_pos],
                     "lift_m": round(lift, 4), "near_ee": near, "success": ok})
        print(f"[win {i0}-{i1}] lift={lift:.4f} near={near} success={ok}")
    out = {"tracking_err_rad": round(track, 4), "n_windows": len(wins), "n_success": succ,
           "success_rate": (succ / len(wins)) if wins else 0.0, "windows": rows,
           "note": "object placed at demonstrated grasp point per window; "
                   "success = lift>=2cm within 9cm of EE"}
    json.dump(out, open(a.out, "w"), indent=2)
    if track > 0.15:
        raise RuntimeError(f"M1 FAIL: tracking err {track:.4f} rad; trajectory too fast for position control")
    print(f"[physics] windows={len(wins)} success={succ} rate={out['success_rate']:.2f}")

if __name__ == "__main__":
    main()
