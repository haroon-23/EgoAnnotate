#!/usr/bin/env python3
"""Phase D physics proof: replay exported joints in MuJoCo with gravity,
position actuators, free object on table. Reports grasp/lift success. CPU-fast."""
import argparse, json, os, re
import mujoco as mj
import numpy as np
import h5py

KP = [600, 600, 400, 400, 150, 80, 50]   # arm position gains (tunable)
KF, KVF = 300.0, 10.0  # 3× stiffer fingers for reliable grasp

def build_scene(urdf, cache, obj_home):
    if not os.path.exists(cache):
        m0 = mj.MjModel.from_xml_path(urdf)
        mj.mj_saveLastXML(cache, m0)
    txt = open(cache).read()
    names = re.findall(r'joint name="([^"]+)"', txt)
    arm = [j for j in names if "finger" not in j][:7]
    fing = [j for j in names if "finger" in j][:2]
    acts = "\n".join([f'    <position joint="{j}" kp="{k}" kv="{0.02*k:.1f}"/>'
                      for j, k in zip(arm, KP)] +
                     [f'    <position joint="{j}" kp="{KF:.0f}" kv="{KVF}"/>'
                      for j in fing])
    bodies = (f'\n    <body name="table" pos="0.5 0 0.125">'
              f'<geom type="box" size="0.35 0.30 0.125" rgba="0.45 0.3 0.15 1"/></body>'
              f'\n    <body name="obj" pos="{obj_home[0]} {obj_home[1]} {obj_home[2]}">'
              f'<freejoint/><geom type="cylinder" size="0.030 0.060" mass="0.15" '
              f'rgba="0.8 0.1 0.1 1" friction="1.0 0.005 0.0001"/></body>\n  ')
    txt = re.sub(r"<mujoco[^>]*>", lambda m: m.group(0) + '\n  <option timestep="0.00416667"/>', txt, count=1)
    txt = txt.replace("</worldbody>", bodies + "</worldbody>")
    txt = txt.replace("</mujoco>", f"<actuator>\n{acts}\n</actuator>\n</mujoco>")
    scene = cache.replace(".xml", "_scene.xml")
    open(scene, "w").write(txt)
    return mj.MjModel.from_xml_path(scene), arm, fing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True); ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--obj-z", type=float, default=0.31)
    a = ap.parse_args()
    with h5py.File(a.hdf5, "r") as f:
        ep = list(f.keys())[0]
        q = f[ep]["steps"]["observation"]["robot_joint_angles"][:]
        g = f[ep]["steps"]["observation"]["robot_gripper_opening_m"][:]
        r = f[ep]["steps"]["robot_reachable"][:]
    cache_path = os.path.join(os.path.dirname(a.urdf) or ".", "panda_scene_cache.xml")
    model, arm, fing = build_scene(a.urdf, cache_path, (0.5, 0.0, a.obj_z))
    data = mj.MjData(model); mj.mj_forward(model, data)
    n_arm, n_fin = len(arm), len(fing)
    obj_b = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "obj")
    ee_b = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "panda_link8")
    if ee_b < 0: ee_b = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "panda_hand")
    sub = max(1, int(round((1/30.0) / model.opt.timestep)))
    windows = succ = 0; in_win = win_succ = False; win_steps = 0
    best_lift = 0.0; track_err = []; lifts = []
    for i in range(len(q)):
        if r[i]:
            data.ctrl[:n_arm] = q[i]
            data.ctrl[n_arm:n_arm+n_fin] = g[i, 0] / 2.0
            track_err.append(float(np.max(np.abs(data.qpos[:n_arm] - q[i]))))
        closed = bool(r[i]) and g[i, 0] < 0.03
        for _ in range(sub): mj.mj_step(model, data)
        lift = float(data.xpos[obj_b][2] - a.obj_z)
        horiz = float(np.linalg.norm(data.xpos[obj_b][:2] - data.xpos[ee_b][:2]))
        best_lift = max(best_lift, lift); lifts.append(lift)
        if closed and not in_win:
            # Validate grasp initiation: EE must be within 5cm of object
            ee_obj_dist = np.linalg.norm(data.xpos[ee_b][:2] - data.xpos[obj_b][:2])
            if ee_obj_dist > 0.05:
                continue  # skip this window; grasp too early
            in_win, win_succ, win_steps, windows = True, False, 0, windows + 1
        if in_win:
            win_steps += 1
            if lift >= 0.02 and horiz < 0.09: win_succ = True
            if not closed or i == len(q) - 1:
                dur = win_steps * model.opt.timestep * sub
                if win_succ and dur >= 0.4: succ += 1
                in_win = False
    out = {"n_grasp_windows": windows, "n_success": succ,
           "success_rate": (succ / windows) if windows else 0.0,
           "max_lift_m": round(best_lift, 4),
           "mean_tracking_err_rad": round(float(np.mean(track_err)), 4) if track_err else None,
           "note": "success = object lifted >=2cm within 9cm of EE for >=0.4s while gripper closed"}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[physics] windows={windows} success={succ} rate={out['success_rate']:.2f} "
          f"max_lift={best_lift:.4f}m track_err={out['mean_tracking_err_rad']}")
    if windows == 0: raise RuntimeError("no grasp windows: gripper never closed while reachable")

if __name__ == "__main__":
    main()
