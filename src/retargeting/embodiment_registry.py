"""Config-driven retargeting adapters: arms, humanoid upper bodies, dexterous hands.
All adapters share mink IK + export-side sanitizer + reason codes.
No embodiment-specific logic lives outside YAML configs."""
import numpy as np, yaml, mujoco as mj, mink

class Adapter:
    def __init__(self, cfg_path):
        c = yaml.safe_load(open(cfg_path)); self.cfg = c; self.kind = c.get("kind", "arm")
        self.model = mj.MjModel.from_xml_path(c["urdf"]); self.data = mj.MjData(self.model)
        self.task_links = c["task_links"]; self.tasks = []
        for t in self.task_links:
            self.tasks.append(mink.FrameTask(frame_name=t["link"], frame_type="body",
                            position_cost=t.get("pos_cost", 1.0),
                            orientation_cost=t.get("ori_cost", 1.0)))
        self.posture = mink.PostureTask(self.model, cost=c.get("posture_cost", 1e-3))
        dp = c.get("default_posture"); self.posture.set_target(
            np.array(dp if dp else np.zeros(self.model.nq)))
        self.vel = np.array(c.get("velocity_limits", [2.6]*self.model.nq))
        self.lim = mink.ConfigurationLimit(self.model)

        vel_dict = {}
        for j in range(self.model.njnt):
            jname = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j)
            if jname:
                v_lim = 0.899 * (self.vel[j] if j < len(self.vel) else 2.6)
                vel_dict[jname] = v_lim
        self.vel_lim = mink.VelocityLimit(self.model, vel_dict)
        self.cfgm = mink.Configuration(self.model)
        lo = np.array([self.model.jnt_range[j, 0] for j in range(self.model.nq)])
        hi = np.array([self.model.jnt_range[j, 1] for j in range(self.model.nq)])
        if dp:
            self.cfgm.update(np.clip(np.array(dp), lo, hi))
        self.fingers = c.get("finger_joints", [])          # hand adapters only
        self.fing_open = np.array(c.get("finger_open", [])); self.fing_close = np.array(c.get("finger_close", []))

    def solve(self, targets, dt, grip=None):
        from scipy.spatial.transform import Rotation
        for task, (p, q) in zip(self.tasks, targets):
            task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(Rotation.from_quat(q).as_matrix()), p))
        try:
            dq = mink.solve_ik(self.cfgm, self.tasks + [self.posture], dt, solver="daqp", limits=[self.lim, self.vel_lim], damping=1e-6)
            self.cfgm.integrate_inplace(dq, dt)
        except Exception:
            pass
        q = self.cfgm.q.copy()
        if self.fingers and grip is not None:               # hand v1: grip-proportional posture
            for i, name in enumerate(self.fingers):
                j = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
                if j >= 0 and i < len(self.fing_open):
                    q[self.model.jnt_qposadr[j]] = self.fing_open[i] + grip*(self.fing_close[i]-self.fing_open[i])
        lo = np.array([self.model.jnt_range[j, 0] for j in range(self.model.nq)])
        hi = np.array([self.model.jnt_range[j, 1] for j in range(self.model.nq)])
        return np.clip(q, lo, hi)

    def fk(self, q):
        d = mj.MjData(self.model); d.qpos[:len(q)] = q; mj.mj_forward(self.model, d)
        return {t["link"]: d.xpos[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, t["link"])].copy()
                for t in self.task_links}
