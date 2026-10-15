"""Config-driven retargeting adapters: single arms, dual-arm humanoids, and dexterous hands.

All adapters share mink IK + MuJoCo FK/collision auditing + anatomical finger curl mapping.
Switching robots is strictly a YAML config change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import yaml
import mujoco as mj
import mink
from scipy.spatial.transform import Rotation


@dataclass
class EmbodimentSolveStep:
    q: np.ndarray
    reachable: bool
    ik_residual_m: float
    joint_limit_violated: bool
    self_collision: bool
    reason: str = "ok"


@dataclass
class EmbodimentEpisodeReport:
    name: str
    kind: str
    license: str
    num_frames: int
    reachable_frames: int
    reachable_pct: float
    mean_ik_residual_mm: float
    max_ik_residual_mm: float
    joint_limit_events: int
    self_collision_events: int
    q_trajectory: np.ndarray
    reachable_mask: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "license": self.license,
            "num_frames": int(self.num_frames),
            "reachable_frames": int(self.reachable_frames),
            "reachable_pct": round(float(self.reachable_pct), 2),
            "mean_ik_residual_mm": round(float(self.mean_ik_residual_mm), 3),
            "max_ik_residual_mm": round(float(self.max_ik_residual_mm), 3),
            "joint_limit_events": int(self.joint_limit_events),
            "self_collision_events": int(self.self_collision_events),
        }


def compute_anatomical_finger_curls(hand_landmarks: Any) -> Dict[str, float]:
    """Compute normalized [0, 1] flexion curls for thumb, index, middle, ring, pinky from 21 keypoints."""
    if hand_landmarks is None:
        return {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0, "opposition": 0.3}
    pts = np.stack([hand_landmarks.x, hand_landmarks.y, hand_landmarks.z], axis=-1).astype(np.float64)
    wrist = pts[0]
    palm_len = float(np.linalg.norm(pts[9] - wrist))
    if palm_len < 1e-5:
        palm_len = 0.1

    def _curl(mcp_idx: int, tip_idx: int) -> float:
        # Extended finger tip-to-wrist distance is ~2.0 * palm_len; fully curled is ~0.85 * palm_len
        dist_ratio = float(np.linalg.norm(pts[tip_idx] - wrist)) / palm_len
        curl = (1.95 - dist_ratio) / 1.05
        return float(np.clip(curl, 0.0, 1.0))

    thumb_pinky_dist = float(np.linalg.norm(pts[4] - pts[17])) / palm_len
    opposition = float(np.clip((1.6 - thumb_pinky_dist) / 1.1, 0.2, 1.0))
    return {
        "thumb": _curl(2, 4),
        "index": _curl(5, 8),
        "middle": _curl(9, 12),
        "ring": _curl(13, 16),
        "pinky": _curl(17, 20),
        "opposition": opposition,
    }


class Adapter:
    """Unified config-driven embodiment adapter for arm, dual-arm humanoid, and dexterous hand."""

    def __init__(self, cfg_path: str):
        with open(cfg_path, "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
        self.cfg_path = cfg_path
        self.cfg = c
        self.name = c.get("name", c.get("kind", "robot"))
        self.kind = c.get("kind", "arm")
        self.license = c.get("license", "Open-Source Bundled URDF")
        self.model = mj.MjModel.from_xml_path(c["urdf"])
        self.data = mj.MjData(self.model)
        self.task_links = c["task_links"]
        self.tasks = []
        for t in self.task_links:
            self.tasks.append(
                mink.FrameTask(
                    frame_name=t["link"],
                    frame_type="body",
                    position_cost=t.get("pos_cost", 1.0),
                    orientation_cost=t.get("ori_cost", 1.0),
                )
            )
        self.posture = mink.PostureTask(self.model, cost=c.get("posture_cost", 1e-3))
        dp = c.get("default_posture")
        self.default_q = np.array(dp if dp else np.zeros(self.model.nq), dtype=np.float64)
        self.posture.set_target(self.default_q)
        self.vel = np.array(c.get("velocity_limits", [2.6] * self.model.nq), dtype=np.float64)
        self.lim = mink.ConfigurationLimit(self.model)

        vel_dict = {}
        for j in range(self.model.njnt):
            jname = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j)
            if jname:
                v_lim = 0.899 * float(self.vel[j] if j < len(self.vel) else 2.6)
                vel_dict[jname] = v_lim
        self.vel_lim = mink.VelocityLimit(self.model, vel_dict)
        self.cfgm = mink.Configuration(self.model)
        self.lo = np.array([self.model.jnt_range[j, 0] for j in range(self.model.nq)], dtype=np.float64)
        self.hi = np.array([self.model.jnt_range[j, 1] for j in range(self.model.nq)], dtype=np.float64)
        if dp:
            self.cfgm.update(np.clip(self.default_q, self.lo, self.hi))

        self.fingers = c.get("finger_joints", [])
        self.fing_open = np.array(c.get("finger_open", []), dtype=np.float64)
        self.fing_close = np.array(c.get("finger_close", []), dtype=np.float64)
        self.full_finger_map: Dict[str, List[str]] = c.get("full_finger_joint_map") or {}
        self.last_step_info: Optional[EmbodimentSolveStep] = None

    def _apply_dexterous_hand_joints(
        self,
        q: np.ndarray,
        grip: Optional[float] = None,
        hand_landmarks: Any = None,
    ) -> np.ndarray:
        """Populate dexterous hand joints using 21-keypoint anatomical curls or scalar grip."""
        if hand_landmarks is not None and self.full_finger_map:
            curls = compute_anatomical_finger_curls(hand_landmarks)
            for finger_name in ("index", "middle", "ring"):
                jnames = self.full_finger_map.get(finger_name, [])
                c_val = curls.get(finger_name, 0.0)
                # Spread (joint 0) stays near 0; MCP/PIP/DIP flex proportionally
                angles = [0.0, 1.25 * c_val, 1.15 * c_val, 0.85 * c_val]
                for jname, ang in zip(jnames, angles):
                    jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jname)
                    if jid >= 0:
                        q[self.model.jnt_qposadr[jid]] = ang
            thumb_jnames = self.full_finger_map.get("thumb", [])
            t_curl = curls.get("thumb", 0.0)
            t_opp = curls.get("opposition", 0.4)
            t_angles = [1.1 * t_opp, 0.5 * t_curl, 0.95 * t_curl, 0.85 * t_curl]
            for jname, ang in zip(thumb_jnames, t_angles):
                jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jname)
                if jid >= 0:
                    adr = self.model.jnt_qposadr[jid]
                    q[adr] = float(np.clip(ang, self.lo[adr], self.hi[adr]))
            for finger_name in ("index", "middle", "ring"):
                for jname in self.full_finger_map.get(finger_name, []):
                    jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jname)
                    if jid >= 0:
                        adr = self.model.jnt_qposadr[jid]
                        q[adr] = float(np.clip(q[adr], self.lo[adr], self.hi[adr]))
            return q

        if self.fingers and grip is not None:
            g = float(np.clip(grip, 0.0, 1.0))
            for i, name in enumerate(self.fingers):
                j = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
                if j >= 0 and i < len(self.fing_open):
                    q[self.model.jnt_qposadr[j]] = self.fing_open[i] + g * (self.fing_close[i] - self.fing_open[i])
            if self.full_finger_map:
                for finger_name in ("index", "middle", "ring"):
                    for sub_j in self.full_finger_map.get(finger_name, [])[2:]:
                        jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, sub_j)
                        if jid >= 0:
                            q[self.model.jnt_qposadr[jid]] = 0.9 * g
        return q

    def solve(
        self,
        targets: Sequence[Tuple[np.ndarray, np.ndarray]],
        dt: float,
        grip: Optional[float] = None,
        hand_landmarks: Any = None,
    ) -> Tuple[np.ndarray, bool]:
        """Solve one step of IK. Preserves (q, ok) return signature while recording full diagnostics."""
        if not targets:
            q_cur = self.cfgm.q.copy() if self.cfgm.q is not None else np.zeros(self.model.nq)
            self.last_step_info = EmbodimentSolveStep(q_cur, False, 0.0, False, False, "no_target")
            return q_cur, False

        # Backwards compatibility: if caller passes 1 target for a dual-arm humanoid, mirror Y for left arm
        active_targets = list(targets)
        if len(active_targets) == 1 and len(self.tasks) == 2:
            p0, q0 = active_targets[0]
            p_left = np.array([p0[0], abs(float(p0[1])) + 0.15, p0[2]], dtype=np.float64)
            active_targets.append((p_left, q0.copy()))

        max_reach = float(self.cfg.get("max_reach", 10.0))
        base = np.zeros(3, dtype=np.float64)
        for (p, _) in active_targets[: len(self.tasks)]:
            if float(np.linalg.norm(np.asarray(p, dtype=np.float64) - base)) > max_reach:
                q_cur = self.cfgm.q.copy() if self.cfgm.q is not None else np.zeros(self.model.nq)
                self.last_step_info = EmbodimentSolveStep(q_cur, False, 999.0, False, False, "out_of_reach")
                return q_cur, False

        ActiveTasks = []
        adjusted_targets = []
        for t_cfg, task, (p, quat) in zip(self.task_links, self.tasks, active_targets):
            offset = np.asarray(t_cfg.get("target_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
            p_arr = np.asarray(p, dtype=np.float64) + offset
            adjusted_targets.append((p_arr, quat))
            q_arr = np.asarray(quat, dtype=np.float64)
            if np.linalg.norm(q_arr) < 1e-8:
                q_arr = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            else:
                q_arr = q_arr / np.linalg.norm(q_arr)
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(Rotation.from_quat(q_arr).as_matrix()), p_arr
                )
            )
            ActiveTasks.append(task)

        try:
            dq = mink.solve_ik(
                self.cfgm,
                ActiveTasks + [self.posture],
                dt,
                solver="daqp",
                limits=[self.lim, self.vel_lim],
                damping=1e-6,
            )
            self.cfgm.integrate_inplace(dq, dt)
        except Exception:
            pass

        q = self.cfgm.q.copy()
        q = self._apply_dexterous_hand_joints(q, grip=grip, hand_landmarks=hand_landmarks)

        # Check joint limit violation before clipping (tolerance 1e-3 rad)
        jlim_violated = bool(np.any(q < (self.lo - 1e-3)) or np.any(q > (self.hi + 1e-3)))
        q_clipped = np.clip(q, self.lo, self.hi)

        # Compute FK residual and MuJoCo self-collision
        self.data.qpos[: len(q_clipped)] = q_clipped
        mj.mj_forward(self.model, self.data)
        mj.mj_collision(self.model, self.data)

        residuals = []
        for t_cfg, (p_tgt, _) in zip(self.task_links, adjusted_targets):
            if t_cfg.get("pos_cost", 1.0) > 0:
                bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, t_cfg["link"])
                if bid >= 0:
                    pos_fk = self.data.xpos[bid].copy()
                    residuals.append(float(np.linalg.norm(pos_fk - np.asarray(p_tgt, dtype=np.float64))))
        mean_res_m = float(np.mean(residuals)) if residuals else 0.0

        # Count non-adjacent body contacts as self-collision (exclude worldbody & dexterous hand grasping)
        self_col = False
        if self.kind != "hand":
            for c_idx in range(self.data.ncon):
                con = self.data.contact[c_idx]
                if con.dist < -0.01:
                    b1 = self.model.geom_bodyid[con.geom1]
                    b2 = self.model.geom_bodyid[con.geom2]
                    if b1 == 0 or b2 == 0:
                        continue
                    if abs(int(b1) - int(b2)) > 2:
                        self_col = True
                        break

        reachable = (not jlim_violated) and (not self_col) and (mean_res_m <= 0.25)
        self.last_step_info = EmbodimentSolveStep(
            q=q_clipped,
            reachable=reachable,
            ik_residual_m=mean_res_m,
            joint_limit_violated=jlim_violated,
            self_collision=self_col,
            reason="ok" if reachable else "gated",
        )
        return q_clipped, reachable

    def fk(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        d = mj.MjData(self.model)
        d.qpos[: len(q)] = q
        mj.mj_forward(self.model, d)
        return {
            t["link"]: d.xpos[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, t["link"])].copy()
            for t in self.task_links
        }

    def solve_episode(
        self,
        right_targets: Sequence[Tuple[np.ndarray, np.ndarray]],
        left_targets: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
        grips: Optional[Sequence[float]] = None,
        hand_landmarks_seq: Optional[Sequence[Any]] = None,
        valid_mask: Optional[Sequence[bool]] = None,
        dt: float = 1.0 / 30.0,
        warmup_steps: int = 20,
    ) -> EmbodimentEpisodeReport:
        """Run full-episode retargeting across right/left trajectories and return audited metrics."""
        n = len(right_targets)
        q_traj = np.zeros((n, self.model.nq), dtype=np.float64)
        reach_mask = np.zeros(n, dtype=bool)
        residuals_mm: List[float] = []
        jlim_events = 0
        col_events = 0

        for i in range(n):
            is_valid = True if valid_mask is None else bool(valid_mask[i])
            if not is_valid:
                q_traj[i] = self.cfgm.q.copy()
                reach_mask[i] = False
                continue

            step_targets = [right_targets[i]]
            if len(self.tasks) >= 2:
                if left_targets is not None and i < len(left_targets):
                    step_targets.append(left_targets[i])
                else:
                    p_r, q_r = right_targets[i]
                    step_targets.append((np.array([p_r[0], abs(float(p_r[1])) + 0.18, p_r[2]]), q_r))

            grip_val = float(grips[i]) if (grips is not None and i < len(grips)) else 0.5
            hlm = hand_landmarks_seq[i] if (hand_landmarks_seq is not None and i < len(hand_landmarks_seq)) else None

            # Converge differential IK on the first valid frame
            iters = max(warmup_steps, 80) if i == 0 else 2
            q_out, ok = None, False
            for step_iter in range(iters):
                q_out, ok = self.solve(step_targets, dt=dt, grip=grip_val, hand_landmarks=hlm)
                if i == 0 and self.last_step_info is not None and self.last_step_info.ik_residual_m < 0.005:
                    break

            info = self.last_step_info
            q_traj[i] = q_out
            reach_mask[i] = bool(ok)
            if info is not None:
                if ok:
                    residuals_mm.append(info.ik_residual_m * 1000.0)
                if info.joint_limit_violated:
                    jlim_events += 1
                if info.self_collision:
                    col_events += 1

        n_reach = int(np.sum(reach_mask))
        return EmbodimentEpisodeReport(
            name=self.name,
            kind=self.kind,
            license=self.license,
            num_frames=n,
            reachable_frames=n_reach,
            reachable_pct=100.0 * n_reach / max(n, 1),
            mean_ik_residual_mm=float(np.mean(residuals_mm)) if residuals_mm else 0.0,
            max_ik_residual_mm=float(np.max(residuals_mm)) if residuals_mm else 0.0,
            joint_limit_events=jlim_events,
            self_collision_events=col_events,
            q_trajectory=q_traj,
            reachable_mask=reach_mask,
        )
