"""MuJoCo + mink: Physics-verified IK with joint limits enforced.
Replaces PyBullet with proper velocity/limit constraints."""
import mujoco
import mink
import numpy as np
from pathlib import Path

class MuJoCoIKSolver:
    def __init__(self, urdf_path: str, ee_link_name: str = "panda_hand"):
        """
        Initialize MuJoCo model and mink IK solver.
        
        Args:
            urdf_path: Path to robot URDF
            ee_link_name: End-effector link name
        """
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        self.data = mujoco.MjData(self.model)
        
        # Get end-effector body ID
        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, ee_link_name
        )
        
        # Create mink IK task
        self.ik_task = mink.FrameTask(
            frame_name=ee_link_name,
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-3
        )
        
        # Get joint limits from URDF
        self.joint_limits = self._extract_joint_limits()
        
        # Velocity limits (from URDF or defaults)
        self.velocity_limits = np.array([2.17, 2.17, 2.17, 2.17, 2.61, 2.61, 2.61])
        
        # Configuration
        self.configuration = mink.Configuration(self.model)
        
    def _extract_joint_limits(self) -> np.ndarray:
        """Extract joint position limits from MuJoCo model."""
        n_joints = self.model.nq
        limits = np.zeros((n_joints, 2))
        
        for i in range(n_joints):
            limits[i, 0] = self.model.jnt_range[i, 0]
            limits[i, 1] = self.model.jnt_range[i, 1]
        
        return limits
    
    def solve_ik(self, 
                 target_pos: np.ndarray, 
                 target_quat: np.ndarray,
                 q_prev: np.ndarray = None,
                 dt: float = 1/30.0) -> dict:
        """
        Solve IK with joint position and velocity limits.
        
        Args:
            target_pos: [x, y, z] in meters (world frame)
            target_quat: [w, x, y, z] quaternion
            q_prev: Previous joint configuration (for warm-start)
            dt: Time step for velocity limit
        
        Returns:
            {
                "joint_angles": np.ndarray,
                "reachable": bool,
                "residual": float,
                "method": str
            }
        """
        # Warm-start from previous solution
        if q_prev is not None:
            self.configuration.update(q_prev)
        else:
            self.configuration.update(np.zeros(self.model.nq))
        
        # Set target pose
        from scipy.spatial.transform import Rotation
        quat_xyzw = [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
        rot_matrix = Rotation.from_quat(quat_xyzw).as_matrix()

        target_pose = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(rot_matrix),
            target_pos
        )
        
        # Solve IK with limits
        try:
            # Use mink's limit-aware solver
            velocity = mink.solve_ik(
                self.configuration,
                [self.ik_task],
                self.model.opt.timestep,
                solver="quadprog",
                limits={
                    "velocity": self.velocity_limits,
                    "configuration": self.joint_limits
                }
            )
            
            # Integrate velocity to get new configuration
            q_new = self.configuration.integrate(velocity, dt)
            
            # Update configuration
            self.configuration.update(q_new)
            
            # Compute residual
            current_pose = self.configuration.get_transform_frame_to_world(
                self.ee_body_id, "body"
            )
            residual = np.linalg.norm(current_pose.translation() - target_pos)
            
            # Check if solution is valid (within limits)
            within_limits = np.all(q_new >= self.joint_limits[:, 0]) and \
                           np.all(q_new <= self.joint_limits[:, 1])
            
            return {
                "joint_angles": q_new,
                "reachable": within_limits and residual < 0.01,  # 1cm tolerance
                "residual": float(residual),
                "method": "mink_limit_enforced"
            }
            
        except Exception as e:
            # IK failed - return fallback
            return {
                "joint_angles": q_prev if q_prev is not None else np.zeros(self.model.nq),
                "reachable": False,
                "residual": float('inf'),
                "method": "fallback"
            }
    
    def solve_sequence(self, targets: list, dt: float = 1/30.0) -> list:
        """
        Solve IK for a sequence of targets with warm-starting.
        
        Args:
            targets: List of {"position": [x,y,z], "quaternion": [w,x,y,z]}
            dt: Time step
        
        Returns:
            List of IK results
        """
        results = []
        q_prev = None
        
        for target in targets:
            result = self.solve_ik(
                target_pos=np.array(target["position"]),
                target_quat=np.array(target["quaternion"]),
                q_prev=q_prev,
                dt=dt
            )
            results.append(result)
            
            if result["reachable"]:
                q_prev = result["joint_angles"]
        
        return results
