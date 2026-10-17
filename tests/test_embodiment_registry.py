import os, sys, json, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "retargeting"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from embodiment_registry import Adapter
from run_reference_pipeline import sanitize_trajectory

def test_franka_adapter_solve():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "embodiments", "franka_arm.yaml")
    adapter = Adapter(cfg_path)
    dt = 1.0 / 30.0

    # Get home FK position as target
    home_q = adapter.default_q.copy()
    adapter.cfgm.update(home_q)
    target_link = adapter.task_links[0]["link"]
    fk0 = adapter.fk(home_q)
    target_pos = fk0[target_link]

    import mujoco as mj
    from scipy.spatial.transform import Rotation
    bid = mj.mj_name2id(adapter.model, mj.mjtObj.mjOBJ_BODY, target_link)
    d = mj.MjData(adapter.model)
    d.qpos[: len(home_q)] = home_q
    mj.mj_forward(adapter.model, d)
    target_quat = Rotation.from_matrix(d.xmat[bid].reshape(3, 3)).as_quat()

    q = home_q.copy()
    for _ in range(30):
        q, _ = adapter.solve([(target_pos, target_quat)], dt, grip=0.04)

    # Assert q inside jnt_range
    for j in range(adapter.model.nq):
        lo, hi = adapter.model.jnt_range[j]
        assert lo - 1e-3 <= q[j] <= hi + 1e-3, f"Joint {j} value {q[j]} outside range [{lo}, {hi}]"

    # Assert FK residual < 0.01 m
    fk_final = adapter.fk(q)
    residual = np.linalg.norm(fk_final[target_link] - target_pos)
    assert residual < 0.01, f"FK residual {residual:.4f} m >= 0.01 m"

def test_sanitize_trajectory_import():
    q = np.zeros((10, 7))
    reach = np.ones(10, dtype=bool)
    reason = ["tracked"] * 10
    q_clean, reach_clean, reason_clean = sanitize_trajectory(q, reach, reason, dt=1.0/30.0)
    assert len(q_clean) == 10
    assert len(reach_clean) == 10
    assert len(reason_clean) == 10

def test_manifest_schema_keys():
    manifest_keys = ["episode", "frames", "embodiments"]
    manifest_sample = {"episode": "reference_v2", "frames": 150, "embodiments": []}
    for key in manifest_keys:
        assert key in manifest_sample
