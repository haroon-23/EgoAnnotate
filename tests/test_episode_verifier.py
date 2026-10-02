"""Hermetic tests for src/retargeting/episode_verifier.py.

mujoco is scripted with a fake module (sys.modules injection): perfect
position-servo tracking, a constant EE pose during M1, and a weld that lifts
the object rigidly with the EE while active. This exercises the verifier's
gate logic — not real physics.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hermetic_stubs import (  # noqa: E402
    scipy_stub_for_import,
    blocked_modules,
    fake_module,
)

with scipy_stub_for_import():
    import h5py  # noqa: E402  (real; installed in the sandbox)

    from src.retargeting.episode_verifier import (  # noqa: E402
        EpisodeVerifier,
        VerifierConfig,
        build_verification_scene,
        frame_has_self_collision,
    )

EE_ID, OBJ_ID, TABLE_ID, WELD_ID = 10, 11, 12, 0
EE_POS = np.array([0.5, 0.0, 0.31])


def _make_fake_mujoco():
    """Build a fake mujoco module with scripted step semantics."""
    mj = types.ModuleType("mujoco")

    class _Obj:
        mjOBJ_JOINT = 1
        mjOBJ_BODY = 2
        mjOBJ_EQUALITY = 3
        mjOBJ_ACTUATOR = 4

    class _Joint:
        mjJNT_FREE = 0

    mj.mjtObj = _Obj()
    mj.mjtJoint = _Joint()
    _kind = {1: "joint", 2: "body", 3: "equality", 4: "actuator"}

    class FakeModel:
        def __init__(self):
            self.nq = 16
            self.nu = 9
            self.njnt = 10
            self.nbody = 13
            self.neq = 1
            self.jnt_type = np.array([1] * 9 + [0])  # [9] is FREE
            self.jnt_qposadr = np.arange(10)
            self.opt = types.SimpleNamespace(timestep=0.00416667)
            self.eq_data = np.zeros((1, 7))
            self.geom_bodyid = np.zeros(30, dtype=int)
            self._ids = {}
            for i in range(7):
                self._ids[("joint", f"panda_joint{i+1}")] = i
                self._ids[("actuator", f"ver_q{i+1}")] = i
            for i in range(2):
                self._ids[("joint", f"panda_finger_joint{i+1}")] = 7 + i
                self._ids[("actuator", f"ver_f{i+1}")] = 7 + i
            self._ids[("body", "panda_link8")] = EE_ID
            self._ids[("body", "obj")] = OBJ_ID
            self._ids[("body", "table")] = TABLE_ID
            self._ids[("equality", "ver_grasp_weld")] = WELD_ID

        @classmethod
        def from_xml_path(cls, path):
            return cls()

    class FakeData:
        def __init__(self, model):
            self._model = model
            self.qpos = np.zeros(model.nq)
            self.ctrl = np.zeros(model.nu)
            self.xpos = np.zeros((model.nbody, 3))
            self.xquat = np.zeros((model.nbody, 4))
            self.xquat[:, 0] = 1.0
            self.eq_active = np.zeros(model.neq, dtype=int)
            self.contact = np.zeros(0, dtype=[("geom1", "i4"), ("geom2", "i4")])
            self.ncon = 0
            self._weld_was_active = False
            self._rel0 = np.zeros(3)
            self._home()

        def _home(self):
            self.xpos[EE_ID] = EE_POS.copy()
            self.xpos[OBJ_ID] = EE_POS.copy()

    def mj_name2id(model, objtype, name):
        return model._ids.get((_kind[objtype], name), -1)

    def mj_saveLastXML(path, model):
        pass  # cache is pre-created by the test

    def mj_resetData(model, data):
        data.qpos[:] = 0.0
        data.ctrl[:] = 0.0
        data.eq_active[:] = 0
        data._weld_was_active = False
        data._rel0[:] = 0.0
        data._home()

    def mj_forward(model, data):
        pass

    def mj_step(model, data):
        # Perfect position-servo tracking.
        data.qpos[:7] = data.ctrl[:7]
        data.qpos[7:9] = data.ctrl[7:9]
        active = bool(data.eq_active[WELD_ID])
        if active and not data._weld_was_active:
            data._rel0 = (data.xpos[OBJ_ID] - data.xpos[EE_ID]).copy()
        data._weld_was_active = active
        if active:
            # Welded object rises rigidly with the EE.
            data.xpos[EE_ID][2] += 0.002
            data.xpos[OBJ_ID] = data.xpos[EE_ID] + data._rel0

    mj.MjModel = FakeModel
    mj.MjData = FakeData
    mj.mj_name2id = mj_name2id
    mj.mj_saveLastXML = mj_saveLastXML
    mj.mj_resetData = mj_resetData
    mj.mj_forward = mj_forward
    mj.mj_step = mj_step
    return mj


CACHE_XML = """<mujoco>
  <worldbody>
    <body name="panda_link8">
      <joint name="panda_joint1"/><joint name="panda_joint2"/><joint name="panda_joint3"/>
      <joint name="panda_joint4"/><joint name="panda_joint5"/><joint name="panda_joint6"/>
      <joint name="panda_joint7"/>
      <joint name="panda_finger_joint1"/><joint name="panda_finger_joint2"/>
    </body>
  </worldbody>
</mujoco>
"""


def _write_cache(tmp_path):
    urdf = tmp_path / "panda.urdf"
    urdf.write_text("<robot/>")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "panda_verifier_cache.xml").write_text(CACHE_XML)
    return str(urdf), str(cache_dir)


def _write_hdf5(path, n=20, closed=(5, 15), legacy_reachable=False):
    q = np.full((n, 7), 0.3)  # arm clearly away from home
    g = np.full((n, 1), 0.08)
    g[closed[0]:closed[1], 0] = 0.01
    r = np.ones(n, dtype=bool)
    with h5py.File(path, "w") as f:
        ep = f.create_group("ep001")
        steps = ep.create_group("steps")
        obs = steps.create_group("observation")
        obs.create_dataset("robot_joint_angles", data=q)
        obs.create_dataset("robot_gripper_opening_m", data=g)
        if legacy_reachable:
            steps.create_dataset("robot_reachable", data=r)
        else:
            obs.create_dataset("robot_reachable", data=r)


def test_mujoco_blocked_is_available_false_and_verify_returns_failed_report(tmp_path):
    with blocked_modules("mujoco"):
        v = EpisodeVerifier()
        assert v.is_available() is False
        rep = v.verify_episode(str(tmp_path / "nope.hdf5"))
        assert rep.passed is False
        assert rep.checks[0].name == "mujoco_available"


def test_build_scene_resolves_ids_and_sanity(tmp_path):
    urdf, cache_dir = _write_cache(tmp_path)
    with fake_module("mujoco", _make_fake_mujoco()):
        model, info = build_verification_scene(urdf, cache_dir=cache_dir)
    assert info.ee_body_id == EE_ID
    assert info.obj_body_id == OBJ_ID
    assert info.table_body_id == TABLE_ID
    assert info.weld_eq_id == WELD_ID
    assert info.substeps == 8
    assert list(info.arm_q) == list(range(7))
    assert list(info.arm_ctrl) == list(range(7))
    # The spliced scene XML passed its sanity checks (no raise above); spot-check.
    scene = next(Path(cache_dir).glob("*_verifier_scene.xml"))
    txt = scene.read_text()
    assert txt.count("<equality>") == 1 and "ver_grasp_weld" in txt
    assert 'timestep="0.00416667"' in txt


def test_build_scene_missing_urdf_raises(tmp_path):
    with fake_module("mujoco", _make_fake_mujoco()):
        try:
            build_verification_scene(str(tmp_path / "missing.urdf"), cache_dir=str(tmp_path))
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")


def _run(tmp_path, cfg=None, **kw):
    urdf, cache_dir = _write_cache(tmp_path)
    hdf5 = tmp_path / "ep.hdf5"
    _write_hdf5(str(hdf5), **kw)
    with fake_module("mujoco", _make_fake_mujoco()):
        v = EpisodeVerifier(cfg or VerifierConfig(urdf_path=urdf))
        # Point the verifier's cache at our pre-created cache dir via monkeypatching
        # build_verification_scene: easiest is to copy cache into the default location.
        import tempfile
        default = Path(tempfile.gettempdir()) / "egoannotate_physics"
        default.mkdir(parents=True, exist_ok=True)
        (default / "panda_verifier_cache.xml").write_text(CACHE_XML)
        rep = v.verify_episode(str(hdf5), episode_id="ep001",
                               object_position=np.array([0.5, 0.0, 0.31]))
    return rep


def test_verify_episode_passes_on_synthetic(tmp_path):
    rep = _run(tmp_path)
    names = {c.name: c for c in rep.checks}
    assert rep.passed is True
    assert names["m1_tracking_err_rad"].passed is True
    assert names["m1_tracking_err_rad"].value == 0.0
    assert names["reachable_pct"].value == 100.0
    assert names["self_collision_frames"].value == 0
    assert names["grasp_windows_found"].value == 1
    assert names["window_0_weld_held"].passed is True
    assert names["window_0_lift_m"].passed is True
    assert names["window_0_lift_m"].value >= 0.02
    assert names["window_0_near_ee_m"].passed is True
    assert len(rep.windows) == 1
    w = rep.windows[0]
    assert (w.start_idx, w.end_idx) == (5, 15) and w.weld_held is True
    assert rep.metrics["best_lift_m"] >= 0.02


def test_threshold_edge_strict_less_than(tmp_path):
    # err == 0.0 with gate == 0.0 must FAIL (strict <).
    rep = _run(tmp_path, cfg=None)
    assert rep.passed is True
    urdf, _ = _write_cache(tmp_path)
    cfg = VerifierConfig(urdf_path=urdf, tracking_err_rad_max=0.0)
    rep2 = _run(tmp_path, cfg=cfg)
    assert rep2.passed is False
    by_name = {c.name: c for c in rep2.checks}
    assert by_name["m1_tracking_err_rad"].passed is False
    assert by_name["reachable_pct"].passed is True  # AND logic: others still pass


def test_and_logic_lift_gate_fails_report(tmp_path):
    urdf, _ = _write_cache(tmp_path)
    rep = _run(tmp_path, cfg=VerifierConfig(urdf_path=urdf, lift_min_m=999.0))
    assert rep.passed is False
    by_name = {c.name: c for c in rep.checks}
    assert by_name["window_0_lift_m"].passed is False
    assert by_name["m1_tracking_err_rad"].passed is True


def test_missing_hdf5_raises_loud(tmp_path):
    urdf, _ = _write_cache(tmp_path)
    with fake_module("mujoco", _make_fake_mujoco()):
        v = EpisodeVerifier(VerifierConfig(urdf_path=urdf))
        try:
            v.verify_episode(str(tmp_path / "missing.hdf5"))
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError, never a silent pass")


def test_legacy_reachable_key_still_reads(tmp_path):
    rep = _run(tmp_path, legacy_reachable=True)
    assert rep.passed is True


def test_object_position_derived_when_none(tmp_path):
    urdf, cache_dir = _write_cache(tmp_path)
    hdf5 = tmp_path / "ep.hdf5"
    _write_hdf5(str(hdf5))
    with fake_module("mujoco", _make_fake_mujoco()):
        import tempfile
        default = Path(tempfile.gettempdir()) / "egoannotate_physics"
        default.mkdir(parents=True, exist_ok=True)
        (default / "panda_verifier_cache.xml").write_text(CACHE_XML)
        v = EpisodeVerifier(VerifierConfig(urdf_path=urdf))
        rep = v.verify_episode(str(hdf5), episode_id="ep001", object_position=None)
    assert rep.passed is True
    assert len(rep.windows) == 1


def test_save_json_roundtrip(tmp_path):
    rep = _run(tmp_path)
    out = tmp_path / "physics_verification.json"
    rep.save_json(str(out))
    payload = json.loads(out.read_text())
    assert payload["passed"] is True
    assert payload["episode_id"] == "ep001"
    assert len(payload["checks"]) == len(rep.checks)
    assert payload["windows"][0]["weld_held"] is True
    assert rep.report_path == str(out)


def test_frame_has_self_collision_helper():
    mj = _make_fake_mujoco()
    model = mj.MjModel.from_xml_path("x")
    data = mj.MjData(model)
    robot = {5, 7}
    model.geom_bodyid[0] = 5
    model.geom_bodyid[1] = 7
    data.contact = np.zeros(1, dtype=[("geom1", "i4"), ("geom2", "i4")])
    data.contact[0] = (0, 1)  # geom 0 (body 5) vs geom 1 (body 7)
    data.ncon = 1
    assert frame_has_self_collision(model, data, robot) is True
    model.geom_bodyid[1] = OBJ_ID  # robot vs object -> not self-collision
    assert frame_has_self_collision(model, data, robot) is False
    data.ncon = 0
    assert frame_has_self_collision(model, data, robot) is False
