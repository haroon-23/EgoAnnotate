"""Hermetic tests for the Stage 9b physics-verification gate (pipeline hook).

The pipeline is built via ``__new__`` with only the attributes the gate needs —
full ``__init__`` would pull every stage. The verifier is faked; the exporter
is real.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hermetic_stubs import scipy_stub_for_import  # noqa: E402

with scipy_stub_for_import():
    from src.pipeline import EgoAnnotatePipeline  # noqa: E402
    from src.datatypes import AnnotatedEpisode  # noqa: E402
    from src.dataset_exporter import DatasetExporter, ExporterConfig  # noqa: E402
    from src.retargeting.episode_verifier import (  # noqa: E402
        VerificationReport,
        CheckResult,
    )


class FakeVerifier:
    def __init__(self, passed: bool):
        self._passed = passed
        self.calls = []

    def verify_episode(self, hdf5_path, episode_id=None, object_position=None,
                       urdf_path=None):
        self.calls.append(
            {"hdf5": hdf5_path, "episode_id": episode_id, "urdf_path": urdf_path})
        return VerificationReport(
            episode_id=episode_id or "ep001", passed=self._passed,
            checks=[CheckResult("fake", self._passed, 1.0, 0.0)],
            windows=[], metrics={})


def _episode():
    return AnnotatedEpisode(
        episode_id="ep001", video_path="v.mp4", task_description="pick up the cup",
        frames=[], segments=[], num_frames=10, duration_seconds=1.0)


def _pipeline(tmp_path, enabled, verifier):
    p = EgoAnnotatePipeline.__new__(EgoAnnotatePipeline)
    p.physics_verify_enabled = enabled
    p.episode_verifier = verifier
    p.dataset_exporter = DatasetExporter(
        ExporterConfig(output_dir=str(tmp_path / "out")))
    p.retargeter = None
    return p


def _write_hdf5(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        f.create_group("ep001")


def test_gate_disabled_never_touches_verifier(tmp_path):
    p = _pipeline(tmp_path, enabled=False, verifier=FakeVerifier(passed=True))
    ep = _episode()
    p._maybe_verify_physics(ep)  # must not raise
    assert p.episode_verifier.calls == []
    assert ep.physics_verified is False
    assert ep.physics_report_path is None
    assert not (Path(p.dataset_exporter.output_path) / "ep001").exists()


def test_gate_pass_sets_label_and_persists(tmp_path):
    p = _pipeline(tmp_path, enabled=True, verifier=FakeVerifier(passed=True))
    ep = _episode()
    _write_hdf5(Path(p.dataset_exporter.output_path) / "ep001" / "episode_rlds.hdf5")
    (Path(p.dataset_exporter.output_path) / "ep001" / "metadata.json").write_text("{}")
    p._maybe_verify_physics(ep)

    assert len(p.episode_verifier.calls) == 1
    call = p.episode_verifier.calls[0]
    assert call["episode_id"] == "ep001"
    assert call["hdf5"].endswith("episode_rlds.hdf5")

    assert ep.physics_verified is True
    assert ep.physics_report_path is not None
    report = json.loads(Path(ep.physics_report_path).read_text())
    assert report["passed"] is True

    meta = json.loads(
        (Path(p.dataset_exporter.output_path) / "ep001" / "metadata.json").read_text())
    assert meta["physics_verified"] is True
    with h5py.File(
            str(Path(p.dataset_exporter.output_path) / "ep001" / "episode_rlds.hdf5"), "r") as f:
        assert bool(f.attrs["physics_verified"]) is True


def test_gate_fail_keeps_episode_and_withholds_label(tmp_path):
    p = _pipeline(tmp_path, enabled=True, verifier=FakeVerifier(passed=False))
    ep = _episode()
    _write_hdf5(Path(p.dataset_exporter.output_path) / "ep001" / "episode_rlds.hdf5")
    p._maybe_verify_physics(ep)  # must not raise: the gate never drops episodes

    assert ep.physics_verified is False
    assert ep.physics_report_path is not None  # report still written
    report = json.loads(Path(ep.physics_report_path).read_text())
    assert report["passed"] is False


def test_gate_verifier_missing_keeps_episode(tmp_path):
    p = _pipeline(tmp_path, enabled=True, verifier=None)
    ep = _episode()
    p._maybe_verify_physics(ep)  # must not raise
    assert ep.physics_verified is False


def test_gate_verifier_error_keeps_episode(tmp_path):
    class Boom:
        def verify_episode(self, *a, **k):
            raise RuntimeError("mujoco exploded")

    p = _pipeline(tmp_path, enabled=True, verifier=Boom())
    ep = _episode()
    p._maybe_verify_physics(ep)  # the gate catches everything
    assert ep.physics_verified is False


def test_persist_physics_verification_real_exporter(tmp_path):
    exp = DatasetExporter(ExporterConfig(output_dir=str(tmp_path / "out")))
    ep_dir = Path(exp.output_path) / "ep001"
    ep_dir.mkdir(parents=True)
    (ep_dir / "metadata.json").write_text(json.dumps({"episode_id": "ep001"}))
    _write_hdf5(ep_dir / "episode_rlds.hdf5")

    ep = _episode()
    ep.physics_verified = True
    ep.physics_report_path = str(ep_dir / "physics_verification.json")
    exp.persist_physics_verification(ep, ep_dir)

    meta = json.loads((ep_dir / "metadata.json").read_text())
    assert meta["physics_verified"] is True
    assert meta["physics_report_path"] == str(ep_dir / "physics_verification.json")
    with h5py.File(str(ep_dir / "episode_rlds.hdf5"), "r") as f:
        assert bool(f.attrs["physics_verified"]) is True
        assert f.attrs["physics_report_path"] == str(ep_dir / "physics_verification.json")


def test_episode_roundtrip_carries_physics_fields():
    ep = _episode()
    ep.physics_verified = True
    ep.physics_report_path = "/tmp/physics_verification.json"
    ep2 = AnnotatedEpisode.from_dict(ep.to_dict())
    assert ep2.physics_verified is True
    assert ep2.physics_report_path == "/tmp/physics_verification.json"
    # Old dicts without the keys still load (backward compatible).
    d = ep.to_dict()
    del d["physics_verified"]
    del d["physics_report_path"]
    ep3 = AnnotatedEpisode.from_dict(d)
    assert ep3.physics_verified is False and ep3.physics_report_path is None
