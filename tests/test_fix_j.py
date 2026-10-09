"""Fix J regression tests.

Run #3 produced a fresh retargeting line (213/729 reachable) but a
physics_verification.json byte-identical to Run #2. These tests pin the three
Fix J changes that make that failure mode impossible to miss and raise the
reachable ceiling:

1. A colliding IK attempt pays a selection-score penalty, so any clean
   attempt beats any table- or self-colliding attempt. The returned residual
   remains the winning attempt's true geometric residual.
2. Parquet frame export is additive: frame_annotations.json (the file the
   RLDS exporter re-reads from disk) is ALWAYS refreshed.
3. The pipeline physics verifier fails closed when episode_rlds.hdf5 is
   missing, predates the export, or disagrees with the in-memory episode's
   robot_reachable count.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tests.hermetic_stubs import scipy_stub_for_import

with scipy_stub_for_import():
    from src.dataset_exporter import DatasetExporter, ExporterConfig
    from src.datatypes import AnnotatedEpisode, AnnotationFrame
    from src.pipeline import EgoAnnotatePipeline
    from src.retargeting.ik_solver import IKSolver, IKSolverConfig
    from src.retargeting.retargeter import RetargetingConfig


# ---------------------------------------------------------------------------
# IK collision penalty
# ---------------------------------------------------------------------------

class _FakePB:
    """Deterministic PyBullet stand-in for one two-joint IK candidate set."""

    ROBOT_ID = 3
    TABLE_ID = 12

    def __init__(self, candidates, positions, colliding, collision_kind):
        self.candidates = [tuple(candidate) for candidate in candidates]
        self.positions = {
            tuple(candidate): tuple(position)
            for candidate, position in zip(candidates, positions)
        }
        self.colliding = tuple(colliding)
        self.collision_kind = collision_kind
        self.current = self.candidates[0]
        self._state = {}
        self._ik_calls = 0

    def calculateInverseKinematics(self, *args, **kwargs):
        candidate = self.candidates[self._ik_calls]
        self._ik_calls += 1
        return candidate

    def resetJointState(self, bodyUniqueId, jointIndex, targetValue, physicsClientId=None):
        self._state[int(jointIndex)] = float(targetValue)
        if len(self._state) == 2:
            self.current = tuple(self._state[index] for index in sorted(self._state))

    def getLinkState(self, *args, **kwargs):
        position = self.positions[self.current]
        return (None, None, None, None, position, (0.0, 0.0, 0.0, 1.0))

    def getContactPoints(self, bodyA, bodyB, physicsClientId=None):
        if self.current != self.colliding:
            return []
        contact = (0,) * 8 + (-0.05,)
        if self.collision_kind == "table" and bodyB == self.TABLE_ID:
            return [contact]
        if (
            self.collision_kind == "self"
            and bodyA == self.ROBOT_ID
            and bodyB == self.ROBOT_ID
        ):
            return [contact]
        return []


def _make_ik_solver(*, collision_kind="table", collision_penalty=1.0):
    colliding = (0.10, 0.20)
    clean = (0.30, 0.40)
    fake = _FakePB(
        candidates=[colliding, clean],
        positions=[(0.50, 0.0, 0.401), (0.50, 0.0, 0.404)],
        colliding=colliding,
        collision_kind=collision_kind,
    )
    kinematics = SimpleNamespace(
        arm_joint_indices=[0, 1],
        active_joints=[0, 1],
        rest_poses=np.zeros(2),
        lower_limits=np.array([-3.0, -3.0]),
        upper_limits=np.array([3.0, 3.0]),
        end_effector_index=2,
    )
    config = IKSolverConfig(
        num_attempts=2,
        collision_penalty=collision_penalty,
    )
    solver = IKSolver(kinematics, config)
    solver._pb = fake
    solver._client_id = 999
    solver._robot_id = _FakePB.ROBOT_ID
    solver._table_id = _FakePB.TABLE_ID
    return solver, colliding, clean


def _solve_two_attempts(solver):
    return solver._solve_ik_single(
        np.array([0.50, 0.0, 0.40]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        q_prev_full=None,
        q_cont_ref=None,
    )


def test_fixj_collision_penalty_defaults_and_franka_yaml():
    assert IKSolverConfig().collision_penalty == 1.0

    config_path = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "retargeting_franka.yaml"
    )
    retargeting_config = RetargetingConfig.from_yaml(config_path)
    assert retargeting_config.ik_solver.collision_penalty == 1.0


def test_fixj_table_colliding_attempt_loses_despite_better_residual():
    solver, colliding, clean = _make_ik_solver(collision_penalty=1.0)

    angles, residual, _ = _solve_two_attempts(solver)

    # The colliding attempt has residual 0.001 m; the clean attempt has
    # 0.004 m. The penalty affects selection only, not the returned residual.
    assert tuple(angles) == clean
    assert tuple(angles) != colliding
    assert residual == pytest.approx(0.004)


def test_fixj_self_colliding_attempt_loses_despite_better_residual():
    solver, colliding, clean = _make_ik_solver(
        collision_kind="self",
        collision_penalty=1.0,
    )

    angles, residual, _ = _solve_two_attempts(solver)

    assert tuple(angles) == clean
    assert tuple(angles) != colliding
    assert residual == pytest.approx(0.004)


def test_fixj_zero_collision_penalty_preserves_residual_only_selection():
    solver, colliding, clean = _make_ik_solver(collision_penalty=0.0)

    angles, residual, _ = _solve_two_attempts(solver)

    assert tuple(angles) == colliding
    assert tuple(angles) != clean
    assert residual == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Exporter freshness
# ---------------------------------------------------------------------------

def _make_episode(episode_id="fixj_export", reachable=(True, False)):
    frames = [
        AnnotationFrame(
            frame_idx=index,
            timestamp=index / 30.0,
            image_path=f"missing_frame_{index}.png",
            robot_joint_angles=[0.1, -0.2],
            robot_gripper_opening_m=0.04,
            robot_reachable=bool(flag),
        )
        for index, flag in enumerate(reachable)
    ]
    return AnnotatedEpisode(
        episode_id=episode_id,
        video_path="missing_video.mp4",
        task_description="Fix J exporter regression",
        frames=frames,
        segments=[],
        num_frames=len(frames),
        duration_seconds=len(frames) / 30.0,
    )


def _patch_parquet_writer(monkeypatch):
    def fake_to_parquet(self, path_or_buf=None, *args, **kwargs):
        Path(path_or_buf).write_bytes(b"PARQUET")

    monkeypatch.setattr("pandas.DataFrame.to_parquet", fake_to_parquet)


def _read_reachable(hdf5_path, episode_id):
    with h5py.File(hdf5_path, "r") as hf:
        return hf[episode_id]["steps"]["observation"]["robot_reachable"][()].tolist()


def test_fixj_parquet_export_still_writes_frame_annotations_json(
    tmp_path, monkeypatch
):
    _patch_parquet_writer(monkeypatch)
    exporter = DatasetExporter(
        ExporterConfig(output_dir=str(tmp_path), format="parquet")
    )
    episode = _make_episode()
    episode_dir = Path(exporter.output_path) / episode.episode_id
    episode_dir.mkdir(parents=True)

    exporter._export_frames(episode, episode_dir)

    assert (episode_dir / "frame_annotations.parquet").exists()
    annotations_path = episode_dir / "frame_annotations.json"
    assert annotations_path.exists()
    annotations = json.loads(annotations_path.read_text())
    assert [row["robot_reachable"] for row in annotations] == [True, False]


def test_fixj_reexport_replaces_stale_json_before_rlds_export(
    tmp_path, monkeypatch
):
    _patch_parquet_writer(monkeypatch)
    exporter = DatasetExporter(
        ExporterConfig(output_dir=str(tmp_path), format="parquet")
    )
    episode = _make_episode(reachable=(True, False))
    episode_dir = Path(exporter.output_path) / episode.episode_id
    episode_dir.mkdir(parents=True)

    exporter._export_frames(episode, episode_dir)
    exporter._export_metadata(episode, episode_dir)
    exporter._export_rlds(episode, episode_dir)
    hdf5_path = episode_dir / "episode_rlds.hdf5"
    assert _read_reachable(hdf5_path, episode.episode_id) == [True, False]

    # Mutate the same in-memory episode, as a later run over the same episode
    # id would. Re-export must write the CURRENT annotations, not reuse the
    # previous run's JSON underneath a fresh-looking HDF5.
    episode.frames[1].robot_reachable = True
    exporter._export_frames(episode, episode_dir)
    exporter._export_rlds(episode, episode_dir)

    assert _read_reachable(hdf5_path, episode.episode_id) == [True, True]


# ---------------------------------------------------------------------------
# Pipeline fail-closed freshness guard
# ---------------------------------------------------------------------------

class _FakeVerifier:
    def __init__(self):
        self.calls = []

    def verify_episode(self, hdf5_path, episode_id=None, **_kwargs):
        from src.retargeting.episode_verifier import CheckResult, VerificationReport

        self.calls.append((hdf5_path, episode_id))
        return VerificationReport(
            episode_id=episode_id or "fixj_pipeline",
            passed=True,
            checks=[CheckResult("fake", True, 1.0, 0.0)],
            windows=[],
            metrics={"fake": 1.0},
        )


def _make_pipeline(tmp_path, verifier):
    pipeline = EgoAnnotatePipeline.__new__(EgoAnnotatePipeline)
    pipeline.physics_verify_enabled = True
    pipeline.episode_verifier = verifier
    pipeline.dataset_exporter = DatasetExporter(
        ExporterConfig(output_dir=str(tmp_path / "out"))
    )
    pipeline.retargeter = None
    return pipeline


def _export_episode(exporter, episode):
    episode_dir = Path(exporter.output_path) / episode.episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    exporter._export_frames(episode, episode_dir)
    exporter._export_metadata(episode, episode_dir)
    exporter._export_rlds(episode, episode_dir)
    return episode_dir, episode_dir / "episode_rlds.hdf5"


def _read_physics_report(episode_dir):
    return json.loads((episode_dir / "physics_verification.json").read_text())


def _freshness_check(report):
    return {check["name"]: check for check in report["checks"]}["export_freshness"]


def test_fixj_missing_hdf5_blocks_verifier_and_fails_closed(tmp_path):
    verifier = _FakeVerifier()
    pipeline = _make_pipeline(tmp_path, verifier)
    episode = _make_episode("fixj_pipeline")
    episode_dir = Path(pipeline.dataset_exporter.output_path) / episode.episode_id
    episode_dir.mkdir(parents=True)

    pipeline._maybe_verify_physics(episode, not_before=time.time())

    assert verifier.calls == []
    assert episode.physics_verified is False
    report = _read_physics_report(episode_dir)
    assert report["passed"] is False
    check = _freshness_check(report)
    assert check["passed"] is False
    assert "missing after export" in check["detail"]


def test_fixj_stale_hdf5_mtime_blocks_verifier_and_fails_closed(tmp_path):
    verifier = _FakeVerifier()
    pipeline = _make_pipeline(tmp_path, verifier)
    episode = _make_episode("fixj_pipeline")
    episode_dir, hdf5_path = _export_episode(
        pipeline.dataset_exporter, episode
    )

    not_before = time.time() + 5.0
    old_time = not_before - 10.0
    os.utime(hdf5_path, (old_time, old_time))

    pipeline._maybe_verify_physics(episode, not_before=not_before)

    assert verifier.calls == []
    assert episode.physics_verified is False
    report = _read_physics_report(episode_dir)
    assert report["passed"] is False
    check = _freshness_check(report)
    assert check["passed"] is False
    assert "predates this run's export" in check["detail"]


def test_fixj_reachable_count_mismatch_blocks_verifier(tmp_path):
    verifier = _FakeVerifier()
    pipeline = _make_pipeline(tmp_path, verifier)
    episode = _make_episode("fixj_pipeline", reachable=(True, False))
    export_started = time.time()
    episode_dir, _ = _export_episode(pipeline.dataset_exporter, episode)

    # The HDF5 is fresh by mtime, but the in-memory episode now disagrees
    # with it — exactly the signature of a stale export underneath.
    episode.frames[1].robot_reachable = True

    pipeline._maybe_verify_physics(episode, not_before=export_started)

    assert verifier.calls == []
    assert episode.physics_verified is False
    report = _read_physics_report(episode_dir)
    assert report["passed"] is False
    check = _freshness_check(report)
    assert check["passed"] is False
    assert "robot_reachable count mismatch" in check["detail"]


def test_fixj_fresh_hdf5_reaches_verifier(tmp_path):
    verifier = _FakeVerifier()
    pipeline = _make_pipeline(tmp_path, verifier)
    episode = _make_episode("fixj_pipeline", reachable=(True, False))

    export_started = time.time()
    episode_dir, hdf5_path = _export_episode(
        pipeline.dataset_exporter, episode
    )

    pipeline._maybe_verify_physics(episode, not_before=export_started)

    assert verifier.calls == [(str(hdf5_path), episode.episode_id)]
    assert episode.physics_verified is True
    assert _read_physics_report(episode_dir)["passed"] is True


def test_fixj_not_before_omitted_preserves_direct_call_behavior(tmp_path):
    verifier = _FakeVerifier()
    pipeline = _make_pipeline(tmp_path, verifier)
    episode = _make_episode("fixj_pipeline", reachable=(True, False))
    episode_dir, hdf5_path = _export_episode(
        pipeline.dataset_exporter, episode
    )

    pipeline._maybe_verify_physics(episode)

    assert verifier.calls == [(str(hdf5_path), episode.episode_id)]
    assert episode.physics_verified is True
    assert _read_physics_report(episode_dir)["passed"] is True
