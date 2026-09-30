"""Test the open-source perception stack (Phase-C hardened components).

The old ``test_end_to_end_pipeline`` mocked ``run_pipeline`` itself (mock
theater) and ``run_pipeline`` has been deleted as dead code — both are gone.
What remains are real component tests using fakes for weights/model objects.
"""
import os
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.perception.detector as det_mod
import src.perception.depth as depth_mod
from src.perception.detector import Detector2DConfig, GroundingDinoDetector
from src.perception.depth import DepthConfig, UniDepthEstimator

# NOTE: MuJoCoIKSolver is imported lazily inside its test: importing
# src.retargeting pulls scipy/pybullet via the package __init__, which is
# unavailable in minimal environments.


@pytest.fixture
def test_scene_img():
    scene_path = "tests/fixtures/test_scene.jpg"
    if os.path.exists(scene_path):
        return cv2.imread(scene_path)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)
    return img


@pytest.fixture
def fake_gdino_backend(monkeypatch, tmp_path):
    weights = tmp_path / "groundingdino_swint_ogc.pth"
    weights.write_bytes(b"fake")
    cfg = tmp_path / "GroundingDINO_SwinT_OGC.py"
    cfg.write_text("# fake")

    def fake_load_model(cfg_path, weights_path, device="cpu"):
        return object()

    def fake_predict(model, image, caption, box_threshold, text_threshold, device):
        boxes = torch.tensor([[0.25, 0.25, 0.2, 0.2]])  # normalized cxcywh
        logits = torch.tensor([0.85])
        return boxes, logits, ["container"]

    class FakeBoxOps:
        @staticmethod
        def box_cxcywh_to_xyxy(boxes):
            cx, cy, w, h = boxes.unbind(-1)
            return torch.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
            )

    monkeypatch.setattr(det_mod, "_GDINO_IMPORT_OK", True)
    monkeypatch.setattr(det_mod, "load_model", fake_load_model, raising=False)
    monkeypatch.setattr(det_mod, "predict", fake_predict, raising=False)
    monkeypatch.setattr(det_mod, "box_ops", FakeBoxOps(), raising=False)
    return str(weights), str(cfg)


@pytest.fixture
def fake_unidepth_backend(monkeypatch, tmp_path):
    weights = tmp_path / "unidepth_v1.onnx"
    weights.write_bytes(b"fake")

    class FakeInput:
        name = "image"
        shape = [1, 3, 448, 448]

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, output_names, input_feed):
            depth_mm = np.ones((448, 448), dtype=np.float32) * 1500.0
            return [np.expand_dims(depth_mm, 0)]

    monkeypatch.setattr(depth_mod, "_ORT_IMPORT_OK", True)
    monkeypatch.setattr(
        depth_mod, "ort",
        type("Ort", (), {"InferenceSession": staticmethod(lambda p, providers=None: FakeSession())})(),
        raising=False,
    )
    return str(weights)


def test_grounding_dino_detects_objects(test_scene_img, fake_gdino_backend):
    weights, cfg = fake_gdino_backend
    detector = GroundingDinoDetector(
        Detector2DConfig(
            backend="grounding_dino",
            grounding_dino_weights=weights,
            grounding_dino_config=cfg,
        )
    )
    assert detector.is_available()
    detections = detector.detect(test_scene_img, ["container"])
    assert len(detections) > 0
    for d in detections:
        assert d.bbox_xyxy_norm.min() >= 0.0 and d.bbox_xyxy_norm.max() <= 1.0
        assert d.label
        assert d.score > 0


def test_unidepth_metric_scale(test_scene_img, fake_unidepth_backend):
    estimator = UniDepthEstimator(DepthConfig(model_path=fake_unidepth_backend))
    assert estimator.is_available()
    depth = estimator.estimate_depth(test_scene_img)
    assert depth.shape == test_scene_img.shape[:2]
    # Fake session returns 1500 mm -> 1.5 m.
    assert 0.1 <= float(np.median(depth)) <= 5.0


def test_mujoco_ik_respects_limits(tmp_path):
    from src.retargeting.mujoco_ik_solver import MuJoCoIKSolver

    xml_path = "configs/franka_panda.xml"
    if not os.path.exists(xml_path):
        # Create minimal valid MuJoCo xml for Panda in tmp_path
        xml_path = str(tmp_path / "panda.xml")
        xml_content = """<mujoco model="panda">
          <compiler angle="radian"/>
          <worldbody>
            <body name="panda_hand" pos="0.5 0 0.3">
              <joint name="j1" type="hinge" range="-2.8973 2.8973"/>
              <geom type="sphere" size="0.05"/>
            </body>
          </worldbody>
        </mujoco>"""
        with open(xml_path, "w") as f:
            f.write(xml_content)

    solver = MuJoCoIKSolver(xml_path, ee_link_name="panda_hand")

    # Test reachable target
    result = solver.solve_ik(
        target_pos=np.array([0.5, 0.0, 0.3]),
        target_quat=np.array([1.0, 0.0, 0.0, 0.0])
    )

    if result["reachable"]:
        # Check within limits
        q = result["joint_angles"]
        assert np.all(q >= solver.joint_limits[:, 0])
        assert np.all(q <= solver.joint_limits[:, 1])
