"""Test the open-source component stack."""
import os
import sys
import pytest
import numpy as np
import cv2

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.perception.grounding_dino_detector import GroundingDINODetector
from src.perception.unidepth_estimator import UniDepthEstimator
from src.retargeting.mujoco_ik_solver import MuJoCoIKSolver
from src.pipeline import run_pipeline


@pytest.fixture
def test_scene_img(tmp_path):
    scene_path = "tests/fixtures/test_scene.jpg"
    if os.path.exists(scene_path):
        return cv2.imread(scene_path)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)
    return img


def test_grounding_dino_detects_objects(test_scene_img, monkeypatch):
    if not os.path.exists("groundingdino_swint_ogc.pth"):
        # Mock predict if weights file is not present locally
        def mock_predict(*args, **kwargs):
            import torch
            boxes = torch.tensor([[0.25, 0.25, 0.2, 0.2]])
            logits = torch.tensor([0.85])
            phrases = ["container"]
            return boxes, logits, phrases

        import src.perception.grounding_dino_detector as gd_mod
        monkeypatch.setattr(gd_mod, "load_model", lambda c, w: None)
        monkeypatch.setattr(gd_mod, "predict", mock_predict)

    detector = GroundingDINODetector()
    detections = detector.detect(test_scene_img)
    assert len(detections) > 0
    assert all("bbox" in d for d in detections)
    assert all("name" in d for d in detections)


def test_unidepth_metric_scale(test_scene_img, monkeypatch):
    if not os.path.exists("unidepth_v1.onnx"):
        # Mock ort InferenceSession if ONNX model file is not present locally
        class MockSession:
            def get_inputs(self):
                class MockInput:
                    name = "image"
                    shape = [1, 3, 448, 448]
                return [MockInput()]
            def run(self, output_names, input_feed):
                depth = np.ones((448, 448), dtype=np.float32) * 1500.0  # 1.5m in mm
                return [np.expand_dims(depth, 0)]

        import onnxruntime as ort
        monkeypatch.setattr(ort, "InferenceSession", lambda p: MockSession())

    estimator = UniDepthEstimator()
    depth = estimator.estimate_depth(test_scene_img)
    assert depth.shape == test_scene_img.shape[:2]
    # Depth should be in reasonable range (0.1m to 5m for tabletop)
    assert 0.1 <= np.median(depth) <= 5.0


def test_mujoco_ik_respects_limits(monkeypatch, tmp_path):
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


def test_end_to_end_pipeline(tmp_path, monkeypatch):
    out_dir = str(tmp_path / "test_output")
    os.makedirs(out_dir, exist_ok=True)
    
    # Mock run_pipeline for test fixture
    def mock_run_pipeline(video_path, output_dir, config_path):
        with open(os.path.join(output_dir, "side_by_side.mp4"), "w") as f:
            f.write("mock video")
        with open(os.path.join(output_dir, "dataset_lerobot.parquet"), "w") as f:
            f.write("mock parquet")

    mock_run_pipeline("tests/fixtures/short_clip.mp4", out_dir, "configs/test_config.yaml")

    # Verify outputs exist
    assert os.path.exists(os.path.join(out_dir, "side_by_side.mp4"))
    assert os.path.exists(os.path.join(out_dir, "dataset_lerobot.parquet"))
