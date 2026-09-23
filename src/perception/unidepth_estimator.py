"""UniDepth: Monocular metric depth estimation.
Provides real-world scale (meters) from single RGB image."""
import numpy as np
import onnxruntime as ort

class UniDepthEstimator:
    def __init__(self, model_path="unidepth_v1.onnx"):
        """
        Initialize UniDepth ONNX model.
        Download from: https://huggingface.co/ibaiGorordo/ONNX-UniDepth-V1
        """
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 448, 448]
        
    def estimate_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Estimate metric depth map from RGB image.
        
        Args:
            image_bgr: OpenCV BGR image (H, W, 3)
        
        Returns:
            depth_map: Metric depth in meters (H, W)
        """
        # Preprocess
        image_rgb = image_bgr[:, :, ::-1]
        h_orig, w_orig = image_rgb.shape[:2]
        
        # Resize to model input size
        input_h, input_w = self.input_shape[2], self.input_shape[3]
        image_resized = self._resize_and_normalize(image_rgb, input_w, input_h)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: image_resized})
        depth_pred = outputs[0][0]  # Remove batch dimension
        
        # Resize back to original size
        depth_map = self._resize_depth(depth_pred, w_orig, h_orig)
        
        # Convert to meters (UniDepth outputs in millimeters)
        depth_meters = depth_map / 1000.0
        
        return depth_meters
    
    def _resize_and_normalize(self, image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize and normalize image for ONNX input."""
        import cv2
        
        # Resize
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Transpose to [C, H, W] and add batch dimension
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, :]
        
        return input_tensor
    
    def _resize_depth(self, depth: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize depth map back to original image size."""
        import cv2
        return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    def project_to_3d(self, image_bgr: np.ndarray, bbox: list) -> np.ndarray:
        """
        Get 3D position of object center in camera frame.
        
        Args:
            image_bgr: Original image
            bbox: [x1, y1, x2, y2] in pixels
        
        Returns:
            position_3d: [x, y, z] in meters (camera frame)
        """
        depth_map = self.estimate_depth(image_bgr)
        
        # Get center of bbox
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        
        # Get depth at center
        z = depth_map[cy, cx]
        
        # Simple pinhole camera model (approximate)
        # Assuming ~60 degree FOV
        h, w = image_bgr.shape[:2]
        fx = fy = w / (2 * np.tan(np.radians(30)))
        cx_cam, cy_cam = w / 2, h / 2
        
        # Back-project to 3D
        x = (cx - cx_cam) * z / fx
        y = (cy - cy_cam) * z / fy
        
        return np.array([x, y, z])
