import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.object_detector import GeminiObjectDetector, ObjectDetectorConfig

def main():
    # Load configuration
    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        
    obj_detect_config = config_data.get("object_detection", {})
    gemini_config = config_data.get("gemini", {})
    gd_config = config_data.get("grounding_dino", {})
    
    config = ObjectDetectorConfig(
        keyframes_per_video=obj_detect_config.get("keyframes_per_video", 3),
        bbox_keyframe_interval=obj_detect_config.get("bbox_keyframe_interval", 15),
        prompt=obj_detect_config.get("prompt"),
        gemini_model=gemini_config.get("model", "gemini-1.5-flash"),
        grounding_dino_model=gd_config.get("model_name", "google/owlvit-base-patch32"),
        grounding_dino_confidence=gd_config.get("confidence_threshold", 0.1),
        grounding_dino_box_threshold=gd_config.get("box_threshold", 0.1),
        grounding_dino_text_threshold=gd_config.get("text_threshold", 0.1)
    )
    
    detector = GeminiObjectDetector(config)
    
    video_path = "data/raw_videos/new_video.mp4"
    print(f"\nRunning detect_objects on {video_path}...")
    try:
        results = detector.detect_objects(video_path)
        print("\n=== SUCCESS ===")
        print(f"Detected object list: {[r.name for r in results]}")
        print("===============")
    except Exception as e:
        print("\n=== FAILURE ===")
        print(f"Error during VLM call: {e}")
        print("===============")
        sys.exit(1)

if __name__ == "__main__":
    main()
