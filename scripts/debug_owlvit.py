import cv2
import numpy as np
import torch
from transformers import AutoProcessor, OwlViTForObjectDetection

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/owlvit-base-patch32"
    
    print(f"Loading {model_name} on {device}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
    model.eval()
    
    video_path = "data/raw_videos/test_10s.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
        
    frames_to_test = [0, 100, 200]
    
    # Use the dynamically derived object list for this sewing video
    target_classes = ["hand", "fabric", "cloth", "sewing machine", "scissors", "thread"]
    
    for frame_idx in frames_to_test:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Error: Cannot read frame {frame_idx}")
            continue
            
        print("\n" + "="*80)
        print(f"FRAME {frame_idx} (Shape: {frame.shape})")
        print("="*80)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Prepare inputs
        inputs = processor(images=image_rgb, text=target_classes, return_tensors="pt").to(device)
        
        # Run model
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Post process to get boxes and scores for all targets at threshold=0.0
        target_sizes = torch.tensor([image_rgb.shape[:2]]).to(device)
        results = processor.post_process_object_detection(
            outputs,
            threshold=0.0,
            target_sizes=target_sizes
        )[0]
        
        # Group detection items by query class
        grouped = {cls: [] for cls in target_classes}
        for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
            idx = label_idx.item()
            if idx < len(target_classes):
                name = target_classes[idx]
                grouped[name].append((score.item(), box.cpu().numpy()))
                
        # Sort each group by score in descending order
        for cls in target_classes:
            grouped[cls].sort(key=lambda x: x[0], reverse=True)
            
        # Print top-5 for each class
        for cls in target_classes:
            print(f"\nTarget Class: {cls.upper()} (Total proposals: {len(grouped[cls])})")
            top_5 = grouped[cls][:5]
            if not top_5:
                print("  No proposals found.")
                continue
            for rank, (score, box) in enumerate(top_5, 1):
                # Normalized bbox conversion
                h, w = image_rgb.shape[:2]
                bbox_norm = [box[0]/w, box[1]/h, box[2]/w, box[3]/h]
                bbox_str = ", ".join([f"{x:.3f}" for x in bbox_norm])
                print(f"  Rank {rank}: score={score:.5f} | bbox=[{bbox_str}]")
                
    cap.release()

if __name__ == "__main__":
    main()
