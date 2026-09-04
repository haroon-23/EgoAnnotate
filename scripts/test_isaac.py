import os
import sys
import time
import psutil
import torch
import cv2
from PIL import Image
import numpy as np

def print_memory():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[Memory] Current RSS: {mem_mb:.2f} MB")
    return mem_mb

print("Starting Isaac-0.1 model loading test...")
print_memory()

start_time = time.time()

try:
    # Inject mock transformers.masking_utils module
    import types
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

    masking_utils = types.ModuleType("transformers.masking_utils")

    def packed_sequence_mask_function(packed_sequence_mask):
        def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
            return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        return inner_mask

    def create_masks_for_generate(
        config,
        inputs_embeds,
        attention_mask,
        past_key_values,
        position_ids=None,
        **kwargs
    ):
        batch_size, q_len = inputs_embeds.shape[:2]
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_len()
            
        mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, q_len),
            inputs_embeds,
            past_key_values_length
        )
        return {"full_attention": mask}

    masking_utils.packed_sequence_mask_function = packed_sequence_mask_function
    masking_utils.create_masks_for_generate = create_masks_for_generate

    sys.modules["transformers.masking_utils"] = masking_utils

    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    # Since modular_isaac is dynamically loaded via trust_remote_code=True,
    # we need modular_isaac from the Hugging Face repo.
    # When AutoModelForCausalLM loads with trust_remote_code=True, 
    # it imports it from the HF cache.
    # We can also import IsaacProcessor from the cache or modular_isaac directly.
    # Let's import AutoModel/Tokenizer first and let it cache.
    
    model_path = "./scripts/isaac_local"
    
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    print(f"Loading config from {model_path}...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Import the processor directly from local modular_isaac.py
    import importlib.util
    module_path = os.path.join(model_path, "modular_isaac.py")
    spec = importlib.util.spec_from_file_location("modular_isaac", module_path)
    modular_isaac = importlib.util.module_from_spec(spec)
    sys.modules["modular_isaac"] = modular_isaac
    spec.loader.exec_module(modular_isaac)
    IsaacProcessor = modular_isaac.IsaacProcessor
    
    processor = IsaacProcessor(tokenizer=tokenizer, config=config)
    
    print(f"Loading model from {model_path}...")
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    load_end = time.time()
    
    print(f"Model loaded successfully in {load_end - load_start:.2f} seconds!")
    print_memory()
    
    # Now let's extract the keyframes from data/raw_videos/test_10s.mp4
    video_path = "data/raw_videos/test_10s.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} does not exist.")
        sys.exit(1)
        
    cap = cv2.VideoCapture(video_path)
    frames = {}
    for frame_idx in [0, 100, 200]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[frame_idx] = Image.fromarray(frame_rgb)
            print(f"Extracted frame {frame_idx} with shape {frame.shape}")
        else:
            print(f"Failed to extract frame {frame_idx}")
    cap.release()
    
    # Categories dynamically generated/requested
    categories = ["sewing machine", "person", "clothing", "hand", "fabric", "thread", "scissors"]
    print(f"Querying for categories: {categories}")
    
    device = "cpu"
    model.to(device)
    model.eval()
    
    for f_idx, img in frames.items():
        print(f"\n--- Frame {f_idx} ---")
        for cat in categories:
            # Grounding prompt format for Isaac:
            # "Your goal is to segment out the following categories: {categories}"
            prompt = f"<image>Your goal is to segment out the following categories: {cat}"
            print(f"Prompt: {prompt}")
            
            inputs = processor(text=prompt, images=img)
            
            # Prepare tensors and move to device
            input_ids = inputs["input_ids"].unsqueeze(0).to(device)
            tensor_stream = inputs["tensor_stream"].to(device)
            
            # Run generation
            inf_start = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    tensor_stream=tensor_stream,
                    max_new_tokens=64,
                    do_sample=False,
                )
            inf_end = time.time()
            
            # Decode response
            # Since input_ids is part of output, let's slice it
            response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            print(f"Response (time: {inf_end - inf_start:.2f}s): {response}")
            
except Exception as e:
    import traceback
    print("\n=== ERROR RUNNING LOCAL ISAAC-0.1 ===")
    traceback.print_exc()
    print("=====================================")
