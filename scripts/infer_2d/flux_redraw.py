import argparse
import os
import torch
import numpy as np
from PIL import Image
from diffusers import Flux2KleinPipeline
from diffusers.models.attention_processor import Attention

import diffusers.models.attention_dispatch as attn_dispatch

original_dispatch = attn_dispatch.dispatch_attention_fn

def hijacked_dispatch(query, key, value, *args, **kwargs):
    # query shape: [B, SeqLen, Heads, HeadDim]
    batch_size = query.shape[0]
    
    if batch_size == 5:
        # Front is index 0
        front_k = key[0:1]
        front_v = value[0:1]
        
        front_k_repeated = front_k.expand(4, -1, -1, -1)
        front_v_repeated = front_v.expand(4, -1, -1, -1)
        
        # Concat along sequence dim (dim=1)
        key_others = torch.cat([key[1:], front_k_repeated], dim=1)
        value_others = torch.cat([value[1:], front_v_repeated], dim=1)
        
        # Pad front view with itself so sequence lengths match across the batch
        front_k_pad = torch.cat([key[0:1], key[0:1]], dim=1)
        front_v_pad = torch.cat([value[0:1], value[0:1]], dim=1)
        
        key = torch.cat([front_k_pad, key_others], dim=0)
        value = torch.cat([front_v_pad, value_others], dim=0)
        
    return original_dispatch(query, key, value, *args, **kwargs)

def set_mv_attention(pipeline):
    attn_dispatch.dispatch_attention_fn = hijacked_dispatch

def load_images(render_dir):
    views = ['front', 'left', 'right', 'back', 'top']
    images = []
    for v in views:
        p = os.path.join(render_dir, f"{v}_rgb.png")
        img = Image.open(p).convert("RGB").resize((1024, 1024))
        images.append(img)
    return images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", type=str, required=True, help="Ground truth 2D front image")
    parser.add_argument("--render_dir", type=str, required=True, help="Directory containing blender renders")
    parser.add_argument("--out_dir", type=str, default="results/multi_view_flux", help="Output directory")
    parser.add_argument("--model", type=str, default="/home/clifford/.cache/modelscope/hub/models/black-forest-labs/FLUX___2-klein-4B/", help="Path to local FLUX model")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading FLUX.2-klein pipeline...")
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
    
    print("Injecting Cross-View Attention...")
    set_mv_attention(pipe)

    ref_img = Image.open(args.ref_image).convert("RGB").resize((1024, 1024))
    init_images = load_images(args.render_dir)
    
    # We replace the front blender render with the actual ground truth 2D reference image
    # so that the Multi-View Attention uses the real image's tokens as the anchor.
    init_images[0] = ref_img

    prompt = ["high quality, ultra realistic, detailed human hair, individual hair strands, 8k resolution, photorealistic"] * 5
    
    with torch.no_grad():
        out = pipe(
            prompt=prompt,
            image=init_images,
            num_inference_steps=30,
            guidance_scale=4.0
        ).images

    views = ['front', 'left', 'right', 'back', 'top']
    for v, img in zip(views, out):
        out_path = os.path.join(args.out_dir, f"{v}_hair.png")
        img.save(out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
