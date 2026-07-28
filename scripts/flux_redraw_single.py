"""
flux_redraw_single.py  —  逐张 Img2Img 重绘
用法:
  python scripts/flux_redraw_single.py \
      --render_dir results/multi_view_blender \
      --out_dir    results/multi_view_flux_single \
      --strength   0.55    # 0=完全保留输入, 1=完全重新生成
"""
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image

# ---- 添加 ReChannel 的 diffusers 到路径 --------------------------------
PIXI_SITE = "/home/clifford/Projects/hair-research/ReChannel/.pixi/envs/default/lib/python3.11/site-packages"
if PIXI_SITE not in sys.path:
    sys.path.insert(0, PIXI_SITE)

from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    compute_empirical_mu,
    retrieve_timesteps,
)

MODEL_PATH = "/home/clifford/.cache/modelscope/hub/models/black-forest-labs/FLUX___2-klein-4B/"

PROMPT = (
    "photorealistic human hair, individual strands, natural lighting, "
    "high resolution, 8k, ultra detailed, realistic hair texture"
)

VIEWS = ["front", "left", "right", "back", "top"]


def encode_image(pipe, image: Image.Image, device, dtype):
    """PIL Image → pipeline-ready latents (1, C*4, H/16, W/16)."""
    vae_dtype = next(pipe.vae.parameters()).dtype
    img_t = (
        torch.from_numpy(np.array(image).astype(np.float32) / 127.5 - 1.0)
        .permute(2, 0, 1).unsqueeze(0)
        .to(device, vae_dtype)
    )
    with torch.no_grad():
        z = pipe.vae.encode(img_t).latent_dist.sample()  # (1, 32, H/8, W/8)
    
    # Pipeline's prepare_latents uses shape (B, C*4, H/16, W/16).
    # We must fold 2×2 spatial patches into the channel dim to match.
    B, C, H, W = z.shape   # (1, 32, 96, 96)
    # (1, 32, 96, 96) → (1, 32, 48, 2, 48, 2) → permute → (1, 48, 48, 32, 2, 2)
    # → (1, 48, 48, 128) → (1, 128, 48, 48)
    z = z.reshape(B, C, H // 2, 2, W // 2, 2)
    z = z.permute(0, 2, 4, 1, 3, 5)          # (B, H/2, W/2, C, 2, 2)
    z = z.reshape(B, H // 2, W // 2, C * 4)  # (B, H/2, W/2, C*4)
    z = z.permute(0, 3, 1, 2)                # (B, C*4, H/2, W/2)
    
    return z.to(dtype)  # (1, 128, 48, 48)


def img2img(pipe, init_image: Image.Image, prompt: str,
            strength: float, num_steps: int, device, dtype):
    """
    Img2Img via sigma-truncation：
    1. encode → latents
    2. pipeline 内部计算 mu → sigmas 全序列
    3. 截取后 (1-strength) 段，在 sigma_start 处加噪
    4. 从截断的 sigma 序列开始去噪
    """
    # 1. Encode
    latents = encode_image(pipe, init_image, device, dtype)   # (1, 32, H/8, W/8)

    # 2. 计算 image_seq_len（pack 后的 token 数）
    # FLUX packs (2, 2) patches → h/16 * w/16 tokens
    h_lat = latents.shape[2]   # H/8
    w_lat = latents.shape[3]   # W/8
    h_pkg = h_lat // 2         # H/16
    w_pkg = w_lat // 2         # W/16
    image_seq_len = h_pkg * w_pkg

    # 3. 计算 mu 并获取完整 sigma 序列
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
    pipe.scheduler.set_timesteps(num_steps, device=device, mu=mu)
    sigmas_full = pipe.scheduler.sigmas  # length = num_steps + 1

    # 4. 截断：t_start = 从哪步开始去噪（跳过前面 strength 比例的步骤）
    t_start = max(1, int(round(num_steps * (1.0 - strength))))
    sigmas_trunc = sigmas_full[t_start:].tolist()
    sigma_start  = sigmas_full[t_start].item()

    # 5. 加噪到 sigma_start
    noise = torch.randn_like(latents)
    noisy_latents = latents + sigma_start * noise

    # 6. 直接传给 pipeline（它内部会自己 pack）
    with torch.no_grad():
        out = pipe(
            prompt=prompt,
            height=init_image.height,
            width=init_image.width,
            num_inference_steps=len(sigmas_trunc) - 1,
            sigmas=sigmas_trunc,
            latents=noisy_latents,   # 4D: (1, 32, H/8, W/8)
            guidance_scale=4.0,
        )
    return out.images[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_dir", default="results/multi_view_blender")
    parser.add_argument("--out_dir",    default="results/multi_view_flux_single")
    parser.add_argument("--strength",   type=float, default=0.55,
                        help="0=完全保留, 1=完全重生成")
    parser.add_argument("--steps",      type=int,   default=30)
    parser.add_argument("--size",       type=int,   default=768)
    parser.add_argument("--views",      nargs="+",  default=VIEWS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16

    print("Loading pipeline...")
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)

    for view in args.views:
        src = os.path.join(args.render_dir, f"{view}_rgb.png")
        if not os.path.exists(src):
            print(f"[SKIP] {src} not found")
            continue

        print(f"Processing {view} (strength={args.strength}) ...")
        img = Image.open(src).convert("RGB").resize((args.size, args.size))

        result = img2img(
            pipe, img, PROMPT,
            strength=args.strength,
            num_steps=args.steps,
            device=device, dtype=dtype,
        )

        dst = os.path.join(args.out_dir, f"{view}_hair.png")
        result.save(dst)
        print(f"  Saved -> {dst}")

    print("Done.")


if __name__ == "__main__":
    main()
