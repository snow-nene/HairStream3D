"""
flux_redraw_multiview.py — Multi-View Hair Inpainting with FLUX.2-klein

Uses Flux2KleinInpaintPipeline with hair masks to redraw ONLY the hair
region while preserving head structure, face, background, and hairstyle.

Requires:
  - RGB renders in results/multi_view_blender/*_rgb.png
  - Hair masks in results/multi_view_renders/*_mask.png (white=hair, black=keep)

Usage:
  python scripts/infer_2d/flux_redraw_multiview.py \\
      --render_dir results/multi_view_blender \\
      --maps_dir   results/multi_view_renders \\
      --out_dir    results/multi_view_flux_multiview \\
      --strength   0.8
"""
import os
import sys

# ── Auto-switch to ReChannel's Python interpreter ──────────────────
_RECHANNEL_PYTHON = "/home/clifford/Projects/hair-research/ReChannel/.pixi/envs/default/bin/python"
if os.path.realpath(sys.executable) != os.path.realpath(_RECHANNEL_PYTHON):
    if os.path.exists(_RECHANNEL_PYTHON):
        os.execv(_RECHANNEL_PYTHON, [_RECHANNEL_PYTHON] + sys.argv)
    else:
        sys.stderr.write(
            f"[WARN] ReChannel python not found at {_RECHANNEL_PYTHON}; "
            f"trying current interpreter.\n"
        )

import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw

from diffusers import Flux2KleinInpaintPipeline

# ── Globals ────────────────────────────────────────────────────────
MODEL_PATH = "/home/clifford/.cache/modelscope/hub/models/black-forest-labs/FLUX___2-klein-4B/"

PROMPT = (
    "high definition, rich hair texture, sharp individual strands, "
    "clear hair fibers, ultra detailed, photorealistic"
)

# Top-down view needs an explicit negative instruction — FLUX.2-klein
# tends to hallucinate faces from hair swirl patterns (pareidolia).
PROMPT_TOP = (
    "top-down view of human hair on scalp, hair parting and crown, "
    "hair strands radiating from center, bird's eye view, this is only hair, "
    "no face, no facial features, no eyes, no nose, no mouth, "
    "purely hair texture, high definition, sharp individual strands, photorealistic"
)

VIEWS = ["front", "left", "right", "back", "top"]


# ── Image Loading ──────────────────────────────────────────────────

def load_images(render_dir, view_names, size):
    """Load RGB renders for each view.

    Returns:
        images:      list of PIL RGB Images
        loaded_views: list of view name strings
    """
    images = []
    loaded_views = []

    for v in view_names:
        rgb_path = os.path.join(render_dir, f"{v}.png")
        if not os.path.exists(rgb_path):
            print(f"  [WARN] Missing render: {rgb_path}")
            continue
        img = Image.open(rgb_path).convert("RGB").resize((size, size))
        images.append(img)
        loaded_views.append(v)

    return images, loaded_views


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-View Hair Inpainting with FLUX.2-klein"
    )
    parser.add_argument("--img_id", required=True,
                        help="Image ID used for multiview_data directory")
    parser.add_argument("--views", nargs="+", default=VIEWS,
                        help="Views to process")
    parser.add_argument("--strength", type=float, default=0.8,
                        help="Inpaint strength: 0=keep hair as-is, 1=full regenerate. "
                             "Higher = more texture detail but may alter hairstyle.")
    parser.add_argument("--steps", type=int, default=30,
                        help="Denoising steps (default: 30)")
    parser.add_argument("--size", type=int, default=1024,
                        help="Output size (default: 1024)")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to FLUX.2-klein model")
    args = parser.parse_args()

    data_dir = os.path.join("results", "multiview_data", args.img_id)
    render_dir = os.path.join(data_dir, "blender_renders")
    out_dir = os.path.join(data_dir, "flux_redrawn")
    os.makedirs(out_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # ── Load renders ────────────────────────────────────────────
    print(f"Loading renders from: {render_dir}")
    images, loaded_views = load_images(render_dir, args.views, args.size)
    if len(images) == 0:
        print("[ERROR] No render+mask pairs found. Aborting.")
        sys.exit(1)
    print(f"  Loaded {len(images)} views: {loaded_views}")

    # ── Load inpainting pipeline ─────────────────────────────────
    print(f"Loading FLUX.2-klein Inpaint pipeline from: {args.model}")
    pipe = Flux2KleinInpaintPipeline.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)

    # ── Process each view ────────────────────────────────────────
    for i, (view, img) in enumerate(zip(loaded_views, images)):
        prompt = PROMPT_TOP if view == "top" else PROMPT
        print(f"  [{i+1}/{len(loaded_views)}] Inpainting {view} "
              f"(strength={args.strength}, steps={args.steps}) ...")
        mask = Image.new("L", (args.size, args.size), 255)

        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                image=img,
                mask_image=mask,
                strength=args.strength,
                num_inference_steps=args.steps,
            ).images[0]

        out_path = os.path.join(out_dir, f"{view}.png")
        result.save(out_path)
        print(f"    Saved {out_path}")

    # ── Composite grid ───────────────────────────────────────────
    if len(loaded_views) >= 3:
        # Re-load the saved results for the grid
        grid_images = []
        for v in loaded_views:
            p = os.path.join(out_dir, f"{v}.png")
            if os.path.exists(p):
                grid_images.append(Image.open(p).convert("RGB"))

        if len(grid_images) >= 3:
            grid = _make_grid(grid_images, loaded_views[:len(grid_images)])
            grid_path = os.path.join(out_dir, "grid_overview.png")
            grid.save(grid_path)
            print(f"  Saved {grid_path}")

    print(f"Done — {len(loaded_views)} views saved to {out_dir}")
    print(f"Tip: lower --strength (e.g. 0.4) = keep hair closer to original")
    print(f"     higher --strength (e.g. 0.9) = more texture regeneration")


def _make_grid(images, view_names):
    """Create a grid overview with view labels."""
    w, h = images[0].size

    if len(images) == 5:
        positions = {
            "front": (0, 0), "left": (1, 0), "right": (2, 0),
            "back": (0, 2), "top": (1, 1),
        }
    else:
        cols = min(len(images), 3)
        rows = (len(images) + cols - 1) // cols
        positions = {vn: (i % cols, i // cols) for i, vn in enumerate(view_names)}

    max_col = max(p[0] for p in positions.values())
    max_row = max(p[1] for p in positions.values())
    grid = Image.new("RGB", ((max_col + 1) * w, (max_row + 1) * h), color=(40, 40, 40))

    from PIL import ImageFont
    for vn, img in zip(view_names, images):
        if vn in positions:
            x, y = positions[vn]
            grid.paste(img, (x * w, y * h))
            draw = ImageDraw.Draw(grid)
            draw.text((x * w + 8, y * h + 8), vn, fill=(255, 255, 255))

    return grid


if __name__ == "__main__":
    main()
