"""
compute_multiview_maps.py - Compute strand_map & depth_map for multi-view images.

Pipeline (per view):
  1. SAM → hair mask (seg) + body mask (body_img)
  2. img2strand → strand_map  (2D orientation)
  3. img2depth  → depth_map   (normalized depth)

Front view: strand_map from HiSa_HiDa is COPIED AS-IS, never modified.
            (Depth is still computed from the original 2D image.)
Left/Right/Back: full pipeline on FLUX-redrawn images.

Output:
  {out_dir}/strand_map/{view}.png    — strand maps (512x512x3, 1 body + 2 orientation)
  {out_dir}/depth_map/{view}.npy     — normalized depth maps (512x512)
  {out_dir}/depth_vis/{view}.png     — depth visualizations
  {out_dir}/mask/{view}_seg.png      — SAM hair masks
  {out_dir}/mask/{view}_body.png     — SAM body masks
  {out_dir}/combined_strand.npz      — all 4 views stacked (4, 512, 512, 3)
  {out_dir}/combined_depth.npz       — all 4 views stacked (4, 512, 512)

Usage:
  python scripts/render/compute_multiview_maps.py \
      --front_img     datasets/HiSa_HiDa/img/0a1ba3dbefc8934ab60577c5c91f66a0.png \
      --front_strand  /home/clifford/Academic/数据集/HiSa_HiDa/strand_map/0a1ba3dbefc8934ab60577c5c91f66a0.png \
      --out_dir       results/multiview_strand_depth
"""
import os
import sys
import argparse
import numpy as np
import imageio.v2 as imageio
import torch
import cv2
from torch.autograd import Variable
from tqdm import tqdm

# HairStep project root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lib.options import BaseOptions
from lib.model.img2hairstep.model_factory import create_img2strand_model
from lib.model.img2hairstep.hourglass import Model as DepthModel
from segment_anything import SamPredictor, sam_model_registry
from skimage.transform import resize as skresize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIEWS = ["front", "left", "right", "back"]
IMG_SIZE = 512


# ── Image utils ────────────────────────────────────────────────────

def pad_to_square(img):
    """Zero-pad an image to square aspect ratio, then resize to IMG_SIZE."""
    img = np.array(img)
    h, w = img.shape[:2]
    if h > w:
        pad_w = (h - w) // 2
        padded = np.pad(img, ((0, 0), (pad_w, pad_w), (0, 0))) if img.ndim == 3 \
                 else np.pad(img, ((0, 0), (pad_w, pad_w)))
    elif w > h:
        pad_h = (w - h) // 2
        padded = np.pad(img, ((pad_h, pad_h), (0, 0), (0, 0))) if img.ndim == 3 \
                 else np.pad(img, ((pad_h, pad_h), (0, 0)))
    else:
        padded = img
    return (skresize(padded, (IMG_SIZE, IMG_SIZE), preserve_range=True)
            .astype(np.uint8))


# ── SAM mask generation ────────────────────────────────────────────

def generate_masks_sam(image_path, sam_predictor, device):
    """Run SAM on a single image → hair mask + body mask.

    Uses point prompts: a center click selects the foreground (hair),
    then hair/body point pairs refine the hair mask.

    Returns:
        hair_mask: (512, 512, 3) uint8, white=hair
        body_mask: (512, 512, 3) uint8, white=body (person/head under hair)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    resized = pad_to_square(image)
    sam_predictor.set_image(resized)

    H, W = resized.shape[:2]
    cx, cy = W // 2, H // 2

    # 1. Body mask: click center → foreground (the hair/head)
    body_masks, _, _ = sam_predictor.predict(
        point_coords=np.array([[cx, cy]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    # body_masks: (3, H, W) — pick the largest mask as body
    body_mask = _largest_mask(body_masks)

    # 2. Hair mask: click center=body(0) + hair-edge=hair(1)
    # Find the leftmost column where body_mask is True, then step inward
    body_cols = np.where(np.sum(body_mask, axis=0) > 0)[0]
    if len(body_cols) > 0:
        hair_x = body_cols[0] + 5
    else:
        hair_x = cx

    hair_pt = np.array([[cx, hair_x]])
    body_pt = np.array([[cx, cy]])
    pts = np.concatenate([hair_pt, body_pt], axis=0)

    hair_masks, _, _ = sam_predictor.predict(
        point_coords=pts,
        point_labels=np.array([1, 0]),
        multimask_output=False,
    )

    # hair_masks: (1, H, W) or (H, W)
    if hair_masks.ndim == 3:
        hair_mask = hair_masks[0]
    else:
        hair_mask = hair_masks

    # Convert to uint8 RGB images
    body_rgb = np.stack([body_mask.astype(np.uint8) * 255] * 3, axis=-1)
    hair_rgb = np.stack([hair_mask.astype(np.uint8) * 255] * 3, axis=-1)

    return hair_rgb, body_rgb


def _largest_mask(masks_3d):
    """Pick the mask with the largest area from a (N, H, W) array."""
    best = masks_3d[0]
    best_area = best.sum()
    for i in range(1, masks_3d.shape[0]):
        area = masks_3d[i].sum()
        if area > best_area:
            best = masks_3d[i]
            best_area = area
    return best


# ── Strand & Depth prediction ──────────────────────────────────────

def predict_strand(model, rgb_img, hair_mask, body_mask, device):
    """Run strand prediction.

    Args:
        rgb_img:   (512, 512, 3) float32 [0,1]
        hair_mask: (512, 512, 1) bool
        body_mask: (512, 512, 1) bool
    Returns:
        strand_map: (512, 512, 3) uint8
          ch0: body label (0=bg, 128=body, 255=hair)
          ch1-2: strand orientation
    """
    masked_rgb = rgb_img * hair_mask
    x = Variable(
        torch.from_numpy(masked_rgb).permute(2, 0, 1).float().unsqueeze(0)
    ).to(device)

    with torch.no_grad():
        strand_pred = model(x)

    strand_pred = np.clip(
        strand_pred.permute(0, 2, 3, 1)[0].cpu().detach().numpy(), 0.0, 1.0
    )
    # R = 255 for hair, 0 for background
    # G = (dy + 1) / 2 * 255
    # B = (-dx + 1) / 2 * 255
    strand_map = np.concatenate([
        hair_mask.astype(np.float32),                      # R: only 0/1
        strand_pred * hair_mask.astype(np.float32),        # G, B
    ], axis=-1)
    return (strand_map * 255).astype(np.uint8)


def predict_depth(model, rgb_img, hair_mask, device):
    """Run depth prediction.

    Args:
        rgb_img:   (512, 512, 3) float32 [0,1]
        hair_mask: (512, 512, 1) bool
    Returns:
        depth_norm: (512, 512) float32 [0,1]
    """
    x = Variable(
        torch.from_numpy(rgb_img).permute(2, 0, 1).float().unsqueeze(0)
    ).to(device)

    with torch.no_grad():
        depth_pred = model(x)
    depth_pred = depth_pred.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    depth_pred = depth_pred[:, :, 0]

    mask_f = hair_mask[:, :, 0].astype(np.float32)
    inv = 1.0 - mask_f
    big = np.abs(np.nanmax(depth_pred)) + np.abs(np.nanmin(depth_pred)) + 1.0

    depth_masked = depth_pred * mask_f - inv * big
    max_val = np.nanmax(depth_masked)
    min_val = np.nanmin(depth_masked + 2 * inv * big)
    depth_norm = (depth_masked - min_val) / (max_val - min_val + 1e-8) * mask_f
    return np.clip(depth_norm, 0.0, 1.0).astype(np.float32)


def depth2vis(mask, depth, path_output):
    """Jet-colormap depth visualization without writing temporary files."""
    m = mask[:, :, 0].astype(np.float32)
    masked = depth * m + (1 - m) * ((depth * m).max())
    norm = masked / (np.nanmax(masked) - np.nanmin(masked) + 1e-8)
    
    cm = plt.get_cmap("jet")
    colored = (cm(norm)[:, :, :3] * 255).astype(np.uint8)
    vis = (colored.astype(np.float32) * mask[:, :, :3]).astype(np.uint8)
    plt.imsave(path_output, vis)


def load_image(path):
    """Load → pad to square → resize 512×512 → float32 [0,1]."""
    img = imageio.imread(path)[:, :, 0:3] / 255.0
    if img.shape[0] != 512 or img.shape[1] != 512:
        img_u8 = (img * 255).astype(np.uint8)
        img_u8 = pad_to_square(img_u8)
        img = img_u8.astype(np.float32) / 255.0
    return img.astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAM masks → strand_map + depth_map for multi-view images"
    )
    parser.add_argument("--front_img", type=str, required=True,
                        help="Original 2D front portrait image (for depth computation)")
    parser.add_argument("--front_strand", type=str, required=True,
                        help="Original front strand_map PNG from HiSa_HiDa (copied as-is)")
    parser.add_argument("--front_depth", type=str, default=None,
                        help="Optional: original front depth_map .npy")
    parser.add_argument("--render_dir", default="results/multi_view_flux_multiview",
                        help="Directory with FLUX-redrawn images ({view}_hair.png)")
    parser.add_argument("--out_dir", default="results/multiview_strand_depth",
                        help="Output directory")
    parser.add_argument("--views", nargs="+", default=["left", "right", "back"],
                        help="Views to predict (front is always from --front_strand)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_sam",
                        default=os.path.join(ROOT, "checkpoints/SAM-models/sam_vit_h_4b8939.pth"))
    parser.add_argument("--checkpoint_img2strand",
                        default=os.path.join(ROOT, "checkpoints/img2hairstep/img2strand.pth"))
    parser.add_argument("--checkpoint_img2depth",
                        default=os.path.join(ROOT, "checkpoints/img2hairstep/img2depth.pth"))
    parser.add_argument("--model_type_sam", default="vit_h")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Output dirs ────────────────────────────────────────────────
    out_strand = os.path.join(args.out_dir, "strand_map")
    out_depth = os.path.join(args.out_dir, "depth_map")
    out_depth_vis = os.path.join(args.out_dir, "depth_vis")
    out_seg = os.path.join(args.out_dir, "seg")
    out_body = os.path.join(args.out_dir, "body_img")
    for d in [out_strand, out_depth, out_depth_vis, out_seg, out_body]:
        os.makedirs(d, exist_ok=True)

    # ── Load SAM ───────────────────────────────────────────────────
    print("Loading SAM...")
    sam = sam_model_registry[args.model_type_sam](checkpoint=args.checkpoint_sam)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # ── Load strand & depth models ─────────────────────────────────
    print("Loading strand model...")
    # BaseOptions reads sys.argv; save & restore to avoid conflicts
    _argv = sys.argv
    sys.argv = ["compute_multiview_maps.py"]
    opt = BaseOptions().parse()
    sys.argv = _argv
    strand_model = create_img2strand_model(opt).to(device)
    strand_model.load_state_dict(torch.load(args.checkpoint_img2strand, map_location=device))
    strand_model.eval()

    print("Loading depth model...")
    depth_model = DepthModel().to(device)
    depth_model = torch.nn.DataParallel(depth_model)
    depth_model.load_state_dict(torch.load(args.checkpoint_img2depth, map_location=device))
    depth_model.eval()

    # ================================================================
    #  1.  FRONT: copy original strand_map; generate mask → depth
    # ================================================================
    print("\n" + "=" * 60)
    print("FRONT — copying original strand_map (read-only)")
    front_strand = imageio.imread(args.front_strand).astype(np.float32) / 255.0
    if front_strand.shape[0] != 512 or front_strand.shape[1] != 512:
        front_strand = skresize(front_strand, (512, 512), preserve_range=True)

    # Convert R channel from 0/128/255 to 0/1.
    # Only R==255 is hair; R==128 is body (face/skin) and should become 0.
    front_strand[:, :, 0] = (front_strand[:, :, 0] > 0.9).astype(np.float32)  # only 255 → 1

    front_strand_u8 = (front_strand * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(out_strand, "front.png"), front_strand_u8)
    print(f"  strand_map: {front_strand_u8.shape} — COPIED (R channel cleaned to 0/255)")

    # SAM mask for front depth
    print("  Running SAM for front mask...")
    hair_mask_rgb, body_mask_rgb = generate_masks_sam(args.front_img, predictor, device)
    cv2.imwrite(os.path.join(out_seg, "front.png"), hair_mask_rgb)
    cv2.imwrite(os.path.join(out_body, "front.png"), body_mask_rgb)
    front_hair_mask = (hair_mask_rgb[:, :, 0:1] / 255.0 > 0.5)

    # Front depth
    if args.front_depth and os.path.exists(args.front_depth):
        front_depth = np.load(args.front_depth)
        print(f"  depth_map: copied from {args.front_depth}")
    else:
        print("  Predicting depth_map...")
        front_rgb = load_image(args.front_img)
        front_depth = predict_depth(depth_model, front_rgb, front_hair_mask, device)
    np.save(os.path.join(out_depth, "front.npy"), front_depth)
    depth2vis(front_hair_mask, front_depth, os.path.join(out_depth_vis, "front.png"))
    print(f"  depth_map: {front_depth.shape}")

    # ================================================================
    #  2.  LEFT / RIGHT / BACK: SAM mask → strand → depth
    # ================================================================
    for view in args.views:
        print(f"\n{'=' * 60}")
        print(f"{view.upper()} — full pipeline")

        img_path = os.path.join(args.render_dir, f"{view}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(args.render_dir, f"{view}_hair.png")

        if not os.path.exists(img_path):
            print(f"  [SKIP] {view}.png / {view}_hair.png not found in {args.render_dir}")
            continue

        # 2a. SAM masks
        print(f"  Running SAM...")
        hair_mask_rgb, body_mask_rgb = generate_masks_sam(img_path, predictor, device)
        cv2.imwrite(os.path.join(out_seg, f"{view}.png"), hair_mask_rgb)
        cv2.imwrite(os.path.join(out_body, f"{view}.png"), body_mask_rgb)

        hair_mask = (hair_mask_rgb[:, :, 0:1] / 255.0 > 0.5)  # (512, 512, 1) bool
        body_mask = (body_mask_rgb[:, :, 0:1] / 255.0 > 0.5)

        rgb = load_image(img_path)

        # 2b. Strand prediction
        print(f"  Predicting strand_map...")
        strand_map = predict_strand(strand_model, rgb, hair_mask, body_mask, device)
        imageio.imwrite(os.path.join(out_strand, f"{view}.png"), strand_map)
        print(f"    strand_map: {strand_map.shape}")

        # 2c. Depth prediction
        print(f"  Predicting depth_map...")
        depth_norm = predict_depth(depth_model, rgb, hair_mask, device)
        np.save(os.path.join(out_depth, f"{view}.npy"), depth_norm)
        depth2vis(hair_mask, depth_norm, os.path.join(out_depth_vis, f"{view}.png"))
        print(f"    depth_map: {depth_norm.shape}")

    # ================================================================
    #  3.  Combine into multi-view arrays
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Combining multi-view maps...")
    all_views = ["front"] + [v for v in args.views]

    strand_stack, depth_stack = [], []
    for v in all_views:
        sp = os.path.join(out_strand, f"{v}.png")
        dp = os.path.join(out_depth, f"{v}.npy")
        if os.path.exists(sp):
            strand_stack.append(imageio.imread(sp).astype(np.float32) / 255.0)
        else:
            print(f"  [WARN] Missing strand_map: {sp}")
        if os.path.exists(dp):
            depth_stack.append(np.load(dp))
        else:
            print(f"  [WARN] Missing depth_map: {dp}")

    if strand_stack:
        s = np.stack(strand_stack, axis=0)
        np.savez_compressed(os.path.join(args.out_dir, "combined_strand.npz"),
                            strand=s, views=np.array(all_views))
        print(f"  combined_strand: {s.shape}")

    if depth_stack:
        d = np.stack(depth_stack, axis=0)
        np.savez_compressed(os.path.join(args.out_dir, "combined_depth.npz"),
                            depth=d, views=np.array(all_views))
        print(f"  combined_depth: {d.shape}")

    print(f"\nDone → {args.out_dir}/")


if __name__ == "__main__":
    main()
