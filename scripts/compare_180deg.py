import os
import numpy as np
from PIL import Image
import cv2

def custom_180deg_colorwheel(theta):
    """
    Map an undirected angle in [0, pi] to a 360-degree colorwheel by doubling it.
    Uses the mirroring (np.pi - theta) to match coordinate system.
    """
    a2 = 2.0 * (np.pi - theta)
    phi_deg = (a2 / np.pi) * 180.0
    phi_deg = phi_deg % 360.0
    
    g = np.zeros_like(phi_deg)
    b = np.zeros_like(phi_deg)
    
    m1 = (phi_deg >= 0) & (phi_deg < 90)
    g[m1] = 0
    b[m1] = (phi_deg[m1] / 90.0) * 255.0
    
    m2 = (phi_deg >= 90) & (phi_deg < 180)
    g[m2] = ((phi_deg[m2] - 90.0) / 90.0) * 255.0
    b[m2] = 255.0
    
    m3 = (phi_deg >= 180) & (phi_deg < 270)
    g[m3] = 255.0
    b[m3] = (1.0 - (phi_deg[m3] - 180.0) / 90.0) * 255.0
    
    m4 = (phi_deg >= 270) & (phi_deg <= 360)
    g[m4] = (1.0 - (phi_deg[m4] - 270.0) / 90.0) * 255.0
    b[m4] = 0
    
    r = np.full_like(g, 255.0)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb

def extract_theta_from_strand_map(img_path):
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
    # G = (dy + 1)/2 * 255 => dy = (G/255)*2 - 1
    # B = (-dx + 1)/2 * 255 => -dx = (B/255)*2 - 1 => dx = 1 - (B/255)*2
    dy = (img[:, :, 1] / 255.0) * 2.0 - 1.0
    dx = 1.0 - (img[:, :, 2] / 255.0) * 2.0
    
    # Calculate undirected theta in [0, pi)
    theta = np.arctan2(dy, dx) % np.pi
    return theta

def main():
    pred_path = "./outputs_gabor/vtracer_lineart_strand_map.png"
    gt_path = "./HiSa_HiDa/strand_map/002c5a69cd4546957b02da0f762ed929.png"
    mask_path = "./HiSa_HiDa/seg/002c5a69cd4546957b02da0f762ed929.png"
    
    print("Loading mask and extracting undirected angles...")
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    
    theta_pred = extract_theta_from_strand_map(pred_path)
    theta_gt = extract_theta_from_strand_map(gt_path)
    
    print("Mapping to 180-degree colorwheel...")
    rgb_pred = custom_180deg_colorwheel(theta_pred)
    rgb_gt = custom_180deg_colorwheel(theta_gt)
    
    # Mask out background
    rgb_pred[~mask] = [0, 0, 0]
    rgb_gt[~mask] = [0, 0, 0]
    
    # Convert to uint8 images
    img_pred = Image.fromarray(rgb_pred.astype(np.uint8))
    img_gt = Image.fromarray(rgb_gt.astype(np.uint8))
    
    # Tile side-by-side
    w, h = img_pred.size
    tiled = Image.new("RGB", (w * 2 + 10, h), (40, 40, 40)) # dark grey separator
    tiled.paste(img_gt, (0, 0))
    tiled.paste(img_pred, (w + 10, 0))
    
    out_dir = "./outputs_gabor"
    os.makedirs(out_dir, exist_ok=True)
    tiled_path = f"{out_dir}/compare_180deg.png"
    tiled.save(tiled_path)
    print(f"Saved tiled comparison to {tiled_path}")

if __name__ == "__main__":
    main()
