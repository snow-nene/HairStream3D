"""
Standalone High-Quality 2D Hair Strand Renderer (HairStep / Laplace PDE Output)

Projects 3D hair strands (.ply) onto the 2D image plane using camera calibration (.npy),
rendering realistic colored hair strands with anti-aliasing and depth sorting.
"""

import os
import sys
import numpy as np
import cv2
import open3d as o3d
from PIL import Image

def load_calib_matrix(param_path, load_size=512):
    """Load calibration data and build 4x4 projection matrix."""
    param = np.load(param_path, allow_pickle=True)
    ortho_ratio = param.item().get('ortho_ratio')
    scale       = param.item().get('scale')
    center      = param.item().get('center')
    R           = param.item().get('R')

    translate = -np.matmul(R, center).reshape(3, 1)
    extrinsic = np.concatenate([R, translate], axis=1)
    extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

    scale_intrinsic = np.identity(4)
    scale_intrinsic[0, 0] =  scale / ortho_ratio
    scale_intrinsic[1, 1] = -scale / ortho_ratio
    scale_intrinsic[2, 2] =  scale / ortho_ratio

    uv_intrinsic = np.identity(4)
    uv_intrinsic[0, 0] = 1.0 / float(load_size // 2)
    uv_intrinsic[1, 1] = 1.0 / float(load_size // 2)
    uv_intrinsic[2, 2] = 1.0 / float(load_size // 2)

    trans_intrinsic = np.identity(4)
    intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
    calib = np.matmul(intrinsic, extrinsic)
    return calib

def project_points(pts, calib, width=512, height=512):
    """Project (N, 3) 3D points to (N, 2) 2D pixel coordinates and Z depth."""
    N = len(pts)
    pts_homo = np.hstack([pts, np.ones((N, 1))])  # (N, 4)
    pts_proj = (calib @ pts_homo.T).T             # (N, 4)

    u = pts_proj[:, 0]
    v = pts_proj[:, 1]
    z = pts_proj[:, 2]

    # Map UV in [-1, 1] to pixel coordinates [0, width]
    px = (u + 1.0) * 0.5 * width
    py = (1.0 - v) * 0.5 * height  # flip y for image space

    return np.column_stack([px, py]), z

def render_strands_to_image(ply_path, param_path, bg_img_path=None, width=512, height=512):
    """Render 3D strands (.ply) into a 2D image."""
    calib = load_calib_matrix(param_path, load_size=width)

    # Load 3D strands
    lineset = o3d.io.read_line_set(ply_path)
    pts = np.asarray(lineset.points)
    lines = np.asarray(lineset.lines)

    if len(pts) == 0 or len(lines) == 0:
        print("Warning: empty strand data!")
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Project all 3D points to 2D
    pts_2d, depths = project_points(pts, calib, width, height)

    # Prepare background image
    if bg_img_path and os.path.exists(bg_img_path):
        canvas = cv2.imread(bg_img_path)
        canvas = cv2.resize(canvas, (width, height))
    else:
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light background

    # Create hair strand layer on PURE WHITE canvas (to clearly show generated structure)
    white_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    hair_only = white_canvas.copy()

    # Create hair strand overlay on original image
    hair_overlay = canvas.copy()

    # Sort lines by depth for painter's algorithm
    line_depths = []
    for line in lines:
        p1_idx, p2_idx = line[0], line[1]
        z_avg = (depths[p1_idx] + depths[p2_idx]) / 2.0
        line_depths.append(z_avg)
    
    sorted_indices = np.argsort(line_depths)[::-1]  # Far to near

    # Draw hair strands with color gradient based on orientation/depth
    for idx in sorted_indices:
        line = lines[idx]
        p1 = pts_2d[line[0]]
        p2 = pts_2d[line[1]]

        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))

        # Skip lines outside image canvas
        if max(x1, x2) < 0 or min(x1, x2) >= width or max(y1, y2) < 0 or min(y1, y2) >= height:
            continue

        # Compute strand 2D direction for dynamic hair coloring
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.arctan2(dy, dx)

        # Hair color: Natural realistic dark brown / chestnut with direction tint
        r = int(60 + 40 * np.cos(angle))
        g = int(40 + 30 * np.sin(angle))
        b = int(30 + 20 * np.cos(angle))

        cv2.line(hair_only, (x1, y1), (x2, y2), (b, g, r), thickness=2, lineType=cv2.LINE_AA)
        cv2.line(hair_overlay, (x1, y1), (x2, y2), (b, g, r), thickness=2, lineType=cv2.LINE_AA)

    # Blend overlay for anti-aliasing
    final_overlay = cv2.addWeighted(canvas, 0.4, hair_overlay, 0.6, 0)
    
    # Combine horizontally: Original | Hair Only | Overlay
    combined = np.hstack([canvas, hair_only, final_overlay])
    return combined

if __name__ == '__main__':
    hair_dir  = "results/real_imgs/hair3D_laplace"
    param_dir = "results/real_imgs/param"
    img_dir   = "results/real_imgs/resized_img"
    out_dir   = "results/final_rendered_hair"

    os.makedirs(out_dir, exist_ok=True)
    items = [f for f in os.listdir(hair_dir) if f.endswith('.ply')]

    print(f"Rendering {len(items)} samples into side-by-side final 2D hair images...")
    for item in items:
        name = item[:-4]
        hair_path  = os.path.join(hair_dir, item)
        param_path = os.path.join(param_dir, name + ".npy")
        bg_path    = os.path.join(img_dir, name + ".png")
        if not os.path.exists(bg_path):
            bg_path = os.path.join(img_dir, name + ".jpg")

        out_path   = os.path.join(out_dir, name + "_final_hair.png")

        if os.path.exists(param_path):
            rendered = render_strands_to_image(hair_path, param_path, bg_path)
            cv2.imwrite(out_path, rendered)
            print(f"  -> Saved final rendered hair: {out_path}")
