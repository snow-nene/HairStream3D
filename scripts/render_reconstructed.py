#!/usr/bin/env python3
"""
重建结果渲染与可视化脚本 (Render Reconstructed 3D Hair & Mesh)

利用 src/utils/render_utils.py 中的工具 (如 load_cameras, render_feature_map,
render_meshes_zbuf 等) 以及 PyTorch3D / Matplotlib 将 3D 重建得到的头发发丝 (.ply/.hair/.obj)
和头部 Mesh (.obj) 重新渲染为 2D 特征图 (Direction Map, Depth Map, Silhouette 等) 及组合预览图。

使用方法:
  python scripts/render_reconstructed.py \
      --hair_path results/real_imgs/hair3D/0d285f5be7fa09c3dbbf1c9334047888.ply \
      --param_path results/real_imgs/param/0d285f5be7fa09c3dbbf1c9334047888.npy \
      --mesh_path data/head_model.obj \
      --output_dir results/rendered_previews
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# 将项目根目录添加到 sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.render_utils import (
    load_cameras,
    render_feature_map,
    render_meshes_zbuf,
)

try:
    from pytorch3d.io import load_objs_as_meshes
    from pytorch3d.structures import Meshes
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False


def read_ply_lines(ply_path):
    """
    读取 Open3D / LineSet 保存的 PLY 文件中的点和线段，
    并将线段按顺序或拓扑重新拆解为连续发丝 strands (N, P, 3)。
    """
    points = []
    lines = []
    
    with open(ply_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = True
        num_verts = 0
        num_lines = 0
        for line in f:
            line_str = line.strip()
            if header:
                if line_str.startswith("element vertex"):
                    num_verts = int(line_str.split()[-1])
                elif line_str.startswith("element line") or line_str.startswith("element edge"):
                    num_lines = int(line_str.split()[-1])
                elif line_str == "end_header":
                    header = False
                continue
            
            parts = line_str.split()
            if len(points) < num_verts:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            elif len(lines) < num_lines:
                lines.append([int(parts[0]), int(parts[1])])
                
    points = np.array(points, dtype=np.float32)
    lines = np.array(lines, dtype=int)
    
    if len(lines) == 0:
        # 如果文件中仅点坐标或无显式 lines 结构，按默认片段拆分
        num_pts = len(points)
        num_strands = max(1, num_pts // 100)
        pts_per_strand = num_pts // num_strands
        strands = points[:num_strands * pts_per_strand].reshape(num_strands, pts_per_strand, 3)
        return torch.from_numpy(strands).float()

    # 根据连通线段组合成链条/发丝
    # 假设线段是连续排列的
    strand_list = []
    curr_strand = [points[lines[0][0]], points[lines[0][1]]]
    
    for i in range(1, len(lines)):
        prev_end = lines[i-1][1]
        curr_start = lines[i][0]
        if curr_start == prev_end:
            curr_strand.append(points[lines[i][1]])
        else:
            if len(curr_strand) >= 2:
                strand_list.append(curr_strand)
            curr_strand = [points[lines[i][0]], points[lines[i][1]]]
    if len(curr_strand) >= 2:
        strand_list.append(curr_strand)
        
    # 对每条发丝重采样到固定点数 (比如 32 点)
    target_P = 32
    resampled_strands = []
    for st in strand_list:
        st_arr = np.array(st, dtype=np.float32)
        if len(st_arr) < 2:
            continue
        # 线性插值重采样
        idx_old = np.linspace(0, 1, len(st_arr))
        idx_new = np.linspace(0, 1, target_P)
        st_new = np.zeros((target_P, 3), dtype=np.float32)
        for d in range(3):
            st_new[:, d] = np.interp(idx_new, idx_old, st_arr[:, d])
        resampled_strands.append(st_new)
        
    if len(resampled_strands) == 0:
        # fallback
        strands = points[:(len(points)//32)*32].reshape(-1, 32, 3)
        return torch.from_numpy(strands).float()
        
    resampled_strands = np.stack(resampled_strands, axis=0) # (N, 32, 3)
    return torch.from_numpy(resampled_strands).float()


def direction_to_rgb(orien_map):
    """
    将渲染输出的方向图 orien_map (H, W, 3) 转为可视化的 RGB Colorwheel 样式图。
    """
    if isinstance(orien_map, torch.Tensor):
        orien_map = orien_map.detach().cpu().numpy()
        
    dx = orien_map[..., 0]
    dy = orien_map[..., 1]
    
    rgb = np.zeros((*dx.shape, 3), dtype=np.uint8)
    mask = (np.abs(dx) + np.abs(dy)) > 1e-5
    
    rgb[..., 0] = (mask * 255).astype(np.uint8)
    rgb[..., 1] = np.clip((dy + 1.0) / 2.0 * 255 * mask, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip((-dx + 1.0) / 2.0 * 255 * mask, 0, 255).astype(np.uint8)
    
def render_fallback_opencv(strands, cameras, img_size):
    import torch.nn.functional as F
    from src.utils.render_utils import transform_points_to_ndc
    
    W, H = img_size
    strands_cpu = strands.cpu()
    N, P, _ = strands_cpu.shape
    
    points_packed = strands_cpu.reshape(-1, 3)
    points_ndc = transform_points_to_ndc(cameras, points_packed).reshape(N, P, 3).numpy()
    
    orien_img = np.zeros((H, W, 3), dtype=np.uint8)
    depth_buffer = np.full((H, W), np.inf, dtype=np.float32)
    
    tangents = F.normalize(strands_cpu[:, 1:, :] - strands_cpu[:, :-1, :], dim=-1).numpy()
    
    for i in range(N):
        for j in range(P - 1):
            p1 = points_ndc[i, j]
            p2 = points_ndc[i, j + 1]
            x1, y1 = int((p1[0] + 1) * 0.5 * W), int((1 - p1[1]) * 0.5 * H)
            x2, y2 = int((p2[0] + 1) * 0.5 * W), int((1 - p2[1]) * 0.5 * H)
            
            if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                z = (p1[2] + p2[2]) / 2.0
                if z < depth_buffer[y1, x1]:
                    depth_buffer[y1, x1] = z
                    
                dx, dy = tangents[i, j, 0], tangents[i, j, 1]
                r = 255
                g = int(np.clip((dy + 1.0) / 2.0 * 255, 0, 255))
                b = int(np.clip((-dx + 1.0) / 2.0 * 255, 0, 255))
                cv2.line(orien_img, (x1, y1), (x2, y2), (b, g, r), 1)

    silh_img = (np.sum(orien_img, axis=-1) > 0).astype(np.uint8) * 255
    valid_depth = depth_buffer[depth_buffer != np.inf]
    if len(valid_depth) > 0:
        dmin, dmax = valid_depth.min(), valid_depth.max()
        depth_norm = np.clip((depth_buffer - dmin) / (dmax - dmin + 1e-5), 0, 1)
        depth_img = ((1 - depth_norm) * (depth_buffer != np.inf) * 255).astype(np.uint8)
    else:
        depth_img = np.zeros((H, W), dtype=np.uint8)
        
    return orien_img, depth_img, silh_img


def render_reconstruction(
    hair_path,
    param_path,
    mesh_path=None,
    output_dir="./results/rendered_previews",
    img_size=(512, 512),
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    主渲染逻辑函数：
      1. 加载相继承包 params (.npy/.json) -> load_cameras
      2. 读取 3D Hair Strands (.ply)
      3. (可选) 加载 3D Head Mesh (.obj) 计算深度 Z-buffer 用于遮挡
      4. 调用 render_feature_map 渲染 2D 贴图 (Direction Map, Depth Map, Silhouette 等)
      5. 保存并输出对比/拼接图片
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(hair_path))[0]
    print(f"[Render] Processing: {base_name}")
    
    # 1. 加载相机参数
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Camera parameter file not found: {param_path}")
    cameras = load_cameras(param_path, device=device)
    
    # 2. 读取发丝数据
    if not os.path.exists(hair_path):
        raise FileNotFoundError(f"Hair 3D file not found: {hair_path}")
        
    if hair_path.endswith('.ply'):
        strands = read_ply_lines(hair_path).to(device)
    elif hair_path.endswith('.npy'):
        strands_np = np.load(hair_path) # (N, P, 3)
        strands = torch.from_numpy(strands_np).float().to(device)
    else:
        raise ValueError(f"Unsupported hair file format: {hair_path}")
        
    print(f"Loaded strands shape: {strands.shape}")
    
    # 3. 头部 Mesh 遮挡 Z-Buffer (若有 PyTorch3D 且提供了 Mesh)
    mesh_zbuf = None
    if mesh_path and os.path.exists(mesh_path) and HAS_PYTORCH3D:
        try:
            print(f"Loading head mesh for occlusion: {mesh_path}")
            mesh = load_objs_as_meshes([mesh_path], device=device)
            mesh_zbuf = render_meshes_zbuf(mesh, cameras, img_size)
        except Exception as e:
            print(f"[Warning] Failed to render mesh zbuf: {e}")
            mesh_zbuf = None

    # 4. 核心渲染 (若无 PyTorch3D 软件环境，自动使用 OpenCV 投影渲染方案)
    if HAS_PYTORCH3D:
        try:
            silh, depth, cov, clump, rgb, label, orien = render_feature_map(
                strands=strands,
                cameras=cameras,
                img_size=img_size,
                mesh_zbuf=mesh_zbuf
            )
            orien_rgb = direction_to_rgb(orien)
            depth_np = (depth.detach().cpu().numpy() * 255).astype(np.uint8)
            silh_np = (silh.detach().cpu().numpy() * 255).astype(np.uint8)
        except Exception as e:
            print(f"[Warning] PyTorch3D render failed ({e}), falling back to OpenCV software rasterizer...")
            orien_rgb, depth_np, silh_np = render_fallback_opencv(strands, cameras, img_size)
    else:
        print("[Info] PyTorch3D rasterizer not available, using OpenCV software rasterizer...")
        orien_rgb, depth_np, silh_np = render_fallback_opencv(strands, cameras, img_size)

    
    # 保存单张结果图
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_rendered_orien.png"), cv2.cvtColor(orien_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_rendered_depth.png"), depth_np)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_rendered_silh.png"), silh_np)
    
    # 组合网格图预览 (Combined Grid)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orien_rgb)
    axes[0].set_title("Rendered Orientation Field")
    axes[0].axis('off')
    
    axes[1].imshow(depth_np, cmap='gray')
    axes[1].set_title("Rendered Depth Map")
    axes[1].axis('off')
    
    axes[2].imshow(silh_np, cmap='gray')
    axes[2].set_title("Rendered Silhouette")
    axes[2].axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, f"{base_name}_preview_grid.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    
    print(f"[Success] Rendered files saved to: {output_dir}")
    return grid_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Reconstructed 3D Hair and Mesh")
    parser.add_argument("--hair_path", type=str, required=True, help="Path to 3D reconstructed hair PLY/NPY file")
    parser.add_argument("--param_path", type=str, required=True, help="Path to camera parameter .npy or .json file")
    parser.add_argument("--mesh_path", type=str, default="data/head_model.obj", help="Path to head mesh .obj file")
    parser.add_argument("--output_dir", type=str, default="results/rendered_previews", help="Directory to save output renderings")
    parser.add_argument("--img_size", type=int, default=512, help="Output image size (width & height)")

    args = parser.parse_args()
    render_reconstruction(
        hair_path=args.hair_path,
        param_path=args.param_path,
        mesh_path=args.mesh_path,
        output_dir=args.output_dir,
        img_size=(args.img_size, args.img_size)
    )
