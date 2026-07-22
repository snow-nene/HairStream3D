#!/usr/bin/env python3
"""
3D 头发/头部模型可视化渲染脚本 (Render 3D Hair & Head Mesh)

使用纯 Python (Matplotlib 3D / OpenCV) 或 Blender (可选) 渲染重建得到的 3D 头发 (PLY/OBJ) 与头部 Mesh，
无需 PyTorch3D 或 可微渲染器 (3D Gaussian Rasterizer)。

使用方法:
  pixi run python scripts/render_hair.py \
      --hair_path results/real_imgs/hair3D/0d285f5be7fa09c3dbbf1c9334047888.ply \
      --mesh_path data/head_model.obj \
      --output_dir results/rendered_previews
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def read_ply_strands(ply_path):
    """
    读取 3D 头发 PLY 文件中的发丝顶点与线段拓扑，还原为连续 3D 发丝 (N_strands, P_points, 3)
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
        # 没有 explicit lines 结构，按默认段数拆分
        num_pts = len(points)
        pts_per_strand = 100
        num_strands = num_pts // pts_per_strand
        if num_strands == 0:
            return points[None, :, :]
        return points[:num_strands * pts_per_strand].reshape(num_strands, pts_per_strand, 3)

    # 根据连续 line 组合成发丝
    strands = []
    curr_strand = [points[lines[0][0]], points[lines[0][1]]]
    
    for i in range(1, len(lines)):
        prev_end = lines[i-1][1]
        curr_start = lines[i][0]
        if curr_start == prev_end:
            curr_strand.append(points[lines[i][1]])
        else:
            if len(curr_strand) >= 2:
                strands.append(np.array(curr_strand, dtype=np.float32))
            curr_strand = [points[lines[i][0]], points[lines[i][1]]]
    if len(curr_strand) >= 2:
        strands.append(np.array(curr_strand, dtype=np.float32))
        
    return strands


def load_obj_verts(obj_path):
    """
    读取 3D 头部 OBJ Mesh 模型的顶点坐标
    """
    verts = []
    with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float32)


def render_hair_3d_views(hair_path, mesh_path=None, output_dir="./results/rendered_previews", stride=5):
    """
    使用 Matplotlib 3D / OpenCV 渲染多视角 3D 头发和头部 Mesh 叠加图像
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(hair_path))[0]
    print(f"[Render] Loading hair from: {hair_path}")
    
    strands = read_ply_strands(hair_path)
    print(f"[Render] Total strands: {len(strands)}")

    mesh_verts = None
    if mesh_path and os.path.exists(mesh_path):
        print(f"[Render] Loading head mesh from: {mesh_path}")
        mesh_verts = load_obj_verts(mesh_path)

    # 创建多视角画布 (Front, Side, Isometric)
    fig = plt.figure(figsize=(15, 5), facecolor='black')
    views = [
        ("Front View", (0, 0)),
        ("Side View", (0, 90)),
        ("Perspective View", (20, -45))
    ]
    
    # 抽取部分发丝进行快速渲染
    rendered_strands = strands[::stride] if isinstance(strands, list) else strands[::stride]

    for idx, (title, (elev, azim)) in enumerate(views):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.set_facecolor('black')
        ax.axis('off')
        
        # 1. 绘制头部 Mesh 顶点云
        if mesh_verts is not None and len(mesh_verts) > 0:
            m_sub = mesh_verts[::10]
            ax.scatter(m_sub[:, 0], m_sub[:, 1], m_sub[:, 2], c='gray', s=0.2, alpha=0.3)

        # 2. 绘制发丝 (根据梯度或方向上色)
        for st in rendered_strands:
            ax.plot(st[:, 0], st[:, 1], st[:, 2], color='#d69c69', linewidth=0.6, alpha=0.8)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, color='white', fontsize=12)

    plt.tight_layout()
    out_png = os.path.join(output_dir, f"{base_name}_rendered_3d.png")
    plt.savefig(out_png, dpi=200, facecolor='black')
    plt.close()
    
    print(f"[Success] 3D hair rendering saved to: {out_png}")
    return out_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render 3D Hair and Head Mesh")
    parser.add_argument("--hair_path", type=str, required=True, help="Path to reconstructed 3D hair PLY/OBJ file")
    parser.add_argument("--mesh_path", type=str, default="data/head_model.obj", help="Path to head mesh OBJ file")
    parser.add_argument("--output_dir", type=str, default="results/rendered_previews", help="Output directory")
    parser.add_argument("--stride", type=int, default=3, help="Subsampling stride for strands")

    args = parser.parse_args()
    render_hair_3d_views(
        hair_path=args.hair_path,
        mesh_path=args.mesh_path,
        output_dir=args.output_dir,
        stride=args.stride
    )
