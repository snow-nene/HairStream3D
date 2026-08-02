"""
可视化相机标定 front.npy 投影人脸网格叠加效果图
"""
import os
import sys
import argparse
import numpy as np
import cv2
import open3d as o3d
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.recon_3d.recon3D import load_calib

def visualize_real_img_alignment(img_id, input_img_path=None, out_dir=None):
    data_dir = os.path.join("results", "multiview_data", img_id)
    calib_path = os.path.join(data_dir, "maps", "param", "front.npy")
    
    if out_dir is None:
        out_dir = os.path.join(data_dir, "maps", "param")
    os.makedirs(out_dir, exist_ok=True)
    
    if input_img_path is None:
        raw_txt = os.path.join(data_dir, "raw_img_path.txt")
        if os.path.exists(raw_txt):
            input_img_path = open(raw_txt).read().strip()
        else:
            input_img_path = f"results/real_imgs/{img_id}.png"
            
    if not os.path.exists(calib_path):
        print(f"[WARN] 相机标定文件未找到: {calib_path}，跳过对齐可视化。")
        return
    if not os.path.exists(input_img_path):
        print(f"[WARN] 原始输入图像未找到: {input_img_path}，跳过对齐可视化。")
        return
        
    print(f"--> [Align Vis] 正在加载标定矩阵: {calib_path}")
    calib = load_calib(calib_path, loadSize=1024)
    if isinstance(calib, np.ndarray):
        calib = torch.from_numpy(calib)
        
    bg_img = cv2.imread(input_img_path)
    if bg_img is None:
        print(f"[ERROR] 无法读取图片: {input_img_path}")
        return
    bg_img = cv2.resize(bg_img, (1024, 1024))
        
    # Load 3D head model
    head_mesh_path = "data/head_model.obj"
    if not os.path.exists(head_mesh_path):
        print(f"[ERROR] 未找到 3D 头部模板: {head_mesh_path}")
        return
        
    head_mesh = o3d.io.read_triangle_mesh(head_mesh_path)
    verts = np.asarray(head_mesh.vertices).T # [3, N]
    
    N = verts.shape[1]
    verts_homo = np.vstack([verts, np.ones((1, N))]) # [4, N]
    
    pts_proj = np.matmul(calib.numpy(), verts_homo) # [4, N]
    uv = pts_proj[:2, :] / (pts_proj[3:4, :] + 1e-8) # [-1, 1]
    
    # Map [-1, 1] to pixel [0, 1023]
    px = ((uv[0] + 1.0) * 0.5 * 1023).astype(np.int32)
    py = ((uv[1] + 1.0) * 0.5 * 1023).astype(np.int32)
    
    canvas = bg_img.copy()
    valid_mask = (px >= 0) & (px < 1024) & (py >= 0) & (py < 1024)
    
    px_valid = px[valid_mask]
    py_valid = py[valid_mask]
    
    for x, y in zip(px_valid[::3], py_valid[::3]): # Subsample green dots
        cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)
        
    out_path = os.path.join(out_dir, "real_face_alignment.png")
    cv2.imwrite(out_path, canvas)
    print(f"  ✓ 标定对齐效果图已保存至: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Head Alignment on Real Image")
    parser.add_argument("--img_id", required=True, help="Image ID")
    parser.add_argument("--input_img", default=None, help="Input real image path")
    parser.add_argument("--out_dir", default=None, help="Output directory")
    args = parser.parse_args()
    
    visualize_real_img_alignment(args.img_id, args.input_img, args.out_dir)

if __name__ == "__main__":
    main()
