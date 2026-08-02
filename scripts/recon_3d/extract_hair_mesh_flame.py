import numpy as np
import open3d as o3d
import torch
import os

def extract_hair_mesh_flame(aligned_mesh_path, out_path):
    # 1. Load meshes
    # Use the pre-aligned FLAME mesh which is perfectly aligned to data/head_model.obj
    flame_mesh = o3d.io.read_triangle_mesh('data/flame_aligned_to_head.obj')
    pixal_mesh = o3d.io.read_triangle_mesh(aligned_mesh_path)
    
    # 2. No ICP needed! pixal_mesh is already aligned to data/head_model.obj
    # by scripts/utils/align_glb_lmk.py
    
    print("  [FLAME Extract] Skipping cropping to preserve the complete head model as requested.")
    
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    o3d.io.write_triangle_mesh(out_path, pixal_mesh)
    print(f"  [FLAME Extract] Saved full mesh to: {out_path}")

if __name__ == '__main__':
    extract_hair_mesh_flame('results/multiview_data/0a1ba3dbefc8934ab60577c5c91f66a0/pixal3d/hair_mesh_aligned_best.obj', 'results/multiview_data/0a1ba3dbefc8934ab60577c5c91f66a0/pixal3d/hair_mesh_sdf.obj')
