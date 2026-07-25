"""
使用 3D SDF 方法提取头发网格 (User's Method)。
算法：
1. 读取使用 LMK 拟合好的 3D Mesh (hair_mesh_aligned_best.obj)
2. 读取标准头部模型 (data/head_model.obj)
3. 计算拟合 Mesh 到头部表面的 SDF (Signed Distance Field)
4. 将头部模型范围以外的（SDF > threshold）且位于脖子以上的（Y > 1.63）判定为头发
"""
import sys, os
import numpy as np
import open3d as o3d
import trimesh
import argparse

def extract_hair_mesh_sdf(aligned_mesh_path, head_model_path, out_path, sdf_thresh=0.003, back_sdf_thresh=-0.015, neck_y_thresh=1.63):
    print(f"[1/4] 加载头部模型: {head_model_path}")
    head = o3d.io.read_triangle_mesh(head_model_path)
    head_t = o3d.t.geometry.TriangleMesh.from_legacy(head)
    
    print(f"[2/4] 加载已对齐的网格: {aligned_mesh_path}")
    aligned = o3d.io.read_triangle_mesh(aligned_mesh_path)
    
    verts = np.asarray(aligned.vertices)
    faces = np.asarray(aligned.triangles)
    print(f"  网格总顶点: {len(verts)}，总面数: {len(faces)}")

    print("[3/4] 计算 SDF 距离场 (前后分区域阈值过滤)...")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(head_t)
    sdf = scene.compute_signed_distance(verts.astype(np.float32)).numpy()

    # 判定条件：
    # 正面 (Z > 0): sdf > sdf_thresh
    # 背部 (Z <= 0): 使用更宽松的 back_sdf_thresh (防止头部模型在后脑勺压穿生成的头发网格)
    is_front = verts[:, 2] > 0.0
    thresh = np.where(is_front, sdf_thresh, back_sdf_thresh)

    outside = sdf > thresh
    not_neck = verts[:, 1] > neck_y_thresh
    is_hair = outside & not_neck
    
    # 提取三个顶点都在头发区域的面
    face_mask = is_hair[faces[:, 0]] & is_hair[faces[:, 1]] & is_hair[faces[:, 2]]
    n_hair_faces = face_mask.sum()
    print(f"  符合条件的头发面数: {n_hair_faces} ({100*n_hair_faces/len(faces):.1f}%)")

    if n_hair_faces == 0:
        print("  [警告] 没有找到符合条件的头发面！")
        sys.exit(1)

    print("[4/4] 保存头发网格...")
    hair_faces = faces[face_mask]
    used = np.unique(hair_faces)
    old2new = np.full(len(verts), -1, dtype=np.int64)
    old2new[used] = np.arange(len(used))
    new_faces = old2new[hair_faces]
    new_verts = verts[used]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        for v3 in new_verts:
            f.write(f'v {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n')
        for face in new_faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
            
    print(f"[完成] 输出保存至: {out_path}")
    print(f"  保存的顶点数: {len(new_verts)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", default="results/test_pixal3d/hair_mesh_aligned_best.obj", help="对齐后的网格")
    parser.add_argument("--head", default="data/head_model.obj", help="头部标准模型")
    parser.add_argument("--out", default="results/test_pixal3d/hair_mesh_sdf.obj", help="输出路径")
    parser.add_argument("--sdf_thresh", type=float, default=0.003, help="正面判定为在外部的距离阈值")
    parser.add_argument("--back_sdf_thresh", type=float, default=-0.015, help="后脑勺放宽的距离阈值 (防止切穿背部)")
    parser.add_argument("--neck_y", type=float, default=1.63, help="脖子截断 Y 坐标")
    args = parser.parse_args()
    
    extract_hair_mesh_sdf(args.aligned, args.head, args.out, args.sdf_thresh, args.back_sdf_thresh, args.neck_y)
