"""
自动对齐并提取新图像的头发网格。
步骤：
1. 使用 Blender 渲染 GLB 获得正面图与相机参数。
2. 用 3DDFA_V2 提取人脸关键点，并使用相机参数反投影到 3D 空间。
3. 使用 Umeyama 算法计算 3D 关键点到标准头部模型关键点的变换（尺度、旋转、平移）。
4. 将该变换应用 to Pixal3D 生成的网格上，得到对齐的 hair_mesh_aligned_best.obj。
5. 运行 3D SDF 提取逻辑，剥离头部范围外的面片并过滤脖子，生成最终的 hair_mesh_sdf.obj。
"""

import sys, os
import numpy as np
import cv2
import trimesh
import open3d as o3d
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'external', '3DDFA_V2'))

from scripts.utils.align_glb_lmk import get_lmk, backproject_render_lmk
from scripts.utils.align_mesh_to_head import umeyama
from lib.mesh_util import load_obj_mesh
from lib.util.opt_lmk import load_point_ids
from scripts.recon_3d.extract_hair_mesh_3d import extract_hair_mesh_sdf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", default="results/test_pixal3d/output.glb", help="Pixal3D输出的GLB路径")
    parser.add_argument("--head", default="data/head_model.obj", help="标准头部模型OBJ路径")
    parser.add_argument("--out_dir", default="results/test_pixal3d", help="输出文件夹")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    glb_path = os.path.abspath(args.glb)
    
    # ─── Step 1: 调用 Blender 渲染 ───
    print("\n[Step 1] 调用 Blender 渲染正面视图...")
    render_cmd = [
        "conda", "run", "-n", "trellis2",
        "blender", "--background", "--python", "scripts/render/render_glb_front.py", "--",
        "--glb", glb_path,
        "--out_dir", os.path.abspath(args.out_dir)
    ]
    subprocess.run(render_cmd, check=True)
    print("Blender 渲染完成！")

    # ─── Step 2: 提取并反投影 3D 关键点 ───
    print("\n[Step 2] 提取渲染图的 3D 关键点...")
    render_img_path = os.path.join(args.out_dir, "render_front.png")
    render_cam_path = os.path.join(args.out_dir, "render_camera.npz")
    
    render_bgr = cv2.imread(render_img_path)
    pixal_lmk_2d = get_lmk(render_bgr)
    if pixal_lmk_2d is None:
        raise ValueError("无法在渲染正面图中检测到人脸关键点，请检查渲染图！")
    
    cam_npz = np.load(render_cam_path)
    ortho_scale = float(cam_npz['ortho_scale'])
    cam_R       = cam_npz['cam_R']
    cam_pos     = cam_npz['cam_pos']
    S           = int(cam_npz['img_size'])

    lx = (pixal_lmk_2d[:, 0] - S/2) / (S/2) * (ortho_scale / 2)
    ly = -(pixal_lmk_2d[:, 1] - S/2) / (S/2) * (ortho_scale / 2)
    lz = np.zeros_like(lx)

    local_origins = np.stack([lx, ly, lz], axis=1)
    ray_origins_blender = (cam_R @ local_origins.T).T + cam_pos
    ray_dir_blender = -cam_R[:, 2]

    # Convert rays from Blender (Z-up) to GLTF (Y-up)
    ray_origins_gltf = np.zeros_like(ray_origins_blender)
    ray_origins_gltf[:, 0] = ray_origins_blender[:, 0]
    ray_origins_gltf[:, 1] = ray_origins_blender[:, 2]
    ray_origins_gltf[:, 2] = -ray_origins_blender[:, 1]

    ray_dir_gltf = np.zeros(3)
    ray_dir_gltf[0] = ray_dir_blender[0]
    ray_dir_gltf[1] = ray_dir_blender[2]
    ray_dir_gltf[2] = -ray_dir_blender[1]

    # Use Open3D raycasting to get true 3D landmarks on the mesh
    raw_mesh = o3d.io.read_triangle_mesh(args.glb)
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(raw_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    rays = np.concatenate([ray_origins_gltf, np.tile(ray_dir_gltf, (len(lx), 1))], axis=1)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays_tensor)
    t_hit = ans['t_hit'].numpy()
    
    valid = t_hit < 1e5
    pixal_lmk_gltf = ray_origins_gltf + t_hit[:, None] * ray_dir_gltf

    # ─── Step 3: 加载头部模型关键点并计算对齐 ───
    print("\n[Step 3] 计算对齐参数 (Umeyama)...")
    head_verts, _ = load_obj_mesh(args.head)
    landmark_ids = load_point_ids("data/landmark_id_uschair.obj")
    head_lmk = head_verts[landmark_ids]
    
    c, R, t = umeyama(pixal_lmk_gltf[valid], head_lmk[valid])
    print(f"  Scale: {c:.4f}")
    print(f"  Rotation:\n{R}")
    print(f"  Translation: {t}")

    # ─── Step 4: 应用对齐到网格 ───
    print("\n[Step 4] 对齐 Pixal3D 几何网格...")
    
    # 应用 Umeyama 仿射变换
    verts = np.asarray(raw_mesh.vertices)
    aligned_verts = c * (verts @ R.T) + t
    raw_mesh.vertices = o3d.utility.Vector3dVector(aligned_verts)
    raw_mesh.compute_vertex_normals()
    
    print("\n[Step 4.5] 运行密集 ICP 对齐 (面部区域微调)...")
    head_mesh = o3d.geometry.TriangleMesh()
    head_mesh.vertices = o3d.utility.Vector3dVector(head_verts)
    head_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(o3d.io.read_triangle_mesh(args.head).triangles))
    head_mesh.compute_vertex_normals()
    
    lmk_min = head_lmk.min(axis=0)
    lmk_max = head_lmk.max(axis=0)
    margin = 0.05
    bbox = o3d.geometry.AxisAlignedBoundingBox(lmk_min - margin, lmk_max + margin)
    
    head_face = head_mesh.crop(bbox)
    raw_face = raw_mesh.crop(bbox)
    
    head_pc = o3d.geometry.PointCloud()
    head_pc.points = head_face.vertices
    head_pc.normals = head_face.vertex_normals
    
    raw_pc = o3d.geometry.PointCloud()
    raw_pc.points = raw_face.vertices
    raw_pc.normals = raw_face.vertex_normals
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        raw_pc, head_pc, 0.05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
    print(f"  ICP 匹配度 (Fitness): {reg_p2p.fitness:.4f}")
    raw_mesh.transform(reg_p2p.transformation)
    
    aligned_out_path = os.path.join(args.out_dir, "hair_mesh_aligned_best.obj")
    o3d.io.write_triangle_mesh(aligned_out_path, raw_mesh)
    print(f"  对齐网格已保存至: {aligned_out_path}")

    # ─── 保存 GLB → 世界坐标的组合变换 ───
    # Umeyama: world_pt = c * R @ glb_pt + t
    # ICP 在世界坐标内微调，变换矩阵为 reg_p2p.transformation [4x4]
    # 合并后: world_pt = icp_T @ [c*R@glb_pt + t; 1]
    # 保存供 align_glb_lmk.py 使用，将 GLB 坐标系的 calib 转换到世界坐标
    glb_to_world_path = os.path.join(args.out_dir, "glb_to_world.npz")
    np.savez(
        glb_to_world_path,
        umeyama_scale=np.array([c], dtype=np.float32),
        umeyama_R=R.astype(np.float32),
        umeyama_t=t.astype(np.float32),
        icp_T=reg_p2p.transformation.astype(np.float32),
    )
    print(f"  GLB→世界坐标变换已保存: {glb_to_world_path}")

    # ─── Step 5: FLAME 头发网格提取 ───
    print("\n[Step 5] 基于 FLAME 模型提取头发表面网格...")
    out_sdf_path = os.path.join(args.out_dir, "hair_mesh_sdf.obj")
    from scripts.recon_3d.extract_hair_mesh_flame import extract_hair_mesh_flame
    extract_hair_mesh_flame(
        aligned_mesh_path=aligned_out_path,
        out_path=out_sdf_path
    )
    print("\n[完成] 头发网格对齐与提取全流程成功结束！")

if __name__ == '__main__':
    main()
