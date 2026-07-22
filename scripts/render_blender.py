#!/usr/bin/env python3
"""
使用 Blender 渲染模板 (assets/render_template.blend) 可视化 3D 头发发丝

主要排查与修正点：
  1. 坐标系与位置对齐：将发丝变换到模板中 `head_smooth` 模型的头部顶端位置 (Z: 1.468 ~ 1.886)。
  2. 发丝倒角与粗细 (Bevel Depth)：曲线必须给 `bevel_depth` (默认 0.0015~0.002)，否则默认为无限细的线段在渲染器中不可见。
  3. 环境贴图重连：修复 Windows 绝对路径导致的 HDRI 贴图丢失问题，重定向连接到 `assets/ENV.Environment.exr`。
  4. 材质分配：为曲线对象正确配置并绑定 Principled BSDF 材质。

使用方法:
  blender -b assets/render_template.blend --python scripts/render_blender.py -- --hair_path <path_to_ply> --output_path <path_to_png>
"""

import bpy
import sys
import os
import argparse
import numpy as np
from mathutils import Matrix


def read_ply_strands(ply_path):
    """
    读取 3D 头发 PLY 文件中的点与线段拓扑 (自动兼容 Open3D 二进制格式 binary_little_endian 与 ASCII 格式)
    """
    with open(ply_path, 'rb') as f:
        header_text = ""
        while True:
            line_bytes = f.readline()
            if not line_bytes:
                break
            line_str = line_bytes.decode('ascii', errors='ignore')
            header_text += line_str
            if line_str.strip() == "end_header":
                break

        is_binary = "binary_little_endian" in header_text
        num_verts = 0
        num_edges = 0
        is_double = "property double" in header_text

        for line in header_text.split('\n'):
            line = line.strip()
            if line.startswith("element vertex"):
                num_verts = int(line.split()[-1])
            elif line.startswith("element edge") or line.startswith("element line"):
                num_edges = int(line.split()[-1])

        if is_binary:
            dtype_vert = '<f8' if is_double else '<f4'
            vert_bytes = f.read(num_verts * 3 * (8 if is_double else 4))
            points = np.frombuffer(vert_bytes, dtype=dtype_vert).reshape(num_verts, 3).astype(np.float32)

            edge_bytes = f.read(num_edges * 2 * 4)
            lines = np.frombuffer(edge_bytes, dtype='<i4').reshape(num_edges, 2)
        else:
            # ASCII 解析
            body_text = f.read().decode('ascii', errors='ignore')
            lines_str = [l.strip() for l in body_text.split('\n') if l.strip()]
            points = []
            lines = []
            for l in lines_str[:num_verts]:
                parts = l.split()
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            for l in lines_str[num_verts:num_verts + num_edges]:
                parts = l.split()
                lines.append([int(parts[0]), int(parts[1])])
            points = np.array(points, dtype=np.float32)
            lines = np.array(lines, dtype=int)

    if len(lines) == 0:
        num_pts = len(points)
        pts_per_strand = 100
        num_strands = num_pts // pts_per_strand
        if num_strands == 0:
            return [points]
        return [points[i*pts_per_strand:(i+1)*pts_per_strand] for i in range(num_strands)]

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


def create_curves_from_strands(name, strands, bevel_depth=0.0018):
    """
    创建带三维管径粗细的 3D Hair Curve 对象，并对齐到场景头顶位置
    """
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = bevel_depth  # 给发丝赋予实体管径粗细
    curve.use_fill_caps = True
    
    for strand in strands:
        # 旋转并变换发丝中心坐标至模板中 head_smooth 模型的顶端 (Z: 1.6 ~ 1.8)
        # PLY coordinates are Y-up (Y is in [1.5, 1.9]). Blender is Z-up.
        # So we map PLY (x, y, z) -> Blender (x, -z, y).
        strand_world = np.zeros_like(strand)
        strand_world[:, 0] = strand[:, 0]
        strand_world[:, 1] = -strand[:, 2]
        strand_world[:, 2] = strand[:, 1]
        
        s = curve.splines.new("POLY")
        s.points.add(strand_world.shape[0] - 1)
        
        # 4D 齐次坐标 [x, y, z, w=1.0]
        strand_4d = np.hstack((strand_world, np.ones((strand_world.shape[0], 1)))).astype(np.float32)
        s.points.foreach_set("co", strand_4d.flatten())

    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    return obj


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render Blender Template with Hair")
    parser.add_argument("--hair_path", type=str, required=True, help="Path to 3D hair PLY/OBJ file")
    parser.add_argument("--template_blend", type=str, default="assets/render_template.blend", help="Path to template blend file")
    parser.add_argument("--output_path", type=str, default="results/rendered_previews/blender_render.png", help="Output PNG path")
    parser.add_argument("--bevel_depth", type=float, default=0.0003, help="Hair strand thickness bevel depth")
    parser.add_argument("--save_blend", type=str, default="results/rendered_previews/scene_with_hair.blend", help="Path to save the modified Blender .blend project file")
    args = parser.parse_args(argv)

    print(f"[Blender Script] Loading template blend: {args.template_blend}")
    bpy.ops.wm.open_mainfile(filepath=args.template_blend)

    print(f"[Blender Script] Reading hair strands from: {args.hair_path}")
    strands = read_ply_strands(args.hair_path)
    print(f"[Blender Script] Imported {len(strands)} strands")

    # 创建发丝曲线对象
    hair_obj = create_curves_from_strands("reconstructed_hair", strands, bevel_depth=args.bevel_depth)

    # 赋予/创建材质
    mat = bpy.data.materials.get("Hair")
    if not mat:
        mat = bpy.data.materials.new(name="HairMat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.2, 0.1, 0.05, 1.0)
    hair_obj.data.materials.append(mat)

    # 修复并重向工程中环境贴图 EXR 路径
    env_img = bpy.data.images.get("studio_small_01_4k.exr")
    if env_img:
        bpy.data.images.remove(env_img)
        
    env_path = os.path.abspath("assets/ENV.Environment.exr")
    if os.path.exists(env_path):
        new_env = bpy.data.images.load(env_path)
        world = bpy.context.scene.world
        if world and world.use_nodes:
            for node in world.node_tree.nodes:
                if node.type == 'TEX_ENVIRONMENT':
                    node.image = new_env
    
    bpy.data.use_autopack = False

    # 保存导入头发后的原始 Blender 工程文件 (.blend)
    if args.save_blend:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_blend)), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend))
        print(f"[Blender Script] Saved modified blend scene file to: {args.save_blend}")

    # 开启渲染输出
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    bpy.context.scene.render.filepath = args.output_path
    
    print(f"[Blender Script] Rendering scene to: {args.output_path}")
    bpy.ops.render.render(write_still=True)
    print("[Blender Script] Render complete!")

if __name__ == "__main__":
    main()

