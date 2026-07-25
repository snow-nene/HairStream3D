import bpy, sys, os
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--glb",    required=True)
parser.add_argument("--out_dir",required=True)
parser.add_argument("--size",   type=int, default=1024)
parser.add_argument("--exr",    default="assets/ENV.Environment.exr")
args = parser.parse_args(argv)

os.makedirs(args.out_dir, exist_ok=True)
exr_path = os.path.abspath(args.exr)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=os.path.abspath(args.glb))
print(f"[Blender] 导入完成, 物体: {[o.name for o in bpy.context.scene.objects if o.type=='MESH']}")

mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
all_verts = []
for obj in mesh_objs:
    for v in obj.data.vertices:
        co = obj.matrix_world @ v.co
        all_verts.append((co.x, co.y, co.z))
verts_np = np.array(all_verts)
center   = verts_np.mean(axis=0)
extent   = (verts_np.max(axis=0) - verts_np.min(axis=0)).max()
print(f"[Blender] center={center.round(4)}, extent={extent:.4f}")

# ── EXR 环境贴图 (IBL) ────────────────────────────────────────
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
nt = world.node_tree
# 清空默认节点
nt.nodes.clear()

bg   = nt.nodes.new("ShaderNodeBackground")
tex  = nt.nodes.new("ShaderNodeTexEnvironment")
out  = nt.nodes.new("ShaderNodeOutputWorld")
mapp = nt.nodes.new("ShaderNodeMapping")
tc   = nt.nodes.new("ShaderNodeTexCoord")

if os.path.exists(exr_path):
    img = bpy.data.images.load(exr_path)
    tex.image = img
    print(f"[Blender] 加载 EXR: {exr_path}")
else:
    print(f"[Blender] 未找到 EXR: {exr_path}，用纯白背景")
    bg.inputs[0].default_value = (1,1,1,1)

bg.inputs[1].default_value = 2.0   # EXR 强度
bg.inputs[0].default_value = (1, 1, 1, 1)  # 默认白色（EXR 加载后会覆盖）

nt.links.new(tc.outputs["Generated"], mapp.inputs["Vector"])
nt.links.new(mapp.outputs["Vector"],  tex.inputs["Vector"])
nt.links.new(tex.outputs["Color"],    bg.inputs["Color"])
nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

# ── 相机（正交，从 +Y 看，对准中心）──────────────────────────
import mathutils
dist = extent * 1.5

cam_data = bpy.data.cameras.new("Camera")
cam_data.type = 'ORTHO'
cam_data.ortho_scale = extent * 1.15   # 留少量边距

cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

cam_pos = mathutils.Vector((float(center[0]),
                             float(center[1]) + dist,
                             float(center[2])))
cam_obj.location = cam_pos
direction = mathutils.Vector(center.tolist()) - cam_pos
rot_quat  = direction.to_track_quat('-Z', 'Y')
cam_obj.rotation_euler = rot_quat.to_euler()

# ── 渲染设置 ──────────────────────────────────────────────────
scene = bpy.context.scene
scene.render.engine        = 'CYCLES'
scene.cycles.samples       = 128
scene.view_settings.view_transform = 'Filmic'
scene.view_settings.exposure       = 0.5
scene.render.resolution_x  = args.size
scene.render.resolution_y  = args.size
scene.render.film_transparent = True          # 透明背景
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'  # 保留 alpha 通道

out_png = os.path.join(args.out_dir, "render_front.png")
scene.render.filepath = out_png
bpy.ops.render.render(write_still=True)
print(f"[Blender] 渲染完成: {out_png}")

# ── 保存相机参数 ───────────────────────────────────────────────
cam_R = np.array(cam_obj.matrix_world.to_3x3())
cam_t = np.array(cam_obj.matrix_world.translation)
np.savez(os.path.join(args.out_dir, "render_camera.npz"),
         cam_R=cam_R, cam_pos=cam_t,
         ortho_scale=float(cam_data.ortho_scale),
         mesh_center=center,
         img_size=args.size)
print("[Blender] 相机参数已保存")
