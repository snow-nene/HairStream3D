import bpy, sys, os, math
import numpy as np
import mathutils

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--glb",    required=True)
parser.add_argument("--out_dir",required=True)
parser.add_argument("--size",   type=int, default=512)
parser.add_argument("--views",  nargs="*", default=["front", "back", "left", "right", "top"])
parser.add_argument("--camera_only", action="store_true",
                    help="只导出相机矩阵，不重复渲染 PNG")
args = parser.parse_args(argv)

os.makedirs(args.out_dir, exist_ok=True)
camera_dir = os.path.join(args.out_dir, "camera_params")
os.makedirs(camera_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)

# Import mesh
if args.glb.lower().endswith(".glb") or args.glb.lower().endswith(".gltf"):
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(args.glb))
elif args.glb.lower().endswith(".obj"):
    bpy.ops.wm.obj_import(filepath=os.path.abspath(args.glb))

mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
all_verts = []
for obj in mesh_objs:
    for v in obj.data.vertices:
        co = obj.matrix_world @ v.co
        all_verts.append((co.x, co.y, co.z))
verts_np = np.array(all_verts)
center   = verts_np.mean(axis=0)
extent   = (verts_np.max(axis=0) - verts_np.min(axis=0)).max()

# ── EXR Environment Lighting (IBL) ──────────────────────────
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
nt = world.node_tree
nt.nodes.clear()

bg   = nt.nodes.new("ShaderNodeBackground")
tex  = nt.nodes.new("ShaderNodeTexEnvironment")
out  = nt.nodes.new("ShaderNodeOutputWorld")
mapp = nt.nodes.new("ShaderNodeMapping")
tc   = nt.nodes.new("ShaderNodeTexCoord")

exr_path = os.path.abspath("assets/ENV.Environment.exr")
if os.path.exists(exr_path):
    img = bpy.data.images.load(exr_path)
    tex.image = img
else:
    bg.inputs[0].default_value = (1,1,1,1)

bg.inputs[1].default_value = 2.0   # EXR strength

nt.links.new(tc.outputs["Generated"], mapp.inputs["Vector"])
nt.links.new(mapp.outputs["Vector"],  tex.inputs["Vector"])
nt.links.new(tex.outputs["Color"],    bg.inputs["Color"])
nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

# Setup Camera
cam_data = bpy.data.cameras.new("Camera")
cam_data.type = 'ORTHO'
cam_data.ortho_scale = extent * 1.4

cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

dist = extent * 2.0
views = {
    'front': (0, dist, 0),
    'back': (0, -dist, 0),
    'left': (-dist, 0, 0),
    'right': (dist, 0, 0),
    'top': (0, 0, dist)
}

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 128
scene.render.resolution_x = args.size
scene.render.resolution_y = args.size
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'

for view_name, offset in views.items():
    if view_name not in args.views:
        continue
    cam_pos = mathutils.Vector((
        float(center[0]) + offset[0],
        float(center[1]) + offset[1],
        float(center[2]) + offset[2]
    ))
    cam_obj.location = cam_pos
    direction = mathutils.Vector(center.tolist()) - cam_pos
    
    if view_name == 'front':
        cam_obj.rotation_euler = (math.pi/2, 0, math.pi)
    elif view_name == 'back':
        cam_obj.rotation_euler = (math.pi/2, 0, 0)
    elif view_name == 'left':
        cam_obj.rotation_euler = (math.pi/2, 0, -math.pi/2)
    elif view_name == 'right':
        cam_obj.rotation_euler = (math.pi/2, 0, math.pi/2)
    elif view_name == 'top':
        cam_obj.rotation_euler = (0, 0, 0)

    # 强制更新依赖图后保存 Blender 渲染实际使用的矩阵。矩阵采用列向量约定：
    # clip = world_to_clip @ world_homogeneous。
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    world_to_camera = cam_obj.matrix_world.inverted()
    projection = cam_obj.calc_matrix_camera(
        depsgraph,
        x=args.size,
        y=args.size,
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    world_to_clip = projection @ world_to_camera
    verts_homogeneous = np.column_stack([verts_np, np.ones(len(verts_np))])
    camera_vertices = verts_homogeneous @ np.asarray(world_to_camera, dtype=np.float64).T
    camera_depth_min = camera_vertices[:, 2].min()
    camera_depth_max = camera_vertices[:, 2].max()
    np.savez(
        os.path.join(camera_dir, f"{view_name}.npz"),
        camera_to_world=np.asarray(cam_obj.matrix_world, dtype=np.float64),
        world_to_camera=np.asarray(world_to_camera, dtype=np.float64),
        projection=np.asarray(projection, dtype=np.float64),
        world_to_clip=np.asarray(world_to_clip, dtype=np.float64),
        camera_location=np.asarray(cam_obj.location, dtype=np.float64),
        ortho_scale=np.float64(cam_data.ortho_scale),
        image_size=np.asarray([args.size, args.size], dtype=np.int32),
        mesh_center=center.astype(np.float64),
        mesh_extent=np.float64(extent),
        camera_depth_min=np.float64(camera_depth_min),
        camera_depth_max=np.float64(camera_depth_max),
    )
    print(f"[Blender] Saved camera parameters for {view_name}")

    if args.camera_only:
        continue

    out_png = os.path.join(args.out_dir, f"{view_name}_rgb.png")
    scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)
    print(f"[Blender] Rendered {view_name}")

if args.camera_only:
    print("[Blender] All camera parameters exported successfully.")
else:
    print("[Blender] All views rendered successfully.")
