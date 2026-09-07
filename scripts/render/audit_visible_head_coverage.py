"""用同一模板相机的平色物体渲染独立测量露头皮，不依赖生长代理分数。"""
import argparse
import json
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Matrix, Vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--view', choices=['front', 'back', 'left', 'right'], default='back')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    target_image = bpy.data.images.load(str(args.target.resolve()))
    width, height = target_image.size
    target = np.array(target_image.pixels[:]).reshape(height, width, 4)[::-1, :, 0] > .5
    report = {'view': args.view, 'target_pixels': int(target.sum()), 'method': 'cycles_emission_object_visibility'}
    for name, path in [('before', args.before), ('after', args.after)]:
        bpy.ops.wm.open_mainfile(filepath=str(path.resolve()))
        scene = bpy.context.scene
        scene.use_nodes = False
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = 4
        scene.cycles.use_denoising = False
        scene.render.threads_mode = 'FIXED'
        scene.render.threads = 4
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = True
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0
        scene.view_settings.gamma = 1
        for obj in scene.objects:
            if obj.type not in {'MESH', 'CURVE', 'CURVES'}:
                continue
            color = (1, 0, 0, 1) if obj.name == 'head_smooth' else (
                (0, 1, 0, 1) if obj.name.startswith('all_valid_root_prefixes') else (0, 0, 0, 1))
            material = bpy.data.materials.new(f'audit_{obj.name}')
            material.use_nodes = True
            # Dense curves must not become millions of sampled area lights.
            material.cycles.emission_sampling = 'NONE'
            material.node_tree.nodes.clear()
            emission = material.node_tree.nodes.new('ShaderNodeEmission')
            emission.inputs['Color'].default_value = color
            output = material.node_tree.nodes.new('ShaderNodeOutputMaterial')
            material.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
            obj.data.materials.clear()
            obj.data.materials.append(material)
        cam = scene.camera
        pivot = Vector((0, 0, 1.73))
        angle = {'front': 0, 'left': np.pi / 2, 'right': -np.pi / 2, 'back': np.pi}[args.view]
        cam.matrix_world = Matrix.Translation(pivot) @ Matrix.Rotation(angle, 4, 'Z') @ Matrix.Translation(-pivot) @ cam.matrix_world
        output_path = args.output_dir / f'{name}.png'
        scene.render.filepath = str(output_path.resolve())
        bpy.ops.render.render(write_still=True)
        rendered = bpy.data.images.load(str(output_path.resolve()), check_existing=False)
        pixels = np.array(rendered.pixels[:]).reshape(height, width, 4)[::-1]
        head = (pixels[..., 0] > .5) & (pixels[..., 0] > pixels[..., 1])
        hair = (pixels[..., 1] > .5) & (pixels[..., 1] > pixels[..., 0])
        report[name] = {'visible_head_pixels_in_target': int((head & target).sum()),
                        'visible_hair_pixels_in_target': int((hair & target).sum())}
    report['head_gap_reduction_pixels'] = report['before']['visible_head_pixels_in_target'] - report['after']['visible_head_pixels_in_target']
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
