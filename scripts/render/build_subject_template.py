"""将经过验收的样例头皮写入派生 Blender 模板，保留源模板。"""
import argparse
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if a.output.exists():
        raise ValueError('派生模板输出已存在')
    with np.load(a.head) as h:
        vertices, faces = h['vertices'], h['faces']
    transformed = vertices[:, [0, 2, 1]]*np.array([1, -1, 1])
    obj = bpy.data.objects['head_smooth']
    materials = list(obj.data.materials)
    mesh = bpy.data.meshes.new('subject_scalp_evaluated')
    mesh.from_pydata(transformed.tolist(), [], faces.tolist())
    mesh.update()
    obj.modifiers.clear()
    obj.data = mesh
    obj.matrix_world = Matrix.Identity(4)
    for material in materials:
        mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    scene = bpy.context.scene
    if scene.world and scene.world.use_nodes:
        for node in scene.world.node_tree.nodes:
            if node.type == 'TEX_ENVIRONMENT':
                node.image = bpy.data.images.load(str(Path('assets/ENV.Environment.exr').resolve()), check_existing=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.output.resolve()))


if __name__ == '__main__':
    main()
