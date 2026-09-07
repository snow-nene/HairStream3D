"""导出与 render_all_prefixes 相同的模板相机到发丝世界坐标。"""
import argparse
import hashlib
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Matrix, Vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    scene = bpy.context.scene
    camera = scene.camera
    original = camera.matrix_world.copy()
    pivot = Vector((0, 0, 1.73))
    world_to_blender = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.]])
    projection = np.array(camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=768, y=768))
    for view, angle in [('front', 0), ('left', np.pi / 2), ('right', -np.pi / 2), ('back', np.pi)]:
        matrix = Matrix.Translation(pivot) @ Matrix.Rotation(angle, 4, 'Z') @ Matrix.Translation(-pivot) @ original
        camera_space = np.array(matrix.inverted()) @ world_to_blender
        # Reverse clip depth so nearer points have greater depth.
        calibration = np.diag([1., -1., -1., 1.]) @ projection @ camera_space
        np.savez_compressed(args.output_dir / f'{view}.npz', camera=calibration, image_shape=[768, 768],
                            template_blend_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
