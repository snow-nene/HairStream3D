"""隔离 ICP 目标顶点／面索引混用，保存变换复现与原图投影对照。"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.mesh_util import load_obj_mesh
from lib.util.opt_lmk import load_point_ids
from scripts.recon_3d.grow_visible_surface_gaps import load_observation_camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出必须按 Image ID 聚合')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    v, f = load_obj_mesh('data/head_model.obj')
    other = o3d.io.read_triangle_mesh('data/head_model.obj')
    mixed = np.asarray(other.triangles)
    lm = v[load_point_ids('data/landmark_id_uschair.obj')]
    box = o3d.geometry.AxisAlignedBoundingBox(lm.min(0)-.05, lm.max(0)+.05)
    saved = np.load(a.data_dir/'pixal3d/glb_to_world.npz')
    icp = saved['icp_T']
    raw = o3d.io.read_triangle_mesh(str(a.data_dir/'pixal3d/hair_mesh_aligned_best.obj'))
    raw.transform(np.linalg.inv(icp))
    raw.compute_vertex_normals()
    crop = raw.crop(box)
    source = o3d.geometry.PointCloud()
    source.points, source.normals = crop.vertices, crop.vertex_normals
    report = {'vertex_order_equal': bool(np.allclose(v, np.asarray(other.vertices))),
              'source_points': len(source.points), 'variants': {}}
    camera = load_observation_camera(a.data_dir, 'front')
    canvas = Image.new('RGB', (1536, 560), 'white')
    draw = ImageDraw.Draw(canvas)
    transforms = {'pre_icp': np.eye(4)}
    for name, faces in [('mixed_indices', mixed), ('consistent_indices', f)]:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(faces))
        mesh.compute_vertex_normals()
        target_crop = mesh.crop(box)
        target = o3d.geometry.PointCloud()
        target.points, target.normals = target_crop.vertices, target_crop.vertex_normals
        reg = o3d.pipelines.registration.registration_icp(source, target, .05, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transforms[name] = reg.transformation
        edges = np.linalg.norm(v[faces]-v[np.roll(faces, 1, axis=1)], axis=2)
        points = np.asarray(source.points)
        delta = points @ (reg.transformation[:3,:3]-icp[:3,:3]).T + reg.transformation[:3,3]-icp[:3,3]
        report['variants'][name] = {'fitness': reg.fitness, 'rmse_m': reg.inlier_rmse,
            'edge_mm_quantiles_50_90_99': (np.quantile(edges, [.5,.9,.99])*1000).tolist(),
            'difference_from_saved_transform_surface_mm_50_90_max': (np.quantile(np.linalg.norm(delta, axis=1), [.5,.9,1])*1000).tolist(),
            'transform': reg.transformation.tolist()}
    points = np.asarray(raw.vertices)
    # 同一 GLB 顶点在三个配准状态下投影；原图只是背景真值，未重新拟合。
    for i, (name, transform) in enumerate(transforms.items()):
        photo = Image.open(a.data_dir/'raw_img.png').convert('RGB').resize((512,512))
        pv = points @ transform[:3,:3].T + transform[:3,3]
        q = np.c_[pv, np.ones(len(pv))] @ camera.T
        uv = (q[:,:2]/q[:,3:]+1)*255.5
        d = ImageDraw.Draw(photo)
        for x,y in uv[::120]:
            if 0 <= x < 512 and 0 <= y < 512:
                d.point((int(x),int(y)), fill=(0,255,255))
        canvas.paste(photo, (512*i, 32))
        draw.text((512*i+8,8), name, fill='black')
    canvas.save(a.output_dir/'registration_comparison.png')
    np.savez(a.output_dir/'transforms.npz', **transforms)
    (a.output_dir/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
