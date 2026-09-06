"""直接反投影 Depth Pro 和二维导向曲线，使用射线深度测试显示自遮挡。"""
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import map_coordinates


def visualize_direct_strand_depth():
    base = Path('results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888')
    root = base / 'pde_governance/sparse_guides'
    out = root / 'step_11_direct_strand_depth'
    out.mkdir(parents=True, exist_ok=True)
    raw = cv2.imread(str(base / 'raw_img.png'))
    depth = np.load(root / 'step_05_matted_depth_pro/corrected_hair_depth.npy')
    seg = cv2.imread(str(base / 'maps/seg/front.png'), 0) > 127
    focal = json.loads((root / 'step_05_matted_depth_pro/report.json').read_text())['fixed_focal_px']
    h, w = depth.shape
    yy, xx = np.indices(depth.shape)
    valid = seg & np.isfinite(depth) & (depth > 0)
    xyz = np.stack([(xx-(w-1)/2)*depth/focal,
                    (yy-(h-1)/2)*depth/focal, depth], axis=-1)
    center = np.median(xyz[valid], axis=0)
    vertices = np.nan_to_num(xyz.reshape(-1, 3)-center).astype(np.float32)
    ids = np.arange(h*w).reshape(h, w)
    triangles = np.concatenate([
        np.stack([ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1]], -1).reshape(-1, 3),
        np.stack([ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]], -1).reshape(-1, 3)])
    triangles = triangles[valid.ravel()[triangles].all(axis=1)]
    # Do not bridge large depth discontinuities into artificial occluders.
    triangles = triangles[np.ptp(depth.ravel()[triangles], axis=1) < .03]
    guides = np.load(root / 'step_01_front_curves/guides_pixel_depth.npz')
    curves = {}
    dense = []
    colors = []
    overlay = raw.copy()
    for index, key in enumerate(guides.files):
        xy = guides[key][:, :2]
        d = map_coordinates(depth, [xy[:, 1], xy[:, 0]], order=1, mode='constant', cval=np.nan)
        points = np.column_stack([(xy[:, 0]-(w-1)/2)*d/focal,
                                  (xy[:, 1]-(h-1)/2)*d/focal, d])
        curves[key] = points
        color = cv2.cvtColor(np.uint8([[[index*37 % 180, 190, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        cv2.polylines(overlay, [np.rint(xy).astype(np.int32)], False, tuple(map(int, color)), 1, cv2.LINE_AA)
        for a, b in zip(points[:-1], points[1:]):
            if not np.isfinite([a, b]).all() or abs(a[2]-b[2]) > .03:
                continue
            count = max(2, int(np.linalg.norm(b-a)/.0003)+1)
            dense.append(np.linspace(a, b, count)-center)
            colors.append(np.tile(color, (count, 1)))
    dense = np.concatenate(dense).astype(np.float32)
    colors = np.concatenate(colors)
    np.savez_compressed(out / 'strand_depth_camera.npz', **curves)
    cv2.imwrite(str(out / 'front_raw_strands.png'), overlay)
    size = 640
    span = float(2.2*np.max(np.linalg.norm(xyz[valid]-center, axis=1)))
    pixels = (np.arange(size)-(size-1)/2)*span/size
    rx, ry = np.meshgrid(pixels, pixels)
    reports = []
    panels = []
    for yaw in [0, -40, 40, -80, 80]:
        angle = np.deg2rad(yaw)
        rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)
        mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices @ rot.T),
                                         o3d.core.Tensor(triangles.astype(np.int32)))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        rays = np.stack([rx, ry, np.full_like(rx, -5), np.zeros_like(rx),
                         np.zeros_like(rx), np.ones_like(rx)], -1).astype(np.float32)
        hits = scene.cast_rays(o3d.core.Tensor(rays))
        zbuffer = hits['t_hit'].numpy()-5
        hit = np.isfinite(zbuffer)
        tri = triangles[hits['primitive_ids'].numpy()[hit]]
        uv = hits['primitive_uvs'].numpy()[hit]
        bary = np.column_stack([1-uv.sum(axis=1), uv])
        rgb = (raw.reshape(-1, 3)[tri]*bary[:, :, None]).sum(axis=1)
        canvas = np.full((size, size, 3), 238, dtype=np.uint8)
        canvas[hit] = np.clip(rgb*.7+50, 0, 255).astype(np.uint8)
        transformed = dense @ rot.T
        screen = np.rint(transformed[:, :2]/span*size+(size-1)/2).astype(int)
        inside = ((screen >= 0) & (screen < size)).all(axis=1)
        screen, z, c = screen[inside], transformed[inside, 2], colors[inside]
        ref = zbuffer[screen[:, 1], screen[:, 0]]
        visible = z <= ref+.002
        # Strand pixels have their own depth buffer as well as surface occlusion.
        strand_z = np.full((size, size), np.inf)
        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            sx, sy = screen[:, 0]+dx, screen[:, 1]+dy
            ok = (sx >= 0) & (sx < size) & (sy >= 0) & (sy < size)
            candidates = np.flatnonzero(ok)
            candidates = candidates[z[candidates] <= zbuffer[sy[candidates], sx[candidates]]+.002]
            np.minimum.at(strand_z, (sy[candidates], sx[candidates]), z[candidates])
        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            sx, sy = screen[:, 0]+dx, screen[:, 1]+dy
            candidates = np.flatnonzero((sx >= 0) & (sx < size) & (sy >= 0) & (sy < size))
            candidates = candidates[z[candidates] <= strand_z[sy[candidates], sx[candidates]]+1e-7]
            canvas[sy[candidates], sx[candidates]] = c[candidates]
        cv2.putText(canvas, f'Yaw {yaw:+d} | depth-tested strands', (14, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f'yaw_{yaw:+03d}.png'), canvas)
        panels.append(canvas)
        reports.append({'yaw_deg': yaw, 'sample_count': len(z), 'occluded_samples': int((~visible).sum())})
    cv2.imwrite(str(out / 'comparison.png'), np.concatenate(panels[:3], axis=1))
    cv2.imwrite(str(out / 'side_views.png'), np.concatenate(panels[3:], axis=1))
    report = {'focal_px': focal, 'curves': len(curves), 'surface_triangles': len(triangles),
              'depth_test_tolerance_m': .002, 'views': reports,
              'note': '直接使用抠图 Depth Pro 米制深度，未平滑、未使用头模或 PDE。表面三角形只用于遮挡测试；单图2.5D，没有隐藏层和头脸遮挡信息。斜视图为正交观察，颜色仅区分导向曲线。'}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    visualize_direct_strand_depth()
