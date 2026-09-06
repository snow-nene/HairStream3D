"""稀疏可见曲线的三维提升、头皮距离检查与保守根部连接候选。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.recon3D import load_calib
from lib.hair_util import save_polyline_strands


def run_sparse_guide_lift_audit():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--root-radius-px", type=float, default=12)
    parser.add_argument("--max-bridge-m", type=float, default=0.02)
    args = parser.parse_args()
    base = Path("results/multiview_data") / args.image_id
    source = base / "pde_governance/sparse_guides/step_01_front_curves/guides_pixel_depth.npz"
    out = base / "pde_governance/sparse_guides/step_02_depth_lift"
    out.mkdir(parents=True, exist_ok=True)
    dense = base / "maps/param/front_dense_silhouette.npy"
    legacy = base / "maps/param/front.npy"
    calib = load_calib(str(dense)).numpy().astype(np.float64)
    calib[2] = load_calib(str(legacy)).numpy()[2]
    inverse = np.linalg.inv(calib)
    raw = cv2.imread(str(base / "raw_img.png"))
    depth = np.load(base / "maps/depth_map/front.npy")
    height, width = depth.shape
    parting_path = base / "pde_governance/front_parting_dinov3_migrated/parting_region_mask.png"
    parting = cv2.imread(str(parting_path), cv2.IMREAD_GRAYSCALE)
    parting = cv2.resize(parting, (width, height), interpolation=cv2.INTER_NEAREST) > 127
    distance_to_parting = distance_transform_edt(~parting)
    mesh_path = Path("data/head_model.obj")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    guides = np.load(source)
    worlds = {}
    candidates = {}
    rows = []
    all_distances = []
    roundtrip_errors = []
    overlay = raw.copy()
    scale = np.array([raw.shape[1] / width, raw.shape[0] / height])
    for index, key in enumerate(guides.files):
        samples = guides[key].astype(np.float64)
        clip = np.column_stack([samples[:, :2] / [width - 1, height - 1] * 2 - 1,
                                samples[:, 2], np.ones(len(samples))])
        world_h = clip @ inverse.T
        world = world_h[:, :3] / world_h[:, 3:4]
        worlds[key] = world.astype(np.float32)
        projected = np.column_stack([world, np.ones(len(world))]) @ calib.T
        pixels = (projected[:, :2] / projected[:, 3:4] + 1) * [width - 1, height - 1] / 2
        error = np.linalg.norm(pixels - samples[:, :2], axis=1)
        roundtrip_errors.extend(error.tolist())
        query = o3d.core.Tensor(world.astype(np.float32))
        closest = scene.compute_closest_points(query)
        surface = closest["points"].numpy()
        normal = closest["primitive_normals"].numpy()
        distance = np.linalg.norm(world - surface, axis=1)
        all_distances.extend(distance.tolist())
        endpoint_dist = map_coordinates(distance_to_parting,
                                       [samples[[0, -1], 1], samples[[0, -1], 0]], order=1)
        endpoint = 0 if endpoint_dist[0] <= endpoint_dist[1] else len(world) - 1
        near_parting = float(endpoint_dist.min()) <= args.root_radius_px
        # 只保存短连接候选；靠近表面不等价于已确认的生发区。
        accepted = near_parting and distance[endpoint] <= args.max_bridge_m
        if accepted:
            root = surface[endpoint] + 0.001 * normal[endpoint]
            bridge = np.linspace(root, world[endpoint], 12)
            # 沿连接线检查是否穿过头部；网格有向距离仅作为候选筛查。
            signed = scene.compute_signed_distance(o3d.core.Tensor(bridge.astype(np.float32))).numpy()
            accepted = bool(np.all(signed >= -0.0005))
            if accepted:
                ordered = world if endpoint == 0 else world[::-1]
                candidates[key] = np.concatenate([bridge[:-1], ordered]).astype(np.float32)
        rows.append({"guide": key, "endpoint": int(endpoint),
                     "parting_distance_px": float(endpoint_dist.min()),
                     "head_distance_m": float(distance[endpoint]),
                     "candidate_accepted": bool(accepted)})
        color = (80, 230, 80) if accepted else (0, 180, 255)
        cv2.polylines(overlay, [np.rint(pixels * scale).astype(np.int32)], False,
                      color, 1, cv2.LINE_AA)
        if near_parting:
            cv2.circle(overlay, tuple(np.rint(pixels[endpoint] * scale).astype(int)),
                       3, (60, 60, 255), -1)
    assert max(roundtrip_errors) < 1e-4, "相机往返投影误差过大"
    assert all(np.isfinite(value).all() for value in worlds.values())
    np.savez_compressed(out / "visible_guides_world.npz", **worlds)
    np.savez_compressed(out / "root_connection_candidates.npz", **candidates)
    np.save(out / "calibration.npy", calib)
    save_polyline_strands(list(worlds.values()), str(out / "visible_guides.ply"))
    if candidates:
        save_polyline_strands(list(candidates.values()), str(out / "root_connection_candidates.ply"))
    cv2.imwrite(str(out / "reprojection_on_raw.png"), overlay)
    fig = plt.figure(figsize=(13, 6))
    vertices = np.asarray(mesh.vertices)[::30]
    for panel, azimuth in enumerate((90, 0), 1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        ax.scatter(vertices[:, 0], vertices[:, 2], vertices[:, 1], s=0.2, c="gray", alpha=0.15)
        for index, curve in enumerate(worlds.values()):
            ax.plot(curve[:, 0], curve[:, 2], curve[:, 1], lw=0.6, color=plt.cm.turbo(index / len(worlds)))
        ax.view_init(elev=10, azim=azimuth)
        points = np.concatenate(list(worlds.values()) + [np.asarray(mesh.vertices)])
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = np.max(np.ptp(points, axis=0)) / 2
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(center[1] - radius, center[1] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title("Front geometry" if panel == 1 else "Side geometry")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("Y (m)")
    fig.tight_layout()
    fig.savefig(out / "geometry_views.png", dpi=160)
    plt.close(fig)
    report = {"parameters": vars(args), "guide_count": len(worlds),
              "candidate_connections": len(candidates),
              "near_parting_endpoints": sum(row["parting_distance_px"] <= args.root_radius_px for row in rows),
              "reprojection_max_px": max(roundtrip_errors),
              "head_distance_m_q10_q50_q90": np.quantile(all_distances, [0.1, 0.5, 0.9]).tolist(),
              "endpoint_audit": rows,
              "limitations": "直接使用预测相对深度及旧标定转换；往返投影只验证代数一致性。连接为未确认头皮归属的候选；未执行曲面场与全曲线碰撞优化。",
              "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (source, dense, legacy, mesh_path, parting_path)}}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in ("endpoint_audit", "input_sha256")}, ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    run_sparse_guide_lift_audit()
