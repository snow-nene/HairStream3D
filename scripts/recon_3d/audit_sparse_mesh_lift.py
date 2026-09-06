"""用已对齐网格的首个可见射线交点提升稀疏曲线，保留缺失与断段。"""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.recon3D import load_calib
from lib.hair_util import save_polyline_strands


def run_sparse_mesh_lift():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--max-step-m", type=float, default=0.01)
    args = parser.parse_args()
    if args.max_step_m <= 0:
        parser.error("max-step-m 必须为正")
    base = Path("results/multiview_data") / args.image_id
    root = base / "pde_governance/sparse_guides"
    source = root / "step_01_front_curves/guides_pixel_depth.npz"
    mesh_path = base / "pixal3d/hair_mesh_aligned_best.obj"
    camera_path = base / "maps/param/front_dense_silhouette.npy"
    out = root / "step_03_mesh_lift"
    out.mkdir(parents=True, exist_ok=True)
    guides = np.load(source)
    camera = load_calib(str(camera_path)).numpy().astype(np.float64)
    inverse = np.linalg.inv(camera)
    seg = cv2.imread(str(base / "maps/seg/front.png"), cv2.IMREAD_GRAYSCALE)
    height, width = seg.shape
    raw = cv2.imread(str(base / "raw_img.png"))
    overlay = raw.copy()
    scale = np.array([raw.shape[1] / width, raw.shape[0] / height])
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    vertices = np.asarray(mesh.vertices)
    mesh_clip = np.column_stack([vertices, np.ones(len(vertices))]) @ camera.T
    near_z, far_z = mesh_clip[:, 2].max() + 1, mesh_clip[:, 2].min() - 1
    fragments, evidence, rows = {}, {}, []
    errors, steps = [], []
    for index, key in enumerate(guides.files):
        xy = guides[key][:, :2].astype(np.float64)
        uv = xy / [width - 1, height - 1] * 2 - 1
        near = np.column_stack([uv, np.full(len(uv), near_z), np.ones(len(uv))]) @ inverse.T
        far = np.column_stack([uv, np.full(len(uv), far_z), np.ones(len(uv))]) @ inverse.T
        near = near[:, :3] / near[:, 3:4]
        far = far[:, :3] / far[:, 3:4]
        direction = far - near
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        hit = scene.cast_rays(o3d.core.Tensor(np.column_stack([near, direction]).astype(np.float32)))
        distance = hit["t_hit"].numpy()
        valid = np.isfinite(distance)
        world = np.full((len(xy), 3), np.nan)
        world[valid] = near[valid] + direction[valid] * distance[valid, None]
        normals = hit["primitive_normals"].numpy()
        incidence = np.abs(np.sum(normals * direction, axis=1))
        lengths = np.linalg.norm(np.diff(world, axis=0), axis=1)
        jump = np.isfinite(lengths) & (lengths > args.max_step_m)
        steps.extend(lengths[np.isfinite(lengths)].tolist())
        cuts = np.flatnonzero(~valid[:-1] | ~valid[1:] | jump) + 1
        kept = 0
        color = tuple(map(int, cv2.cvtColor(np.uint8([[[index * 47 % 180, 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]))
        for number, ids in enumerate(np.split(np.arange(len(xy)), cuts)):
            if len(ids) < 6 or not valid[ids].all():
                continue
            name = f"{key}_segment_{number:03d}"
            fragments[name] = world[ids].astype(np.float32)
            evidence[name + "_source_indices"] = ids
            kept += len(ids)
            cv2.polylines(overlay, [np.rint(xy[ids] * scale).astype(np.int32)], False, color, 1, cv2.LINE_AA)
        for pt in xy[~valid]:
            cv2.circle(overlay, tuple(np.rint(pt * scale).astype(int)), 1, (30, 30, 255), -1)
        projected = np.column_stack([world[valid], np.ones(valid.sum())]) @ camera.T
        pixels = (projected[:, :2] / projected[:, 3:4] + 1) * [width - 1, height - 1] / 2
        errors.extend(np.linalg.norm(pixels - xy[valid], axis=1).tolist())
        evidence[key + "_world_raw"] = world
        evidence[key + "_hit_mask"] = valid
        evidence[key + "_incidence"] = incidence
        rows.append({"guide": key, "samples": len(xy), "hits": int(valid.sum()),
                     "kept_samples": kept, "jump_segments": int(jump.sum()),
                     "grazing_hits": int((valid & (incidence < 0.1)).sum())})
    assert fragments and max(errors) < 0.001
    assert all(np.isfinite(curve).all() and np.max(np.linalg.norm(np.diff(curve, axis=0), axis=1)) <= args.max_step_m + 1e-6 for curve in fragments.values())
    np.savez_compressed(out / "mesh_guides_world.npz", **fragments)
    np.savez_compressed(out / "raycast_evidence.npz", **evidence)
    save_polyline_strands(list(fragments.values()), str(out / "mesh_guides.ply"))
    cv2.imwrite(str(out / "reprojection_on_raw.png"), overlay)
    cv2.imwrite(str(out / "raw_comparison.png"), np.concatenate([raw, overlay], axis=1))
    old = np.load(root / "step_02_depth_lift/visible_guides_world.npz")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    displayed = vertices[::max(1, len(vertices) // 18000)]
    for ax, curves, plane, title in zip(axes, (old, fragments, fragments), ((2, 1), (2, 1), (0, 1)),
                                       ("Previous direct-depth lift: side", "Mesh raycast: side", "Mesh raycast: front")):
        ax.scatter(displayed[:, plane[0]], displayed[:, plane[1]], s=0.2, color="gray", alpha=0.2)
        for index, key in enumerate(curves):
            curve = curves[key]
            ax.plot(curve[:, plane[0]], curve[:, plane[1]], lw=0.6, color=plt.cm.turbo(index / len(curves)))
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("Z (m)" if plane[0] == 2 else "X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_ylim(1.54, 1.96)
        ax.set_xlim(-0.23, 0.3)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "geometry_comparison.png", dpi=150)
    plt.close(fig)
    report = {"parameters": vars(args), "source_guides": len(rows), "output_fragments": len(fragments),
              "samples": sum(row["samples"] for row in rows), "hits": sum(row["hits"] for row in rows),
              "kept_samples": sum(row["kept_samples"] for row in rows),
              "jump_segments": sum(row["jump_segments"] for row in rows),
              "reprojection_max_px": max(errors),
              "raw_step_m_q50_q90_q99": np.quantile(steps, [0.5, 0.9, 0.99]).tolist(),
              "limitations": "仅恢复网格可见外层。seg 是二维筛选，网格未提供逐面头发语义；首交点可能属于其他表面。掠射角只记录未删除。缺失与大跳变断开，未平滑、未连接发根。",
              "guides": rows,
              "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, mesh_path, camera_path)}}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("guides", "input_sha256")}, ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    run_sparse_mesh_lift()
