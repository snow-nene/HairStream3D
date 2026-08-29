"""Run the S4 topology-preserving parting smoke on a real scalp crop."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.recon_strategy.parting_topology import (
    build_scalp_cut_graph,
    integrate_surface_prefix,
    solve_cut_graph_laplace,
    surface_scalar_gradient,
)


def load_curve(path):
    data = np.load(path)
    if "points" not in data:
        raise ValueError("parting curve NPZ must contain a 'points' array")
    points = np.asarray(data["points"], dtype=np.float64).reshape(-1, 3)
    if len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError("parting curve must contain at least three finite points")
    return points


def crop_mesh_around_curve(mesh, curve, radius, vertical_margin):
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    distance, _ = cKDTree(curve).query(centers)
    lower_y = float(np.min(curve[:, 1]) - vertical_margin)
    keep_faces = (distance <= float(radius)) & (centers[:, 1] >= lower_y)
    selected_faces = np.asarray(mesh.faces, dtype=np.int64)[keep_faces]
    if len(selected_faces) == 0:
        raise RuntimeError("real scalp crop contains no faces")
    used, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)[used]
    faces = inverse.reshape(-1, 3)
    # data/head_model.obj is exported as triangle soup (three independent
    # vertices per face).  A PDE graph over those raw indices contains one
    # disconnected component per triangle, so weld coordinate-identical
    # vertices before constructing any topology.
    cropped = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    # The welded crown crop is deliberately low resolution (about 243
    # vertices for the reference head).  Two midpoint subdivisions preserve
    # its topology while providing enough distinct bank roots for the fixed
    # 256-root smoke without expanding to a full reconstruction domain.
    for _ in range(2):
        refined_vertices, refined_faces = trimesh.remesh.subdivide(
            np.asarray(cropped.vertices), np.asarray(cropped.faces)
        )
        cropped = trimesh.Trimesh(
            vertices=refined_vertices,
            faces=refined_faces,
            process=True,
        )
    return cropped


def curve_lateral_frame(mesh, curve):
    closest, _, face_ids = trimesh.proximity.closest_point(mesh, curve)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[face_ids]
    tangent = np.gradient(closest, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12
    lateral = np.cross(tangent, normals)
    lateral /= np.linalg.norm(lateral, axis=1, keepdims=True) + 1e-12
    reverse = lateral[:, 0] < 0.0
    lateral[reverse] *= -1.0
    return closest, lateral


def signed_curve_distance(vertices, curve, lateral):
    distance, nearest = cKDTree(curve).query(vertices)
    offset = vertices - curve[nearest]
    signed = np.sum(offset * lateral[nearest], axis=1)
    return signed, distance, nearest


def select_balanced_roots(graph, tangent, count, minimum_bank, maximum_bank, seed):
    rng = np.random.default_rng(seed)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    bank = np.abs(graph.signed_distance)
    eligible = (
        ~graph.cap_mask
        & (bank >= float(minimum_bank))
        & (bank <= float(maximum_bank))
        & (tangent_norm >= 0.5)
    )
    each_side = int(count) // 2
    selected = []
    for side in (-1, 1):
        candidates = np.flatnonzero(eligible & (graph.side == side))
        if len(candidates) < each_side:
            raise RuntimeError(
                f"not enough real scalp roots on side {side}: "
                f"{len(candidates)} < {each_side}"
            )
        selected.append(rng.choice(candidates, each_side, replace=False))
    return np.concatenate(selected)


def write_prefix_obj(path, points):
    lines = []
    vertex_offset = 1
    for strand in points:
        lines.extend(f"v {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}" for point in strand)
        ids = " ".join(str(index) for index in range(vertex_offset, vertex_offset + len(strand)))
        lines.append(f"l {ids}")
        vertex_offset += len(strand)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_roots_obj(path, points):
    lines = [
        f"v {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}"
        for point in np.asarray(points)[:, 0]
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_real_smoke(args):
    if args.root_count > 256 or args.num_points > 30:
        raise ValueError("S4 is limited to at most 256 roots and 30 points")
    mesh = trimesh.load(args.head_mesh, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("head_mesh must be a non-empty triangle mesh")
    curve_raw = load_curve(args.parting_curve)
    curve, lateral = curve_lateral_frame(mesh, curve_raw)
    scalp = crop_mesh_around_curve(
        mesh, curve, args.crop_radius, args.vertical_margin
    )
    signed, curve_distance, nearest_curve = signed_curve_distance(
        np.asarray(scalp.vertices), curve, lateral
    )
    endpoint_distance = np.linalg.norm(
        np.asarray(scalp.vertices) - curve[0], axis=1
    )
    cap = (
        (endpoint_distance <= args.cap_length)
        & (nearest_curve <= args.cap_curve_points)
    )
    graph = build_scalp_cut_graph(
        np.asarray(scalp.vertices), np.asarray(scalp.faces), signed, cap
    )
    if graph.metrics.crossing_edges_outside_cap != 0:
        raise AssertionError("real cut graph retained an outside-cap crossing")
    if graph.metrics.mixed_side_components_outside_cap != 0:
        raise AssertionError("real cut graph has a mixed-side outside-cap component")

    bank = np.abs(signed)
    near = bank <= args.pde_near_bank
    far_threshold = float(np.quantile(bank, args.pde_far_quantile))
    far = bank >= far_threshold
    boundary_mask = near | far
    boundary_ids = np.flatnonzero(boundary_mask)
    boundary_values = far[boundary_ids].astype(np.float64)
    potential, pde = solve_cut_graph_laplace(
        graph,
        boundary_ids,
        boundary_values,
        tolerance=args.pde_tolerance,
        max_iterations=args.pde_max_iterations,
    )
    if not pde.converged:
        raise AssertionError(
            f"real scalp PDE did not converge: {pde.final_relative_residual}"
        )
    normals = np.asarray(scalp.vertex_normals, dtype=np.float64)
    tangent = surface_scalar_gradient(graph, potential, normals=normals)
    roots = select_balanced_roots(
        graph,
        tangent,
        count=args.root_count,
        minimum_bank=args.root_bank_min,
        maximum_bank=args.root_bank_max,
        seed=args.seed,
    )
    prefix = integrate_surface_prefix(
        graph,
        roots,
        tangent,
        normals,
        num_points=args.num_points,
        root_height=args.root_height,
        release_height=args.release_height,
        release_steps=args.release_steps,
    )

    closest, surface_distance, _ = trimesh.proximity.closest_point(
        scalp, prefix.points.reshape(-1, 3)
    )
    surface_distance = surface_distance.reshape(len(roots), args.num_points)
    height_error = np.abs(
        surface_distance - prefix.target_heights[None, :]
    )
    root_error = float(np.max(height_error[:, 0]))
    prefix_error = float(np.max(height_error[:, :6]))
    stalled_fraction = float(
        prefix.stalled_steps / max(args.root_count * (args.num_points - 1), 1)
    )
    if prefix.side_violations_before_cap != 0:
        raise AssertionError("real prefixes crossed the parting before the cap")
    if root_error > 0.0015:
        raise AssertionError(f"real root attachment error too large: {root_error}")
    if stalled_fraction > 0.02:
        raise AssertionError(
            f"real prefix stalled fraction too large: {stalled_fraction}"
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "root_prefixes.npz",
        points=prefix.points.astype(np.float32),
        surface_vertex_ids=prefix.surface_vertex_ids,
        root_vertex_ids=roots,
        side=graph.side[roots],
        target_heights=prefix.target_heights.astype(np.float32),
        parting_curve=curve.astype(np.float32),
    )
    write_prefix_obj(output / "root_prefixes.obj", prefix.points)
    write_roots_obj(output / "roots.obj", prefix.points)
    metrics = {
        "suite": "parting_topology_smoke_s4_real_crown",
        "passed": True,
        "full_reconstruction_run": False,
        "root_count": int(args.root_count),
        "num_points": int(args.num_points),
        "voxel_resolution": None,
        "styling": {
            "kmeans": False,
            "divergence_noise": False,
            "laplacian_smoothing": False,
            "parting_trim": False,
        },
        "crop": {
            "vertices": int(len(scalp.vertices)),
            "faces": int(len(scalp.faces)),
            "curve_points": int(len(curve)),
            "curve_vertex_distance_q95_m": float(np.quantile(curve_distance, 0.95)),
        },
        "cut_graph": graph.metrics.to_dict(),
        "pde": pde.to_dict(),
        "root_layer": {
            "side_violations_before_cap": prefix.side_violations_before_cap,
            "stalled_steps": prefix.stalled_steps,
            "stalled_fraction": stalled_fraction,
            "root_height_error_max_m": root_error,
            "prefix_height_error_q95_m": float(np.quantile(height_error[:, :6], 0.95)),
            "prefix_height_error_max_m": prefix_error,
            "surface_distance_root_q50_m": float(np.quantile(surface_distance[:, 0], 0.50)),
            "surface_distance_root_q95_m": float(np.quantile(surface_distance[:, 0], 0.95)),
            "surface_distance_root_max_m": float(np.max(surface_distance[:, 0])),
        },
        "outputs": [
            "metrics.json",
            "root_prefixes.npz",
            "root_prefixes.obj",
            "roots.obj",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_mesh", default="data/head_model.obj")
    parser.add_argument("--parting_curve", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--root_count", type=int, default=256)
    parser.add_argument("--num_points", type=int, default=30)
    parser.add_argument("--crop_radius", type=float, default=0.10)
    parser.add_argument("--vertical_margin", type=float, default=0.04)
    parser.add_argument("--cap_length", type=float, default=0.018)
    parser.add_argument("--cap_curve_points", type=int, default=16)
    parser.add_argument("--pde_near_bank", type=float, default=0.004)
    parser.add_argument("--pde_far_quantile", type=float, default=0.82)
    parser.add_argument("--pde_tolerance", type=float, default=1e-9)
    parser.add_argument("--pde_max_iterations", type=int, default=3000)
    parser.add_argument("--root_bank_min", type=float, default=0.004)
    parser.add_argument("--root_bank_max", type=float, default=0.028)
    parser.add_argument("--root_height", type=float, default=0.0005)
    parser.add_argument("--release_height", type=float, default=0.006)
    parser.add_argument("--release_steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def main():
    metrics = run_real_smoke(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
