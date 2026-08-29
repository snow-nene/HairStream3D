"""Generate full-density topology-preserving scalp root prefixes.

This is the production-scale companion to ``run_parting_topology_smoke.py``.
It keeps the smoke limits intact and refines the real crown mesh until each
parting bank contains enough distinct vertices for balanced, no-replacement
root sampling.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.recon_strategy.parting_topology import (
    build_scalp_cut_graph,
    integrate_surface_prefix,
    solve_cut_graph_laplace,
    surface_scalar_gradient,
)
from scripts.recon_3d.run_parting_topology_smoke import (
    crop_mesh_around_curve,
    curve_lateral_frame,
    load_curve,
    select_balanced_roots,
    signed_curve_distance,
    write_prefix_obj,
    write_roots_obj,
)


def refine_mesh(mesh, levels):
    """Apply midpoint subdivision while preserving welded connectivity."""

    refined = mesh
    for _ in range(int(levels)):
        vertices, faces = trimesh.remesh.subdivide(
            np.asarray(refined.vertices), np.asarray(refined.faces)
        )
        refined = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return refined


def run_full(args):
    if args.root_count < 2 or args.root_count % 2:
        raise ValueError("root_count must be an even integer >= 2")
    if args.num_points < 2:
        raise ValueError("num_points must be >= 2")

    mesh = trimesh.load(args.head_mesh, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("head_mesh must be a non-empty triangle mesh")
    curve_raw = load_curve(args.parting_curve)
    curve, lateral = curve_lateral_frame(mesh, curve_raw)
    scalp = crop_mesh_around_curve(
        mesh, curve, args.crop_radius, args.vertical_margin
    )
    scalp = refine_mesh(scalp, args.extra_subdivisions)

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
        raise AssertionError("full cut graph retained an outside-cap crossing")
    if graph.metrics.mixed_side_components_outside_cap != 0:
        raise AssertionError("full cut graph has a mixed-side outside-cap component")

    bank = np.abs(signed)
    near = bank <= args.pde_near_bank
    far_threshold = float(np.quantile(bank, args.pde_far_quantile))
    far = bank >= far_threshold
    boundary_ids = np.flatnonzero(near | far)
    potential, pde = solve_cut_graph_laplace(
        graph,
        boundary_ids,
        far[boundary_ids].astype(np.float64),
        tolerance=args.pde_tolerance,
        max_iterations=args.pde_max_iterations,
    )
    if not pde.converged:
        raise AssertionError(
            f"full scalp PDE did not converge: {pde.final_relative_residual}"
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

    _, surface_distance, _ = trimesh.proximity.closest_point(
        scalp, prefix.points.reshape(-1, 3)
    )
    surface_distance = surface_distance.reshape(args.root_count, args.num_points)
    height_error = np.abs(surface_distance - prefix.target_heights[None, :])
    stalled_fraction = float(
        prefix.stalled_steps / max(args.root_count * (args.num_points - 1), 1)
    )
    if prefix.side_violations_before_cap != 0:
        raise AssertionError("full prefixes crossed the parting before the cap")
    if float(np.max(height_error[:, 0])) > args.root_error_max:
        raise AssertionError("full root attachment error exceeds the configured gate")
    if stalled_fraction > args.stalled_fraction_max:
        raise AssertionError("full prefix stalled fraction exceeds the configured gate")

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
        "suite": "parting_topology_full_real_crown",
        "passed": True,
        "root_count": int(args.root_count),
        "num_points": int(args.num_points),
        "extra_subdivisions": int(args.extra_subdivisions),
        "crop": {
            "vertices": int(len(scalp.vertices)),
            "faces": int(len(scalp.faces)),
            "curve_points": int(len(curve)),
            "curve_vertex_distance_q95_m": float(
                np.quantile(curve_distance, 0.95)
            ),
        },
        "cut_graph": graph.metrics.to_dict(),
        "pde": pde.to_dict(),
        "root_layer": {
            "side_violations_before_cap": prefix.side_violations_before_cap,
            "stalled_steps": prefix.stalled_steps,
            "stalled_fraction": stalled_fraction,
            "root_height_error_max_m": float(np.max(height_error[:, 0])),
            "prefix_height_error_q95_m": float(
                np.quantile(height_error[:, :6], 0.95)
            ),
            "prefix_height_error_max_m": float(np.max(height_error[:, :6])),
            "surface_distance_root_q50_m": float(
                np.quantile(surface_distance[:, 0], 0.50)
            ),
            "surface_distance_root_q95_m": float(
                np.quantile(surface_distance[:, 0], 0.95)
            ),
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
    parser.add_argument("--root_count", type=int, default=10000)
    parser.add_argument("--num_points", type=int, default=30)
    parser.add_argument("--extra_subdivisions", type=int, default=2)
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
    parser.add_argument("--root_error_max", type=float, default=0.0015)
    parser.add_argument("--stalled_fraction_max", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def main():
    print(json.dumps(run_full(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
