"""S0-S3 smoke tests for the topology-preserving scalp parting core."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.recon_strategy.parting_topology import (
    blend_parting_cap_directions,
    build_scalp_cut_graph,
    integrate_surface_prefix,
    line_tensors,
    principal_line_directions,
    solve_cut_graph_laplace,
    surface_scalar_gradient,
)


def _grid_mesh(nx=33, ny=33, extent=1.0):
    x = np.linspace(-extent, extent, nx)
    y = np.linspace(-extent, extent, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])
    faces = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            lower = row * nx + column
            faces.append([lower, lower + 1, lower + nx + 1])
            faces.append([lower, lower + nx + 1, lower + nx])
    return vertices, np.asarray(faces, dtype=np.int64), xx.ravel(), yy.ravel()


def _hemisphere_mesh(latitude_count=21, longitude_count=65, radius=0.10):
    latitude = np.linspace(0.18, 1.35, latitude_count)
    longitude = np.linspace(-1.35, 1.35, longitude_count)
    lon, lat = np.meshgrid(longitude, latitude, indexing="xy")
    vertices = radius * np.column_stack([
        np.sin(lon).ravel() * np.sin(lat).ravel(),
        np.cos(lat).ravel(),
        np.cos(lon).ravel() * np.sin(lat).ravel(),
    ])
    faces = []
    for row in range(latitude_count - 1):
        for column in range(longitude_count - 1):
            lower = row * longitude_count + column
            faces.append([lower, lower + 1, lower + longitude_count + 1])
            faces.append([lower, lower + longitude_count + 1, lower + longitude_count])
    normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    return (
        vertices,
        np.asarray(faces, dtype=np.int64),
        lon.ravel(),
        lat.ravel(),
        normals,
    )


def scenario_s0_cut_graph():
    vertices, faces, x, y = _grid_mesh()
    seam = 0.16 * np.sin(np.pi * y)
    graph = build_scalp_cut_graph(vertices, faces, x - seam)
    assert graph.metrics.removed_crossing_edges > 0
    assert graph.metrics.crossing_edges_outside_cap == 0
    assert graph.metrics.mixed_side_components_outside_cap == 0

    boundary = np.flatnonzero(np.isclose(np.abs(y), 1.0))
    values = (y[boundary] + 1.0) * 0.5
    potential, pde = solve_cut_graph_laplace(graph, boundary, values)
    assert pde.converged
    assert pde.final_relative_residual <= 5e-10
    assert pde.dirichlet_max_error <= 1e-12
    gradient = surface_scalar_gradient(
        graph,
        potential,
        normals=np.tile([0.0, 0.0, 1.0], (len(vertices), 1)),
    )
    assert np.mean(gradient[:, 1]) > 0.85
    return {
        "name": "S0_cut_graph",
        "passed": True,
        "cut_graph": graph.metrics.to_dict(),
        "pde": pde.to_dict(),
        "mean_gradient_y": float(np.mean(gradient[:, 1])),
    }


def scenario_s1_root_layer():
    vertices, faces, longitude, latitude, normals = _hemisphere_mesh()
    graph = build_scalp_cut_graph(vertices, faces, longitude)
    screen_x = np.tile([1.0, 0.0, 0.0], (len(vertices), 1))
    tangent = screen_x - np.sum(screen_x * normals, axis=1, keepdims=True) * normals
    tangent *= graph.side[:, None]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12

    near_seam = np.argsort(np.abs(longitude))
    left = near_seam[graph.side[near_seam] < 0][:128]
    right = near_seam[graph.side[near_seam] > 0][:128]
    roots = np.concatenate([left, right])
    prefix = integrate_surface_prefix(
        graph,
        roots,
        tangent,
        normals,
        num_points=24,
        root_height=0.0005,
        release_height=0.006,
        release_steps=12,
    )
    radii = np.linalg.norm(prefix.points, axis=2)
    measured_height = radii - 0.10
    error = np.abs(measured_height - prefix.target_heights[None])
    root_error = float(np.max(error[:, 0]))
    prefix_error = float(np.max(error[:, :6]))
    assert len(roots) == 256
    assert prefix.side_violations_before_cap == 0
    assert root_error <= 0.0005
    assert prefix_error <= 0.001
    return {
        "name": "S1_root_layer",
        "passed": True,
        "strand_count": int(len(roots)),
        "side_violations_before_cap": prefix.side_violations_before_cap,
        "stalled_steps": prefix.stalled_steps,
        "root_error_max_m": root_error,
        "prefix_error_max_m": prefix_error,
    }


def scenario_s2_endpoint_cap():
    vertices, faces, x, y = _grid_mesh(nx=41, ny=41)
    seam = 0.12 * np.sin(np.pi * y)
    cap = y >= 0.75
    graph = build_scalp_cut_graph(vertices, faces, x - seam, cap_mask=cap)
    assert graph.metrics.crossing_edges_outside_cap == 0
    assert graph.metrics.cap_crossing_edges > 0

    distances = np.linspace(0.20, 0.0, 41)
    left = np.tile([-1.0, 0.0, 0.0], (len(distances), 1))
    right = np.tile([1.0, 0.0, 0.0], (len(distances), 1))
    endpoint = np.array([0.0, 1.0, 0.0])
    left_blend = blend_parting_cap_directions(left, endpoint, distances, 0.20)
    right_blend = blend_parting_cap_directions(right, endpoint, distances, 0.20)

    def max_turn(directions):
        cosine = np.sum(directions[:-1] * directions[1:], axis=1)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).max())

    max_turn_degrees = max(max_turn(left_blend), max_turn(right_blend))
    endpoint_disagreement = float(np.linalg.norm(left_blend[-1] - right_blend[-1]))
    assert endpoint_disagreement <= 1e-12
    assert max_turn_degrees <= 5.0
    return {
        "name": "S2_endpoint_cap",
        "passed": True,
        "cut_graph": graph.metrics.to_dict(),
        "max_turn_degrees": max_turn_degrees,
        "endpoint_direction_disagreement": endpoint_disagreement,
    }


def scenario_s3_sign_invariance():
    rng = np.random.default_rng(20260826)
    directions = rng.normal(size=(256, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    flips = rng.choice([-1.0, 1.0], size=(256, 1))
    reference = directions.copy()
    original_q = line_tensors(directions)
    flipped_q = line_tensors(directions * flips)
    tensor_error = float(np.max(np.abs(original_q - flipped_q)))
    original = principal_line_directions(original_q, reference)
    flipped = principal_line_directions(flipped_q, reference)
    trajectory_error = float(
        np.max(np.linalg.norm(np.cumsum(original, axis=0) - np.cumsum(flipped, axis=0), axis=1))
    )
    assert tensor_error <= 1e-12
    assert trajectory_error <= 1e-10
    return {
        "name": "S3_sign_invariance",
        "passed": True,
        "random_flip_count": int(np.sum(flips < 0.0)),
        "tensor_max_error": tensor_error,
        "trajectory_max_error": trajectory_error,
    }


def run_smoke_suite(output_dir):
    results = [
        scenario_s0_cut_graph(),
        scenario_s1_root_layer(),
        scenario_s2_endpoint_cap(),
        scenario_s3_sign_invariance(),
    ]
    summary = {
        "suite": "parting_topology_smoke_s0_s3",
        "passed": all(result["passed"] for result in results),
        "full_reconstruction_run": False,
        "scenarios": results,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def test_s0_cut_graph():
    scenario_s0_cut_graph()


def test_s1_root_layer():
    scenario_s1_root_layer()


def test_s2_endpoint_cap():
    scenario_s2_endpoint_cap()


def test_s3_sign_invariance():
    scenario_s3_sign_invariance()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="tests/outputs/parting_topology_smoke",
    )
    args = parser.parse_args()
    summary = run_smoke_suite(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
