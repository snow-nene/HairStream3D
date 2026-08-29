"""Topology-preserving scalp parting primitives.

This module is intentionally independent from the production reconstruction
entry point.  It provides the small, auditable core needed to prove that a
parting changes the discrete PDE topology instead of merely adding a finite
penalty to an otherwise connected domain.
"""

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg


@dataclass(frozen=True)
class CutGraphMetrics:
    """Structural evidence emitted while constructing a scalp cut graph."""

    vertex_count: int
    full_edge_count: int
    kept_edge_count: int
    removed_crossing_edges: int
    cap_crossing_edges: int
    crossing_edges_outside_cap: int
    outside_cap_components: int
    mixed_side_components_outside_cap: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SurfacePDEMetrics:
    """Numerical evidence for one cut-graph Laplace solve."""

    unknown_count: int
    dirichlet_count: int
    component_count: int
    iterations: int
    initial_relative_residual: float
    final_relative_residual: float
    dirichlet_max_error: float
    converged: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ScalpCutGraph:
    """Triangle-mesh graph with all non-cap parting crossings removed."""

    vertices: np.ndarray
    faces: np.ndarray
    full_edges: np.ndarray
    edges: np.ndarray
    side: np.ndarray
    cap_mask: np.ndarray
    signed_distance: np.ndarray
    adjacency: sparse.csr_matrix
    metrics: CutGraphMetrics


@dataclass(frozen=True)
class SurfacePrefixResult:
    """Root prefixes represented by surface vertices plus normal height."""

    points: np.ndarray
    surface_vertex_ids: np.ndarray
    target_heights: np.ndarray
    side_violations_before_cap: int
    stalled_steps: int


def _unique_mesh_edges(faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
        axis=0,
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _binary_adjacency(vertex_count: int, edges: np.ndarray) -> sparse.csr_matrix:
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return sparse.csr_matrix((vertex_count, vertex_count), dtype=np.float64)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    values = np.ones(len(rows), dtype=np.float64)
    return sparse.csr_matrix(
        (values, (rows, cols)), shape=(vertex_count, vertex_count)
    )


def build_scalp_cut_graph(
    vertices: np.ndarray,
    faces: np.ndarray,
    signed_distance: np.ndarray,
    cap_mask: Optional[np.ndarray] = None,
) -> ScalpCutGraph:
    """Remove every mesh edge that crosses the parting outside its rear cap.

    Zero signed-distance vertices deterministically belong to the positive
    chart.  This makes their negative-side incident edges crossing edges, so a
    vertex exactly on the sampled seam cannot silently bridge both charts.
    Crossing edges are retained only when both endpoints are inside the cap.
    """

    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    distance = np.asarray(signed_distance, dtype=np.float64).reshape(-1)
    if len(distance) != len(vertices):
        raise ValueError("signed_distance must match the vertex count")
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(distance)):
        raise ValueError("vertices and signed_distance must be finite")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("faces contain an invalid vertex index")

    cap = (
        np.zeros(len(vertices), dtype=bool)
        if cap_mask is None
        else np.asarray(cap_mask, dtype=bool).reshape(-1)
    )
    if len(cap) != len(vertices):
        raise ValueError("cap_mask must match the vertex count")

    side = np.where(distance < 0.0, -1, 1).astype(np.int8)
    full_edges = _unique_mesh_edges(faces)
    opposite = side[full_edges[:, 0]] != side[full_edges[:, 1]]
    cap_crossing = opposite & cap[full_edges[:, 0]] & cap[full_edges[:, 1]]
    removed = opposite & ~cap_crossing
    kept_edges = full_edges[~removed]
    adjacency = _binary_adjacency(len(vertices), kept_edges)

    kept_opposite = side[kept_edges[:, 0]] != side[kept_edges[:, 1]]
    kept_outside_cap = kept_opposite & ~(
        cap[kept_edges[:, 0]] & cap[kept_edges[:, 1]]
    )

    outside_vertices = ~cap
    outside_ids = np.flatnonzero(outside_vertices)
    if len(outside_ids):
        outside_adjacency = adjacency[outside_ids][:, outside_ids]
        component_count, labels = connected_components(
            outside_adjacency, directed=False
        )
        mixed_components = 0
        for component in range(component_count):
            component_sides = side[outside_ids[labels == component]]
            mixed_components += int(
                np.any(component_sides < 0) and np.any(component_sides > 0)
            )
    else:
        component_count = 0
        mixed_components = 0

    metrics = CutGraphMetrics(
        vertex_count=int(len(vertices)),
        full_edge_count=int(len(full_edges)),
        kept_edge_count=int(len(kept_edges)),
        removed_crossing_edges=int(removed.sum()),
        cap_crossing_edges=int(cap_crossing.sum()),
        crossing_edges_outside_cap=int(kept_outside_cap.sum()),
        outside_cap_components=int(component_count),
        mixed_side_components_outside_cap=int(mixed_components),
    )
    return ScalpCutGraph(
        vertices=vertices,
        faces=faces,
        full_edges=full_edges,
        edges=kept_edges,
        side=side,
        cap_mask=cap,
        signed_distance=distance,
        adjacency=adjacency,
        metrics=metrics,
    )


def solve_cut_graph_laplace(
    graph: ScalpCutGraph,
    dirichlet_indices: Sequence[int],
    dirichlet_values: Sequence[float],
    tolerance: float = 1e-10,
    max_iterations: int = 2000,
) -> tuple[np.ndarray, SurfacePDEMetrics]:
    """Solve a metric graph Laplace PDE with exact Dirichlet elimination."""

    count = len(graph.vertices)
    boundary_ids = np.asarray(dirichlet_indices, dtype=np.int64).reshape(-1)
    boundary_values = np.asarray(dirichlet_values, dtype=np.float64).reshape(-1)
    if len(boundary_ids) != len(boundary_values) or len(boundary_ids) == 0:
        raise ValueError("Dirichlet indices and values must be non-empty and match")
    if np.any(boundary_ids < 0) or np.any(boundary_ids >= count):
        raise ValueError("Dirichlet index outside the graph")
    if len(np.unique(boundary_ids)) != len(boundary_ids):
        raise ValueError("Dirichlet indices must be unique")
    if tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("PDE tolerance and max_iterations must be positive")

    edges = graph.edges
    lengths = np.linalg.norm(
        graph.vertices[edges[:, 1]] - graph.vertices[edges[:, 0]], axis=1
    )
    weights = 1.0 / np.maximum(lengths, 1e-12)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([weights, weights])
    weighted_adjacency = sparse.csr_matrix(
        (data, (rows, cols)), shape=(count, count)
    )
    laplacian = sparse.diags(
        np.asarray(weighted_adjacency.sum(axis=1)).reshape(-1)
    ) - weighted_adjacency

    component_count, labels = connected_components(
        graph.adjacency, directed=False
    )
    anchored = set(labels[boundary_ids].tolist())
    missing = sorted(set(range(component_count)) - anchored)
    if missing:
        raise ValueError(f"Cut-graph PDE has unanchored components: {missing}")

    is_boundary = np.zeros(count, dtype=bool)
    is_boundary[boundary_ids] = True
    unknown_ids = np.flatnonzero(~is_boundary)
    solution = np.zeros(count, dtype=np.float64)
    solution[boundary_ids] = boundary_values
    if not len(unknown_ids):
        metrics = SurfacePDEMetrics(
            unknown_count=0,
            dirichlet_count=int(len(boundary_ids)),
            component_count=int(component_count),
            iterations=0,
            initial_relative_residual=0.0,
            final_relative_residual=0.0,
            dirichlet_max_error=0.0,
            converged=True,
        )
        return solution, metrics

    system = laplacian[unknown_ids][:, unknown_ids].tocsr()
    rhs = -laplacian[unknown_ids][:, boundary_ids] @ boundary_values
    rhs_norm = max(float(np.linalg.norm(rhs)), 1e-15)
    initial_relative_residual = float(np.linalg.norm(rhs) / rhs_norm)
    iterations = 0

    def count_iteration(_):
        nonlocal iterations
        iterations += 1

    solved, info = cg(
        system,
        rhs,
        rtol=float(tolerance),
        atol=0.0,
        maxiter=int(max_iterations),
        callback=count_iteration,
    )
    solution[unknown_ids] = solved
    final_relative_residual = float(
        np.linalg.norm(system @ solved - rhs) / rhs_norm
    )
    dirichlet_error = float(
        np.max(np.abs(solution[boundary_ids] - boundary_values), initial=0.0)
    )
    metrics = SurfacePDEMetrics(
        unknown_count=int(len(unknown_ids)),
        dirichlet_count=int(len(boundary_ids)),
        component_count=int(component_count),
        iterations=int(iterations),
        initial_relative_residual=initial_relative_residual,
        final_relative_residual=final_relative_residual,
        dirichlet_max_error=dirichlet_error,
        converged=bool(info == 0 and final_relative_residual <= tolerance * 5.0),
    )
    return solution, metrics


def surface_scalar_gradient(
    graph: ScalpCutGraph,
    scalar: np.ndarray,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate a per-vertex metric gradient using only retained edges."""

    values = np.asarray(scalar, dtype=np.float64).reshape(-1)
    if len(values) != len(graph.vertices):
        raise ValueError("scalar must match the graph vertex count")
    gradient = np.zeros_like(graph.vertices)
    weight_sum = np.zeros(len(graph.vertices), dtype=np.float64)
    for start, stop in graph.edges:
        edge = graph.vertices[stop] - graph.vertices[start]
        length_sq = max(float(np.dot(edge, edge)), 1e-15)
        contribution = (values[stop] - values[start]) * edge / length_sq
        gradient[start] += contribution
        gradient[stop] += contribution
        edge_weight = 1.0 / np.sqrt(length_sq)
        weight_sum[start] += edge_weight
        weight_sum[stop] += edge_weight
    valid = weight_sum > 0.0
    gradient[valid] /= weight_sum[valid, None]
    if normals is not None:
        normal = np.array(
            normals, dtype=np.float64, copy=True
        ).reshape(-1, 3)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-15
        gradient -= np.sum(gradient * normal, axis=1, keepdims=True) * normal
    norms = np.linalg.norm(gradient, axis=1, keepdims=True)
    gradient = np.divide(
        gradient,
        norms,
        out=np.zeros_like(gradient),
        where=norms > 1e-12,
    )
    return gradient


def smooth_release_heights(
    num_points: int,
    root_height: float,
    release_height: float,
    release_steps: int,
) -> np.ndarray:
    """Return a monotone C1 smoothstep root-to-free-layer height schedule."""

    if num_points < 1 or release_steps < 1:
        raise ValueError("num_points and release_steps must be positive")
    indices = np.arange(num_points, dtype=np.float64)
    t = np.clip(indices / float(release_steps), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return float(root_height) + (
        float(release_height) - float(root_height)
    ) * smooth


def integrate_surface_prefix(
    graph: ScalpCutGraph,
    root_vertex_ids: Sequence[int],
    tangent_field: np.ndarray,
    vertex_normals: np.ndarray,
    num_points: int = 24,
    root_height: float = 0.0005,
    release_height: float = 0.006,
    release_steps: int = 12,
) -> SurfacePrefixResult:
    """Trace discrete same-chart roots and lift them along surface normals."""

    roots = np.asarray(root_vertex_ids, dtype=np.int64).reshape(-1)
    tangent = np.asarray(tangent_field, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(vertex_normals, dtype=np.float64).reshape(-1, 3)
    if len(tangent) != len(graph.vertices) or len(normals) != len(graph.vertices):
        raise ValueError("tangent_field and vertex_normals must match the graph")
    if np.any(roots < 0) or np.any(roots >= len(graph.vertices)):
        raise ValueError("root vertex outside the graph")
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(
        normals, normal_norm, out=np.zeros_like(normals), where=normal_norm > 1e-12
    )
    tangent -= np.sum(tangent * normals, axis=1, keepdims=True) * normals
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.divide(
        tangent,
        tangent_norm,
        out=np.zeros_like(tangent),
        where=tangent_norm > 1e-12,
    )

    heights = smooth_release_heights(
        num_points, root_height, release_height, release_steps
    )
    vertex_paths = np.empty((len(roots), num_points), dtype=np.int64)
    vertex_paths[:, 0] = roots
    violations = 0
    stalled = 0
    indptr = graph.adjacency.indptr
    indices = graph.adjacency.indices

    for strand_id, root in enumerate(roots):
        root_side = graph.side[root]
        previous = -1
        current = int(root)
        for step in range(1, num_points):
            candidates = indices[indptr[current]:indptr[current + 1]]
            if not graph.cap_mask[current]:
                candidates = candidates[
                    (graph.side[candidates] == root_side)
                    | graph.cap_mask[candidates]
                ]
            if len(candidates) > 1 and previous >= 0:
                without_previous = candidates[candidates != previous]
                if len(without_previous):
                    candidates = without_previous
            if not len(candidates):
                vertex_paths[strand_id, step:] = current
                stalled += num_points - step
                break
            edges = graph.vertices[candidates] - graph.vertices[current]
            edge_norm = np.linalg.norm(edges, axis=1, keepdims=True)
            edge_unit = np.divide(
                edges, edge_norm, out=np.zeros_like(edges), where=edge_norm > 1e-12
            )
            scores = edge_unit @ tangent[current]
            next_vertex = int(candidates[int(np.argmax(scores))])
            previous, current = current, next_vertex
            vertex_paths[strand_id, step] = current
            if not graph.cap_mask[current] and graph.side[current] != root_side:
                violations += 1

    base = graph.vertices[vertex_paths]
    path_normals = normals[vertex_paths]
    points = base + heights[None, :, None] * path_normals
    return SurfacePrefixResult(
        points=points,
        surface_vertex_ids=vertex_paths,
        target_heights=heights,
        side_violations_before_cap=int(violations),
        stalled_steps=int(stalled),
    )


def smooth_cap_weight(distance_to_endpoint, cap_length: float) -> np.ndarray:
    """C1 blend weight: one at the endpoint and zero outside the cap."""

    if cap_length <= 0.0:
        raise ValueError("cap_length must be positive")
    distance = np.asarray(distance_to_endpoint, dtype=np.float64)
    t = np.clip(distance / float(cap_length), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return 1.0 - smooth


def blend_parting_cap_directions(
    side_directions: np.ndarray,
    endpoint_direction: np.ndarray,
    distance_to_endpoint,
    cap_length: float,
) -> np.ndarray:
    """Blend both parting banks into one common endpoint direction."""

    side = np.asarray(side_directions, dtype=np.float64)
    endpoint = np.asarray(endpoint_direction, dtype=np.float64)
    endpoint = np.broadcast_to(endpoint, side.shape)
    weight = smooth_cap_weight(distance_to_endpoint, cap_length)
    while weight.ndim < side.ndim:
        weight = weight[..., None]
    blended = (1.0 - weight) * side + weight * endpoint
    norm = np.linalg.norm(blended, axis=-1, keepdims=True)
    return np.divide(blended, norm, out=np.zeros_like(blended), where=norm > 1e-12)


def line_tensors(directions: np.ndarray) -> np.ndarray:
    """Represent signless observations as Q = vv^T."""

    direction = np.asarray(directions, dtype=np.float64)
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    unit = np.divide(
        direction, norm, out=np.zeros_like(direction), where=norm > 1e-12
    )
    return np.einsum("...i,...j->...ij", unit, unit)


def principal_line_directions(
    tensors: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Recover Q's principal direction and orient it toward a reference."""

    q = np.asarray(tensors, dtype=np.float64)
    reference = np.broadcast_to(np.asarray(reference, dtype=np.float64), q.shape[:-1])
    _, vectors = np.linalg.eigh(q)
    direction = vectors[..., -1]
    reverse = np.sum(direction * reference, axis=-1) < 0.0
    direction[reverse] *= -1.0
    return direction
