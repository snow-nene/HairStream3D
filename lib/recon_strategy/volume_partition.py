"""Versioned world-space volume contract and depth-local partition propagation.

Arrays use XYZ node order. Labels describe nearest-node cells; zero is outside
the growth domain. Evidence confidence is an engineering weight, not probability.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra


EVIDENCE_CLASSES = {"unknown": 0, "hair": 1, "solid": 2, "air": 3}


def physical_sdf(domain, spacing):
    """Positive-inside metric distance, including the box exterior."""
    padded = np.pad(np.asarray(domain, bool), 1, constant_values=False)
    signed = (ndimage.distance_transform_edt(padded, sampling=spacing)
              - ndimage.distance_transform_edt(~padded, sampling=spacing))
    return signed[1:-1, 1:-1, 1:-1].astype(np.float32)


def validate_bundle(arrays, metadata):
    """Reject ambiguous coordinates, labels, evidence and incompatible grids."""
    required = {"version", "axis_order", "units", "origin", "spacing",
                "world_to_grid", "evidence_classes", "sources", "depth_convention"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"Missing volume metadata: {sorted(missing)}")
    if (metadata["version"] != 1 or metadata["axis_order"] != "XYZ"
            or metadata["units"] != "m" or metadata["evidence_classes"] != EVIDENCE_CLASSES):
        raise ValueError("Unsupported volume version, axes, units or evidence classes")
    origin = np.asarray(metadata["origin"], float)
    spacing = np.asarray(metadata["spacing"], float)
    if (origin.shape != (3,) or spacing.shape != (3,)
            or not np.isfinite(origin).all() or not np.isfinite(spacing).all()
            or np.any(spacing <= 0)):
        raise ValueError("Invalid volume origin/spacing")
    transform = np.eye(4)
    transform[:3, :3] = np.diag(1 / spacing)
    transform[:3, 3] = -origin / spacing
    supplied = np.asarray(metadata["world_to_grid"], float)
    if supplied.shape != (4, 4) or not np.allclose(supplied, transform):
        raise ValueError("world_to_grid disagrees with origin/spacing")
    if not metadata["sources"] or not metadata["depth_convention"]:
        raise ValueError("Sources and depth convention must be explicit")
    names = ("domain_mask", "domain_sdf", "solid_mask", "evidence_class",
             "evidence_confidence", "partition_labels", "partition_confidence", "conflict")
    if any(name not in arrays for name in names):
        raise ValueError("Incomplete volume arrays")
    shape = arrays["domain_mask"].shape
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError("Volume must have at least two nodes on each axis")
    for name in names:
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Invalid shape/nonfinite values: {name}")
    for name in ("domain_mask", "solid_mask", "conflict"):
        if arrays[name].dtype != bool:
            raise ValueError(f"{name} must be boolean")
    domain = arrays["domain_mask"]
    labels = arrays["partition_labels"]
    if (not domain.any() or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels[domain] <= 0) or np.any(labels[~domain] != 0)):
        raise ValueError("Partition labels must be positive exactly inside domain")
    if np.any(domain & arrays["solid_mask"]):
        raise ValueError("Domain intersects solid")
    if not np.isin(arrays["evidence_class"], list(EVIDENCE_CLASSES.values())).all():
        raise ValueError("Unknown evidence class")
    for name in ("evidence_confidence", "partition_confidence"):
        if np.any((arrays[name] < 0) | (arrays[name] > 1)):
            raise ValueError(f"{name} must be in [0, 1]")
    if np.any(arrays["domain_sdf"][domain] <= 0) or np.any(arrays["domain_sdf"][~domain] >= 0):
        raise ValueError("SDF sign disagrees with domain")
    return origin, spacing


def save_bundle(path, arrays, metadata):
    validate_bundle(arrays, metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, metadata_json=np.asarray(json.dumps(metadata)))


def load_bundle(path):
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data:
            raise ValueError("Missing volume metadata; use an explicit legacy adapter")
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: data[key] for key in data.files if key != "metadata_json"}
    validate_bundle(arrays, metadata)
    return arrays, metadata


def adapt_refined_evidence(refined_path, original_path, envelope_path):
    """Explicit adapter for refine_volume_evidence.py / step_06 / step_05.

    Restores the original solid prior even where later conflict handling changed
    the evidence class to unknown. Envelope is resampled by nearest node.
    """
    with np.load(refined_path, allow_pickle=False) as f:
        evidence, conflict = f["labels"].copy(), f["conflict"].copy()
        low, high = f["b_min"].copy(), f["b_max"].copy()
    with np.load(original_path, allow_pickle=False) as f:
        if not np.allclose(f["b_min"], low) or not np.allclose(f["b_max"], high):
            raise ValueError("Legacy evidence bounds differ")
        solid = f["head_core_prior"].astype(bool)
    if solid.shape != evidence.shape or not np.isin(evidence, [0, 1, 2, 3]).all():
        raise ValueError("Legacy evidence shape/classes differ")
    spacing = (high - low) / (np.asarray(evidence.shape) - 1)
    with np.load(envelope_path, allow_pickle=False) as f:
        if not np.allclose(f["b_min"], low) or not np.allclose(f["b_max"], high):
            raise ValueError("Legacy envelope bounds differ")
        axes = [np.linspace(0, n - 1, r) for n, r in zip(f["domain"].shape, evidence.shape)]
        coords = np.asarray(np.meshgrid(*axes, indexing="ij"))
        envelope = ndimage.map_coordinates(f["domain"].astype(np.uint8), coords, order=0) > 0
    domain = envelope & ~solid & ~((evidence == 3) & ~conflict)
    weights = np.where(evidence == 0, 0.25, 1.0).astype(np.float32)
    weights[conflict] = 0
    transform = np.eye(4)
    transform[:3, :3] = np.diag(1 / spacing)
    transform[:3, 3] = -low / spacing
    metadata = {"version": 1, "axis_order": "XYZ", "units": "m",
                "origin": low.tolist(), "spacing": spacing.tolist(),
                "world_to_grid": transform.tolist(), "evidence_classes": EVIDENCE_CLASSES,
                "sources": [str(refined_path), str(original_path), str(envelope_path)],
                "depth_convention": "legacy mesh ray distance in metres; neural depth not calibrated",
                "adapter": "refine_volume_evidence_step07_v1", "solid_is_prior": True}
    arrays = {"domain_mask": domain, "domain_sdf": physical_sdf(domain, spacing),
              "solid_mask": solid, "evidence_class": evidence, "conflict": conflict,
              "evidence_confidence": weights,
              "partition_labels": np.zeros(evidence.shape, np.int32),
              "partition_confidence": np.zeros(evidence.shape, np.float32)}
    return arrays, metadata


def surface_partition_seeds(domain, origin, spacing, surface_points, labels, radius):
    """Seed only a metric neighbourhood of calibrated visible surface points."""
    from scipy.spatial import cKDTree
    points = np.asarray(surface_points, float)
    labels = np.asarray(labels)
    if points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),):
        raise ValueError("Surface points/labels shape mismatch")
    if radius <= 0 or not np.isfinite(points).all() or np.any(labels <= 0):
        raise ValueError("Invalid surface evidence")
    seeds = np.zeros(domain.shape, np.int32)
    if not len(points):
        raise ValueError("No visible surface seeds")
    indices = np.argwhere(domain)
    world = np.asarray(origin) + indices * np.asarray(spacing)
    distance, nearest = cKDTree(points).query(world)
    supported = distance <= radius
    seeds[tuple(indices[supported].T)] = labels[nearest[supported]]
    return seeds


def propagate_partitions(domain, seeds, spacing, confidence=None, blocked_edges=None):
    """Shortest paths on the six-neighbour domain, never through empty space.

    blocked_edges[a, i, j, k] blocks the positive-axis edge at that node.
    Computes best and second-best label costs without a labels-by-volume array.
    """
    domain = np.asarray(domain, bool)
    seeds = np.asarray(seeds)
    if seeds.shape != domain.shape or np.any(seeds[~domain] != 0):
        raise ValueError("Seeds must be inside domain")
    ids = np.unique(seeds[seeds > 0])
    if not len(ids):
        raise ValueError("No partition seeds")
    confidence = np.ones(domain.shape) if confidence is None else np.asarray(confidence)
    if confidence.shape != domain.shape or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Invalid evidence confidence")
    blocked = np.zeros((3, *domain.shape), bool) if blocked_edges is None else np.asarray(blocked_edges, bool)
    if blocked.shape != (3, *domain.shape):
        raise ValueError("Invalid blocked edge shape")
    index = np.full(domain.shape, -1, np.int64)
    count = int(domain.sum())
    index[domain] = np.arange(count)
    rows, cols, values = [], [], []
    for axis in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        active = domain[a] & domain[b] & ~blocked[(axis, *a)]
        weight = float(spacing[axis]) * (1 + 2 - confidence[a] - confidence[b])
        rows.extend([index[a][active], index[b][active]])
        cols.extend([index[b][active], index[a][active]])
        values.extend([weight[active], weight[active]])
    graph = sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(count, count))
    best = np.full(count, np.inf)
    second = best.copy()
    winner = np.zeros(count, np.int32)
    for label_id in ids:
        cost = dijkstra(graph, directed=False, indices=index[seeds == label_id], min_only=True)
        better = cost < best
        second = np.where(better, best, np.minimum(second, cost))
        best = np.minimum(best, cost)
        winner[better] = label_id
    if not np.isfinite(best).all():
        raise ValueError(f"Unseeded domain components: {int((~np.isfinite(best)).sum())} voxels")
    result = np.zeros(domain.shape, np.int32)
    result[domain] = winner
    scale = max(float(np.max(spacing)), 1e-12)
    certainty = np.ones(count)
    finite_second = np.isfinite(second)
    certainty[finite_second] = (second[finite_second] - best[finite_second]) / (second[finite_second] + scale)
    certainty *= 1 / (1 + best / (5 * scale))
    out_conf = np.zeros(domain.shape, np.float32)
    out_conf[domain] = certainty
    return result, out_conf


def merge_view_partition_seeds(existing, existing_directions, local, local_directions,
                               min_overlap=10, min_axial_dot=0.8660254):
    """Match unique overlapping regions only when world line directions agree.

    Existing (front-first) seed assignments win conflicts. IDs from disjoint
    views are allocated afresh, regardless of local integer equality.
    """
    existing, local = np.asarray(existing), np.asarray(local)
    if existing.shape != local.shape or existing_directions.shape != (*existing.shape, 3) or local_directions.shape != existing_directions.shape:
        raise ValueError("Seed/direction grids must agree")
    if min_overlap < 1 or not 0 <= min_axial_dot <= 1:
        raise ValueError("Invalid overlap/direction thresholds")
    output, directions = existing.copy(), existing_directions.copy()
    conflicts = np.zeros(existing.shape, bool)
    mapping = {}
    next_id = int(existing.max()) + 1
    for local_id in np.unique(local[local > 0]):
        mask = local == local_id
        overlap = mask & (existing > 0)
        owners = np.unique(existing[overlap])
        matched = False
        if len(owners) == 1 and overlap.sum() >= min_overlap:
            a, b = existing_directions[overlap], local_directions[overlap]
            norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            dot = np.abs(np.sum(a*b, axis=1)) / np.maximum(norms, 1e-12)
            matched = bool(np.all(norms > 1e-8) and np.mean(dot >= min_axial_dot) >= .9)
        mapped = int(owners[0]) if matched else next_id
        if not matched:
            next_id += 1
        mapping[str(int(local_id))] = mapped
        conflicts |= overlap & (existing != mapped)
        added = mask & (existing == 0)
        output[added] = mapped
        directions[added] = local_directions[added]
    return output, directions, conflicts, mapping
