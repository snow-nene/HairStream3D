import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.vis.audit_strand_depth_partitions import (
    axial_direction_score,
    compute_partition_evidence,
    decode_axial_field,
    partition_from_barrier,
    select_topological_edge_components,
)


def _strand_map(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    strand = np.zeros((*dx.shape, 3), dtype=np.float32)
    strand[:, :, 0] = 1.0
    strand[:, :, 1] = 0.5 * (dy + 1.0)
    strand[:, :, 2] = 0.5 * (1.0 - dx)
    return strand


def test_axial_field_is_invariant_to_direction_sign():
    dx = np.ones((16, 16), dtype=np.float32)
    dx[:, 8:] = -1.0
    dy = np.zeros_like(dx)
    seg = np.ones_like(dx, dtype=bool)
    axial, valid = decode_axial_field(_strand_map(dx, dy), seg)
    score = axial_direction_score(axial, valid, scales=(1,))
    assert float(score.max()) < 1e-6


def test_axial_score_detects_perpendicular_line_change():
    dx = np.ones((16, 16), dtype=np.float32)
    dy = np.zeros_like(dx)
    dx[:, 8:] = 0.0
    dy[:, 8:] = 1.0
    seg = np.ones_like(dx, dtype=bool)
    axial, valid = decode_axial_field(_strand_map(dx, dy), seg)
    score = axial_direction_score(axial, valid, scales=(1,))
    assert float(score[:, 7:9].mean()) > 0.9
    assert float(score[:, :6].max()) < 1e-6


def test_depth_jump_outside_eroded_hair_is_suppressed():
    height = width = 32
    seg = np.zeros((height, width), dtype=bool)
    seg[8:24, 8:24] = True
    dx = np.ones((height, width), dtype=np.float32)
    dy = np.zeros_like(dx)
    depth = np.ones((height, width), dtype=np.float32)
    depth[:, :8] = 4.0
    evidence = compute_partition_evidence(
        _strand_map(dx, dy), depth, seg, interior_margin=3
    )
    assert not np.any(np.asarray(evidence["valid"])[:, :11])
    assert float(np.asarray(evidence["depth"])[:, :11].max()) == 0.0


def test_empty_barrier_keeps_one_connected_hair_region():
    seg = np.zeros((32, 32), dtype=bool)
    seg[4:28, 4:28] = True
    labels, metrics = partition_from_barrier(
        seg,
        np.zeros_like(seg),
        min_region_pixels=16,
    )
    assert metrics["effective_region_count"] == 1
    assert np.all(labels[seg] == 1)


def test_topological_filter_keeps_only_partitioning_edge():
    seg = np.zeros((40, 40), dtype=bool)
    seg[4:36, 4:36] = True
    edge = np.zeros_like(seg)
    edge[4:36, 20] = True
    edge[12:15, 10:13] = True
    retained, metrics = select_topological_edge_components(
        seg,
        edge,
        barrier_radius=1,
        min_region_pixels=64,
    )
    assert metrics["retained_component_count"] == 1
    assert retained[20, 20]
    assert not retained[13, 11]
    labels, partition = partition_from_barrier(
        seg,
        retained,
        barrier_radius=1,
        min_region_pixels=64,
    )
    assert partition["effective_region_count"] == 2
    assert labels[20, 10] != labels[20, 30]


if __name__ == "__main__":
    tests = [
        test_axial_field_is_invariant_to_direction_sign,
        test_axial_score_detects_perpendicular_line_change,
        test_depth_jump_outside_eroded_hair_is_suppressed,
        test_empty_barrier_keeps_one_connected_hair_region,
        test_topological_filter_keeps_only_partitioning_edge,
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} strand/depth partition tests")
