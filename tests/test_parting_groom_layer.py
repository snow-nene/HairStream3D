import numpy as np
import trimesh

from lib.recon_strategy.parting_groom import (
    GroomConfig,
    build_curve_lateral_frame,
    build_local_parting_layer,
    curve_coordinates,
    select_balanced_roots,
)


def _synthetic_case():
    curve_y = np.linspace(-0.04, 0.04, 33)
    curve = np.column_stack([np.zeros_like(curve_y), curve_y, np.zeros_like(curve_y)])
    roots = []
    sides = []
    prefixes = []
    guides = []
    for side in (-1, 1):
        for y in np.linspace(-0.038, 0.038, 48):
            root = np.array([side * 0.006, y, 0.0005])
            prefix = np.repeat(root[None, :], 4, axis=0)
            prefix[:, 2] += np.linspace(0.0, 0.003, 4)
            roots.append(root)
            sides.append(side)
            prefixes.append(prefix)

            arc = np.linspace(0.0, 0.09, 31)
            widening = np.minimum(arc / 0.03, 1.0) * 0.025
            guide = np.column_stack(
                [side * (0.010 + widening), np.full_like(arc, y), arc + 0.0005]
            )
            guides.append(guide)
    vertices = np.array(
        [[-0.08, -0.08, 0.0], [0.08, -0.08, 0.0], [0.08, 0.08, 0.0], [-0.08, 0.08, 0.0]]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    return (
        guides,
        np.asarray(prefixes),
        np.asarray(sides, dtype=np.int8),
        curve,
        mesh,
    )


def test_balanced_selection_is_curve_uniform_and_equal():
    _, prefixes, side, curve, _, = _synthetic_case()
    roots = prefixes[:, 0]
    lateral = build_curve_lateral_frame(curve, roots, side)
    config = GroomConfig(strands_per_side=12)
    selected = select_balanced_roots(roots, side, curve, lateral, config)

    assert len(selected) == 24
    assert np.sum(side[selected] == -1) == 12
    assert np.sum(side[selected] == 1) == 12
    _, _, nearest = curve_coordinates(roots[selected], curve, lateral)
    for bank in (-1, 1):
        bank_curve = nearest[side[selected] == bank]
        assert np.ptp(bank_curve) >= 24


def test_local_layer_caps_opening_and_preserves_tail():
    guides, prefixes, side, curve, mesh = _synthetic_case()
    config = GroomConfig(
        strands_per_side=12,
        guide_root_max_m=0.05,
        max_added_opening_m=0.002,
        handoff_m=0.06,
    )
    layer, metrics = build_local_parting_layer(
        guides, prefixes, side, curve, mesh, config
    )

    assert metrics["side_counts"] == {"-1": 12, "1": 12}
    assert metrics["root_layer"]["side_violations"] == 0
    assert metrics["opening_ratio_30mm"] <= 1.5
    assert metrics["tail_max_deviation_m"] == 0.0
    assert metrics["root_surface_distance_m"]["q50"] <= 0.001
    assert metrics["passed"], metrics["failed_gates"]
    assert all(len(strand) > 31 for strand in layer)
