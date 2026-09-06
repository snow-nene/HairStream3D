"""Generate an opt-in local parting groom layer over an existing v30 PLY."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.recon_strategy.parting_groom import (
    GroomConfig,
    build_local_parting_layer,
    load_topology_roots,
    read_ordered_strands,
    write_ordered_strands,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v30_ply", required=True)
    parser.add_argument("--topology_npz", required=True)
    parser.add_argument("--head_mesh", default="data/head_model.obj")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--strands_per_side", type=int, default=128)
    parser.add_argument("--root_bank_min_mm", type=float, default=4.0)
    parser.add_argument("--root_bank_max_mm", type=float, default=12.0)
    parser.add_argument("--guide_root_max_mm", type=float, default=35.0)
    parser.add_argument("--dense_length_mm", type=float, default=20.0)
    parser.add_argument("--root_spacing_mm", type=float, default=1.0)
    parser.add_argument("--side_lock_length_mm", type=float, default=12.0)
    parser.add_argument("--opening_control_mm", type=float, default=30.0)
    parser.add_argument("--handoff_mm", type=float, default=60.0)
    parser.add_argument("--max_added_opening_mm", type=float, default=2.0)
    parser.add_argument("--surface_root_clearance_mm", type=float, default=0.5)
    parser.add_argument("--surface_peak_clearance_mm", type=float, default=4.0)
    return parser.parse_args()


def _mm(value: float) -> float:
    return float(value) / 1000.0


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    expected_root = (Path.cwd() / "results" / "multiview_data").resolve()
    if expected_root not in output.parents:
        raise ValueError(f"输出目录必须位于 {expected_root} 下")
    input_paths = {
        Path(args.v30_ply).resolve(),
        Path(args.topology_npz).resolve(),
        Path(args.head_mesh).resolve(),
    }
    if output in input_paths:
        raise ValueError("输出目录不能覆盖输入文件")

    config = GroomConfig(
        strands_per_side=args.strands_per_side,
        root_bank_min_m=_mm(args.root_bank_min_mm),
        root_bank_max_m=_mm(args.root_bank_max_mm),
        guide_root_max_m=_mm(args.guide_root_max_mm),
        dense_length_m=_mm(args.dense_length_mm),
        root_spacing_m=_mm(args.root_spacing_mm),
        side_lock_length_m=_mm(args.side_lock_length_mm),
        opening_control_m=_mm(args.opening_control_mm),
        handoff_m=_mm(args.handoff_mm),
        max_added_opening_m=_mm(args.max_added_opening_mm),
        surface_root_clearance_m=_mm(args.surface_root_clearance_mm),
        surface_peak_clearance_m=_mm(args.surface_peak_clearance_mm),
    )
    guides = read_ordered_strands(args.v30_ply)
    topology = load_topology_roots(args.topology_npz)
    head_mesh = trimesh.load(args.head_mesh, process=False)
    if not isinstance(head_mesh, trimesh.Trimesh) or not len(head_mesh.faces):
        raise ValueError("head_mesh 必须是非空三角网格")

    layer, metrics = build_local_parting_layer(
        guides,
        topology["points"],
        topology["side"],
        topology["parting_curve"],
        head_mesh,
        config,
    )
    output.mkdir(parents=True, exist_ok=True)
    layer_path = output / "parting_groom_layer.ply"
    combined_path = output / "hair_multiview.ply"
    metrics_path = output / "metrics.json"
    write_ordered_strands(layer, layer_path)
    write_ordered_strands([*guides, *layer], combined_path)
    report = {
        "suite": "local_parting_groom_layer",
        "inputs": {
            "v30_ply": str(Path(args.v30_ply).resolve()),
            "topology_npz": str(Path(args.topology_npz).resolve()),
            "head_mesh": str(Path(args.head_mesh).resolve()),
        },
        "config": asdict(config),
        "base_strand_count": len(guides),
        "outputs": {
            "layer_ply": str(layer_path),
            "combined_ply": str(combined_path),
            "metrics": str(metrics_path),
        },
        **metrics,
    }
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
