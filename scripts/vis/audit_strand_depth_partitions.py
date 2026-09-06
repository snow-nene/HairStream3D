#!/usr/bin/env python3
"""用 strand_map 与 depth_map 审计头发内部突变边缘及候选分区。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy import ndimage


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
DEFAULT_DATA_ROOT = Path("results/multiview_data") / IMAGE_ID
DEFAULT_QUANTILES = (0.85, 0.90, 0.93, 0.96)


def decode_axial_field(
    strand_map_rgb: np.ndarray,
    seg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """将 RGB strand map 解码为符号不变的双角无向线场。"""

    strand = np.asarray(strand_map_rgb, dtype=np.float32)
    if strand.max(initial=0.0) > 1.5:
        strand = strand / 255.0
    mask = np.asarray(seg, dtype=bool)
    if strand.shape[:2] != mask.shape or strand.ndim != 3 or strand.shape[2] < 3:
        raise ValueError("strand_map 与 seg 的形状不匹配")

    dx = 1.0 - 2.0 * strand[:, :, 2]
    dy = 2.0 * strand[:, :, 1] - 1.0
    norm = np.hypot(dx, dy)
    valid = mask & (strand[:, :, 0] > 0.1) & (norm > 1e-6)
    dx = np.divide(dx, norm, out=np.zeros_like(dx), where=norm > 1e-6)
    dy = np.divide(dy, norm, out=np.zeros_like(dy), where=norm > 1e-6)
    axial = np.stack([dx * dx - dy * dy, 2.0 * dx * dy], axis=-1)
    axial[~valid] = 0.0
    return axial.astype(np.float32), valid


def _pairwise_axial_difference(
    axial: np.ndarray,
    valid: np.ndarray,
    offset_y: int,
    offset_x: int,
) -> np.ndarray:
    """计算一对平移邻域的无向角度差，并把差值回写到两端。"""

    height, width = valid.shape
    y0a, y1a = max(0, -offset_y), min(height, height - offset_y)
    x0a, x1a = max(0, -offset_x), min(width, width - offset_x)
    y0b, y1b = y0a + offset_y, y1a + offset_y
    x0b, x1b = x0a + offset_x, x1a + offset_x
    pair_valid = valid[y0a:y1a, x0a:x1a] & valid[y0b:y1b, x0b:x1b]
    dot = np.sum(
        axial[y0a:y1a, x0a:x1a] * axial[y0b:y1b, x0b:x1b],
        axis=-1,
    )
    difference = 0.5 * np.arccos(np.clip(dot, -1.0, 1.0)) / (0.5 * np.pi)
    difference[~pair_valid] = 0.0
    result = np.zeros(valid.shape, dtype=np.float32)
    result[y0a:y1a, x0a:x1a] = np.maximum(
        result[y0a:y1a, x0a:x1a], difference
    )
    result[y0b:y1b, x0b:x1b] = np.maximum(
        result[y0b:y1b, x0b:x1b], difference
    )
    return result


def axial_direction_score(
    axial: np.ndarray,
    valid: np.ndarray,
    scales: tuple[int, ...] = (1, 2, 4),
) -> np.ndarray:
    """以多尺度邻域最大无向夹角构造方向突变分数。"""

    score = np.zeros(valid.shape, dtype=np.float32)
    for scale in scales:
        if scale <= 0:
            raise ValueError("方向差尺度必须为正整数")
        for dy, dx in ((0, scale), (scale, 0), (scale, scale), (scale, -scale)):
            score = np.maximum(
                score,
                _pairwise_axial_difference(axial, valid, dy, dx),
            )
    score[~valid] = 0.0
    return score


def masked_gaussian(
    values: np.ndarray,
    valid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """只用有效 mask 邻域做归一化 Gaussian 平滑。"""

    weights = ndimage.gaussian_filter(valid.astype(np.float32), sigma=sigma)
    weighted = ndimage.gaussian_filter(
        np.where(valid, values, 0.0).astype(np.float32), sigma=sigma
    )
    return np.divide(
        weighted,
        weights,
        out=np.zeros_like(weighted),
        where=weights > 1e-5,
    )


def depth_discontinuity_score(
    depth: np.ndarray,
    valid: np.ndarray,
    sigma: float = 1.5,
    scales: tuple[int, ...] = (1, 2, 4),
) -> np.ndarray:
    """计算 mask 内多尺度相对深度跳变，屏蔽无效深度及外轮廓。"""

    depth = np.asarray(depth, dtype=np.float32)
    depth_valid = valid & np.isfinite(depth) & (depth > 0.05)
    smooth = masked_gaussian(depth, depth_valid, sigma=sigma)
    score = np.zeros(depth.shape, dtype=np.float32)
    height, width = depth.shape
    for scale in scales:
        for offset_y, offset_x in ((0, scale), (scale, 0), (scale, scale), (scale, -scale)):
            y0a, y1a = max(0, -offset_y), min(height, height - offset_y)
            x0a, x1a = max(0, -offset_x), min(width, width - offset_x)
            y0b, y1b = y0a + offset_y, y1a + offset_y
            x0b, x1b = x0a + offset_x, x1a + offset_x
            pair_valid = (
                depth_valid[y0a:y1a, x0a:x1a]
                & depth_valid[y0b:y1b, x0b:x1b]
            )
            a = smooth[y0a:y1a, x0a:x1a]
            b = smooth[y0b:y1b, x0b:x1b]
            relative = np.abs(a - b) / np.maximum(0.5 * (np.abs(a) + np.abs(b)), 1e-4)
            relative[~pair_valid] = 0.0
            score[y0a:y1a, x0a:x1a] = np.maximum(
                score[y0a:y1a, x0a:x1a], relative
            )
            score[y0b:y1b, x0b:x1b] = np.maximum(
                score[y0b:y1b, x0b:x1b], relative
            )
    score[~depth_valid] = 0.0
    return score


def robust_unit_score(
    score: np.ndarray,
    valid: np.ndarray,
    upper_quantile: float = 0.98,
) -> tuple[np.ndarray, float]:
    """用有效像素稳健高分位数把非负分数压到 [0, 1]。"""

    values = np.asarray(score, dtype=np.float32)[valid]
    positive = values[np.isfinite(values) & (values > 0.0)]
    scale = float(np.quantile(positive, upper_quantile)) if len(positive) else 1.0
    scale = max(scale, 1e-8)
    normalized = np.clip(np.asarray(score, dtype=np.float32) / scale, 0.0, 1.0)
    normalized[~valid] = 0.0
    return normalized, scale


def compute_partition_evidence(
    strand_map_rgb: np.ndarray,
    depth: np.ndarray,
    seg: np.ndarray,
    interior_margin: int = 5,
    orientation_weight: float = 0.75,
    reject_depth_planes: bool = False,
) -> dict[str, np.ndarray | float]:
    """仅从 strand/depth/seg 计算方向、深度与融合证据。"""

    seg = np.asarray(seg, dtype=bool)
    kernel_size = 2 * max(int(interior_margin), 0) + 1
    interior = (
        cv2.erode(seg.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8))
        > 0
        if interior_margin > 0
        else seg.copy()
    )
    axial, strand_valid = decode_axial_field(strand_map_rgb, seg)
    valid = interior & strand_valid & np.isfinite(depth) & (np.asarray(depth) > 0.05)
    orientation_raw = axial_direction_score(axial, valid)
    depth_raw = depth_discontinuity_score(depth, valid)
    if reject_depth_planes:
        from lib.recon_strategy.partition_evidence import depth_plane_residual
        depth_raw = depth_plane_residual(depth, valid)
    orientation, orientation_scale = robust_unit_score(orientation_raw, valid)
    depth_score, depth_scale = robust_unit_score(depth_raw, valid)
    weight = float(np.clip(orientation_weight, 0.0, 1.0))
    fused = weight * orientation + (1.0 - weight) * depth_score
    if reject_depth_planes:
        # Either reliable cue can define a boundary; do not require both.
        fused = np.maximum(orientation, depth_score)
    fused[~valid] = 0.0
    return {
        "axial": axial,
        "valid": valid,
        "interior": interior,
        "orientation_raw": orientation_raw,
        "depth_raw": depth_raw,
        "orientation": orientation,
        "depth": depth_score,
        "fused": fused.astype(np.float32),
        "orientation_scale": orientation_scale,
        "depth_scale": depth_scale,
    }


def clean_candidate_edge(
    fused: np.ndarray,
    valid: np.ndarray,
    quantile: float,
    min_component_pixels: int = 16,
) -> tuple[np.ndarray, float]:
    """按有效区分位数生成、闭合并去除短小候选边缘。"""

    if not 0.0 < quantile < 1.0:
        raise ValueError("边缘分位数必须位于 (0, 1)")
    values = fused[valid]
    threshold = float(np.quantile(values, quantile)) if len(values) else 1.0
    edge = (fused >= threshold) & valid & (fused > 0.0)
    edge = cv2.morphologyEx(
        edge.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(edge, connectivity=8)
    cleaned = np.zeros(edge.shape, dtype=bool)
    for label_id in range(1, count):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= int(min_component_pixels):
            cleaned |= labels == label_id
    return cleaned, threshold


def partition_from_barrier(
    seg: np.ndarray,
    edge: np.ndarray,
    barrier_radius: int = 1,
    min_region_pixels: int = 256,
) -> tuple[np.ndarray, dict[str, object]]:
    """从 edge 阻隔后的 seg 连通分量生成可解释分区。"""

    edge_u8 = np.asarray(edge, dtype=np.uint8)
    radius = max(int(barrier_radius), 0)
    if radius:
        size = 2 * radius + 1
        barrier = cv2.dilate(edge_u8, np.ones((size, size), np.uint8)) > 0
    else:
        barrier = edge_u8 > 0
    domain = np.asarray(seg, dtype=bool) & ~barrier
    raw_labels, count = ndimage.label(domain, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(raw_labels.ravel())
    candidates = [
        label_id for label_id in range(1, count + 1)
        if int(sizes[label_id]) >= int(min_region_pixels)
    ]
    if not candidates and count:
        candidates = [int(np.argmax(sizes[1:]) + 1)]

    labels = np.zeros(raw_labels.shape, dtype=np.int32)
    for new_id, old_id in enumerate(candidates, start=1):
        labels[raw_labels == old_id] = new_id
    small = domain & (labels == 0)
    if small.any() and candidates:
        nearest = ndimage.distance_transform_edt(
            labels == 0, return_distances=False, return_indices=True
        )
        labels[small] = labels[tuple(nearest[:, small])]
    region_sizes = [int(np.sum(labels == label_id)) for label_id in range(1, len(candidates) + 1)]
    return labels, {
        "raw_component_count": int(count),
        "effective_region_count": int(len(candidates)),
        "region_sizes": region_sizes,
        "barrier_pixels": int(barrier.sum()),
        "unassigned_seg_pixels": int(np.sum(np.asarray(seg, dtype=bool) & (labels == 0))),
    }


def select_topological_edge_components(
    seg: np.ndarray,
    edge: np.ndarray,
    barrier_radius: int = 5,
    min_region_pixels: int = 256,
) -> tuple[np.ndarray, dict[str, object]]:
    """只保留能够把头发 mask 切成至少两个大区域的边缘分量。"""

    count, components = cv2.connectedComponents(
        np.asarray(edge, dtype=np.uint8), connectivity=8
    )
    retained = np.zeros(np.asarray(edge).shape, dtype=bool)
    component_reports = []
    for component_id in range(1, count):
        component = components == component_id
        _, metrics = partition_from_barrier(
            seg,
            component,
            barrier_radius=barrier_radius,
            min_region_pixels=min_region_pixels,
        )
        region_sizes = sorted(metrics["region_sizes"], reverse=True)
        is_partitioning = (
            metrics["effective_region_count"] >= 2
            and len(region_sizes) >= 2
            and region_sizes[1] >= int(min_region_pixels)
        )
        if is_partitioning:
            retained |= component
        component_reports.append(
            {
                "component_id": int(component_id),
                "edge_pixels": int(component.sum()),
                "effective_region_count": int(metrics["effective_region_count"]),
                "region_sizes": [int(value) for value in region_sizes],
                "retained": bool(is_partitioning),
            }
        )
    return retained, {
        "input_component_count": int(max(count - 1, 0)),
        "retained_component_count": int(
            sum(item["retained"] for item in component_reports)
        ),
        "components": component_reports,
    }


def evaluate_against_reference(
    edge: np.ndarray,
    seg: np.ndarray,
    reference: np.ndarray,
    tolerance_px: int = 5,
) -> dict[str, float | int]:
    """检测完成后，用参考发缝邻域计算事后精度与召回率。"""

    size = 2 * max(int(tolerance_px), 0) + 1
    kernel = np.ones((size, size), np.uint8)
    edge_dilated = cv2.dilate(edge.astype(np.uint8), kernel) > 0
    reference = np.asarray(reference, dtype=bool) & np.asarray(seg, dtype=bool)
    reference_dilated = cv2.dilate(reference.astype(np.uint8), kernel) > 0
    edge_count = int(edge.sum())
    reference_count = int(reference.sum())
    return {
        "reference_pixels": reference_count,
        "edge_pixels": edge_count,
        "reference_recall_at_tolerance": float(
            np.sum(edge_dilated & reference) / max(reference_count, 1)
        ),
        "edge_precision_at_tolerance": float(
            np.sum(edge & reference_dilated) / max(edge_count, 1)
        ),
    }


def _heatmap(score: np.ndarray, raw_bgr: np.ndarray, title: str) -> np.ndarray:
    colored = cv2.applyColorMap(np.uint8(np.clip(score, 0.0, 1.0) * 255), cv2.COLORMAP_TURBO)
    panel = cv2.addWeighted(raw_bgr, 0.42, colored, 0.58, 0.0)
    cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _edge_overlay(raw_bgr: np.ndarray, edge: np.ndarray, title: str) -> np.ndarray:
    panel = raw_bgr.copy()
    panel[edge] = (0, 0, 255)
    cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _partition_overlay(raw_bgr: np.ndarray, labels: np.ndarray, title: str) -> np.ndarray:
    palette = np.asarray(
        [[0, 0, 0], [235, 90, 40], [40, 190, 245], [80, 220, 90], [210, 80, 220], [40, 220, 220]],
        dtype=np.uint8,
    )
    colors = palette[np.mod(labels, len(palette))]
    overlay = cv2.addWeighted(raw_bgr, 0.58, colors, 0.42, 0.0)
    overlay[labels == 0] = raw_bgr[labels == 0]
    cv2.putText(overlay, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def run_partition_audit(args: argparse.Namespace) -> dict[str, object]:
    """运行隔离审计并输出多阈值边缘、分区和事后评估。"""

    strand = imageio.imread(args.strand_map)
    depth = np.load(args.depth_map).astype(np.float32)
    seg_image = imageio.imread(args.seg)
    if seg_image.ndim == 3:
        seg_image = seg_image[:, :, 0]
    seg = seg_image > 127
    raw_bgr = cv2.imread(str(args.raw_image), cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise FileNotFoundError(f"缺少原图: {args.raw_image}")
    height, width = seg.shape
    if raw_bgr.shape[:2] != (height, width):
        raw_bgr = cv2.resize(raw_bgr, (width, height), interpolation=cv2.INTER_AREA)
    if strand.shape[:2] != seg.shape or depth.shape != seg.shape:
        raise ValueError("strand_map、depth_map 与 seg 必须具有相同分辨率")

    # 检测阶段到此为止，不读取 reference_parting_mask。
    evidence = compute_partition_evidence(
        strand,
        depth,
        seg,
        interior_margin=args.interior_margin,
        orientation_weight=args.orientation_weight,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "strand_depth_partition_evidence.npz",
        orientation=evidence["orientation"],
        depth=evidence["depth"],
        fused=evidence["fused"],
        valid=evidence["valid"],
        interior=evidence["interior"],
    )
    score_grid = np.concatenate(
        [
            _heatmap(evidence["orientation"], raw_bgr, "axial orientation discontinuity"),
            _heatmap(evidence["depth"], raw_bgr, "masked depth discontinuity"),
            _heatmap(evidence["fused"], raw_bgr, "fused evidence"),
        ],
        axis=1,
    )
    cv2.imwrite(str(args.output_dir / "score_components_on_raw_img.png"), score_grid)

    reference = None
    if args.reference_parting_mask and args.reference_parting_mask.exists():
        reference_image = imageio.imread(args.reference_parting_mask)
        if reference_image.ndim == 3:
            reference_image = reference_image[:, :, 0]
        if reference_image.shape != seg.shape:
            reference_image = cv2.resize(
                reference_image.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        reference = reference_image > 0

    threshold_reports: dict[str, object] = {}
    edge_panels = []
    partition_panels = []
    default_payload = None
    for quantile in args.quantiles:
        key = f"q{int(round(100 * quantile)):02d}"
        raw_edge, threshold = clean_candidate_edge(
            evidence["fused"],
            evidence["valid"],
            quantile,
            min_component_pixels=args.min_edge_pixels,
        )
        edge, topology_metrics = select_topological_edge_components(
            seg,
            raw_edge,
            barrier_radius=args.barrier_radius,
            min_region_pixels=args.min_region_pixels,
        )
        labels, partition_metrics = partition_from_barrier(
            seg,
            edge,
            barrier_radius=args.barrier_radius,
            min_region_pixels=args.min_region_pixels,
        )
        np.save(args.output_dir / f"{key}_partition_labels.npy", labels)
        imageio.imwrite(
            args.output_dir / f"{key}_candidate_edge.png",
            raw_edge.astype(np.uint8) * 255,
        )
        imageio.imwrite(
            args.output_dir / f"{key}_topological_edge.png",
            edge.astype(np.uint8) * 255,
        )
        edge_panels.append(
            _edge_overlay(
                raw_bgr,
                edge,
                f"{key} topology edge / threshold={threshold:.3f}",
            )
        )
        partition_panels.append(
            _partition_overlay(
                raw_bgr,
                labels,
                f"{key} partitions={partition_metrics['effective_region_count']}",
            )
        )
        report_item: dict[str, object] = {
            "quantile": float(quantile),
            "score_threshold": threshold,
            "candidate_edge_pixels": int(raw_edge.sum()),
            "topological_edge_pixels": int(edge.sum()),
            "topological_filter": topology_metrics,
            "partition": partition_metrics,
        }
        if reference is not None:
            report_item["reference_evaluation"] = {
                "raw_candidate": evaluate_against_reference(
                    raw_edge,
                    seg,
                    reference,
                    tolerance_px=args.reference_tolerance_px,
                ),
                "topological_edge": evaluate_against_reference(
                    edge,
                    seg,
                    reference,
                    tolerance_px=args.reference_tolerance_px,
                ),
            }
        threshold_reports[key] = report_item
        if abs(float(quantile) - float(args.default_quantile)) < 1e-6:
            default_payload = (key, edge, labels)

    cv2.imwrite(
        str(args.output_dir / "candidate_edges_on_raw_img.png"),
        np.concatenate(edge_panels, axis=1),
    )
    cv2.imwrite(
        str(args.output_dir / "candidate_partitions_on_raw_img.png"),
        np.concatenate(partition_panels, axis=1),
    )
    if default_payload is None:
        raise ValueError("default_quantile 必须包含在 quantiles 中")
    default_key, default_edge, default_labels = default_payload
    cv2.imwrite(
        str(args.output_dir / "default_edge_on_raw_img.png"),
        _edge_overlay(raw_bgr, default_edge, f"default {default_key} candidate edge"),
    )
    cv2.imwrite(
        str(args.output_dir / "default_partition_on_raw_img.png"),
        _partition_overlay(
            raw_bgr,
            default_labels,
            f"default {default_key} candidate partitions",
        ),
    )

    report = {
        "step": "strand_depth_partition_audit_2d",
        "purpose": "仅从 strand_map、depth_map 和 seg 检测内部突变并审计候选分区；不修改 PDE",
        "inputs": {
            "strand_map": str(args.strand_map.resolve()),
            "depth_map": str(args.depth_map.resolve()),
            "seg": str(args.seg.resolve()),
            "raw_image": str(args.raw_image.resolve()),
            "reference_parting_mask": (
                str(args.reference_parting_mask.resolve())
                if args.reference_parting_mask and args.reference_parting_mask.exists()
                else None
            ),
        },
        "detection_input_policy": "reference_parting_mask is loaded only after evidence computation",
        "parameters": {
            "interior_margin": int(args.interior_margin),
            "orientation_weight": float(args.orientation_weight),
            "quantiles": [float(value) for value in args.quantiles],
            "default_quantile": float(args.default_quantile),
            "min_edge_pixels": int(args.min_edge_pixels),
            "barrier_radius": int(args.barrier_radius),
            "min_region_pixels": int(args.min_region_pixels),
            "reference_tolerance_px": int(args.reference_tolerance_px),
        },
        "normalization": {
            "orientation_raw_q98": float(evidence["orientation_scale"]),
            "depth_raw_q98": float(evidence["depth_scale"]),
            "valid_pixels": int(np.sum(evidence["valid"])),
        },
        "thresholds": threshold_reports,
        "default_candidate": default_key,
        "human_review_status": "pending",
        "outputs": {
            "evidence": str((args.output_dir / "strand_depth_partition_evidence.npz").resolve()),
            "score_components": str((args.output_dir / "score_components_on_raw_img.png").resolve()),
            "edge_comparison": str((args.output_dir / "candidate_edges_on_raw_img.png").resolve()),
            "partition_comparison": str((args.output_dir / "candidate_partitions_on_raw_img.png").resolve()),
            "default_edge": str((args.output_dir / "default_edge_on_raw_img.png").resolve()),
            "default_partition": str((args.output_dir / "default_partition_on_raw_img.png").resolve()),
        },
    }
    report_path = args.output_dir / "strand_depth_partition_audit_report.json"
    report["outputs"]["report"] = str(report_path.resolve())
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """构造当前样例可直接运行、同时允许替换输入的命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--strand-map", type=Path, default=DEFAULT_DATA_ROOT / "maps/strand_map/front.png")
    parser.add_argument("--depth-map", type=Path, default=DEFAULT_DATA_ROOT / "maps/depth_map/front.npy")
    parser.add_argument("--seg", type=Path, default=DEFAULT_DATA_ROOT / "maps/seg/front.png")
    parser.add_argument("--raw-image", type=Path, default=DEFAULT_DATA_ROOT / "raw_img.png")
    parser.add_argument(
        "--reference-parting-mask",
        type=Path,
        default=DEFAULT_DATA_ROOT / "pde_governance/front_parting_dinov3_temp_rerun/parting_region_mask.png",
        help="仅用于检测完成后的事后评估",
    )
    parser.add_argument("--interior-margin", type=int, default=5)
    parser.add_argument("--orientation-weight", type=float, default=0.75)
    parser.add_argument("--quantiles", type=float, nargs="+", default=list(DEFAULT_QUANTILES))
    parser.add_argument("--default-quantile", type=float, default=0.96)
    parser.add_argument("--min-edge-pixels", type=int, default=16)
    parser.add_argument("--barrier-radius", type=int, default=5)
    parser.add_argument("--min-region-pixels", type=int, default=1024)
    parser.add_argument("--reference-tolerance-px", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "pde_governance/strand_depth_partition/step_01_edge_partition_audit",
    )
    return parser


def main() -> None:
    report = run_partition_audit(build_arg_parser().parse_args())
    print(
        json.dumps(
            {
                "default_candidate": report["default_candidate"],
                "thresholds": report["thresholds"],
                "human_review_status": report["human_review_status"],
                "outputs": report["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
