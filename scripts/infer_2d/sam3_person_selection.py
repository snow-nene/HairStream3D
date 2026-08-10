"""SAM3 多人场景中的主人物与头发实例筛选工具。"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def deduplicate_masks(masks, iou_threshold=0.90, area_ratio_threshold=1.25):
    """去除跨文本提示产生的近重复 mask，不合并不同人物。

    只有 IoU 很高且面积接近的 mask 才被视为重复。刻意不使用包含率，避免
    把一个覆盖多人的大 mask 与其中的单个人物误判为同一实例。
    """
    unique = []
    for mask in masks:
        candidate = np.asarray(mask, dtype=bool)
        candidate_area = int(candidate.sum())
        if candidate_area == 0:
            continue

        duplicate_idx = None
        for i, existing in enumerate(unique):
            existing_area = int(existing.sum())
            intersection = int(np.logical_and(candidate, existing).sum())
            union = candidate_area + existing_area - intersection
            iou = intersection / max(union, 1)
            area_ratio = max(candidate_area, existing_area) / max(
                min(candidate_area, existing_area), 1
            )
            if iou >= iou_threshold and area_ratio <= area_ratio_threshold:
                duplicate_idx = i
                break

        if duplicate_idx is None:
            unique.append(candidate.copy())
        elif candidate_area > int(unique[duplicate_idx].sum()):
            unique[duplicate_idx] = candidate.copy()
    return unique


def _mask_center_score(mask):
    """返回中心先验与中心区域覆盖率，适配默认的正面单人输入。"""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0

    target_x = 0.5 * (w - 1)
    target_y = 0.45 * (h - 1)
    dx = (float(xs.mean()) - target_x) / max(0.5 * w, 1.0)
    dy = (float(ys.mean()) - target_y) / max(0.5 * h, 1.0)
    center_score = float(np.exp(-1.5 * (dx * dx + dy * dy)))

    yy, xx = np.ogrid[:h, :w]
    center_region = (
        ((xx - target_x) / max(0.18 * w, 1.0)) ** 2
        + ((yy - target_y) / max(0.24 * h, 1.0)) ** 2
        <= 1.0
    )
    center_coverage = float(mask[center_region].mean())
    return center_score, center_coverage


def _border_occupancy(mask):
    h, w = mask.shape
    border = np.concatenate(
        [mask[0], mask[-1], mask[1:-1, 0], mask[1:-1, -1]]
    )
    return float(border.mean()) if border.size else 0.0


def score_person_instances(person_instances, hair_union=None):
    """为人物实例评分；中心、主要头发占比和合理轮廓共同决定主角。"""
    if not person_instances:
        return []

    shape = np.asarray(person_instances[0]).shape
    image_area = float(shape[0] * shape[1])
    if hair_union is None:
        hair = np.zeros(shape, dtype=bool)
    else:
        hair = np.asarray(hair_union, dtype=bool)
    hair_area = int(hair.sum())

    scores = []
    for mask in person_instances:
        person = np.asarray(mask, dtype=bool)
        area_ratio = float(person.sum()) / max(image_area, 1.0)
        center_score, center_coverage = _mask_center_score(person)
        hair_share = (
            float(np.logical_and(person, hair).sum()) / hair_area
            if hair_area > 0
            else 0.0
        )
        border_penalty = _border_occupancy(person)
        oversized_penalty = max(0.0, (area_ratio - 0.78) / 0.22)

        score = (
            0.30 * center_score
            + 0.20 * center_coverage
            + 0.20 * np.sqrt(min(area_ratio, 1.0))
            + 0.40 * hair_share
            - 0.45 * border_penalty
            - 0.45 * oversized_penalty
        )
        scores.append(float(score))
    return scores


def select_primary_person(person_instances, hair_union=None):
    """选择最可能的主人物，返回实例下标。"""
    if not person_instances:
        raise ValueError("person_instances 不能为空")
    scores = score_person_instances(person_instances, hair_union)
    return int(np.argmax(scores))


def select_primary_hair(hair_instances):
    """人物检测失败时，仅保留最居中且面积较大的头发实例。"""
    if not hair_instances:
        raise ValueError("hair_instances 不能为空")
    scores = []
    image_area = float(np.asarray(hair_instances[0]).size)
    for mask in hair_instances:
        hair = np.asarray(mask, dtype=bool)
        center_score, center_coverage = _mask_center_score(hair)
        area_score = np.sqrt(float(hair.sum()) / max(image_area, 1.0))
        scores.append(
            0.55 * center_score + 0.25 * center_coverage + 0.20 * area_score
        )
    return int(np.argmax(scores))


def assign_hair_to_groups(hair_instances, group_masks, max_distance_ratio=0.12):
    """按整体重叠与稳健距离分配头发，无法可靠归属时返回 -1。

    距离使用头发像素的第 10 百分位，而非最小值，避免单个擦边像素把整个
    头发实例分配给错误人物。
    """
    if not group_masks:
        return [-1] * len(hair_instances)

    groups = [np.asarray(group, dtype=bool) for group in group_masks]
    group_areas = [int(group.sum()) for group in groups]
    dist_maps = [distance_transform_edt(~group) for group in groups]
    h, w = groups[0].shape
    max_distance = max_distance_ratio * float(np.hypot(h, w))
    assignments = []

    for mask in hair_instances:
        hair = np.asarray(mask, dtype=bool)
        hair_area = int(hair.sum())
        if hair_area == 0:
            assignments.append(-1)
            continue

        overlaps = [int(np.logical_and(hair, group).sum()) for group in groups]
        if max(overlaps, default=0) > 0:
            best = max(
                range(len(groups)),
                key=lambda i: (
                    overlaps[i] / hair_area,
                    overlaps[i] / max(group_areas[i], 1),
                    -float(np.percentile(dist_maps[i][hair], 10)),
                ),
            )
            assignments.append(best)
            continue

        distances = [float(np.percentile(dm[hair], 10)) for dm in dist_maps]
        best = int(np.argmin(distances))
        assignments.append(best if distances[best] <= max_distance else -1)
    return assignments
