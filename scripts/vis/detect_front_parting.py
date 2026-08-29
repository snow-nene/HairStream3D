"""从 front 原始图像提取可见头皮发缝区域并输出诊断。

默认模式将发缝建模为宽度随位置变化的二维可见头皮区域。DINOv3 只提供
低分辨率区域一致性，最终像素边界由原图 Lab 色度和亮脊两侧的强度下降确定。
旧版固定宽度暗谷中心线保留为 ``legacy_line`` 回退模式。
"""
import argparse
import os
import warnings

import cv2
import imageio.v2 as imageio
import numpy as np


DINO_MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"


def _normalize01(values, valid):
    picked = np.asarray(values)[valid]
    if picked.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(picked, [5, 95])
    return np.clip(
        (values - low) / max(float(high - low), 1e-6), 0.0, 1.0
    ).astype(np.float32)


def _solve_smooth_path(score, valid, y0, y1, x0, x1, max_step=3):
    """在二维响应图中寻找一条允许弯曲、但不逐像素跳变的纵向路径。"""
    cost = -score[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    cost += (~valid[y0:y1 + 1, x0:x1 + 1]).astype(np.float32) * 5.0
    dp = np.full_like(cost, np.inf)
    previous = np.zeros_like(cost, dtype=np.int16)
    dp[0] = cost[0]
    for row in range(1, len(cost)):
        for column in range(cost.shape[1]):
            lo = max(0, column - max_step)
            hi = min(cost.shape[1], column + max_step + 1)
            offsets = np.arange(lo, hi) - column
            choices = dp[row - 1, lo:hi] + 0.035 * offsets * offsets
            best = lo + int(np.argmin(choices))
            dp[row, column] = cost[row, column] + choices[best - lo]
            previous[row, column] = best

    column = int(np.argmin(dp[-1]))
    path_x = np.empty(len(cost), dtype=np.float32)
    for row in range(len(cost) - 1, -1, -1):
        path_x[row] = x0 + column
        if row:
            column = int(previous[row, column])
    return cv2.GaussianBlur(path_x[:, None], (1, 9), 0).ravel()


def _sample_feature_grid(features, xs, ys, scale_x, scale_y):
    ix = np.clip((xs / scale_x).astype(np.int32), 0, features.shape[1] - 1)
    iy = np.clip((ys / scale_y).astype(np.int32), 0, features.shape[0] - 1)
    return features[iy, ix]


def _extract_dinov3_response(
    image,
    initial_path_x,
    path_y,
    roi,
    model_name=DINO_MODEL_NAME,
    feature_size=768,
):
    """用路径与左右发束构造 DINOv3 原型差异响应。"""
    import timm
    import torch

    height, width = roi.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=True).eval().to(device)
    resized = cv2.resize(
        image, (feature_size, feature_size), interpolation=cv2.INTER_CUBIC
    )
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    mean = torch.tensor(model.pretrained_cfg["mean"])
    std = torch.tensor(model.pretrained_cfg["std"])
    tensor = ((tensor - mean) / std).permute(2, 0, 1)[None].to(device)
    with torch.inference_mode():
        tokens = model.forward_features(tensor)[:, model.num_prefix_tokens:]
        tokens = torch.nn.functional.normalize(tokens.float(), dim=-1)
    grid = int(round(np.sqrt(tokens.shape[1])))
    if grid * grid != tokens.shape[1]:
        raise RuntimeError(f"DINOv3 patch token 数量无法还原为方形网格: {tokens.shape[1]}")
    features = tokens[0].cpu().numpy().reshape(grid, grid, -1)
    scale_x = width / float(grid)
    scale_y = height / float(grid)

    positive = _sample_feature_grid(
        features, initial_path_x, path_y, scale_x, scale_y
    ).mean(axis=0)
    negative = np.concatenate(
        [
            _sample_feature_grid(
                features, initial_path_x - 24, path_y, scale_x, scale_y
            ),
            _sample_feature_grid(
                features, initial_path_x + 24, path_y, scale_x, scale_y
            ),
        ],
        axis=0,
    ).mean(axis=0)
    positive /= np.linalg.norm(positive) + 1e-8
    negative /= np.linalg.norm(negative) + 1e-8
    patch_score = features @ positive - features @ negative
    response = cv2.resize(
        patch_score, (width, height), interpolation=cv2.INTER_CUBIC
    )
    return _normalize01(response, roi)


def _rolling_median(values, window=9):
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.asarray(
        [np.median(padded[index:index + window]) for index in range(len(values))],
        dtype=np.float32,
    )


def _smooth_region_boundaries(region, seg, roi, x0, x1):
    """纵向平滑左右边界，去掉单根高光导致的方块凸起。"""
    active = np.flatnonzero(region.any(axis=1))
    if active.size == 0:
        return region
    rows = np.arange(active.min(), active.max() + 1)
    left = np.full(len(rows), np.nan, dtype=np.float32)
    right = np.full(len(rows), np.nan, dtype=np.float32)
    for index, row_index in enumerate(rows):
        columns = np.flatnonzero(region[row_index])
        if columns.size:
            left[index], right[index] = columns.min(), columns.max()
    known = np.flatnonzero(np.isfinite(left))
    if known.size == 0:
        return region
    left = np.interp(np.arange(len(rows)), known, left[known])
    right = np.interp(np.arange(len(rows)), known, right[known])
    center = _rolling_median((left + right) / 2.0)
    half_width = np.clip(_rolling_median((right - left + 1.0) / 2.0), 2.0, 10.0)
    kernel = np.ones(5, dtype=np.float32) / 5.0
    center = np.convolve(np.pad(center, (2, 2), mode="edge"), kernel, mode="valid")
    half_width = np.convolve(
        np.pad(half_width, (2, 2), mode="edge"), kernel, mode="valid"
    )

    smoothed = np.zeros_like(region)
    for row_index, center_x, radius in zip(rows, center, half_width):
        lo = max(x0, int(round(center_x - radius)))
        hi = min(x1, int(round(center_x + radius)))
        smoothed[row_index, lo:hi + 1] = True
    return smoothed & seg & roi


def _detect_scalp_region(
    image,
    seg,
    use_dinov3=True,
    dino_model_name=DINO_MODEL_NAME,
    dino_feature_size=768,
):
    """检测弯曲且宽度自适应的可见头皮发缝区域。"""
    height, width = seg.shape
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.2)
    # 4 px 邻域能稳定捕捉真实发缝的局部亮脊；更大的尺度容易把相邻
    # 金色发束高光误当成头皮，并让可见区域向后脑勺过度延伸。
    flank = (
        np.roll(smooth, 4, axis=1)
        + np.roll(smooth, -4, axis=1)
    ) / 2.0
    bright_ridge = np.maximum(smooth - flank, 0.0)

    hair_y, hair_x = np.where(seg)
    if hair_x.size == 0:
        return np.zeros_like(seg, dtype=bool), np.zeros_like(gray)
    y0 = max(0, int(np.percentile(hair_y, 2)))
    y1 = min(height - 1, int(np.percentile(hair_y, 35)))
    x0 = max(0, int(np.percentile(hair_x, 35)))
    x1 = min(width - 1, int(np.percentile(hair_x, 72)))
    if y1 <= y0 or x1 <= x0:
        return np.zeros_like(seg, dtype=bool), np.zeros_like(gray)
    roi = np.zeros_like(seg, dtype=bool)
    roi[y0:y1 + 1, x0:x1 + 1] = True
    roi &= seg

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    skin_patch = lab[
        int(0.38 * height):int(0.48 * height),
        int(0.42 * width):int(0.58 * width),
        1:3,
    ]
    if skin_patch.size:
        skin_ab = np.median(skin_patch.reshape(-1, 2), axis=0)
        chroma_distance = np.linalg.norm(lab[:, :, 1:3] - skin_ab, axis=-1)
        skin_chroma = np.exp(-(chroma_distance ** 2) / (2.0 * 12.0 ** 2))
    else:
        skin_chroma = np.ones_like(gray, dtype=np.float32)

    initial_score = 0.72 * _normalize01(bright_ridge, roi) + 0.28 * skin_chroma
    path_y = np.arange(y0, y1 + 1)
    initial_path_x = _solve_smooth_path(
        initial_score, seg, y0, y1, x0, x1
    )
    dino_response = np.zeros_like(gray, dtype=np.float32)
    if use_dinov3:
        try:
            dino_response = _extract_dinov3_response(
                image,
                initial_path_x,
                path_y,
                roi,
                model_name=dino_model_name,
                feature_size=dino_feature_size,
            )
        except Exception as error:
            warnings.warn(
                f"DINOv3 发缝区域响应不可用，回退 RGB/Lab: {error}",
                RuntimeWarning,
            )

    refined_score = (
        0.58 * _normalize01(bright_ridge, roi)
        + 0.24 * skin_chroma
        + 0.18 * dino_response
    )
    path_x = _solve_smooth_path(refined_score, seg, y0, y1, x0, x1)

    region = np.zeros_like(seg, dtype=bool)
    strengths = []
    for row_index, x_float in zip(path_y, path_x):
        x = int(round(float(x_float)))
        peak = float(smooth[row_index, x])
        drop = max(9.0, 0.10 * peak)
        left = x
        while (
            left > x - 16
            and left > x0
            and smooth[row_index, left - 1] >= peak - drop
            and skin_chroma[row_index, left - 1] > 0.32
        ):
            left -= 1
        right = x
        while (
            right < x + 16
            and right < x1
            and smooth[row_index, right + 1] >= peak - drop
            and skin_chroma[row_index, right + 1] > 0.32
        ):
            right += 1
        strengths.append(float(bright_ridge[row_index, x]))
        region[row_index, left:right + 1] = True

    strengths = np.asarray(strengths, dtype=np.float32)
    reliable = strengths > max(2.0, float(np.percentile(strengths, 18)))
    region[path_y[~reliable]] = False
    region = cv2.morphologyEx(
        region.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((7, 3), dtype=np.uint8),
    ) > 0
    region &= seg & roi
    region = _smooth_region_boundaries(region, seg, roi, x0, x1)
    return region, refined_score


def _detect_legacy_parting(
    image,
    seg,
    width=2,
    center_x=0.56,
    smooth_window=9,
    dark_threshold=80,
    strand_map=None,
):
    """旧版固定宽度暗谷中心线，仅用于回退和历史复现。"""
    h, w = seg.shape
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = np.zeros((h, w), np.float32)
    for channel in range(3):
        gx = cv2.Sobel(lab[:, :, channel], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab[:, :, channel], cv2.CV_32F, 0, 1, ksize=3)
        grad += gx * gx + gy * gy
    grad = np.sqrt(grad)
    orientation_jump = np.zeros((h, w), np.float32)
    if strand_map is not None:
        strand = np.asarray(strand_map).astype(np.float32)
        if strand.max() > 1.5:
            strand /= 255.0
        for channel in (2, 1):
            gx = cv2.Sobel(strand[:, :, channel], cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(strand[:, :, channel], cv2.CV_32F, 0, 1, ksize=3)
            orientation_jump += gx * gx + gy * gy
        orientation_jump = np.sqrt(orientation_jump)

    yy, xx = np.where(seg)
    if len(xx) == 0:
        return np.zeros_like(seg, bool), np.zeros((h, w), np.float32)
    y0 = max(0, int(np.percentile(yy, 2)))
    y1 = min(h - 1, int(np.percentile(yy, 35)))
    center = float(center_x) * (w - 1)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    dark = ((gray_u8 < int(dark_threshold)) & seg).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    candidates = []
    for label in range(1, count):
        _, y, component_width, component_height, area = stats[label]
        if component_height < 35 or area < 80 or component_width > 28:
            continue
        center_distance = abs(float(centroids[label][0]) - center) / max(w, 1)
        if center_distance < 0.13 and y < int(0.35 * h):
            candidates.append((center_distance, -component_height, -area, label))
    if candidates:
        label = sorted(candidates)[0][3]
        component = labels == label
        component = cv2.dilate(
            component.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=max(1, int(np.ceil(width / 2))),
        ) > 0
        return component & seg, grad

    band = max(18.0, 0.09 * w)
    xlo, xhi = max(0, int(center - band)), min(w - 1, int(center + band))
    cost = np.full((y1 - y0 + 1, xhi - xlo + 1), 1e6, np.float32)
    for row, y in enumerate(range(y0, y1 + 1)):
        valid = seg[y, xlo:xhi + 1]
        darkness = 1.0 - gray[y, xlo:xhi + 1] / 255.0
        transition = grad[y, xlo:xhi + 1]
        x_coords = np.arange(xlo, xhi + 1, dtype=np.float32)
        expected_x = center - 0.10 * float(y - y0)
        center_penalty = 2.0 * ((x_coords - expected_x) / band) ** 2
        jump = orientation_jump[y, xlo:xhi + 1]
        cost[row] = (
            1.5 * (1.0 - darkness)
            - 0.006 * transition
            - 0.10 * jump
            + center_penalty
        )
        cost[row, ~valid] += 0.5

    dp = np.full_like(cost, np.inf)
    previous = np.zeros_like(cost, np.int16)
    dp[0] = cost[0]
    for row in range(1, len(cost)):
        for column in range(cost.shape[1]):
            lo, hi = max(0, column - 2), min(cost.shape[1], column + 3)
            best = lo + int(np.argmin(dp[row - 1, lo:hi]))
            dp[row, column] = cost[row, column] + dp[row - 1, best]
            previous[row, column] = best
    column = int(np.argmin(dp[-1]))
    path = []
    for row in range(len(cost) - 1, -1, -1):
        path.append((y0 + row, xlo + column))
        column = int(previous[row, column]) if row else column

    path_by_y = {y: x for y, x in path}
    ys = np.array(sorted(path_by_y), dtype=np.int32)
    xs = np.array([path_by_y[y] for y in ys], dtype=np.float32)
    if len(xs) >= 3:
        window = max(3, int(smooth_window) | 1)
        kernel = np.ones(window, dtype=np.float32) / float(window)
        pad = window // 2
        xs = np.convolve(np.pad(xs, (pad, pad), mode="edge"), kernel, mode="valid")
    mask = np.zeros((h, w), bool)
    for y, x_float in zip(ys, xs):
        x = int(round(float(x_float)))
        xa, xb = max(0, x - width), min(w, x + width + 1)
        mask[y, xa:xb] |= seg[y, xa:xb]
    return mask, grad


def detect_parting(
    image,
    seg,
    width=2,
    center_x=0.56,
    smooth_window=9,
    dark_threshold=80,
    strand_map=None,
    mode="scalp_region",
    use_dinov3=True,
    dino_model_name=DINO_MODEL_NAME,
    dino_feature_size=768,
):
    """检测发缝；默认返回二维可见头皮区域，旧版模式返回固定宽中心线。"""
    image = np.asarray(image)[..., :3]
    seg = np.asarray(seg).astype(bool)
    if mode == "legacy_line":
        return _detect_legacy_parting(
            image,
            seg,
            width,
            center_x,
            smooth_window,
            dark_threshold,
            strand_map,
        )
    if mode != "scalp_region":
        raise ValueError(f"未知发缝检测模式: {mode}")
    mask, response = _detect_scalp_region(
        image,
        seg,
        use_dinov3=use_dinov3,
        dino_model_name=dino_model_name,
        dino_feature_size=dino_feature_size,
    )
    if mask.any():
        return mask, response
    warnings.warn("自适应发缝区域为空，回退 legacy_line", RuntimeWarning)
    return _detect_legacy_parting(
        image,
        seg,
        width,
        center_x,
        smooth_window,
        dark_threshold,
        strand_map,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--seg", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--width", type=int, default=2, help="legacy_line 半宽")
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--dark_threshold", type=int, default=80)
    parser.add_argument("--strand_map", default=None)
    parser.add_argument("--center_x", type=float, default=0.56, help="legacy_line 中心先验")
    parser.add_argument(
        "--mode", choices=["scalp_region", "legacy_line"], default="scalp_region"
    )
    parser.add_argument("--no_dinov3", action="store_true")
    parser.add_argument("--dino_model_name", default=DINO_MODEL_NAME)
    parser.add_argument("--dino_feature_size", type=int, default=768)
    parser.add_argument("--centerline_output", default=None)
    args = parser.parse_args()

    image = imageio.imread(args.image)[..., :3]
    seg = imageio.imread(args.seg)
    if seg.ndim == 3:
        seg = seg[:, :, 0]
    strand_map = imageio.imread(args.strand_map) if args.strand_map else None
    mask, response = detect_parting(
        image,
        seg > 127,
        args.width,
        args.center_x,
        args.smooth_window,
        args.dark_threshold,
        strand_map,
        mode=args.mode,
        use_dinov3=not args.no_dinov3,
        dino_model_name=args.dino_model_name,
        dino_feature_size=args.dino_feature_size,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.overlay)), exist_ok=True)
    imageio.imwrite(args.output, mask.astype(np.uint8) * 255)

    overlay = image.astype(np.float32).copy()
    overlay[mask] = 0.50 * overlay[mask] + 0.50 * np.array([40, 90, 255])
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 235, 20), 1)
    centerline = np.zeros_like(mask, dtype=np.uint8)
    active_rows = np.flatnonzero(mask.any(axis=1))
    for row in active_rows:
        columns = np.flatnonzero(mask[row])
        centerline[row, int(round(float(np.median(columns))))] = 255
    overlay[centerline > 0] = (30, 255, 40)
    imageio.imwrite(args.overlay, overlay)
    if args.centerline_output:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.centerline_output)), exist_ok=True
        )
        imageio.imwrite(args.centerline_output, centerline)

    widths = np.asarray([int(mask[row].sum()) for row in active_rows])
    print(f"parting pixels: {int(mask.sum())}")
    if widths.size:
        print(
            "parting rows: "
            f"{int(active_rows.min())}..{int(active_rows.max())}, "
            f"width median/p90/max={np.median(widths):.1f}/"
            f"{np.percentile(widths, 90):.1f}/{int(widths.max())}"
        )
    print(
        f"mode={args.mode}, dinov3={not args.no_dinov3}, "
        f"response_max={float(np.max(response)):.4f}"
    )


if __name__ == "__main__":
    main()
