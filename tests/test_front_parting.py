import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.vis.detect_front_parting import detect_parting


def _synthetic_parting_image(size=192):
    image = np.full((size, size, 3), (205, 176, 165), dtype=np.uint8)
    yy, xx = np.mgrid[:size, :size]
    outer = ((xx - 96) / 75) ** 2 + ((yy - 78) / 65) ** 2 < 1.0
    face = ((xx - 96) / 39) ** 2 + ((yy - 105) / 48) ** 2 < 1.0
    seg = outer & ~face & (yy < 133)
    image[seg] = (132, 97, 72)

    expected_center = np.full(size, np.nan, dtype=np.float32)
    expected_width = np.zeros(size, dtype=np.int32)
    for y in range(20, 53):
        center = int(round(104 - 0.09 * (y - 20) + 4 * np.sin((y - 20) / 10)))
        radius = max(2, int(round(7 - 0.12 * (y - 20))))
        lo, hi = center - radius, center + radius
        image[y, lo:hi + 1] = (178, 157, 150)
        expected_center[y] = center
        expected_width[y] = hi - lo + 1

    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image, seg, expected_center, expected_width


def test_scalp_region_is_curved_and_adaptive_without_dinov3():
    image, seg, expected_center, _ = _synthetic_parting_image()
    mask, response = detect_parting(
        image, seg, mode="scalp_region", use_dinov3=False
    )
    rows = np.flatnonzero(mask.any(axis=1))
    assert rows.size >= 18
    predicted_center = np.asarray(
        [np.median(np.flatnonzero(mask[row])) for row in rows], dtype=np.float32
    )
    valid = np.isfinite(expected_center[rows])
    assert valid.sum() >= 15
    assert np.median(np.abs(predicted_center[valid] - expected_center[rows][valid])) < 5
    widths = np.asarray([mask[row].sum() for row in rows])
    assert widths.max() - widths.min() >= 3
    assert np.median(widths) > 5
    assert response.shape == seg.shape


def test_legacy_line_mode_remains_available():
    image, seg, _, _ = _synthetic_parting_image()
    mask, response = detect_parting(
        image,
        seg,
        width=2,
        center_x=0.54,
        mode="legacy_line",
        use_dinov3=False,
    )
    assert mask.any()
    assert mask.shape == seg.shape
    assert response.shape == seg.shape


if __name__ == "__main__":
    test_scalp_region_is_curved_and_adaptive_without_dinov3()
    test_legacy_line_mode_remains_available()
    print("test_front_parting: 2 passed")
