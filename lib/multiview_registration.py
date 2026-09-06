"""Camera/mesh registration gates for visible 3-D observations."""
from __future__ import annotations

import numpy as np


def validate_camera(camera, atol=1e-6):
    camera = np.asarray(camera, dtype=float)
    if camera.shape != (4, 4) or not np.isfinite(camera).all():
        raise ValueError("camera must be a finite 4x4 matrix")
    if np.linalg.matrix_rank(camera) < 4:
        raise ValueError("camera must be invertible")
    return camera


def project_world(points_world, camera):
    camera = validate_camera(camera)
    points = np.asarray(points_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_world must be finite N-by-3")
    clip = np.c_[points, np.ones(len(points))] @ camera.T
    if np.any(np.abs(clip[:, 3]) < 1e-12):
        raise ValueError("world point projects to an invalid homogeneous depth")
    return clip[:, :2] / clip[:, 3:4]


def registration_report(points_world, pixel_uv, camera, image_shape,
                       max_error_px=1.0, min_inlier_fraction=0.95):
    """Measure reprojection, image bounds, and axis/scale consistency.

    ``pixel_uv`` uses image pixel coordinates (x, y). The gate is deliberately
    independent of any learned depth calibration: it tests only the stored
    camera and visible mesh intersections.
    """
    points = np.asarray(points_world, dtype=float)
    pixels = np.asarray(pixel_uv, dtype=float)
    height, width = tuple(image_shape)
    if pixels.shape != (len(points), 2) or height < 2 or width < 2:
        raise ValueError("pixel/world correspondence shape mismatch")
    expected = project_world(points, camera)
    error = np.linalg.norm(expected - pixels, axis=1)
    finite = np.isfinite(error)
    in_image = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    valid = finite & in_image
    fraction = float(valid.mean()) if len(valid) else 0.0
    q50 = float(np.quantile(error[finite], .5)) if finite.any() else float("inf")
    q95 = float(np.quantile(error[finite], .95)) if finite.any() else float("inf")
    passed = bool(len(points) > 0 and fraction >= min_inlier_fraction
                  and q95 <= max_error_px)
    return {
        "points": int(len(points)), "in_image_fraction": fraction,
        "reprojection_median_px": q50, "reprojection_q95_px": q95,
        "max_error_px": float(max_error_px),
        "min_inlier_fraction": float(min_inlier_fraction), "passed": passed,
        "axis_order": "image_xy; world_xyz; camera_homogeneous",
    }


def gate_registration(points_world, pixel_uv, camera, image_shape,
                      max_error_px=1.0, min_inlier_fraction=.95):
    report = registration_report(points_world, pixel_uv, camera, image_shape,
                                 max_error_px, min_inlier_fraction)
    if not report["passed"]:
        raise ValueError("camera/mesh registration gate failed: " + str(report))
    return report
