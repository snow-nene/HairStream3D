# Hair Generation Pipeline (Laplace PDE + RK4) Pitfalls and Solutions

This document records the major pitfalls encountered and resolved during the development of the 3D hair generation pipeline using Laplace PDE orientation fields and 2D-driven RK4 integration.

## 1. Out-of-Bounds Camera Projection in RK4
**Issue:** The RK4 integrator would crash or produce `nan` values because strands growing beyond the 2D camera viewport resulted in UV coordinates outside `[-1, 1]`. When querying 2D maps (e.g., divergence, clustering maps), this caused array out-of-bounds errors or undefined behavior.
**Solution:** Used `torch.nn.functional.grid_sample` with `padding_mode='border'` and `align_corners=True` to safely clamp UV queries. Also added safety clamps to pixel coordinate conversions:
```python
px = np.clip(((uv[0] + 1.0) / 2.0 * W).astype(int), 0, W - 1)
```

## 2. Hair Tip Oscillations (180° Loops) due to Unscaled Noise
**Issue:** Generated strands looked like "noodles" with chaotic 180-degree loops near the hair tips.
**Cause:** The Laplace PDE gradient magnitude drops to near zero at the hair tips (boundary condition sink). However, the divergence noise force was applied with a constant or strictly time-dependent weight. When the natural PDE growth force died out, the pure noise force took over, sending the strands spinning chaotically.
**Solution:** The noise vector must be scaled by the magnitude of the PDE vector (`k_mag`). As the underlying PDE field fades out at the tip, the noise field fades with it proportionally, resulting in a natural stop without oscillation.

## 3. Unnatural Tip Bending due to Constant Clustering Force
**Issue:** Strands exhibited sharp 45°-90° bends at their very tips, aggressively snapping toward guide strands.
**Cause:** Similar to the noise issue, the clustering pull (lerp toward `C_guide`) had a base weight of `0.005` that did not account for the strand's current growth momentum. When the PDE force hit zero at the tip, the clustering force became the sole driving factor, dragging the stalled tip sideways and increasing segment length artificially.
**Solution:** The clustering `total_clump` weight was multiplied by `k_mag / 0.9` (clamped to `[0, 1]`). This ensures that clustering only applies while the hair is actively growing. When growth stalls at the tip, the clustering pull also smoothly decays to zero.

## 4. Sharp Kinks at Hair Roots (Discretization Error)
**Issue:** Strands had sharp, visible "kinks" (10°-15° abrupt angles) within the first 2-3 steps of growth from the scalp.
**Cause:** The Laplace orientation field was solved on a relatively coarse `64x64x64` voxel grid (approx. 1.5cm per voxel). At the scalp boundary, the electric field (vector field) has high curvature. RK4 steps crossing voxel boundaries experienced drastic vector shifts under this coarse resolution.
**Solution:** Increased the PDE and collision SDF volumetric resolution to `128x128x128` (approx. 3-4mm per voxel). The much denser grid provides a naturally smooth vector field transition, eliminating root kinks without needing post-process Gaussian smoothing.

## 5. Collision Normal Skew from Anisotropic Voxels
**Issue:** Collision response (pushing strands out of the head SDF) sometimes pushed hair in slightly incorrect/skewed directions.
**Cause:** The bounding box for the head is `X: [-0.3, 0.3], Y: [1.0, 2.0], Z: [-0.3, 0.3]`, making it a non-cube (0.6 x 1.0 x 0.6). Using `np.gradient(sdf, axis=...)` assumes unit spacing across all axes, which severely distorted the resulting normal vectors because `dx != dy`.
**Solution:** Always compute explicit voxel spacing `dx, dy, dz` and pass it to gradient calculations:
```python
dx = (b_max[0] - b_min[0]) / (R - 1)
dy = (b_max[1] - b_min[1]) / (R - 1)
dz = (b_max[2] - b_min[2]) / (R - 1)
grad_x = np.gradient(sdf, dx, axis=0)
grad_y = np.gradient(sdf, dy, axis=1)
grad_z = np.gradient(sdf, dz, axis=2)
```
