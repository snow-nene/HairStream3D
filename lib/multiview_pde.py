"""
Multi-View Laplace PDE Strategy.

Extends LaplacePDEStrategy to accept pre-fused multi-view boundary conditions.
The fusion is done externally by lib/multiview_fusion.py, and this strategy
uses the fused 3D orientation volume as the Dirichlet boundary for the PDE.

Flow:
  1. multiview_fusion.fuse_multiview_orientation() → fused 3D orien_vol + boundary_mask
  2. MultiViewLaplacePDEStrategy.filter() → PDE solve on fused boundary
  3. strategy.query_orien() → trilinear interpolation for RK4 strand tracing
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
import taichi as ti

from .recon_strategy.laplace_pde import LaplacePDEStrategy


class MultiViewLaplacePDEStrategy(LaplacePDEStrategy):
    """Multi-view variant that uses pre-fused 3D orientation from 4 cameras.

    Instead of projecting 3D voxels to a single 2D strand_map + depth_map,
    this strategy takes a pre-computed fused orientation volume and boundary
    mask, then runs the same Laplace CG solver to fill unseen interior voxels.
    """

    def __init__(self, opt, cuda: torch.device):
        super().__init__(opt, cuda)
        self._fused_orien_vol = None   # (3, R, R, R) from fusion
        self._fused_boundary = None    # (R, R, R) bool boundary mask
        self._is_multiview = False

    def set_fused_data(self, orien_vol: np.ndarray, boundary_mask: np.ndarray,
                        hair_volume: np.ndarray = None,
                        view_ownership: np.ndarray = None):
        R = self.resolution
        assert orien_vol.shape == (3, R, R, R)
        self._fused_orien_vol = orien_vol.copy()
        self._fused_boundary = boundary_mask.copy()
        self._fused_hair_volume = (hair_volume.copy() if hair_volume is not None else None)
        self._fused_view_owner = (view_ownership.copy() if view_ownership is not None else None)
        self._is_multiview = True

    def filter(self, data: dict, mesh_path: str = None) -> None:
        """Build the 3D orientation field from fused multi-view data.

        data dict is still passed for compatibility but only used for:
          - 'hairstep': [4, H, W] → depth_map for occupancy volume
          - 'calib':    [4, 4]   → for occupancy projection

        The orientation field comes from the pre-fused multi-view data,
        NOT from the single-view strand_map in data.
        """
        if not self._is_multiview:
            # Fallback to single-view parent behavior
            return super().filter(data, mesh_path=mesh_path)

        print('[MultiViewPDE] Solving directly from fused mesh-surface boundary...')
        self._orien_vol = self._solve_pde_on_fused_boundary(
            self._fused_orien_vol,
            self._fused_boundary,
            mesh_path=mesh_path,
        )
        self._occ_vol = (
            self._fused_hair_volume.astype(np.float32)
            if self._fused_hair_volume is not None
            else self._fused_boundary.astype(np.float32)
        )
        print('[MultiViewPDE] Field build complete.')

    def _apply_side_residual(self, front_field: np.ndarray) -> np.ndarray:
        """Apply a bounded local correction from side views to a front field.

        Side orientations are signless and may contain view-synthesis errors.
        They are aligned to the front hemisphere, converted to a smooth residual,
        and capped to a small angular change. Voxels where the front solver has
        no valid direction are deliberately left unchanged.
        """
        from scipy.ndimage import gaussian_filter

        if self._fused_view_owner is None:
            return front_field
        is_front = self._fused_boundary & (self._fused_view_owner == 0)
        is_side = self._fused_boundary & (self._fused_view_owner > 0)
        valid_front = np.linalg.norm(front_field, axis=0) > 0.05
        usable_side = is_side & valid_front
        if not usable_side.any():
            print('  No side boundary overlaps the valid front field; keeping baseline.')
            return front_field

        side = self._fused_orien_vol[:, usable_side].copy()
        reference = front_field[:, usable_side]
        dots = np.sum(side * reference, axis=0)
        side[:, dots < 0.0] *= -1.0

        # Reject severe disagreements instead of bending the field toward an
        # unreliable synthesized-view direction.
        agreement = np.abs(np.sum(side * reference, axis=0))
        accepted = agreement >= np.cos(np.deg2rad(45.0))
        residual = np.zeros_like(front_field)
        residual[:, usable_side] = side - reference
        rejected_mask = np.zeros_like(usable_side)
        rejected_mask[usable_side] = ~accepted
        residual[:, rejected_mask] = 0.0

        weights = np.zeros_like(usable_side, dtype=np.float32)
        weights[usable_side] = accepted.astype(np.float32)
        smooth_weight = gaussian_filter(weights, sigma=2.0)
        smooth_residual = np.empty_like(residual)
        for c in range(3):
            smooth_residual[c] = gaussian_filter(residual[c], sigma=2.0)
        smooth_residual /= np.maximum(smooth_weight[None], 1e-6)

        # At most a 10% residual correction, localized around side evidence.
        influence = np.clip(smooth_weight / 0.25, 0.0, 1.0)
        result = front_field + 0.10 * influence[None] * smooth_residual
        result[:, is_front] = self._fused_orien_vol[:, is_front]
        norms = np.linalg.norm(result, axis=0, keepdims=True)
        valid = norms[0] > 0.05
        result[:, valid] /= norms[:, valid]
        result[:, ~valid] = 0.0

        print(f'  Side residual: accepted={accepted.sum()}/{usable_side.sum()}, '
              f'max influence=0.10')
        return result

    def _solve_pde_on_fused_boundary(
        self,
        fused_orien: np.ndarray,
        boundary_mask: np.ndarray,
        mesh_path: str = None,
    ) -> np.ndarray:
        """Build a stable, confidence-weighted multi-view orientation field.

        Front is the trusted reference. Side-view directions are sign-aligned to
        the nearest front direction and blended at low confidence before a small
        number of clamped harmonic-relaxation steps. This prevents noisy side
        predictions from becoming hard, piecewise-constant regions.
        """
        from scipy.ndimage import distance_transform_edt, gaussian_filter

        is_all_boundary = boundary_mask
        if not is_all_boundary.any():
            raise RuntimeError("Multi-view orientation requires a non-empty fused boundary")

        if self._fused_hair_volume is not None:
            is_hair = self._fused_hair_volume | is_all_boundary
        else:
            is_hair = is_all_boundary.copy()
            struct = np.ones((3, 3, 3), dtype=bool)
            for _ in range(min(3, self.dilation_iters // 2)):
                is_hair = binary_dilation(is_hair, structure=struct)

        print(f'  Dir BC: fused={is_all_boundary.sum()}, Hair domain={is_hair.sum()}')

        # Initialize from the nearest contribution of any view. Directions are
        # already fused per voxel before entering this solver.
        import time
        t0 = time.time()
        print('  Fused-boundary EDT...', end=' ', flush=True)
        _, nearest_idx = distance_transform_edt(
            ~is_all_boundary, return_indices=True
        )
        orien_vol = np.zeros_like(fused_orien)
        idx_hair = nearest_idx[:, is_hair]
        for c in range(3):
            orien_vol[c][is_hair] = fused_orien[
                c, idx_hair[0], idx_hair[1], idx_hair[2]
            ]

        # Harmonic relaxation spreads all views symmetrically while clamping
        # every fused surface contribution as a Dirichlet boundary.
        print(f' {time.time()-t0:.1f}s, harmonic relax...', end=' ', flush=True)
        for _ in range(8):
            smoothed = np.empty_like(orien_vol)
            for c in range(3):
                smoothed[c] = gaussian_filter(orien_vol[c], sigma=1.0)
            orien_vol[:, is_hair] = smoothed[:, is_hair]
            orien_vol[:, is_all_boundary] = fused_orien[:, is_all_boundary]
        print(f'{time.time()-t0:.1f}s')

        # ── Normalize ────────────────────────────────────────────
        norms = np.sqrt(np.sum(orien_vol ** 2, axis=0, keepdims=True)) + 1e-8
        valid = (norms.squeeze(0) > 0.05) & is_hair
        for c in range(3):
            orien_vol[c][valid] /= norms[0][valid]
        orien_vol[:, ~valid] = 0.0

        return orien_vol
