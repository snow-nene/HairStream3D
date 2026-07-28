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
        # Front camera (index 0) is the ONLY trusted direction source
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

        self._data = data
        hairstep = data['hairstep'].cpu().numpy()
        depth_map = hairstep[3]
        hair_mask = (depth_map > 0.05).astype(np.float32)

        # ── Occupancy volume (same as single-view, from front depth) ──
        print('[MultiViewPDE] Building occupancy volume (from front depth)...')
        self._occ_vol = self._build_occupancy_volume(
            hair_mask, depth_map, mesh_path=mesh_path
        )

        # ── Orientation volume: Solve PDE on fused boundary ──
        print('[MultiViewPDE] Solving Laplace PDE on fused multi-view boundary...')
        self._orien_vol = self._solve_pde_on_fused_boundary(
            self._fused_orien_vol,
            self._fused_boundary,
            mesh_path=mesh_path,
        )
        print('[MultiViewPDE] Field build complete.')

    def _solve_pde_on_fused_boundary(
        self,
        fused_orien: np.ndarray,
        boundary_mask: np.ndarray,
        mesh_path: str = None,
    ) -> np.ndarray:
        """Fill hair volume via EDT from FRONT boundary only.

        - direction_boundary: front camera voxels → trusted direction
        - Other boundary voxels: occupancy only, direction left at 0
        - EDT propagates front direction into entire hair volume
        - Light Gaussian smooth to avoid EDT Voronoi artifacts
        """
        R = self.resolution
        from scipy.ndimage import distance_transform_edt, gaussian_filter

        # ── Classify ──────────────────────────────────────────────
        is_all_boundary = boundary_mask  # all camera surfaces
        if self._fused_view_owner is not None:
            is_dir_boundary = is_all_boundary & (self._fused_view_owner == 0)
        else:
            is_dir_boundary = is_all_boundary

        if self._fused_hair_volume is not None:
            is_hair = self._fused_hair_volume | is_all_boundary
        else:
            is_hair = is_all_boundary.copy()
            struct = np.ones((3, 3, 3), dtype=bool)
            for _ in range(min(3, self.dilation_iters // 2)):
                is_hair = binary_dilation(is_hair, structure=struct)

        is_interior = is_hair & ~is_dir_boundary
        print(f'  Dir BC (front): {is_dir_boundary.sum()}, '
              f'All surf: {is_all_boundary.sum()}, Interior: {is_interior.sum()}')

        # ── EDT from front-only boundary → fill entire hair volume ─
        import time
        t0 = time.time()
        print(f'  EDT front→interior...', end=' ', flush=True)
        _, nearest_idx = distance_transform_edt(
            ~is_dir_boundary, return_indices=True
        )
        orien_vol = fused_orien.copy()
        idx_i = nearest_idx[:, is_interior]
        for c in range(3):
            orien_vol[c][is_interior] = fused_orien[
                c, idx_i[0], idx_i[1], idx_i[2]
            ]
        print(f'{time.time()-t0:.1f}s', end='', flush=True)

        # ── Light Gaussian smooth ─────────────────────────────────
        print(f', smooth...', end=' ', flush=True)
        for c in range(3):
            orien_vol[c] = gaussian_filter(orien_vol[c], sigma=2.0)
        # Re-fix direction boundary
        orien_vol[:, is_dir_boundary] = fused_orien[:, is_dir_boundary]
        print(f'{time.time()-t0:.1f}s')

        # ── Normalize ────────────────────────────────────────────
        norms = np.sqrt(np.sum(orien_vol ** 2, axis=0, keepdims=True)) + 1e-8
        valid = (norms.squeeze(0) > 0.05) & is_hair
        for c in range(3):
            orien_vol[c][valid] /= norms[0][valid]
        orien_vol[:, ~valid] = 0.0

        return orien_vol
