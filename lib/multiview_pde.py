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
from .multiview_fusion import limit_direction_normal_component


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
        self._head_mesh_path = "data/head_model.obj"
        self._fallback_orien_vol = None
        self.max_normal_component = float(
            getattr(opt, "pde_max_normal_component", 0.3)
        )
        self._is_multiview = False

    def set_fused_data(self, orien_vol: np.ndarray, boundary_mask: np.ndarray,
                        hair_volume: np.ndarray = None,
                        view_ownership: np.ndarray = None,
                        head_mesh_path: str = "data/head_model.obj"):
        R = self.resolution
        assert orien_vol.shape == (3, R, R, R)
        self._fused_orien_vol = orien_vol.copy()
        self._fused_boundary = boundary_mask.copy()
        self._fused_hair_volume = (hair_volume.copy() if hair_volume is not None else None)
        self._fused_view_owner = (view_ownership.copy() if view_ownership is not None else None)
        self._head_mesh_path = head_mesh_path
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

        scalp_boundary, scalp_normals = self._build_scalp_boundary(is_hair)
        _, nearest_fused = distance_transform_edt(
            ~is_all_boundary, return_indices=True
        )
        scalp_orien = np.zeros_like(fused_orien)
        if scalp_boundary.any():
            nearest_scalp = nearest_fused[:, scalp_boundary]
            scalp_strands = fused_orien[
                :, nearest_scalp[0], nearest_scalp[1], nearest_scalp[2]
            ]
            # Match the single-view inner boundary: strand direction remains
            # dominant, while a smaller head-normal component restores volume.
            blended = (
                0.7 * scalp_strands
                + 0.3 * scalp_normals[:, scalp_boundary]
            )
            blended /= np.linalg.norm(blended, axis=0, keepdims=True) + 1e-8
            scalp_orien[:, scalp_boundary] = blended
        all_boundary = is_all_boundary | scalp_boundary
        boundary_orien = fused_orien.copy()
        boundary_orien[:, scalp_boundary] = scalp_orien[:, scalp_boundary]

        print(
            f'  Dir BC: fused={is_all_boundary.sum()}, '
            f'scalp={scalp_boundary.sum()}, Hair domain={is_hair.sum()}'
        )

        # Initialize from the nearest contribution of any view. Directions are
        # already fused per voxel before entering this solver.
        import time
        t0 = time.time()
        print('  Fused-boundary EDT...', end=' ', flush=True)
        _, nearest_idx = distance_transform_edt(
            ~all_boundary, return_indices=True
        )
        orien_vol = np.zeros_like(fused_orien)
        idx_hair = nearest_idx[:, is_hair]
        for c in range(3):
            orien_vol[c][is_hair] = boundary_orien[
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
            orien_vol[:, all_boundary] = boundary_orien[:, all_boundary]
        print(f'{time.time()-t0:.1f}s')

        # ── Normalize ────────────────────────────────────────────
        norms = np.sqrt(np.sum(orien_vol ** 2, axis=0, keepdims=True)) + 1e-8
        valid = (norms.squeeze(0) > 0.05) & is_hair
        for c in range(3):
            orien_vol[c][valid] /= norms[0][valid]
        orien_vol[:, ~valid] = 0.0

        if mesh_path and self.max_normal_component < 1.0:
            self._limit_field_surface_normal(
                orien_vol,
                valid,
                mesh_path,
            )

        if valid.any():
            from scipy.ndimage import distance_transform_edt
            _, nearest = distance_transform_edt(~valid, return_indices=True)
            fallback = np.empty_like(orien_vol)
            for c in range(3):
                fallback[c] = orien_vol[c, nearest[0], nearest[1], nearest[2]]
            self._fallback_orien_vol = fallback

        return orien_vol

    def query_fallback(self, points: torch.Tensor, calib: torch.Tensor):
        """Sample the nearest-valid direction field for small domain gaps."""
        if self._fallback_orien_vol is None:
            return self.query_orien(points, calib)
        vol_tensor = torch.from_numpy(self._fallback_orien_vol).float()
        return self._trilinear_query(vol_tensor, points, out_channels=3)

    def _limit_field_surface_normal(self, orien_vol, valid, mesh_path):
        """Keep the solved field mostly tangent to the reconstructed hair mesh."""
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not mesh.has_vertices() or not mesh.has_triangles():
            raise ValueError(f"Hair mesh is empty: {mesh_path}")
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

        indices = np.argwhere(valid)
        shape = np.asarray(valid.shape, dtype=np.float64)
        b_min = np.asarray(self.b_min, dtype=np.float64)
        b_max = np.asarray(self.b_max, dtype=np.float64)
        changed = 0
        for start in range(0, len(indices), 250_000):
            chunk_indices = indices[start:start + 250_000]
            points = b_min + chunk_indices / np.maximum(shape - 1.0, 1.0) * (
                b_max - b_min
            )
            closest = scene.compute_closest_points(
                o3d.core.Tensor(points.astype(np.float32))
            )
            normals = closest["primitive_normals"].numpy().astype(np.float32)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
            coordinates = tuple(chunk_indices.T)
            directions = orien_vol[:, coordinates[0], coordinates[1], coordinates[2]].T
            limited, was_changed = limit_direction_normal_component(
                directions,
                normals,
                self.max_normal_component,
            )
            orien_vol[:, coordinates[0], coordinates[1], coordinates[2]] = limited.T
            changed += int(was_changed.sum())
        print(
            f'  Tangent field clamp: changed={changed}/{len(indices)}, '
            f'|normal|<={self.max_normal_component:.2f}'
        )

    def _build_scalp_boundary(self, is_hair: np.ndarray):
        """Create the outward scalp boundary used by the single-view PDE."""
        import open3d as o3d

        scalp_mask = np.zeros_like(is_hair, dtype=bool)
        scalp_orien = np.zeros((3,) + is_hair.shape, dtype=np.float32)
        hair_indices = np.argwhere(is_hair)
        if not len(hair_indices):
            return scalp_mask, scalp_orien

        head_mesh = o3d.io.read_triangle_mesh(str(self._head_mesh_path))
        if not head_mesh.has_vertices() or not head_mesh.has_triangles():
            raise ValueError(f"Head mesh is empty: {self._head_mesh_path}")
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(head_mesh))

        shape = np.asarray(is_hair.shape, dtype=np.float64)
        b_min = np.asarray(self.b_min, dtype=np.float64)
        b_max = np.asarray(self.b_max, dtype=np.float64)
        points = b_min + hair_indices / np.maximum(shape - 1.0, 1.0) * (
            b_max - b_min
        )
        query = o3d.core.Tensor(points.astype(np.float32))
        signed_distance = scene.compute_signed_distance(query).numpy()
        voxel_size = np.linalg.norm((b_max - b_min) / np.maximum(shape - 1.0, 1.0))
        near_scalp = (
            (signed_distance >= -voxel_size)
            & (signed_distance <= 0.020)
        )
        if not near_scalp.any():
            return scalp_mask, scalp_orien

        scalp_indices = hair_indices[near_scalp]
        scalp_points = points[near_scalp]
        closest = scene.compute_closest_points(
            o3d.core.Tensor(scalp_points.astype(np.float32))
        )
        normals = closest["primitive_normals"].numpy().astype(np.float32)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8

        head_center = np.asarray(head_mesh.get_center(), dtype=np.float32)
        inward = np.sum(normals * (scalp_points - head_center), axis=1) < 0.0
        normals[inward] *= -1.0
        coordinates = tuple(scalp_indices.T)
        scalp_mask[coordinates] = True
        scalp_orien[:, coordinates[0], coordinates[1], coordinates[2]] = normals.T
        return scalp_mask, scalp_orien
