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
from .recon_strategy.weighted_poisson import (
    build_signed_domain_distance,
    label_partition_components,
    solve_weighted_screened_poisson,
)
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
        self._internal_interface_mask = None
        self._internal_interface_normals = None
        self._internal_interface_confidence = 0.0
        self._internal_interface_length = 0.006
        self.max_normal_component = float(
            getattr(opt, "pde_max_normal_component", 0.3)
        )
        self.harmonic_relax_iters = int(
            getattr(opt, "pde_harmonic_relax_iters", 8)
        )
        self.scalp_boundary_width = float(
            getattr(opt, "pde_scalp_boundary_width", 0.005)
        )
        self.solver_mode = str(
            getattr(opt, "pde_solver_mode", "legacy_smooth")
        )
        if self.solver_mode not in {"legacy_smooth", "screened_poisson"}:
            raise ValueError(f"Unknown multi-view PDE solver: {self.solver_mode}")
        self.side_soft_confidence = float(
            getattr(opt, "pde_side_soft_confidence", 0.10)
        )
        self.side_screening_length = float(
            getattr(opt, "pde_side_screening_length", 0.02)
        )
        self.side_max_angle_degrees = float(
            getattr(opt, "pde_side_max_angle_degrees", 60.0)
        )
        self.side_constraint_mode = str(
            getattr(opt, "pde_side_constraint_mode", "soft")
        )
        if self.side_constraint_mode not in {"soft", "hard"}:
            raise ValueError(
                f"Unknown PDE side constraint mode: {self.side_constraint_mode}"
            )
        self.domain_tangent_confidence = float(
            getattr(opt, "pde_domain_tangent_confidence", 0.0)
        )
        self.domain_tangent_length = float(
            getattr(opt, "pde_domain_tangent_length", 0.01)
        )
        self.domain_padding_voxels = int(
            getattr(opt, "pde_domain_padding_voxels", 0)
        )
        self.solver_metrics = None
        self._pde_domain = None
        self._is_multiview = False

    def set_fused_data(self, orien_vol: np.ndarray, boundary_mask: np.ndarray,
                        hair_volume: np.ndarray = None,
                        view_ownership: np.ndarray = None,
                        head_mesh_path: str = "data/head_model.obj",
                        internal_interface_mask: np.ndarray = None,
                        internal_interface_normals: np.ndarray = None,
                        internal_interface_confidence: float = 0.0,
                        internal_interface_length: float = 0.006,
                        partition_labels: np.ndarray = None):
        R = self.resolution
        assert orien_vol.shape == (3, R, R, R)
        self._fused_orien_vol = orien_vol.copy()
        self._fused_boundary = boundary_mask.copy()
        self._fused_hair_volume = (hair_volume.copy() if hair_volume is not None else None)
        self._fused_view_owner = (view_ownership.copy() if view_ownership is not None else None)
        self._head_mesh_path = head_mesh_path
        self._internal_interface_mask = (
            internal_interface_mask.copy()
            if internal_interface_mask is not None else None
        )
        self._internal_interface_normals = (
            internal_interface_normals.copy()
            if internal_interface_normals is not None else None
        )
        self._internal_interface_confidence = float(
            internal_interface_confidence
        )
        self._internal_interface_length = float(internal_interface_length)
        self._partition_labels = (
            np.asarray(partition_labels).copy()
            if partition_labels is not None else None
        )
        if (
            self._partition_labels is not None
            and self._partition_labels.shape != (R, R, R)
        ):
            raise ValueError(
                "partition_labels must have shape (R, R, R)"
            )
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

        print(f'[MultiViewPDE] Solver mode: {self.solver_mode}')
        if self.solver_mode == "screened_poisson":
            self._orien_vol = self._solve_weighted_screened_poisson(
                mesh_path=mesh_path
            )
        else:
            print(
                '[MultiViewPDE] legacy_smooth is EDT + Gaussian relaxation; '
                'it is retained only as a non-PDE regression baseline.'
            )
            self._orien_vol = self._solve_pde_on_fused_boundary(
                self._fused_orien_vol,
                self._fused_boundary,
                mesh_path=mesh_path,
            )
        self._occ_vol = (
            self._pde_domain.astype(np.float32)
            if self._pde_domain is not None
            else self._fused_hair_volume.astype(np.float32)
            if self._fused_hair_volume is not None
            else self._fused_boundary.astype(np.float32)
        )
        print('[MultiViewPDE] Field build complete.')

    def _solve_weighted_screened_poisson(self, mesh_path: str = None) -> np.ndarray:
        """Solve the true masked PDE with hard front and soft side evidence."""
        from scipy.ndimage import distance_transform_edt, label

        observed = np.asarray(self._fused_boundary, dtype=bool)
        if not observed.any():
            raise RuntimeError(
                "Multi-view orientation requires a non-empty fused boundary"
            )
        domain = (
            np.asarray(self._fused_hair_volume, dtype=bool) | observed
            if self._fused_hair_volume is not None
            else binary_dilation(observed, iterations=max(1, self.dilation_iters))
        )
        constraint_domain = domain.copy()
        strict_volume = getattr(self, "strict_volume_contract", False)
        if strict_volume:
            from lib.recon_strategy.partition_direction import nearest_component_sources
            strict_spacing = (np.asarray(self.b_max)-np.asarray(self.b_min))/(np.asarray(domain.shape)-1)
        unpadded_domain_voxels = int(domain.sum())
        if self.domain_padding_voxels > 0:
            domain = binary_dilation(
                domain,
                structure=np.ones((3, 3, 3), dtype=bool),
                iterations=self.domain_padding_voxels,
            ) | observed
        padding_added_voxels = int(domain.sum()) - unpadded_domain_voxels
        owner = self._fused_view_owner
        if owner is None:
            front_boundary = observed.copy()
            side_boundary = np.zeros_like(observed)
        else:
            front_boundary = observed & (owner == 0)
            side_boundary = observed & (owner > 0)
        if not front_boundary.any():
            raise RuntimeError(
                "screened_poisson requires a trusted front boundary"
            )

        # Padding extends only the PDE unknown region. Artificial scalp
        # Dirichlet constraints must stay tied to the original hair volume,
        # otherwise a padding ablation also changes the root boundary data.
        scalp_boundary, scalp_normals = self._build_scalp_boundary(
            constraint_domain
        )
        scalp_boundary &= ~observed
        _, nearest_observed = distance_transform_edt(
            ~observed, return_indices=True
        )
        if strict_volume:
            nearest_observed = nearest_component_sources(
                domain, self._partition_labels, observed, strict_spacing)
        scalp_values = np.zeros_like(self._fused_orien_vol)
        if scalp_boundary.any():
            nearest_scalp = nearest_observed[:, scalp_boundary]
            nearest_directions = self._fused_orien_vol[
                :,
                nearest_scalp[0],
                nearest_scalp[1],
                nearest_scalp[2],
            ]
            blended = (
                0.7 * nearest_directions
                + 0.3 * scalp_normals[:, scalp_boundary]
            )
            blended /= np.linalg.norm(blended, axis=0, keepdims=True) + 1e-8
            scalp_values[:, scalp_boundary] = blended

        hard_boundary = front_boundary | scalp_boundary
        hard_values = np.zeros_like(self._fused_orien_vol, dtype=np.float32)
        hard_values[:, front_boundary] = self._fused_orien_vol[:, front_boundary]
        hard_values[:, scalp_boundary] = scalp_values[:, scalp_boundary]

        # A line field is signless. Align synthesized directions to the nearest
        # trusted front/scalp direction before turning them into soft sources.
        _, nearest_hard = distance_transform_edt(
            ~hard_boundary, return_indices=True
        )
        if strict_volume:
            nearest_hard = nearest_component_sources(
                domain, self._partition_labels, hard_boundary, strict_spacing)
        soft_values = np.zeros_like(hard_values)
        soft_weight = np.zeros_like(domain, dtype=np.float32)
        accepted_side = np.zeros_like(domain, dtype=bool)
        if side_boundary.any() and self.side_soft_confidence > 0.0:
            nearest_side = nearest_hard[:, side_boundary]
            reference = hard_values[
                :,
                nearest_side[0],
                nearest_side[1],
                nearest_side[2],
            ]
            side = self._fused_orien_vol[:, side_boundary].copy()
            signed_dot = np.sum(side * reference, axis=0)
            side[:, signed_dot < 0.0] *= -1.0
            agreement = np.abs(np.sum(side * reference, axis=0))
            accepted = agreement >= np.cos(
                np.deg2rad(self.side_max_angle_degrees)
            )
            side_coordinates = np.flatnonzero(side_boundary)
            accepted_ids = side_coordinates[accepted]
            accepted_side.ravel()[accepted_ids] = True
            soft_values.reshape(3, -1)[:, accepted_ids] = side[:, accepted]
            screening = self.side_soft_confidence / max(
                self.side_screening_length ** 2, 1e-8
            )
            soft_weight.ravel()[accepted_ids] = screening

        side_hard = np.zeros_like(domain, dtype=bool)
        if self.side_constraint_mode == "hard" and accepted_side.any():
            side_hard = accepted_side.copy()
            hard_boundary |= side_hard
            hard_values[:, side_hard] = soft_values[:, side_hard]
            soft_values[:, side_hard] = 0.0
            soft_weight[side_hard] = 0.0

        # Components with no hard or soft observation make the operator
        # singular and cannot produce meaningful root-to-tip flow. Remove them
        # explicitly and report the governance action.
        partition_labels = self._partition_labels
        components, component_count = label_partition_components(
            domain, partition_labels
        )
        anchors = hard_boundary | (soft_weight > 0.0)
        anchored_ids = np.unique(components[anchors])
        anchored_ids = anchored_ids[anchored_ids > 0]
        kept_domain = np.isin(components, anchored_ids)
        removed_voxels = int(domain.sum() - kept_domain.sum())
        removed_components = int(component_count - len(anchored_ids))
        if getattr(self, "strict_volume_contract", False) and removed_components:
            raise ValueError(
                f"Strict volume has {removed_components} unanchored components "
                f"({removed_voxels} voxels); refusing domain pruning"
            )
        domain = kept_domain | hard_boundary | accepted_side
        self._pde_domain = domain

        initial_sources = hard_boundary | accepted_side
        initial_values = hard_values + soft_values
        _, nearest_source = distance_transform_edt(
            ~initial_sources, return_indices=True
        )
        if strict_volume:
            nearest_source = nearest_component_sources(
                domain, self._partition_labels, initial_sources, strict_spacing)
        initial = np.zeros_like(hard_values)
        source_ids = nearest_source[:, domain]
        initial[:, domain] = initial_values[
            :, source_ids[0], source_ids[1], source_ids[2]
        ]
        spacing = (
            np.asarray(self.b_max, dtype=np.float64)
            - np.asarray(self.b_min, dtype=np.float64)
        ) / np.maximum(np.asarray(domain.shape) - 1, 1)

        tangent_normals = None
        tangent_weight = None
        tangent_boundary = np.zeros_like(domain, dtype=bool)
        if self.domain_tangent_confidence > 0.0:
            _, tangent_normals, _ = build_signed_domain_distance(
                domain,
                self.b_min,
                self.b_max,
            )
            tangent_boundary = (
                domain
                & ~binary_erosion(domain, structure=np.ones((3, 3, 3), dtype=bool))
                & ~hard_boundary
            )
            tangent_weight = np.zeros_like(domain, dtype=np.float32)
            tangent_weight[tangent_boundary] = (
                self.domain_tangent_confidence
                / max(self.domain_tangent_length ** 2, 1e-8)
            )

        interface_boundary = np.zeros_like(domain, dtype=bool)
        if (
            self._internal_interface_mask is not None
            and self._internal_interface_normals is not None
            and self._internal_interface_confidence > 0.0
        ):
            interface_boundary = (
                np.asarray(self._internal_interface_mask, dtype=bool)
                & domain
                & ~hard_boundary
            )
            interface_normals = np.asarray(
                self._internal_interface_normals, dtype=np.float32
            )
            if interface_normals.shape != hard_values.shape:
                raise ValueError(
                    "internal_interface_normals must have shape (3, X, Y, Z)"
                )
            if tangent_normals is None:
                tangent_normals = np.zeros_like(hard_values, dtype=np.float32)
                tangent_weight = np.zeros_like(domain, dtype=np.float32)
            tangent_normals[:, interface_boundary] = (
                interface_normals[:, interface_boundary]
            )
            tangent_weight[interface_boundary] = (
                self._internal_interface_confidence
                / max(self._internal_interface_length ** 2, 1e-8)
            )

        print(
            '  True PDE constraints: '
            f'domain={int(domain.sum())}, front_dirichlet={int(front_boundary.sum())}, '
            f'scalp_dirichlet={int(scalp_boundary.sum())}, '
            f'side_{self.side_constraint_mode}='
            f'{int(accepted_side.sum())}/{int(side_boundary.sum())}, '
            f'domain_padding={self.domain_padding_voxels}vox/'
            f'+{padding_added_voxels}vox, '
            f'domain_tangent={int(tangent_boundary.sum())}, '
            f'parting_interface={int(interface_boundary.sum())}, '
            f'pruned_components={removed_components}, '
            f'pruned_voxels={removed_voxels}'
        )
        solved, metrics = solve_weighted_screened_poisson(
            domain,
            hard_values,
            hard_boundary,
            soft_values=soft_values,
            soft_weight=soft_weight,
            normal_penalty_normals=tangent_normals,
            normal_penalty_weight=tangent_weight,
            spacing=spacing,
            tolerance=self.cg_tol,
            max_iterations=self.cg_maxiter,
            initial_field=initial,
            partition_labels=partition_labels,
            require_component_convergence=getattr(self, "strict_volume_contract", False),
            device=self.cuda,
            dtype=torch.float32,
            validate_components=True,
        )
        self.solver_metrics = metrics.to_dict()
        self.solver_metrics.update({
            "domain_tangent_voxels": int(tangent_boundary.sum()),
            "domain_tangent_confidence": self.domain_tangent_confidence,
            "domain_tangent_length": self.domain_tangent_length,
            "parting_interface_voxels": int(interface_boundary.sum()),
            "parting_interface_confidence": (
                self._internal_interface_confidence
            ),
            "parting_interface_length": self._internal_interface_length,
            "domain_padding_voxels": self.domain_padding_voxels,
            "unpadded_domain_voxels": unpadded_domain_voxels,
            "side_constraint_mode": self.side_constraint_mode,
            "side_hard_voxels": int(side_hard.sum()),
            "partition_count": (
                int(len(np.unique(partition_labels[domain])))
                if partition_labels is not None else 1
            ),
        })
        print(
            '  PDE solved: '
            f'converged={metrics.converged}, iterations={metrics.iterations}, '
            f'residual={metrics.initial_relative_residual:.3e}'
            f'->{metrics.final_relative_residual:.3e}, '
            f'time={metrics.elapsed_seconds:.2f}s'
        )
        if not metrics.converged:
            component_detail = ""
            components = metrics.component_residuals or []
            failed = [item for item in components if not item.get("converged", False)]
            if failed:
                worst = max(
                    failed,
                    key=lambda item: float(
                        item.get("relative_residual")
                        if item.get("relative_residual") is not None
                        else item.get("absolute_residual", 0.0)
                    ),
                )
                component_detail = (
                    f", failed_components={len(failed)}, "
                    f"worst_component={worst.get('component_id')}, "
                    f"worst_relative_residual={worst.get('relative_residual')}, "
                    f"worst_absolute_residual={worst.get('absolute_residual')}"
                )
            raise RuntimeError(
                "screened-Poisson did not converge: "
                f"residual={metrics.final_relative_residual:.3e}, "
                f"breakdown={metrics.breakdown}{component_detail}"
            )

        norms = np.linalg.norm(solved, axis=0, keepdims=True)
        valid = domain & (norms[0] > 1e-6)
        solved[:, valid] /= norms[:, valid]
        solved[:, ~valid] = 0.0

        if mesh_path and self.max_normal_component < 1.0:
            self._limit_field_surface_normal(
                solved,
                valid,
                mesh_path,
            )

        if valid.any():
            _, nearest_valid = distance_transform_edt(
                ~valid, return_indices=True
            )
            self._fallback_orien_vol = solved[
                :,
                nearest_valid[0],
                nearest_valid[1],
                nearest_valid[2],
            ]
        return solved

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
        scalp_overlap = scalp_boundary & is_all_boundary
        scalp_boundary &= ~is_all_boundary
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
            f'scalp={scalp_boundary.sum()}, protected_fused={scalp_overlap.sum()}, '
            f'Hair domain={is_hair.sum()}'
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

        # Match the scalp_protected_5mm baseline: every observed surface
        # direction and the artificial scalp shell are hard Dirichlet data.
        print(f' {time.time()-t0:.1f}s, harmonic relax...', end=' ', flush=True)
        for _ in range(self.harmonic_relax_iters):
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
            & (signed_distance <= self.scalp_boundary_width)
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
