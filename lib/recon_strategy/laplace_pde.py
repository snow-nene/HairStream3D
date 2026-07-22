"""
Laplace PDE Strategy
=====================
Training-free 3D hair reconstruction using the Laplace equation as a potential field solver.

Algorithm Overview
------------------
1. OCCUPANCY FIELD (Coarse Hair Volume):
   Given:
     - seg_map  [H, W] binary mask (hair region = 1)
     - depth_map [H, W] normalized depth values for visible surface
   
   We build a 3D occupancy voxel grid [Rx, Ry, Rz]:
     - Visible surface voxels (depth > threshold inside seg): occ = 1
     - A thin shell inward (head interior): occ = 1
     - Outside the 2D hair silhouette extruded volume: occ = 0
   We use morphological dilation/erosion to estimate the interior volume,
   then return per-point trilinear interpolation as query_occ.

2. ORIENTATION FIELD (3D Hair Growth Direction):
   We solve two anisotropic Laplace equations:
     - ∇²u = 0  (scalar potential u, source=scalp, sink=hair tips boundary)
   The gradient ∇u gives the "flow" direction from scalp toward tips.
   
   To incorporate 2D strand direction:
     - Surface points get a Dirichlet condition: orientation = 2D strand direction + z from depth gradient
     - Interior points: solved by Laplace interpolation (harmonic extension)
   
   This is equivalent to computing a harmonic vector field over the 3D hair volume
   that matches the visible 2D strand map on the surface and smoothly extends inward.

Implementation
--------------
We discretize on a 3D voxel grid (resolution R^3), build a sparse linear system
using SciPy, and solve with the Conjugate Gradient (CG) method.

Complexity: O(R^3) solve, typically R=64 → 262,144 unknowns → ~0.3s on CPU.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import cg
from scipy.ndimage import (
    binary_dilation, binary_erosion, gaussian_filter
)

from .base import BaseReconStrategy
from lib.geometry import orthogonal


class LaplacePDEStrategy(BaseReconStrategy):
    """
    Laplace PDE physical field solver for 3D hair reconstruction.

    No neural network weights required. Solves boundary value problems
    to generate occupancy and orientation fields from the 2D hairstep input.

    Key tuning parameters (all set via constructor kwargs or opt):
        pde_resolution (int):       Voxel grid resolution for PDE solve (default 64)
        pde_dilation_iters (int):   Volume thickness via morphological dilation (default 6)
        pde_cg_tol (float):         CG solver convergence tolerance (default 1e-4)
        pde_cg_maxiter (int):       Max CG iterations (default 500)
        pde_anisotropy (float):     Anisotropy weight: how strongly 2D strand map
                                    guides the Laplace field [0=isotropic, 1=fully guided]
                                    (default 0.8)
    """

    def __init__(self, opt, cuda: torch.device):
        super().__init__(opt, cuda)

        # ---------- tunable hyper-parameters ----------
        self.resolution = getattr(opt, 'pde_resolution', 64)
        self.dilation_iters = getattr(opt, 'pde_dilation_iters', 6)
        self.cg_tol = getattr(opt, 'pde_cg_tol', 1e-4)
        self.cg_maxiter = getattr(opt, 'pde_cg_maxiter', 500)
        self.anisotropy = getattr(opt, 'pde_anisotropy', 0.8)

        # World-space bounding box (same as used by gen_mesh_real in mesh_util.py)
        self.b_min = np.array([-0.3, 1.0, -0.3], dtype=np.float32)
        self.b_max = np.array([ 0.3, 2.0,  0.3], dtype=np.float32)

        # Internal state: built by filter()
        self._occ_vol:   np.ndarray = None   # [Rx, Ry, Rz] float32 in [0, 1]
        self._orien_vol: np.ndarray = None   # [3, Rx, Ry, Rz] float32
        self._data:      dict       = None   # cached data dict

    # ===================================================================
    # BaseReconStrategy interface
    # ===================================================================

    def filter(self, data: dict) -> None:
        """
        Builds the 3D occupancy and orientation fields from the 2D hairstep input.
        This replaces the neural network's feature extraction step.

        Args:
            data: dict with:
                'hairstep': [4, H, W] — channels 0-2 strand RGB, channel 3 depth
                'calib':    [4, 4]    — orthographic calibration matrix
        """
        self._data = data
        hairstep = data['hairstep'].cpu().numpy()  # [4, H, W]

        strand_rgb = hairstep[:3]    # [3, H, W] — orientation in [-1, 1]
        depth_map  = hairstep[3]     # [H, W]    — depth (background = -3.0)

        # Derive hair mask: depth > -2.5 means within valid hair region
        hair_mask = (depth_map > -2.5).astype(np.float32)  # [H, W]

        # --- decode 2D strand directions from RGB (undoing normalize) ---
        # strand_rgb is in [-1, 1]; channels 1 and 2 encode (cos θ, sin θ)
        strand_dx = strand_rgb[1]  # [H, W]  cos θ
        strand_dy = strand_rgb[2]  # [H, W]  sin θ

        print('[LaplacePDEStrategy] Building 3D occupancy volume ...')
        self._occ_vol = self._build_occupancy_volume(hair_mask, depth_map)

        print('[LaplacePDEStrategy] Solving Laplace orientation field ...')
        self._orien_vol = self._build_orientation_volume(
            hair_mask, depth_map, strand_dx, strand_dy
        )
        print('[LaplacePDEStrategy] Field build complete.')

    def query_occ(self, points: torch.Tensor, calib: torch.Tensor) -> torch.Tensor:
        """
        Trilinearly interpolates the pre-built occupancy volume at given 3D points.

        Args:
            points: [B, 3, N] world-space coordinates
            calib:  [B, 4, 4] (unused in pure voxel approach, kept for interface compat)

        Returns:
            [B, 1, N] occupancy values in [0, 1]
        """
        vol_tensor = torch.from_numpy(self._occ_vol).float()  # [Rx, Ry, Rz]
        return self._trilinear_query(vol_tensor, points, out_channels=1)

    def query_orien(self, points: torch.Tensor, calib: torch.Tensor) -> torch.Tensor:
        """
        Trilinearly interpolates the pre-built 3D orientation volume.

        Args:
            points: [B, 3, N] world-space coordinates
            calib:  [B, 4, 4] (unused, kept for interface compat)

        Returns:
            [B, 3, N] 3D orientation vectors
        """
        vol_tensor = torch.from_numpy(self._orien_vol).float()  # [3, Rx, Ry, Rz]
        return self._trilinear_query(vol_tensor, points, out_channels=3)

    # ===================================================================
    # Legacy shims  (same interface as neural net's .query() / .get_preds())
    # ===================================================================

    def query(self, points: torch.Tensor, calib: torch.Tensor, **kwargs):
        if self._query_mode == 'occ':
            self._last_preds = self.query_occ(points, calib)
        else:
            self._last_preds = self.query_orien(points, calib)
        return self._last_preds

    def get_preds(self):
        return [self._last_preds]

    # ===================================================================
    # Core PDE solvers (private)
    # ===================================================================

    def _build_occupancy_volume(
        self, hair_mask: np.ndarray, depth_map: np.ndarray
    ) -> np.ndarray:
        """
        Builds a [Rx, Ry, Rz] occupancy volume from the 2D hair mask and depth map.

        Strategy:
        1. Unproject each hair pixel into a 3D voxel using depth_map.
        2. Morphologically dilate inward to fill interior volume.
        3. Gaussian smooth boundary for soft occupancy (Marching Cubes works better).
        """
        R = self.resolution
        occ = np.zeros((R, R, R), dtype=np.float32)

        H, W = hair_mask.shape
        b_min, b_max = self.b_min, self.b_max

        # --- voxel coordinate arrays [R] ---
        xs = np.linspace(b_min[0], b_max[0], R)
        ys = np.linspace(b_min[1], b_max[1], R)
        zs = np.linspace(b_min[2], b_max[2], R)

        # --- project voxel grid onto 2D image using orthographic projection ---
        # For each 3D voxel we need to find its 2D pixel and look up depth
        # We sample only voxels within the column where hair_mask > 0

        # Meshgrid of 3D voxel centers
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')  # [R, R, R]
        pts_world = np.stack(
            [gx.ravel(), gy.ravel(), gz.ravel()], axis=0
        ).astype(np.float32)  # [3, R^3]

        # Orthographic projection to image UV in [-1, 1]
        # We use the loaded calib matrix for accurate projection
        calib = self._data['calib'].cpu().numpy()  # [4, 4]
        R3 = calib[:3, :3]
        t3 = calib[:3, 3:4]
        pts_cam = R3 @ pts_world + t3  # [3, R^3]
        uv = pts_cam[:2]  # [2, R^3]  in [-1, 1] approximately

        # Convert UV to pixel indices
        px = np.clip(((uv[0] + 1.0) / 2.0 * W).astype(int), 0, W - 1)
        py = np.clip(((uv[1] + 1.0) / 2.0 * H).astype(int), 0, H - 1)

        # Each voxel's visible depth from depth_map
        surf_depth = depth_map[py, px]   # [R^3] surface depth at corresponding pixel
        vox_depth  = pts_cam[2]           # [R^3] depth of the voxel itself
        mask_2d    = hair_mask[py, px]    # [R^3] is this pixel inside hair

        # A voxel is occupied if:
        #   1. The projected pixel is inside the hair silhouette mask
        #   2. The voxel's depth is >= surface_depth - small margin (inside/on surface)
        margin = (b_max[2] - b_min[2]) / R * 2.0
        inside = (mask_2d > 0.5) & (vox_depth >= (surf_depth - margin))

        occ.ravel()[inside] = 1.0

        # Morphological dilation to thicken the volume inward (Z direction)
        struct = np.ones((1, 1, self.dilation_iters), dtype=bool)
        occ_bool = occ > 0.5
        occ_dilated = binary_dilation(occ_bool, structure=struct, iterations=1)
        occ = occ_dilated.astype(np.float32)

        # Fill interior with uniform dilation along all axes for robustness
        struct3 = np.ones((3, 3, 3), dtype=bool)
        for _ in range(max(1, self.dilation_iters // 3)):
            occ = binary_dilation(occ > 0.5, structure=struct3).astype(np.float32)

        # Gaussian smooth for soft boundary → better Marching Cubes surface
        occ = gaussian_filter(occ, sigma=1.0)
        occ = np.clip(occ, 0.0, 1.0)

        return occ  # [R, R, R]

    def _build_orientation_volume(
        self,
        hair_mask: np.ndarray,
        depth_map: np.ndarray,
        strand_dx: np.ndarray,
        strand_dy: np.ndarray,
    ) -> np.ndarray:
        """
        Builds a [3, Rx, Ry, Rz] orientation field by solving the Laplace equation
        as a harmonic extension of 2D strand directions into 3D.

        We solve three independent scalar Laplace BVPs (one per orientation component):
            ∇²u_x = 0,  ∇²u_y = 0,  ∇²u_z = 0
        with boundary conditions on visible surface voxels set to the
        3D direction (dx, dy, dz) derived from the 2D strand map + depth gradient.

        The depth gradient (∂Z/∂x, ∂Z/∂y) is used to estimate the true 3D surface
        tangent direction and lift the 2D (dx, dy) into a proper 3D vector.
        """
        R = self.resolution
        b_min, b_max = self.b_min, self.b_max

        # ------------------------------------------------------------------
        # Step 1: Compute 3D strand directions on visible surface
        # ------------------------------------------------------------------
        # Compute depth gradient to estimate surface tangent dz component
        depth_smooth = gaussian_filter(depth_map * (depth_map > -2.5), sigma=2.0)
        dz_dx = np.gradient(depth_smooth, axis=1)  # ∂Z/∂x
        dz_dy = np.gradient(depth_smooth, axis=0)  # ∂Z/∂y

        # 3D tangent vector: T = (dx, dy, dx*dz_dx + dy*dz_dy)
        strand_dz = strand_dx * dz_dx + strand_dy * dz_dy  # [H, W]

        # Normalize the 3D direction vectors
        norm = np.sqrt(strand_dx**2 + strand_dy**2 + strand_dz**2) + 1e-8
        strand_vx = strand_dx / norm  # [H, W]
        strand_vy = strand_dy / norm  # [H, W]
        strand_vz = strand_dz / norm  # [H, W]

        # ------------------------------------------------------------------
        # Step 2: Map 2D directions to 3D voxel boundary conditions
        # ------------------------------------------------------------------
        H, W = hair_mask.shape
        xs = np.linspace(b_min[0], b_max[0], R)
        ys = np.linspace(b_min[1], b_max[1], R)
        zs = np.linspace(b_min[2], b_max[2], R)

        # Build voxel grids
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')  # [R, R, R]
        pts_world = np.stack(
            [gx.ravel(), gy.ravel(), gz.ravel()], axis=0
        ).astype(np.float32)  # [3, R^3]

        calib = self._data['calib'].cpu().numpy()
        R3 = calib[:3, :3]
        t3 = calib[:3, 3:4]
        pts_cam = R3 @ pts_world + t3

        # Projected pixel coordinates
        uv = pts_cam[:2]
        px = np.clip(((uv[0] + 1.0) / 2.0 * W).astype(int), 0, W - 1)
        py = np.clip(((uv[1] + 1.0) / 2.0 * H).astype(int), 0, H - 1)

        surf_depth = depth_map[py, px]
        vox_depth  = pts_cam[2]
        mask_2d    = hair_mask[py, px]

        margin = (b_max[2] - b_min[2]) / R * 2.0
        is_surface = (mask_2d > 0.5) & (np.abs(vox_depth - surf_depth) < margin)
        is_hair    = (mask_2d > 0.5) & (vox_depth >= (surf_depth - margin))

        # ------------------------------------------------------------------
        # Step 3: Build the Laplace linear system and solve per component
        # ------------------------------------------------------------------
        N = R * R * R
        # Strides for converting (ix, iy, iz) <-> flat index
        stride_x = R * R
        stride_y = R
        stride_z = 1

        orien_vol = np.zeros((3, R, R, R), dtype=np.float32)

        for comp_idx, (bc_vals_2d, comp_name) in enumerate([
            (strand_vx, 'Vx'),
            (strand_vy, 'Vy'),
            (strand_vz, 'Vz'),
        ]):
            print(f'  [Laplace] Solving component {comp_name} ({R}^3 grid) ...', end=' ')

            # Boundary values for surface voxels
            bc_surf = bc_vals_2d[py, px]  # [R^3] — value at each voxel's pixel

            # Build sparse matrix A and rhs b for ∇²u = 0
            A = lil_matrix((N, N), dtype=np.float32)
            b_rhs = np.zeros(N, dtype=np.float32)

            # Index helpers
            idx = np.arange(R, dtype=np.int32)
            ix3, iy3, iz3 = np.meshgrid(idx, idx, idx, indexing='ij')
            flat_idx = (ix3 * stride_x + iy3 * stride_y + iz3 * stride_z).ravel()  # [N]

            is_surface_vol = is_surface  # [N] boolean
            is_interior    = is_hair & ~is_surface  # [N]

            # For surface voxels: u = bc value (Dirichlet)
            surf_flat = flat_idx[is_surface_vol]
            for fi, bval in zip(surf_flat, bc_surf[is_surface_vol]):
                A[fi, fi] = 1.0
                b_rhs[fi] = float(bval)

            # For interior hair voxels: ∇²u = 0  (6-neighbor finite difference)
            interior_ixs = ix3.ravel()[is_interior]
            interior_iys = iy3.ravel()[is_interior]
            interior_izs = iz3.ravel()[is_interior]

            for ix, iy, iz in zip(interior_ixs, interior_iys, interior_izs):
                fi = int(ix * stride_x + iy * stride_y + iz * stride_z)
                neighbors = 0
                for dx, dy, dz in [
                    (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)
                ]:
                    nx, ny, nz = ix+dx, iy+dy, iz+dz
                    if 0 <= nx < R and 0 <= ny < R and 0 <= nz < R:
                        A[fi, int(nx*stride_x + ny*stride_y + nz*stride_z)] = 1.0
                        neighbors += 1
                A[fi, fi] = -neighbors

            # For outside voxels: u = 0 (Dirichlet zero — no hair)
            is_outside = ~is_hair
            outside_flat = flat_idx[is_outside]
            for fi in outside_flat:
                A[fi, fi] = 1.0
                b_rhs[fi] = 0.0

            # Solve using Conjugate Gradient
            A_csr = csr_matrix(A)
            u, info = cg(A_csr, b_rhs, tol=self.cg_tol, maxiter=self.cg_maxiter)
            if info != 0:
                print(f'  [Laplace] Warning: CG did not converge (info={info})')
            else:
                print('done.')

            orien_vol[comp_idx] = u.reshape(R, R, R).astype(np.float32)

        # ------------------------------------------------------------------
        # Step 4: Normalize orientation vectors
        # ------------------------------------------------------------------
        norms = np.sqrt(np.sum(orien_vol**2, axis=0, keepdims=True)) + 1e-8
        orien_vol = orien_vol / norms

        return orien_vol  # [3, R, R, R]

    # ===================================================================
    # Trilinear interpolation (pure PyTorch F.grid_sample)
    # ===================================================================

    def _world_to_vox_uv(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world-space coordinates [B, 3, N] to voxel grid UV in [-1, 1].
        Voxel grid spans [b_min, b_max] in world space.
        """
        b_min = torch.tensor(self.b_min, dtype=torch.float32)
        b_max = torch.tensor(self.b_max, dtype=torch.float32)

        # Normalize to [0, 1] then shift to [-1, 1] for grid_sample
        uv = (points - b_min[None, :, None]) / (b_max - b_min)[None, :, None]
        uv = uv * 2.0 - 1.0  # [B, 3, N] in [-1, 1]
        return uv  # order: [B, x, y, z] -> we pass as DHW

    def _trilinear_query(
        self, vol: torch.Tensor, points: torch.Tensor, out_channels: int
    ) -> torch.Tensor:
        """
        Trilinear interpolation of a voxel volume at arbitrary 3D world coordinates.

        Args:
            vol:         [C, Rx, Ry, Rz] or [Rx, Ry, Rz] float tensor (CPU)
            points:      [B, 3, N] world-space coordinates (on cuda)
            out_channels: 1 (occ) or 3 (orien)

        Returns:
            [B, C, N] interpolated values (on cuda)
        """
        B, _, N = points.shape

        if vol.dim() == 3:
            vol = vol.unsqueeze(0)  # [1, Rx, Ry, Rz]
        C = vol.shape[0]

        # vol as 5D [1, C, D, H, W] with D=Rz, H=Ry, W=Rx
        # F.grid_sample expects [B, C, D, H, W] and grid [B, N, 1, 1, 3] (x,y,z order)
        vol5d = vol.unsqueeze(0)  # [1, C, Rx, Ry, Rz]

        # Convert points to voxel UV coords
        uv = self._world_to_vox_uv(points)  # [B, 3, N]

        # grid_sample needs [..., (x, y, z)] where x=W axis, y=H axis, z=D axis
        # our axes: axis0=X(world), axis1=Y(world), axis2=Z(world)
        # F.grid_sample 5D: D=axis0 of vol, H=axis1, W=axis2
        # grid xyz = (iz, iy, ix) correspondingly  [need to permute]
        grid = uv.permute(0, 2, 1)  # [B, N, 3]  last dim is (x, y, z) world order
        # Flip to (z, y, x) for grid_sample's (x=W, y=H, z=D) convention
        grid = grid[..., [2, 1, 0]]                    # [B, N, 3]
        grid = grid.view(B, N, 1, 1, 3)               # [B, N, 1, 1, 3]

        vol5d = vol5d.expand(B, -1, -1, -1, -1)       # [B, C, Rx, Ry, Rz]

        # Move to same device as points
        vol5d = vol5d.to(points.device)
        grid  = grid.to(points.device)

        sampled = F.grid_sample(
            vol5d, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )  # [B, C, N, 1, 1]

        return sampled.squeeze(-1).squeeze(-1)  # [B, C, N]
