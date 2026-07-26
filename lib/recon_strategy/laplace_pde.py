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
import taichi as ti

ti.init(arch=ti.cuda)

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

        # World-space bounding box
        self.b_min = getattr(opt, 'b_min', np.array([-0.3, 1.0, -0.3], dtype=np.float32))
        self.b_max = getattr(opt, 'b_max', np.array([ 0.3, 2.0,  0.3], dtype=np.float32))

        # Internal state: built by filter()
        self._occ_vol:   np.ndarray = None   # [Rx, Ry, Rz] float32 in [0, 1]
        self._orien_vol: np.ndarray = None   # [3, Rx, Ry, Rz] float32
        self._data:      dict       = None   # cached data dict

    # ===================================================================
    # BaseReconStrategy interface
    # ===================================================================

    def filter(self, data: dict, mesh_path: str = None) -> None:
        """
        Builds the 3D occupancy and orientation fields from the 2D hairstep input.
        This replaces the neural network's feature extraction step.

        Args:
            data: dict with:
                'hairstep': [4, H, W] — channels 0-2 strand RGB, channel 3 depth
                'calib':    [4, 4]    — orthographic calibration matrix
            mesh_path: Optional path to true 3D mesh volume for PDE boundary.
        """
        self._data = data
        hairstep = data['hairstep'].cpu().numpy()  # [4, H, W]

        strand_rgb = hairstep[:3]    # [3, H, W] — orientation in [-1, 1]
        depth_map  = hairstep[3]     # [H, W]    — depth (background = -3.0)

        # Derive hair mask: new depth format uses 0.0 for background
        hair_mask = (depth_map > 0.05).astype(np.float32)  # [H, W]

        # --- decode 2D strand directions from RGB (undoing normalize) ---
        # strand_rgb is in [-1, 1]. cv2 reads BGR, so 0=B, 1=G, 2=R
        # According to the user:
        # B = (-dx + 1) / 2 * 255 -> B_norm = -dx -> dx = -strand_rgb[0]
        # G = (dy + 1) / 2 * 255  -> G_norm = dy  -> dy = strand_rgb[1]
        # However, Image Y (dy) points DOWN, and World Y points UP.
        # So we MUST invert dy to match World coordinates!
        strand_dx = -strand_rgb[0] # [H, W]  X direction
        strand_dy = -strand_rgb[1] # [H, W]  Y direction (inverted for World Up)
        
        # Compute depth gradient to estimate surface tangent dz component
        # Avoid boundary step edges by extrapolating depth or just accepting the gradient 
        # (the PDE surface voxels might pick up boundary gradients, but let's just mask it).
        depth_smooth = gaussian_filter(depth_map, sigma=2.0) 
        dz_dx = np.gradient(depth_smooth, axis=1)  # ∂Z/∂x
        dz_dy = np.gradient(depth_smooth, axis=0)  # ∂Z/∂y
        strand_dz = (strand_dx * dz_dx + strand_dy * dz_dy) * hair_mask

        # 2. Build volumes
        print('[LaplacePDEStrategy] Building 3D occupancy volume ...')
        self._occ_vol = self._build_occupancy_volume(hair_mask, depth_map, mesh_path=mesh_path)

        print('[LaplacePDEStrategy] Solving Laplace orientation field ...')
        self._orien_vol = self._build_orientation_volume(
            hair_mask, depth_map, strand_dx, strand_dy, strand_dz, mesh_path=mesh_path
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
        self, hair_mask: np.ndarray, depth_map: np.ndarray, mesh_path: str = None
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
        # Generate entirely on GPU to bypass single-threaded Numpy bottleneck
        xs = torch.linspace(b_min[0], b_max[0], R, dtype=torch.float32, device=self.cuda)
        ys = torch.linspace(b_min[1], b_max[1], R, dtype=torch.float32, device=self.cuda)
        zs = torch.linspace(b_min[2], b_max[2], R, dtype=torch.float32, device=self.cuda)

        # Meshgrid of 3D voxel centers
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')  # [R, R, R]
        pts_world_t = torch.stack(
            [gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=0
        )  # [3, R^3] on GPU
        del gx, gy, gz

        # Orthographic projection to image UV in [-1, 1]
        calib = torch.from_numpy(self._data['calib'].cpu().numpy()).float().to(self.cuda)
        R3 = calib[:3, :3]
        t3 = calib[:3, 3:4]
        pts_cam_t = torch.matmul(R3, pts_world_t) + t3  # [3, R^3]
        
        uv_t = pts_cam_t[:2]  # [2, R^3]  in [-1, 1] approximately

        # Convert UV to pixel indices
        px_t = torch.clamp(((uv_t[0] + 1.0) / 2.0 * W).long(), 0, W - 1)
        py_t = torch.clamp(((uv_t[1] + 1.0) / 2.0 * H).long(), 0, H - 1)
        del uv_t
        
        # Download variables back to CPU for numpy operations
        px = px_t.cpu().numpy()
        py = py_t.cpu().numpy()
        vox_depth = pts_cam_t[2].cpu().numpy()
        pts_world = pts_world_t.cpu().numpy()
        
        del pts_world_t, pts_cam_t, px_t, py_t
        torch.cuda.empty_cache()

        mask_2d    = hair_mask[py, px]    # [R^3] is this pixel inside hair
        surf_depth = depth_map[py, px]
        
        if mesh_path is not None:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh_t)
            
            # Narrow Band SDF optimization - only use 2D mask
            valid_mask = (mask_2d > 0.5)
            valid_indices = np.where(valid_mask)[0]
            
            query_points = o3d.core.Tensor(pts_world[:, valid_indices].T, dtype=o3d.core.Dtype.Float32)
            del pts_world  # Free massive arrays
            
            distances_narrow = scene.compute_distance(query_points).cpu().numpy()
            del query_points
            
            # Initialize global distance array to a large value (outside the narrow band)
            distances = np.ones(mask_2d.shape, dtype=np.float32) * 100.0
            distances[valid_indices] = distances_narrow
            
            # A 0.3 shell bounded by front surface depth completely fills the distance between front mesh and scalp
            inside = (distances < 0.3) & (mask_2d > 0.5) & (vox_depth >= surf_depth - 0.02)
            occ.ravel()[inside] = 1.0
        else:
            # A voxel is occupied if:
            #   1. The projected pixel is inside the hair silhouette mask
            #   2. The voxel's depth is <= surface_depth + small margin (inside/on surface)
            # In our data, smaller depth means further away (deeper into the head).
            margin = (b_max[2] - b_min[2]) / R * 2.0
            inside = (mask_2d > 0.5) & (vox_depth <= (surf_depth + margin))
    
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
        strand_dz: np.ndarray,
        mesh_path: str = None
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
        depth_smooth = gaussian_filter(depth_map, sigma=2.0)
        dz_dx = np.gradient(depth_smooth, axis=1)  # ∂Z/∂x
        dz_dy = np.gradient(depth_smooth, axis=0)  # ∂Z/∂y

        # 3D tangent vector: T = (dx, dy, dx*dz_dx + dy*dz_dy)
        hair_mask_float = (depth_map > 0.05).astype(np.float32)
        strand_dz = (strand_dx * dz_dx + strand_dy * dz_dy) * hair_mask_float  # [H, W]

        # Normalize the 3D direction vectors
        norm = np.sqrt(strand_dx**2 + strand_dy**2 + strand_dz**2) + 1e-8
        strand_vx = strand_dx / norm  # [H, W]
        strand_vy = strand_dy / norm  # [H, W]
        strand_vz = strand_dz / norm  # [H, W]

        # ------------------------------------------------------------------
        # Step 2: Map 2D directions to 3D voxel boundary conditions
        # ------------------------------------------------------------------
        H, W = hair_mask.shape
        # Build voxel grids on GPU to bypass single-threaded Numpy bottleneck
        xs = torch.linspace(b_min[0], b_max[0], R, dtype=torch.float32, device=self.cuda)
        ys = torch.linspace(b_min[1], b_max[1], R, dtype=torch.float32, device=self.cuda)
        zs = torch.linspace(b_min[2], b_max[2], R, dtype=torch.float32, device=self.cuda)

        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')  # [R, R, R]
        pts_world_t = torch.stack(
            [gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=0
        )  # [3, R^3]
        del gx, gy, gz

        calib = torch.from_numpy(self._data['calib'].cpu().numpy()).float().to(self.cuda)
        R3 = calib[:3, :3]
        t3 = calib[:3, 3:4]
        pts_cam_t = torch.matmul(R3, pts_world_t) + t3
        
        uv_t = pts_cam_t[:2]

        # Projected pixel coordinates
        px_t = torch.clamp(((uv_t[0] + 1.0) / 2.0 * W).long(), 0, W - 1)
        py_t = torch.clamp(((uv_t[1] + 1.0) / 2.0 * H).long(), 0, H - 1)
        del uv_t
        
        # Download variables back to CPU for numpy operations
        px = px_t.cpu().numpy()
        py = py_t.cpu().numpy()
        vox_depth = pts_cam_t[2].cpu().numpy()
        pts_world = pts_world_t.cpu().numpy()
        
        del pts_world_t, pts_cam_t, px_t, py_t
        torch.cuda.empty_cache()
        mask_2d    = hair_mask[py, px]
        surf_depth = depth_map[py, px]

        margin = (b_max[2] - b_min[2]) / R * 2.0
        
        if mesh_path is not None:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh_t)
            
            # Narrow Band SDF optimization - only use 2D mask, don't cut off Z axis!
            valid_mask = (mask_2d > 0.5)
            valid_indices = np.where(valid_mask)[0]
            
            query_points = o3d.core.Tensor(pts_world[:, valid_indices].T, dtype=o3d.core.Dtype.Float32)
            
            distances_narrow = scene.compute_distance(query_points).cpu().numpy()
            distances = np.ones(mask_2d.shape, dtype=np.float32) * 100.0
            distances[valid_indices] = distances_narrow
            
            # The surface voxels are those very close to the mesh
            is_surface = (distances < 0.015) & (mask_2d > 0.5)
            # The interior voxels should span the entire gap from the front surface to the scalp
            is_hair = (distances < 0.05) & (mask_2d > 0.5)
            
            # Also compute inner boundary (scalp) to push hair outwards
            head_mesh = o3d.io.read_triangle_mesh('data/head_model.obj')
            head_t = o3d.t.geometry.TriangleMesh.from_legacy(head_mesh)
            head_scene = o3d.t.geometry.RaycastingScene()
            head_scene.add_triangles(head_t)
            
            # Compute head SDF on the full grid so gradients (normals) are accurate and smooth!
            head_query_points = o3d.core.Tensor(pts_world.T, dtype=o3d.core.Dtype.Float32)
            head_sdf = head_scene.compute_signed_distance(head_query_points).cpu().numpy()
            del head_query_points
            
            # Restrict inner normal (outward puffiness) to the front of the head (Z > 0)
            # For the back of the head (or outside 2D mask), we want it to extrapolate adjacent flow and cling to scalp.
            is_front_vol = (pts_world[2] > 0.0)
            is_inner_surface = (np.abs(head_sdf) < 0.015) & is_hair & is_front_vol
            # Exclude inner surface from outer surface
            is_surface = is_surface & ~is_inner_surface
            
            head_sdf_3d = head_sdf.reshape(R, R, R)
            
            # Offload heavy gradient calculation to GPU
            head_sdf_t = torch.from_numpy(head_sdf_3d).to(self.cuda)
            grad_x, grad_y, grad_z = torch.gradient(head_sdf_t, spacing=1, dim=(0, 1, 2))
            
            grad_x = grad_x.cpu().numpy().ravel()
            grad_y = grad_y.cpu().numpy().ravel()
            grad_z = grad_z.cpu().numpy().ravel()
            del head_sdf_t
            torch.cuda.empty_cache()
            
            head_normals = np.stack([grad_x, grad_y, grad_z], axis=1)
            norms = np.linalg.norm(head_normals, axis=1, keepdims=True) + 1e-8
            head_normals = head_normals / norms
            
        else:
            is_surface = (mask_2d > 0.5) & (np.abs(vox_depth - surf_depth) < margin)
            is_hair    = (mask_2d > 0.5) & (vox_depth <= (surf_depth + margin))
            is_inner_surface = np.zeros_like(is_surface, dtype=bool)

        # ------------------------------------------------------------------
        # Step 3: Build the Laplace linear system and solve per component (Matrix-Free GPU)
        # ------------------------------------------------------------------
        is_interior = is_hair & ~is_surface & ~is_inner_surface
        
        is_surface_t = torch.from_numpy(is_surface.reshape(R, R, R)).bool().to(self.cuda)
        is_inner_surface_t = torch.from_numpy(is_inner_surface.reshape(R, R, R)).bool().to(self.cuda)
        is_interior_t = torch.from_numpy(is_interior.reshape(R, R, R)).bool().to(self.cuda)
        
        # Prepare boundary values (Dirichlet)
        b_vol = torch.zeros((3, R, R, R), dtype=torch.float32, device=self.cuda)
        
        # Outer surface
        b_vol[0, is_surface_t] = torch.from_numpy(strand_dx[py[is_surface], px[is_surface]]).float().to(self.cuda)
        b_vol[1, is_surface_t] = torch.from_numpy(strand_dy[py[is_surface], px[is_surface]]).float().to(self.cuda)
        b_vol[2, is_surface_t] = torch.from_numpy(strand_dz[py[is_surface], px[is_surface]]).float().to(self.cuda)
        
        # Inner surface
        if is_inner_surface.any():
            hn = torch.from_numpy(head_normals[is_inner_surface]).float().to(self.cuda)
            dx_t = torch.from_numpy(strand_dx[py[is_inner_surface], px[is_inner_surface]]).float().to(self.cuda)
            dy_t = torch.from_numpy(strand_dy[py[is_inner_surface], px[is_inner_surface]]).float().to(self.cuda)
            dz_t = torch.from_numpy(strand_dz[py[is_inner_surface], px[is_inner_surface]]).float().to(self.cuda)
            b_vol[0, is_inner_surface_t] = 0.7 * hn[:, 0] + 0.3 * dx_t
            b_vol[1, is_inner_surface_t] = 0.7 * hn[:, 1] + 0.3 * dy_t
            b_vol[2, is_inner_surface_t] = 0.7 * hn[:, 2] + 0.3 * dz_t
            
        # ------------------------------------------------------------------
        # Taichi Sparse CG Solver
        # ------------------------------------------------------------------
        print(f'  [Taichi Laplace] Initializing Sparse SNode for {R}^3 grid...', end=' ', flush=True)
        
        x_ti = ti.Vector.field(3, dtype=ti.f32)
        p_ti = ti.Vector.field(3, dtype=ti.f32)
        r_ti = ti.Vector.field(3, dtype=ti.f32)
        Ap_ti = ti.Vector.field(3, dtype=ti.f32)
        is_int_ti = ti.field(dtype=ti.i32)
        dot_res = ti.Vector.field(3, dtype=ti.f64, shape=())
        
        block = ti.root.pointer(ti.ijk, (R//8, R//8, R//8))
        block.dense(ti.ijk, (8, 8, 8)).place(x_ti, p_ti, r_ti, Ap_ti, is_int_ti)
        
        @ti.kernel
        def init_taichi_fields(
            is_interior_arr: ti.types.ndarray(dtype=ti.u8),
            rhs_arr: ti.types.ndarray(dtype=ti.f32)
        ):
            for i, j, k in ti.ndrange(R, R, R):
                if is_interior_arr[i, j, k] > 0:
                    is_int_ti[i, j, k] = 1
                    x_ti[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                    r_val = ti.Vector([rhs_arr[0, i, j, k], rhs_arr[1, i, j, k], rhs_arr[2, i, j, k]])
                    r_ti[i, j, k] = r_val
                    p_ti[i, j, k] = r_val

        @ti.kernel
        def apply_L():
            for i, j, k in x_ti:
                if is_int_ti[i, j, k] == 1:
                    Ap_ti[i, j, k] = 6.0 * p_ti[i, j, k] - p_ti[i+1, j, k] - p_ti[i-1, j, k] - p_ti[i, j+1, k] - p_ti[i, j-1, k] - p_ti[i, j, k+1] - p_ti[i, j, k-1]

        @ti.kernel
        def compute_dot(v1: ti.template(), v2: ti.template()):
            dot_res[None] = ti.Vector([0.0, 0.0, 0.0])
            for i, j, k in x_ti:
                if is_int_ti[i, j, k] == 1:
                    v = ti.cast(v1[i, j, k] * v2[i, j, k], ti.f64)
                    dot_res[None] += v

        @ti.kernel
        def update_x_r(a0: ti.f32, a1: ti.f32, a2: ti.f32):
            alpha = ti.Vector([a0, a1, a2])
            for i, j, k in x_ti:
                if is_int_ti[i, j, k] == 1:
                    x_ti[i, j, k] += alpha * p_ti[i, j, k]
                    r_ti[i, j, k] -= alpha * Ap_ti[i, j, k]

        @ti.kernel
        def update_p(b0: ti.f32, b1: ti.f32, b2: ti.f32):
            beta = ti.Vector([b0, b1, b2])
            for i, j, k in x_ti:
                if is_int_ti[i, j, k] == 1:
                    p_ti[i, j, k] = beta * p_ti[i, j, k] + r_ti[i, j, k]

        @ti.kernel
        def copy_back_x(out: ti.types.ndarray(dtype=ti.f32)):
            for i, j, k in x_ti:
                if is_int_ti[i, j, k] == 1:
                    out[0, i, j, k] = x_ti[i, j, k][0]
                    out[1, i, j, k] = x_ti[i, j, k][1]
                    out[2, i, j, k] = x_ti[i, j, k][2]
                    
        # RHS computation in PyTorch
        kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32, device=self.cuda)
        kernel[0, 0, 1, 1, 1] = 6.0
        kernel[0, 0, 0, 1, 1] = -1.0; kernel[0, 0, 2, 1, 1] = -1.0
        kernel[0, 0, 1, 0, 1] = -1.0; kernel[0, 0, 1, 2, 1] = -1.0
        kernel[0, 0, 1, 1, 0] = -1.0; kernel[0, 0, 1, 1, 2] = -1.0
        
        neighbor_kernel = -kernel.clone()
        neighbor_kernel[0, 0, 1, 1, 1] = 0.0
        b_in = b_vol.unsqueeze(1)
        rhs = torch.nn.functional.conv3d(b_in, neighbor_kernel, padding=1).squeeze(1)
        rhs[:, ~is_interior_t] = 0.0
        
        init_taichi_fields(is_interior_t.to(torch.uint8).contiguous(), rhs.contiguous())
        print('done.')
        
        print(f'  [Laplace] Solving CG with Taichi Sparse SNodes...', end=' ', flush=True)
        
        compute_dot(r_ti, r_ti)
        rsold = dot_res[None].to_numpy()
        
        max_iter = self.cg_maxiter
        tol = self.cg_tol
        
        for i in range(max_iter):
            apply_L()
            compute_dot(p_ti, Ap_ti)
            pAp = dot_res[None].to_numpy() + 1e-10
            alpha = rsold / pAp
            
            update_x_r(float(alpha[0]), float(alpha[1]), float(alpha[2]))
            
            compute_dot(r_ti, r_ti)
            rsnew = dot_res[None].to_numpy()
            
            if np.max(np.sqrt(rsnew)) < tol:
                print(f'converged at iter {i}.')
                break
                
            beta = rsnew / (rsold + 1e-10)
            update_p(float(beta[0]), float(beta[1]), float(beta[2]))
            rsold = rsnew
        else:
            print(f'Warning: CG did not fully converge after {max_iter} iterations (max res: {np.max(np.sqrt(rsnew)):.4f})')
            
        # Copy solution back to PyTorch
        x = torch.zeros_like(b_vol)
        copy_back_x(x.contiguous())
        
        # Combine solution with boundary values
        orien_vol = x + b_vol
        
        # ------------------------------------------------------------------
        # Step 4: Normalize orientation vectors
        # ------------------------------------------------------------------
        del kernel, neighbor_kernel, b_vol, b_in, rhs, is_surface_t, is_inner_surface_t, is_interior_t
        torch.cuda.empty_cache()
        
        # Move to CPU for normalization to save VRAM
        orien_vol_cpu = orien_vol.cpu().numpy()
        del orien_vol, x
        torch.cuda.empty_cache()
        
        norms = np.sqrt(np.sum(orien_vol_cpu**2, axis=0, keepdims=True)) + 1e-8
        orien_vol_cpu = orien_vol_cpu / norms
        
        return orien_vol_cpu  # [3, R, R, R]

    # ===================================================================
    # Trilinear interpolation (pure PyTorch F.grid_sample)
    # ===================================================================

    def _world_to_vox_uv(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world-space coordinates [B, 3, N] to voxel grid UV in [-1, 1].
        Voxel grid spans [b_min, b_max] in world space.
        """
        b_min = torch.tensor(self.b_min, dtype=torch.float32, device=points.device)
        b_max = torch.tensor(self.b_max, dtype=torch.float32, device=points.device)

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
