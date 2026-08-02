import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
import trimesh
import cv2
import open3d as o3d
from lib.recon_strategy.laplace_pde import LaplacePDEStrategy

def query_grid(vol_5d, points_3n, b_min_tensor, b_max_tensor):
    # vol_5d: [1, C, R, R, R]
    # points_3n: [3, N]
    uv = (points_3n.unsqueeze(0) - b_min_tensor) / (b_max_tensor - b_min_tensor)
    uv = uv * 2.0 - 1.0  # [-1, 1]
    grid = uv.permute(0, 2, 1)
    grid = grid[..., [2, 1, 0]].unsqueeze(2).unsqueeze(2)  # [1, N, 1, 1, 3]
    val = torch.nn.functional.grid_sample(vol_5d, grid, padding_mode='border', align_corners=True)
    return val.squeeze(-1).squeeze(-1).squeeze(0)  # [C, N]


def project_points(pts_3d, calib_tensor):
    # pts_3d: [3, N]    
    # calib_tensor: [1, 4, 4]
    N = pts_3d.shape[1]
    pts_homo = torch.cat([pts_3d, torch.ones(1, N, device=pts_3d.device)], dim=0) # [4, N]
    uv = torch.matmul(calib_tensor.squeeze(0), pts_homo) # [4, N]
    uv = uv[:2, :] / (uv[3:4, :] + 1e-8) # [2, N]
    # Map from [-1, 1] to [0, H-1]
    # In PyTorch grid_sample, we need [-1, 1], so uv is actually exactly what we need for grid_sample!
    # Because calib matrix projects 3D points into NDC space [-1, 1].
    # Let's verify: In recon_strategy/base.py, it uses `uv = torch.bmm(calib, p)` then `uv = uv[:, :2, :] / uv[:, 2:3, :]`
    # and then uses it directly for grid_sample. So uv is in [-1, 1].
    return uv # [2, N]

def query_2d_map(map_tensor, uv):
    # map_tensor: [1, C, H, W]
    # uv: [2, N] in [-1, 1]
    grid = uv.unsqueeze(0).unsqueeze(2).permute(0, 3, 2, 1) # [1, N, 1, 2]
    # grid_sample expects [x, y] in [-1, 1]. uv from calib is [x, y].
    val = torch.nn.functional.grid_sample(map_tensor, grid, padding_mode='border', align_corners=True)
    return val.squeeze(-1).squeeze(0) # [C, N]

def hair_synthesis_rk4(strategy, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006, sdf_vol=None, normal_vol=None, b_min_t=None, b_max_t=None, noise_vol=None, guide_strands=None, guide_indices=None, div_map_t=None, valid_cluster_mask=None):
    """
    RK4 integration for hair synthesis with continuous collision response.
    root_tensor: [1, 3, N]
    """
    num_strand = root_tensor.shape[2]
    hair_strands = torch.zeros(num_sample, 3, num_strand).to(device=cuda)
    
    curr_node = root_tensor.squeeze(0) # [3, N]
    hair_strands[0] = curr_node
    
    for i in range(1, num_sample):
        # k1
        k1 = strategy.query(curr_node.unsqueeze(0), calib_tensor).squeeze(0) # [3, N]
        
        # k2
        p2 = curr_node + 0.5 * hair_unit * k1
        k2 = strategy.query(p2.unsqueeze(0), calib_tensor).squeeze(0)
        
        # k3
        p3 = curr_node + 0.5 * hair_unit * k2
        k3 = strategy.query(p3.unsqueeze(0), calib_tensor).squeeze(0)
        
        # k4
        p4 = curr_node + hair_unit * k3
        k4 = strategy.query(p4.unsqueeze(0), calib_tensor).squeeze(0)
        
        # base derivative
        k_avg = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        # normalized step
        t_ratio = i / float(num_sample)
        
        # Data-driven 2D Divergence Query
        if div_map_t is not None:
            uv = project_points(curr_node, calib_tensor)
            div_val = query_2d_map(div_map_t, uv).squeeze(0) # [N]
        else:
            div_val = torch.zeros(num_strand, device=cuda)

        # Divergence (Noise Field)
        # Always compute k_mag here so it can be reused by clustering below.
        k_mag = torch.norm(k_avg, dim=0, keepdim=True)  # [1, N]
        if noise_vol is not None:
            noise_val = query_grid(noise_vol, curr_node, b_min_t, b_max_t) # [3, N]
            # Only apply noise if divergence > 0 (spreading)
            is_div = (div_val > 0.1).float()
            # magnitude of divergence maps to noise strength
            div_mag = torch.clamp(div_val, 0.0, 1.0)
            divergence_weight = 0.5 * (t_ratio ** 2) * div_mag * is_div
            # Scale noise by k_mag: noise fades with PDE field → no oscillation at tip
            k_avg = k_avg + divergence_weight.unsqueeze(0) * noise_val * k_mag
        
        # update node
        curr_node = curr_node + hair_unit * k_avg
        
        # Clustering (Attractors)
        if guide_strands is not None and guide_indices is not None:
            # guide_strands shape: [num_sample, 3, K]
            C_guide = guide_strands[i, :, guide_indices] # [3, N]
            
            # If divergence < -0.1 (clumping), increase lerp weight
            is_clump = (div_val < -0.1).float()
            clump_mag = torch.clamp(-div_val, 0.0, 1.0)
            
            # Base clustering weight, ramps up over first half of the strand
            base_clump_weight = 0.005 * min(1.0, t_ratio / 0.5)
            # Add extra weight where the 2D map says it should clump
            extra_clump = 0.01 * clump_mag * is_clump * min(1.0, t_ratio / 0.5)
            
            total_clump = torch.clamp(base_clump_weight + extra_clump, 0.0, 0.02)
            
            # Disable clumping completely for roots outside the reliable 2D valid clusters
            if valid_cluster_mask is not None:
                total_clump = total_clump * valid_cluster_mask

            # KEY FIX: scale clustering weight by PDE field magnitude (k_mag).
            # When the PDE force dies at the tip (k_avg → 0), clustering weight also
            # dies → prevents the guide from dragging the stalled tip in a wrong direction.
            # k_mag is [1,N], normalize by expected root magnitude (~0.9) so it's a [0,1] weight.
            k_mag_weight = torch.clamp(k_mag / 0.9, 0.0, 1.0).squeeze(0)  # [N]
            total_clump = total_clump * k_mag_weight
                
            curr_node = torch.lerp(curr_node, C_guide, total_clump.unsqueeze(0))
        
        # Collision Response
        if sdf_vol is not None:
            # Query SDF and Normal
            sdf_val = query_grid(sdf_vol, curr_node, b_min_t, b_max_t) # [1, N]
            normal_val = query_grid(normal_vol, curr_node, b_min_t, b_max_t) # [3, N]
            
            # Normalize the normal
            normal_val = torch.nn.functional.normalize(normal_val, dim=0)
            
            # Moderate collision margin (5mm) to prevent piercing without introducing excessive bulkiness
            margin = 0.005
            penetration = margin - sdf_val
            mask = (penetration > 0).float() # [1, N]
            
            curr_node = curr_node + mask * penetration * normal_val
            
        hair_strands[i] = curr_node
        
    return hair_strands.permute(2, 0, 1).cpu().detach().numpy()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_name", default="0a1ba3dbefc8934ab60577c5c91f66a0")
    parser.add_argument("--mesh_obj", help="Optional custom hair mesh to restrict strands")
    parser.add_argument("--out_ply", default="results/test_pde_rk4/hair_mv_256.ply")
    parser.add_argument("--roots", default="data/roots10k.obj", help="Path to roots obj file")
    parser.add_argument("--pde_resolution", type=int, default=256, help="PDE resolution")
    parser.add_argument("--min_len", type=float, default=0.05, help="Minimum physical length threshold (in meters) to prune short strands")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_ply), exist_ok=True)
    
    # 1. Load data
    img_name = args.img_name
    # strand_path = f"results/multiview_data/{img_name}/maps/strand_map/gt.png"
    # depth_path  = f"results/multiview_data/{img_name}/maps/depth_map/gt.npy"
    # calib_path  = f"results/multiview_data/{img_name}/maps/param/front.npy"
    strand_path = f"results/multiview_data/real_imgs/strand_map/{img_name}.png"
    depth_path  = f"results/multiview_data/real_imgs/depth_map/{img_name}.npy"
    calib_path  = f"results/multiview_data/real_imgs/param/{img_name}.npy"
    strand_img = cv2.imread(strand_path).astype(np.float32) / 255.0 * 2.0 - 1.0 # [-1, 1]
    strand_rgb = strand_img.transpose(2, 0, 1) # [3, H, W]
    
    # Depth in HairStep is .npy
    depth_map = np.load(depth_path).astype(np.float32)
    
    hairstep = np.concatenate([strand_rgb, depth_map[None, :, :]], axis=0) # [4, H, W]
    
    calib_dict = np.load(calib_path, allow_pickle=True).item()
    # Construct calib matrix
    from scripts.recon_3d.recon3D import load_calib
    calib = load_calib(calib_path, loadSize=1024)
    
    data = {
        'hairstep': torch.from_numpy(hairstep).float(),
        'calib': calib.float() if isinstance(calib, torch.Tensor) else torch.from_numpy(calib).float()
    }
    
    cuda = torch.device('cuda:0')
    
    # 2. Run Laplace PDE Strategy
    class DummyOpt:
        pde_resolution = args.pde_resolution
        pde_dilation_iters = 6
        pde_cg_tol = 1e-4
        pde_cg_maxiter = 2000
        pde_anisotropy = 0.8
        
    mesh_path = args.mesh_obj
    
    # Compute dynamic bounding box
    head_mesh = o3d.io.read_triangle_mesh('data/head_model.obj')
    head_bbox = head_mesh.get_axis_aligned_bounding_box()
    b_min_dyn = head_bbox.get_min_bound()
    b_max_dyn = head_bbox.get_max_bound()
    
    if mesh_path and os.path.exists(mesh_path):
        hair_mesh = o3d.io.read_triangle_mesh(mesh_path)
        hair_bbox = hair_mesh.get_axis_aligned_bounding_box()
        b_min_dyn = np.minimum(b_min_dyn, hair_bbox.get_min_bound())
        b_max_dyn = np.maximum(b_max_dyn, hair_bbox.get_max_bound())
        
    # Add a generous 3cm padding
    padding = 0.03
    b_min_dyn -= padding
    b_max_dyn += padding
    
    b_min_dyn = b_min_dyn.astype(np.float32)
    b_max_dyn = b_max_dyn.astype(np.float32)
    
    DummyOpt.b_min = b_min_dyn
    DummyOpt.b_max = b_max_dyn
    print(f"Dynamic Bounding Box: min {b_min_dyn}, max {b_max_dyn}")
        
    strategy = LaplacePDEStrategy(DummyOpt(), cuda)
    strategy.filter(data, mesh_path=mesh_path)
    strategy.set_query_mode('orien')
    np.save("results/test_pde_rk4/debug_orien_vol.npy", strategy._orien_vol)
    
    # 3. Build inner collision SDF and Normal volume for head_model.obj
    print("Building inner collision volume for head_model...")
    # head_mesh was already loaded above
    head_t = o3d.t.geometry.TriangleMesh.from_legacy(head_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(head_t)
    
    R = 384
    b_min = b_min_dyn
    b_max = b_max_dyn
    xs = np.linspace(b_min[0], b_max[0], R)
    ys = np.linspace(b_min[1], b_max[1], R)
    zs = np.linspace(b_min[2], b_max[2], R)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts_grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)
    
    queries = o3d.core.Tensor(pts_grid, dtype=o3d.core.Dtype.Float32)
    sdf = scene.compute_signed_distance(queries).numpy().reshape(R, R, R)
    
    dx = (b_max[0] - b_min[0]) / (R - 1)
    dy = (b_max[1] - b_min[1]) / (R - 1)
    dz = (b_max[2] - b_min[2]) / (R - 1)
    
    grad_x = np.gradient(sdf, dx, axis=0)
    grad_y = np.gradient(sdf, dy, axis=1)
    grad_z = np.gradient(sdf, dz, axis=2)
    normals = np.stack([grad_x, grad_y, grad_z], axis=0)
    
    sdf_vol = torch.from_numpy(sdf).float().unsqueeze(0).unsqueeze(0).to(cuda) # [1, 1, R, R, R]
    normal_vol = torch.from_numpy(normals).float().unsqueeze(0).to(cuda) # [1, 3, R, R, R]
    b_min_t = torch.tensor(b_min, dtype=torch.float32, device=cuda).unsqueeze(1) # [3, 1]
    b_max_t = torch.tensor(b_max, dtype=torch.float32, device=cuda).unsqueeze(1) # [3, 1]
    
    # 4. Load roots
    from lib.hair_util import get_hair_root, save_strands_with_mesh
    root_tensor = torch.from_numpy(get_hair_root(args.roots)).float().unsqueeze(0).to(cuda)
    calib_tensor = calib.clone().detach().float().unsqueeze(0).to(cuda) if isinstance(calib, torch.Tensor) else torch.from_numpy(calib).float().unsqueeze(0).to(cuda)
    

    print("Extracting 2D Clusters and Divergence...")
    # Compute 2D features just like debug script
    strand_norm = strand_img
    strand_dx = -strand_norm[:, :, 0]
    strand_dy = -strand_norm[:, :, 1]
    mask = (depth_map > 0.05).astype(np.float32)
    
    # Divergence
    dx_dx = cv2.Sobel(strand_dx, cv2.CV_32F, 1, 0, ksize=3)
    dy_dy = cv2.Sobel(strand_dy, cv2.CV_32F, 0, 1, ksize=3)
    divergence = (dx_dx + dy_dy) * mask
    divergence = cv2.GaussianBlur(divergence, (5, 5), 0)    # Skip watershed and use KMeans directly for exactly 1024 clusters
    
    # Upload features to PyTorch
    div_map_t = torch.from_numpy(divergence).float().unsqueeze(0).unsqueeze(0).to(cuda) # [1, 1, 1024, 1024]
    
    print("Generating Divergence Noise Field...")
    import scipy.ndimage
    np.random.seed(42)
    raw_noise = np.random.randn(3, 64, 64, 64).astype(np.float32)
    smooth_noise = np.zeros_like(raw_noise)
    for c in range(3):
        smooth_noise[c] = scipy.ndimage.gaussian_filter(raw_noise[c], sigma=3.0)
    smooth_noise = smooth_noise / (np.std(smooth_noise) + 1e-8)
    noise_vol = torch.from_numpy(smooth_noise).unsqueeze(0).to(cuda)
    
    print("Projecting Roots to 2D for Data-Driven Clustering...")
    # calib is [4, 4]. Need to project roots
    roots_3d = root_tensor.squeeze(0) # [3, N]
    N_roots = roots_3d.shape[1]
    pts_homo = torch.cat([roots_3d, torch.ones(1, N_roots, device=cuda)], dim=0)
    uv = torch.matmul(calib_tensor.squeeze(0), pts_homo)
    uv = uv[:2, :] / (uv[3:4, :] + 1e-8) # [-1, 1]
    
    # Map uv [-1, 1] to pixel indices [0, 511]
    uv_px = ((uv + 1.0) * 0.5 * 511).long()
    uv_px = torch.clamp(uv_px, 0, 511)
    
    print("Selecting 1024 Guide Strands using KMeans...")
    from sklearn.cluster import MiniBatchKMeans
    
    uv_np = uv_px.float().cpu().numpy().T # [N, 2]
    
    kmeans = MiniBatchKMeans(n_clusters=1024, random_state=42, n_init="auto", batch_size=2048).fit(uv_np)
    centroids_list = kmeans.cluster_centers_.tolist()
    
    guide_idx_list = []
    
    for cx, cy in centroids_list:
        dist_sq = (uv_px[0].float() - cx)**2 + (uv_px[1].float() - cy)**2
        best_global_idx = torch.argmin(dist_sq)
        guide_idx_list.append(best_global_idx.item())
        
    num_clusters = len(guide_idx_list)
    print(f"Extracted {num_clusters} Data-Driven Guide Strands.")
    
    guide_idx_tensor = torch.tensor(guide_idx_list, device=cuda)
    guide_roots = root_tensor[:, :, guide_idx_tensor] # [1, 3, K]
    
    centroids_t = torch.tensor(centroids_list, device=cuda) # [K, 2]
    
    # Map EVERY root to its nearest centroid to prevent chaotic criss-crossing of background roots
    uv_px_t = uv_px.float() # [2, N]
    # Distances: [K, N]
    dist_sq_all = (uv_px_t[0].unsqueeze(0) - centroids_t[:, 0].unsqueeze(1))**2 + \
                  (uv_px_t[1].unsqueeze(0) - centroids_t[:, 1].unsqueeze(1))**2
    
    guide_indices_t = torch.argmin(dist_sq_all, dim=0) # [N]
            
    guide_strands = hair_synthesis_rk4(
        strategy, cuda, guide_roots, calib_tensor, 
        num_sample=100, hair_unit=0.006,
        sdf_vol=sdf_vol, normal_vol=normal_vol, b_min_t=b_min_t, b_max_t=b_max_t
    )
    guide_strands_t = torch.from_numpy(guide_strands).to(cuda).permute(1, 2, 0) # [num_sample, 3, K]

    print("Integrating Full Strands with 2D-Driven RK4 (Clustering & Divergence)...")
    valid_cluster_mask = torch.from_numpy(mask).to(cuda)[uv_px[1], uv_px[0]] # [N] # [N]
    
    strands = hair_synthesis_rk4(
        strategy, cuda, root_tensor, calib_tensor, 
        num_sample=100, hair_unit=0.006,
        sdf_vol=sdf_vol, normal_vol=normal_vol, b_min_t=b_min_t, b_max_t=b_max_t,
        noise_vol=noise_vol, guide_strands=guide_strands_t, guide_indices=guide_indices_t,
        div_map_t=div_map_t, valid_cluster_mask=valid_cluster_mask
    )
    
    print(f"Pruning short strands (< {args.min_len*100:.1f} cm) and saving PLY...")
    save_strands_with_mesh(strands, mesh_path, args.out_ply, 0.3, is_eval=False, min_len=args.min_len)
    print(f"Saved to {args.out_ply}")

if __name__ == '__main__':
    main()
