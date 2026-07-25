import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
            
            # If inside the head (SDF < 0.001), push out along the normal
            margin = 0.001
            penetration = margin - sdf_val
            mask = (penetration > 0).float() # [1, N]
            
            curr_node = curr_node + mask * penetration * normal_val
            
        hair_strands[i] = curr_node
        
    return hair_strands.permute(2, 0, 1).cpu().detach().numpy()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_name", default="0d285f5be7fa09c3dbbf1c9334047888")
    parser.add_argument("--mesh_obj", help="Optional custom hair mesh to restrict strands")
    parser.add_argument("--out_ply", default="results/test_pde_rk4/hair.ply")
    parser.add_argument("--roots", default="data/roots10k.obj", help="Path to roots obj file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_ply), exist_ok=True)
    
    # 1. Load data
    img_name = args.img_name
    strand_path = f"results/real_imgs/strand_map/{img_name}.png"
    depth_path  = f"results/real_imgs/depth_map/{img_name}.npy"
    calib_path  = f"results/real_imgs/param/{img_name}.npy"
    
    strand_img = cv2.imread(strand_path).astype(np.float32) / 255.0 * 2.0 - 1.0 # [-1, 1]
    strand_rgb = strand_img.transpose(2, 0, 1) # [3, H, W]
    
    # Depth in HairStep is .npy
    depth_map = np.load(depth_path).astype(np.float32)
    
    hairstep = np.concatenate([strand_rgb, depth_map[None, :, :]], axis=0) # [4, H, W]
    
    calib_dict = np.load(calib_path, allow_pickle=True).item()
    # Construct calib matrix
    from scripts.recon3D import load_calib
    calib = load_calib(calib_path, loadSize=1024)
    
    data = {
        'hairstep': torch.from_numpy(hairstep).float(),
        'calib': calib.float() if isinstance(calib, torch.Tensor) else torch.from_numpy(calib).float()
    }
    
    cuda = torch.device('cuda:0')
    
    # 2. Run Laplace PDE Strategy
    class DummyOpt:
        pde_resolution = 512
        pde_dilation_iters = 6
        pde_cg_tol = 1e-4
        pde_cg_maxiter = 500
        pde_anisotropy = 0.8
        
    mesh_path = args.mesh_obj
        
    strategy = LaplacePDEStrategy(DummyOpt(), cuda)
    strategy.filter(data, mesh_path=mesh_path)
    strategy.set_query_mode('orien')
    np.save("results/test_pde_rk4/debug_orien_vol.npy", strategy._orien_vol)
    
    # 3. Build inner collision SDF and Normal volume for head_model.obj
    print("Building inner collision volume for head_model...")
    head_mesh = o3d.io.read_triangle_mesh('data/head_model.obj')
    head_t = o3d.t.geometry.TriangleMesh.from_legacy(head_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(head_t)
    
    R = 512
    b_min = np.array([-0.3, 1.0, -0.3], dtype=np.float32)
    b_max = np.array([ 0.3, 2.0,  0.3], dtype=np.float32)
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
    divergence = cv2.GaussianBlur(divergence, (5, 5), 0)
    
    # Edges & Watershed
    dz_dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    depth_grad_mag = np.sqrt(dz_dx**2 + dz_dy**2) * mask
    depth_grad_mag = np.clip(depth_grad_mag / (np.percentile(depth_grad_mag[mask>0], 98) + 1e-5), 0, 1)
    
    dx_dy = cv2.Sobel(strand_dx, cv2.CV_32F, 0, 1, ksize=3)
    dy_dx = cv2.Sobel(strand_dy, cv2.CV_32F, 1, 0, ksize=3)
    orien_grad_mag = np.sqrt(dx_dx**2 + dx_dy**2 + dy_dx**2 + dy_dy**2) * mask
    orien_grad_mag = np.clip(orien_grad_mag / (np.percentile(orien_grad_mag[mask>0], 98) + 1e-5), 0, 1)
    
    combined_edges = np.maximum(depth_grad_mag, orien_grad_mag)
    binary_edges = (combined_edges > 0.05).astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)
    binary_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel)
    
    sure_bg = cv2.dilate((1 - mask).astype(np.uint8), kernel, iterations=3) * 255
    sure_fg = ((mask > 0) & (binary_edges == 0)).astype(np.uint8) * 255
    sure_fg = cv2.erode(sure_fg, kernel, iterations=2)
    unknown = cv2.subtract(mask.astype(np.uint8)*255, cv2.add(sure_bg, sure_fg))
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_for_ws = np.zeros((strand_img.shape[0], strand_img.shape[1], 3), dtype=np.uint8)
    img_for_ws[mask > 0] = [255, 255, 255]
    markers = cv2.watershed(img_for_ws, markers)
    
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
    
    # Get cluster ID for each root
    markers_t = torch.from_numpy(markers).to(cuda) # [1024, 1024]
    root_cluster_ids = markers_t[uv_px[1], uv_px[0]] # [N]
    
    print("Selecting Guide Strands from 2D Centroids...")
    # Find all unique clusters in the 2D image (not just the ones hit by roots)
    markers_t = torch.from_numpy(markers).to(cuda) # [512, 512]
    unique_clusters = torch.unique(markers_t)
    
    guide_idx_list = []
    cluster_to_guide_arr_idx = {}
    valid_idx = 0
    
    # Pre-calculate a grid of XY coordinates for centroid calculation
    yy, xx = torch.meshgrid(torch.arange(512, device=cuda), torch.arange(512, device=cuda), indexing='ij')
    centroids_list = []
    
    for cid in unique_clusters:
        if cid <= 1: continue 
        
        # Pixels belonging to this cluster
        mask_cid = (markers_t == cid)
        if not mask_cid.any(): continue
        
        # Calculate centroid of the cluster in 2D
        centroid_y = yy[mask_cid].float().mean()
        centroid_x = xx[mask_cid].float().mean()
        
        centroids_list.append((centroid_x.item(), centroid_y.item()))
        
        # Find the globally closest root in 2D projection
        # uv_px is [2, N] (x, y)
        dist_sq = (uv_px[0].float() - centroid_x)**2 + (uv_px[1].float() - centroid_y)**2
        best_global_idx = torch.argmin(dist_sq)
        
        guide_idx_list.append(best_global_idx.item())
        cluster_to_guide_arr_idx[cid.item()] = valid_idx
        valid_idx += 1
        
    num_clusters = len(guide_idx_list)
    print(f"Extracted {num_clusters} Data-Driven Guide Strands.")
    
    guide_idx_tensor = torch.tensor(guide_idx_list, device=cuda)
    guide_roots = root_tensor[:, :, guide_idx_tensor] # [1, 3, K]
    
    # Map every root to the index in the guide_strands array
    # guide_idx_tensor contains the global index of the guides.
    # For a root in cluster CID, its guide index is the position of CID in unique_clusters.

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
    valid_cluster_mask = (root_cluster_ids > 1).float() # [N]
    
    strands = hair_synthesis_rk4(
        strategy, cuda, root_tensor, calib_tensor, 
        num_sample=100, hair_unit=0.006,
        sdf_vol=sdf_vol, normal_vol=normal_vol, b_min_t=b_min_t, b_max_t=b_max_t,
        noise_vol=noise_vol, guide_strands=guide_strands_t, guide_indices=guide_indices_t,
        div_map_t=div_map_t, valid_cluster_mask=valid_cluster_mask
    )
    
    print(f"Clipping strands with mesh {mesh_path}...")
    save_strands_with_mesh(strands, mesh_path, args.out_ply, 0.3, is_eval=False)
    print(f"Saved to {args.out_ply}")

if __name__ == '__main__':
    main()
