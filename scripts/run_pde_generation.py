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

def hair_synthesis_rk4(strategy, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006, sdf_vol=None, normal_vol=None, b_min_t=None, b_max_t=None):
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
        
        # update
        curr_node = curr_node + (hair_unit / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
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
    depth_map = np.load(depth_path)
    
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
        pde_resolution = 64
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
    
    R = 64
    b_min = np.array([-0.3, 1.0, -0.3], dtype=np.float32)
    b_max = np.array([ 0.3, 2.0,  0.3], dtype=np.float32)
    xs = np.linspace(b_min[0], b_max[0], R)
    ys = np.linspace(b_min[1], b_max[1], R)
    zs = np.linspace(b_min[2], b_max[2], R)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts_grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)
    
    queries = o3d.core.Tensor(pts_grid, dtype=o3d.core.Dtype.Float32)
    sdf = scene.compute_signed_distance(queries).numpy().reshape(R, R, R)
    
    grad_x = np.gradient(sdf, axis=0)
    grad_y = np.gradient(sdf, axis=1)
    grad_z = np.gradient(sdf, axis=2)
    normals = np.stack([grad_x, grad_y, grad_z], axis=0)
    
    sdf_vol = torch.from_numpy(sdf).float().unsqueeze(0).unsqueeze(0).to(cuda) # [1, 1, R, R, R]
    normal_vol = torch.from_numpy(normals).float().unsqueeze(0).to(cuda) # [1, 3, R, R, R]
    b_min_t = torch.tensor(b_min, dtype=torch.float32, device=cuda).unsqueeze(1) # [3, 1]
    b_max_t = torch.tensor(b_max, dtype=torch.float32, device=cuda).unsqueeze(1) # [3, 1]
    
    # 4. Load roots
    from lib.hair_util import get_hair_root, save_strands_with_mesh
    root_tensor = torch.from_numpy(get_hair_root(args.roots)).float().unsqueeze(0).to(cuda)
    calib_tensor = calib.clone().detach().float().unsqueeze(0).to(cuda) if isinstance(calib, torch.Tensor) else torch.from_numpy(calib).float().unsqueeze(0).to(cuda)
    
    print("Integrating with RK4...")
    strands = hair_synthesis_rk4(
        strategy, cuda, root_tensor, calib_tensor, 
        num_sample=300, hair_unit=0.006,
        sdf_vol=sdf_vol, normal_vol=normal_vol, b_min_t=b_min_t, b_max_t=b_max_t
    )
    
    print(f"Clipping strands with mesh {mesh_path}...")
    save_strands_with_mesh(strands, mesh_path, args.out_ply, 0.3, is_eval=False)
    print(f"Saved to {args.out_ply}")

if __name__ == '__main__':
    main()
