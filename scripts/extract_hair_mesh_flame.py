import numpy as np
import open3d as o3d
import torch

def main():
    # 1. Load meshes
    flame_mesh = o3d.io.read_triangle_mesh('assets/FLAME/flame_neutral.obj')
    pixal_mesh = o3d.io.read_triangle_mesh('results/test_pixal3d/hair_mesh_aligned_best.obj')
    
    # 2. Initial alignment (translate center of FLAME to center of Pixal3D)
    flame_center = flame_mesh.get_center()
    pixal_center = pixal_mesh.get_center()
    flame_mesh.translate(pixal_center - flame_center)
    
    flame_pcd = o3d.geometry.PointCloud()
    flame_pcd.points = flame_mesh.vertices
    pixal_pcd = o3d.geometry.PointCloud()
    pixal_pcd.points = pixal_mesh.vertices
    
    # Run ICP
    threshold = 0.2 # 20cm
    trans_init = np.eye(4)
    print("Running ICP...")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        flame_pcd, pixal_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000)
    )
    print("ICP Transformation:")
    print(reg_p2p.transformation)
    
    flame_mesh.transform(reg_p2p.transformation)
    o3d.io.write_triangle_mesh('results/test_pixal3d/debug_flame_aligned.obj', flame_mesh)
    
    # 3. Load scalp indices
    scalp_idx = torch.load('assets/FLAME/NHC_scalp_vertex_idx.pth').numpy()
    
    # 4. Find nearest neighbor on FLAME
    flame_tree = o3d.geometry.KDTreeFlann(flame_mesh)
    pixal_vertices = np.asarray(pixal_mesh.vertices)
    keep_mask = np.ones(len(pixal_vertices), dtype=bool)
    
    is_scalp = np.zeros(len(flame_mesh.vertices), dtype=bool)
    is_scalp[scalp_idx] = True
    
    print("Cropping neck and shoulders...")
    for i in range(len(pixal_vertices)):
        y = pixal_vertices[i][1]
        # Keep everything above the neck (face, scalp, hair volume)
        if y <= 1.62:
            keep_mask[i] = False
            
    pixal_mesh.remove_vertices_by_mask(~keep_mask)
    pixal_mesh.remove_unreferenced_vertices()
        
    out_path = 'results/test_pixal3d/hair_mesh_flame_extracted.obj'
    o3d.io.write_triangle_mesh(out_path, pixal_mesh)
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
