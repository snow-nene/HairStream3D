import torch
import numpy as np
import os
import cv2
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)
from pytorch3d.ops import interpolate_face_attributes

def render_multiview_maps(mesh_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load mesh
    mesh = load_objs_as_meshes([mesh_path], device=device)
    
    # Define 5 views: Front, Back, Left, Right, Top
    views = {
        'front': (0, 0),
        'back': (0, 180),
        'left': (0, 90),
        'right': (0, -90),
        'top': (89.9, 0) # slightly off 90 to avoid singularity
    }
    
    dist = 0.4 # head is roughly ~0.2 units across, distance 0.4 is a good portrait crop
    
    # Calculate center of the mesh to point cameras at
    verts = mesh.verts_packed()
    center = (verts.max(dim=0)[0] + verts.min(dim=0)[0]) / 2.0
    center = center.unsqueeze(0).cpu() # [1, 3]
    
    for view_name, (elev, azim) in views.items():
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=center)
        R = R.to(device)
        T = T.to(device)
        
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=45.0)
        
        raster_settings = RasterizationSettings(
            image_size=512, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(mesh)
        
        # Depth
        depth = fragments.zbuf[0, :, :, 0] # [H, W]
        mask = (depth > 0).float()
        
        # Normals
        faces = mesh.faces_packed()
        vertex_normals = mesh.verts_normals_packed()
        faces_normals = vertex_normals[faces] # [num_faces, 3, 3]
        
        pix_to_face = fragments.pix_to_face
        bary_coords = fragments.bary_coords
        
        pixel_normals = interpolate_face_attributes(pix_to_face, bary_coords, faces_normals)
        pixel_normals = pixel_normals[0, :, :, 0, :] # [H, W, 3]
        
        # Transform normals from world space to camera space
        world_normals = pixel_normals.reshape(-1, 3)
        # R is World to Cam rotation
        cam_normals = torch.matmul(world_normals, R[0]) 
        cam_normals = cam_normals.reshape(512, 512, 3)
        
        # Normalize to [0, 1] for RGB
        cam_normals = torch.nn.functional.normalize(cam_normals, p=2, dim=-1)
        normal_rgb = (cam_normals + 1.0) / 2.0
        
        # Convert to numpy for saving
        depth_np = depth.cpu().numpy()
        mask_np = mask.cpu().numpy()
        normal_rgb_np = normal_rgb.cpu().numpy()
        
        # Normalize depth for visualization / ControlNet
        depth_norm = depth_np.copy()
        if depth_norm.max() > 0:
            valid_depth = depth_norm[depth_norm > 0]
            depth_norm[depth_norm > 0] = (valid_depth - valid_depth.min()) / (valid_depth.max() - valid_depth.min() + 1e-8)
            
        depth_img = (depth_norm * 255 * mask_np).astype(np.uint8)
        normal_img = (normal_rgb_np * 255 * mask_np[:, :, None]).astype(np.uint8)
        
        # ControlNet standardizes normals: R=X(right), G=Y(up/down), B=Z(forward)
        # OpenCV uses BGR. So we convert RGB to BGR.
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(out_dir, f"{view_name}_depth.png"), depth_img)
        cv2.imwrite(os.path.join(out_dir, f"{view_name}_normal.png"), normal_img)
        cv2.imwrite(os.path.join(out_dir, f"{view_name}_mask.png"), (mask_np * 255).astype(np.uint8))
        
        # Save camera matrix
        cam_dict = {
            'R': R.cpu().numpy(),
            'T': T.cpu().numpy(),
            'K': cameras.get_projection_transform().get_matrix().cpu().numpy()
        }
        np.save(os.path.join(out_dir, f"{view_name}_cam.npy"), cam_dict)

    print(f"Saved renders to {out_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        mesh_path = sys.argv[1]
    else:
        mesh_path = "results/test_pixal3d/hair_mesh_flame_extracted.obj"
    render_multiview_maps(mesh_path, "results/multi_view_renders")
