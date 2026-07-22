import os
import torch
import json
import numpy as np

try:
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        FoVOrthographicCameras,
    )
    from pytorch3d.structures import Meshes, Curves
    from pytorch3d.renderer import (
        MeshRasterizer,
        RasterizationSettings,
    )
    from pytorch3d.renderer.curve import (
        CurveFragments,
        rasterize_curves,
    )
    HAS_PYTORCH3D = True
except Exception as e:
    HAS_PYTORCH3D = False
    print(f"[Warning] PyTorch3D import failed: {e}")

class SimpleCamera:
    def __init__(self, R, T, K=None):
        self.R = R
        self.T = T
        self.K = K

def load_cameras(camera_path, device="cpu"):
    if camera_path.endswith(".json"):
        with open(camera_path) as f:
            cam_params = json.load(f)
        if HAS_PYTORCH3D:
            R, T = look_at_view_transform(
                eye=(cam_params["position"],),
                at=(cam_params["look_at"],),
                up=(cam_params["up"],),
                device=device,
            )
            cameras = FoVPerspectiveCameras(
                R=R,
                T=T,
                fov=cam_params["fov"],
                device=device,
            )
            return cameras
        else:
            return SimpleCamera(torch.eye(3, device=device).unsqueeze(0), torch.zeros(1, 3, device=device))
    
    if camera_path.endswith(".npy"):
        cam_params = np.load(camera_path, allow_pickle=True).item()
        scale = cam_params["scale"][0]
        ortho_ratio = cam_params["ortho_ratio"]
        center = torch.tensor(cam_params["center"])
        
        R = torch.tensor(cam_params["R"])
        T = - (R @ center).reshape(3, 1)
        
        K = torch.eye(3)
        K[2, 2] = -K[2, 2]
        K[0, 0] = -K[0, 0]
        K *= scale / ortho_ratio / 512
        
        R = K @ R
        T = K @ T
        T[2, 0] += 1
        
        if HAS_PYTORCH3D:
            cameras = FoVOrthographicCameras(
                R=R.T.unsqueeze(0),
                T=T.T,
                K=torch.eye(4).unsqueeze(0),
                device=device,
            )
        else:
            cameras = SimpleCamera(R=R.T.unsqueeze(0).to(device), T=T.T.to(device), K=torch.eye(4, device=device).unsqueeze(0))
        return cameras

    

def transform_points_to_ndc(cameras, points_world):
    # NOTE: Retaining view space z coordinate for now.
    if hasattr(cameras, "get_world_to_view_transform"):
        points_view = cameras.get_world_to_view_transform().transform_points(points_world)
        to_ndc_transform = cameras.get_ndc_camera_transform()
        points_proj = cameras.transform_points(points_world)
        points_ndc = to_ndc_transform.transform_points(points_proj)
        points_ndc[..., 2] = points_view[..., 2]
        return points_ndc
    else:
        # Fallback projection transform using camera R and T
        R, T = cameras.R[0], cameras.T[0]
        pts_cam = points_world @ R + T
        return pts_cam


def transform_to_ndc(cameras, curves_world):
    points_world = curves_world.points_packed()
    points_ndc = transform_points_to_ndc(cameras, points_world)
    curves_ndc = curves_world.copy().update_packed(points_ndc)
    return curves_ndc


def render_meshes_zbuf(meshes, cameras, image_size):
    rasterizer = MeshRasterizer(
        cameras, RasterizationSettings(image_size=image_size, blur_radius=0, faces_per_pixel=1)
    ).to(meshes.device)
    fragments = rasterizer(meshes)
    zbuf = fragments.zbuf.reshape(image_size[0], image_size[1])
    zbuf[zbuf == -1] = np.inf
    return zbuf

def filter_fragments_zbuf(fragments, zbuf: torch.Tensor):
    mask = (fragments.zbuf > zbuf[..., None])
    fragments.dists[mask] = -1
    fragments.zbuf[mask] = -1
    fragments.bary_coords[mask] = -1
    
    return fragments


def curve_softmax_rgb_blend(
    colors: torch.Tensor,
    fragments,
    znear: float = 1.0,
    zfar: float = 100,
    sigma: float = 1e-6,
    gamma: float = 1e-6,
    background_color: torch.Tensor = 0,
) -> torch.Tensor:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
        colors: (H, W, K, C) RGB color for each of the top K lines per pixel.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_line: LongTensor of shape (H, W, K) specifying the indices
              of the lines (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping lines.
            - zbuf: FloatTensor of shape (H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping lines.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction
        sigma: float, parameter which controls the width of the sigmoid
          function used to calculate the 2D distance based probability.
          Sigma controls the sharpness of the edges of the shape.
        gamma: float, parameter which controls the scaling of the
          exponential function used to control the opacity of the color.
        background_color: (C) element list/tuple/torch.Tensor specifying
          the RGB values for the background color.

    Returns:
        RGBA pixel_colors: (H, W, 4)

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """
    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = fragments.dists >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = 2 * torch.sigmoid(-fragments.dists / sigma) * mask
    # alpha = 1 - torch.prod((1.0 - prob_map), dim=-1)
    alpha = torch.tanh(prob_map.mean(dim=-1) * 16)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / gamma)

    # Also apply exp normalize trick for the background color weight.
    # Clamp to ensure delta is never 0.
    delta = torch.exp((eps - z_inv_max) / gamma).clamp(min=eps)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta

    # Sum: weights * textures + background color
    weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
    weighted_background = delta * background_color
    pixel_colors = (weighted_colors + weighted_background) / denom
    pixel_colors = torch.cat([pixel_colors, alpha[..., None]], dim=-1)

    return pixel_colors


def calc_orientation_color(curves_proj, fragments, clump_scale=None):
    strands = curves_proj.points_packed().reshape(-1, 32, 3)
    lines_packed = curves_proj.lines_packed()
    direction = strands[:, 1:, :] - strands[:, :-1, :]
    tangent = torch.cat([direction, direction[:, -1:, :]], dim=1)
    tangent[..., 0] *= -1
    tangent[..., 2] = 0
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-10)
    tangent = tangent * 0.5 + 0.5
    if clump_scale is not None:
        tangent[..., 2] = clump_scale[..., 0]

    H, W, K = fragments.pix_to_line.shape
    pix_to_verts = lines_packed[fragments.pix_to_line]  # (H, W, K, 2)
    # use gather to accelerate backward
    idx = pix_to_verts.reshape(-1, 1).repeat(1, 3)
    colors = tangent.reshape(-1, 3).gather(0, idx).reshape(H, W, K, 2, 3)
    # colors = tangent.reshape(-1, 3)[pix_to_verts]
    colors = (
            fragments.bary_coords[..., None] * colors[:, :, :, 0, :]
            + (1 - fragments.bary_coords[..., None]) * colors[:, :, :, 1, :]
        )
    colors[fragments.dists < 0] = 0
    return colors


import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from src.utils.gaussian_utils import strands_to_gaussians

def render_feature_map(strands, cameras, img_size, mesh_zbuf=None, clump_scale=None, features_dc=None, semantic_label=None, hair_model=None):
    W, H = img_size
    camera = cameras[0] if isinstance(cameras, list) else cameras

    P = strands.shape[1]
    if P > 35:
        step = max(2, P // 33)
        strands = strands[:, ::step, :]
        if clump_scale is not None:
            clump_scale = clump_scale[:, ::step, :]
        if features_dc is not None:
            # features_dc has shape (N * (P_old-1), 3)
            # Since segments are step size, we need to subsample features_dc
            N = strands.shape[0]
            M_old = features_dc.shape[0]
            P_old = M_old // N + 1
            features_dc = features_dc.reshape(N, P_old - 1, 3)
            features_dc = features_dc[:, ::step, :].reshape(-1, 3)

    device = strands.device
    
    R = camera.R[0] # (3, 3)
    T = camera.T[0] # (3,)
    viewmatrix = torch.eye(4, device=device)
    viewmatrix[:3, :3] = R
    viewmatrix[3, :3] = T
    
    is_ortho = not hasattr(camera, 'focal_length')
    
    if is_ortho:
        # Simulate orthographic using perspective by moving camera far away
        D = 10000.0
        viewmatrix[3, 2] += D
        
        proj_trans = camera.get_projection_transform().get_matrix()[0]
        fx = abs(proj_trans[0, 0].item() * D)
        fy = abs(proj_trans[1, 1].item() * D)
        
        znear = D - 100.0
        zfar = D + 100.0
        P_persp = torch.zeros(4, 4, device=device)
        P_persp[0, 0] = fx
        P_persp[1, 1] = fy
        P_persp[2, 3] = 1.0 
        P_persp[2, 2] = zfar / (zfar - znear)
        P_persp[3, 2] = -(zfar * znear) / (zfar - znear)
        
        projmatrix = viewmatrix @ P_persp
    else:
        proj_trans = camera.get_projection_transform().get_matrix()[0]
        projmatrix = viewmatrix @ proj_trans
        fx = camera.focal_length[0, 0].item() if camera.focal_length.shape[1] > 1 else camera.focal_length[0].item()
        fy = camera.focal_length[0, 1].item() if camera.focal_length.shape[1] > 1 else camera.focal_length[0].item()

    flip_xy = torch.tensor([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    projmatrix = projmatrix @ flip_xy
    
    tanfovx = 1.0 / fx
    tanfovy = 1.0 / fy
    
    camera.image_height = H
    camera.image_width = W
    camera.FoVx = 2 * math.atan(tanfovx)
    camera.FoVy = 2 * math.atan(tanfovy)
    camera.world_view_transform = viewmatrix
    camera.full_proj_transform = projmatrix
    
    scale_cam = torch.norm(camera.R[0][:, 0]).item()
    means3D, scales, rotations, opacities, colors_precomp, conic_precomp, means2D_precomp = strands_to_gaussians(
        strands, clump_scale=clump_scale, radius=0.002/scale_cam, length_scale=0.015/scale_cam, camera=camera, features_dc=features_dc, semantic_label=semantic_label)
        
    if hair_model is not None and hair_model.has_noise_components:
        # Construct Unstructured Noise Components
        num_noise = hair_model.noise_xyz.shape[0]
        device = means3D.device
        
        n_means3D = hair_model.noise_xyz
        n_scales = torch.exp(hair_model.noise_scales)
        n_rotations = torch.nn.functional.normalize(hair_model.noise_rotations)
        n_opacities = torch.sigmoid(hair_model.noise_opacities)
        
        n_colors_precomp = torch.zeros((num_noise, 10), device=device)
        n_colors_precomp[:, 0:3] = hair_model.noise_features
        n_colors_precomp[:, 3] = 1.0 # Semantic Label 1.0 (Noise)
        n_colors_precomp[:, 5] = 1.0 # Silh
        
        # Computing Depth
        viewmatrix = camera.world_view_transform
        t = (n_means3D[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        # Depth is populated by rasterizer, but we can also manually put it in 4 if we want, wait, the standard strands put it in 4 just before rasterizer!
        n_colors_precomp[:, 6:9] = 0.0 # Cov2D
        n_colors_precomp[:, 8] = hair_model.noise_clump.view(-1)
        
        means3D = torch.cat([means3D, n_means3D], dim=0)
        scales = torch.cat([scales, n_scales], dim=0)
        rotations = torch.cat([rotations, n_rotations], dim=0)
        opacities = torch.cat([opacities, n_opacities], dim=0)
        colors_precomp = torch.cat([colors_precomp, n_colors_precomp], dim=0)
        
        conic_precomp = None
        means2D_precomp = None
    
    campos = camera.get_camera_center()[0]
    if is_ortho:
        # adjust campos based on viewmatrix change
        campos = campos - camera.R[0][:, 2] * D
    
    tanfovx = 1.0 / fx
    tanfovy = 1.0 / fy
    
    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx, 
        tanfovy=tanfovy,
        bg=torch.tensor([0.0]*10, device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    view_z = (means3D @ R + T)[:, 2]
    colors_precomp[:, 4] = view_z
    
    means2D = means2D_precomp if means2D_precomp is not None else torch.zeros_like(means3D[:, :2])
    
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        shs=None,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        conic_precomp=conic_precomp
    )
    
    image_rgb = rendered_image[0:3].permute(1, 2, 0) # (H, W, 3)
    image_label = rendered_image[3] # (H, W)
    image_silh = rendered_image[12] # (H, W)
    mask_div = image_silh + 1e-6
    
    image_depth = rendered_image[4] / mask_div # (H, W)
    image_orien = torch.zeros((img_size[1], img_size[0], 3), device=rendered_image.device)
    image_orien[..., 0:2] = (rendered_image[10:12] / mask_div).permute(1, 2, 0)
    image_cov = (rendered_image[6:9] / mask_div).permute(1, 2, 0) # (H, W, 3)
    image_clump = rendered_image[9] / mask_div # (H, W)
    mask = image_silh > 1e-4
    if mesh_zbuf is not None:
        mask = mask & (image_depth < mesh_zbuf)
        image_silh = image_silh * mask.float()
        image_orien = image_orien * mask.float().unsqueeze(-1)
        image_rgb = image_rgb * mask.float().unsqueeze(-1)
        image_cov = image_cov * mask.float().unsqueeze(-1)
        image_clump = image_clump * mask.float()
        image_label = image_label * mask.float()

    if mask.any():
        zbuf_min = image_depth[mask].min().item()
        zbuf_max = image_depth[mask].max().item()
        zbuf_range = max(zbuf_max - zbuf_min, 1e-5)
        image_depth = (image_depth - zbuf_min) / zbuf_range
        image_depth = 1 - image_depth
    image_depth = image_depth.clamp(0, 1) * mask.float()
    
    return image_silh, image_depth, image_cov, image_clump, image_rgb, image_label, image_orien
def render_feature_map_pt3d(config, curves, cameras, img_size, mesh_zbuf=None, clump_scale=None):
    from src.utils.render_utils import transform_to_ndc
    curves_ndc = transform_to_ndc(cameras, curves)
    fragments = CurveFragments(
            *rasterize_curves(
                curves_ndc,
                image_size=img_size,
                blur_radius=(config["blur_radius"] / img_size[0]) ** 2,
                lines_per_pixel=config["lines_per_pixel"],
                bin_size=config["bin_size"],
                perspective_correct=False,
                clip_barycentric_coords=True,
            )
        )
    if mesh_zbuf is not None:
        from src.utils.render_utils import filter_fragments_zbuf
        fragments = filter_fragments_zbuf(fragments, mesh_zbuf)
        
    from src.utils.render_utils import calc_orientation_color
    orien_colors = calc_orientation_color(curves_ndc, fragments, clump_scale)
    image = curve_softmax_rgb_blend(
            torch.cat([orien_colors, fragments.zbuf[..., None]], dim=-1),
            fragments,
            sigma=config["sigma"],
            gamma=config["gamma"],
        )
    image_silh = image[..., -1]
    image_feat = image[..., :-1]
    image_orien = image_feat[..., :3]
    
    image_depth = image_feat[..., 3]
    if (fragments.zbuf > 0).any():
        zbuf_min = fragments.zbuf[fragments.zbuf > 0].min().item()
        zbuf_range = fragments.zbuf.max().item() - zbuf_min
        if zbuf_range > 0:
            image_depth = (image_depth - zbuf_min) / zbuf_range
    image_depth = 1 - image_depth
    image_depth[image_depth > 1] = 0
    
    return image_silh, image_depth, image_orien

