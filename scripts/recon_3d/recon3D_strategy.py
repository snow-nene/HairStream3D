"""
Strategy-based 3D hair reconstruction pipeline.

Supports swapping between reconstruction backends via --recon_strategy flag:
    neural  (default): Original HairStep NeuralHDHair* method (PIFu neural network)
    laplace:           Training-free Laplace PDE physical field solver

Usage examples:
    # Run with original neural network (same as original recon3D.py)
    CUDA_VISIBLE_DEVICES=0 python -m scripts.recon_3d.recon3D_strategy --recon_strategy neural

    # Run with Laplace PDE solver (no pretrained weights needed)
    CUDA_VISIBLE_DEVICES=0 python -m scripts.recon_3d.recon3D_strategy --recon_strategy laplace

    # Laplace with custom PDE resolution
    CUDA_VISIBLE_DEVICES=0 python -m scripts.recon_3d.recon3D_strategy \
        --recon_strategy laplace \
        --pde_resolution 64 \
        --pde_dilation_iters 8 \
        --pde_anisotropy 0.85

    # Produce side-by-side comparison: run neural first, then laplace
    CUDA_VISIBLE_DEVICES=0 python -m scripts.recon_3d.recon3D_strategy --recon_strategy neural \
        --root_real_imgs ./results/real_imgs
    CUDA_VISIBLE_DEVICES=0 python -m scripts.recon_3d.recon3D_strategy --recon_strategy laplace \
        --root_real_imgs ./results/real_imgs
    # Outputs are stored in separate subdirs:
    #   ./results/real_imgs/mesh_neural/      ./results/real_imgs/hair3D_neural/
    #   ./results/real_imgs/mesh_laplace/     ./results/real_imgs/hair3D_laplace/
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision.transforms as transforms

from tqdm import tqdm
import imageio.v2 as imageio
from PIL import Image

from lib.options import BaseOptions
from lib.mesh_util import gen_mesh_strategy, load_obj_mesh
from lib.hair_util import export_hair_strategy
from lib.recon_strategy import get_strategy


# ---------------------------------------------------------------------------
# Data loading helpers (identical to original recon3D.py)
# ---------------------------------------------------------------------------

def load_hairstep(orien2d_path, depth_path, seg_path, load_size=512):
    """Load and concatenate strand_map + depth_map into 4-channel hairstep tensor."""
    raw_orien2d = Image.open(orien2d_path).convert('RGB')
    img_to_tensor = transforms.Compose([
        transforms.Resize(load_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    orien2d = img_to_tensor(raw_orien2d).float()  # [3, H, W] in [-1, 1]

    out_mask = ((np.array(imageio.imread(seg_path)) / 255) < 0.5)
    if len(out_mask.shape) == 3:
        out_mask = out_mask[:, :, 0]
    depth = np.load(depth_path)
    depth = depth + out_mask * opt.depth_out_mask  # background gets -3.0
    depth = torch.from_numpy(depth).float()        # [H, W]

    hairstep = torch.cat([orien2d, depth.unsqueeze(0)], dim=0)  # [4, H, W]
    return hairstep


def load_calib(calib_path, loadSize=1024):
    """Load camera calibration parameters and build 4x4 calibration matrix."""
    param = np.load(calib_path, allow_pickle=True)
    ortho_ratio = param.item().get('ortho_ratio')
    scale       = param.item().get('scale')
    center      = param.item().get('center')
    R           = param.item().get('R')

    translate = -np.matmul(R, center).reshape(3, 1)
    extrinsic = np.concatenate([R, translate], axis=1)
    extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

    scale_intrinsic = np.identity(4)
    scale_intrinsic[0, 0] =  scale / ortho_ratio
    scale_intrinsic[1, 1] = -scale / ortho_ratio
    scale_intrinsic[2, 2] =  scale / ortho_ratio

    uv_intrinsic = np.identity(4)
    uv_intrinsic[0, 0] = 1.0 / float(loadSize // 2)
    uv_intrinsic[1, 1] = 1.0 / float(loadSize // 2)
    uv_intrinsic[2, 2] = 1.0 / float(loadSize // 2)

    trans_intrinsic = np.identity(4)
    intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
    calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
    return calib


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def recon3D_strategy(opt):
    """
    3D hair reconstruction using the selected strategy.

    Output directories are named with the strategy suffix for easy comparison:
        mesh_{strategy}/     — coarse 3D hair mesh (.obj)
        hair3D_{strategy}/   — final 3D hair strands (.ply)
    """
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    strategy_name = opt.recon_strategy  # 'neural' or 'laplace'
    print(f'\n[recon3D_strategy] Using strategy: {strategy_name}')

    # ---- instantiate the chosen strategy ----
    strategy = get_strategy(strategy_name, opt, cuda)

    # ---- directory layout ----
    seg_dir    = os.path.join(opt.root_real_imgs, 'seg')
    depth_dir  = os.path.join(opt.root_real_imgs, 'depth_map')
    strand_dir = os.path.join(opt.root_real_imgs, 'strand_map')
    calib_dir  = os.path.join(opt.root_real_imgs, 'param')

    # Strategy-specific output directories (allows side-by-side comparison!)
    output_mesh_dir   = os.path.join(opt.root_real_imgs, f'mesh_{strategy_name}')
    output_hair3D_dir = os.path.join(opt.root_real_imgs, f'hair3D_{strategy_name}')
    os.makedirs(output_mesh_dir,   exist_ok=True)
    os.makedirs(output_hair3D_dir, exist_ok=True)

    items = os.listdir(strand_dir)
    print(f'[recon3D_strategy] Processing {len(items)} samples from {strand_dir}')

    with torch.no_grad():
        for item in tqdm(items):
            strand_path = os.path.join(strand_dir, item[:-3] + 'png')
            seg_path    = os.path.join(seg_dir,    item[:-3] + 'png')
            mesh_path   = os.path.join(output_mesh_dir,   item[:-3] + 'obj')
            hair3D_path = os.path.join(output_hair3D_dir, item[:-3] + 'ply')
            calib_path  = os.path.join(calib_dir,  item[:-3] + 'npy')
            depth_path  = os.path.join(depth_dir,  item[:-3] + 'npy')

            if os.path.exists(hair3D_path):
                continue

            # Load 2D inputs
            calib    = load_calib(calib_path)
            hairstep = load_hairstep(strand_path, depth_path, seg_path, opt.loadSize)
            data     = {'hairstep': hairstep, 'calib': calib}

            # ---- Strategy: filter (one-time per sample, shared by occ + orien) ----
            strategy.filter(data)

            # ---- Step 1: Generate coarse occupancy mesh ----
            gen_mesh_strategy(opt, strategy, cuda, data, mesh_path)

            # ---- Step 2: Grow 3D hair strands ----
            if os.path.exists(mesh_path):
                export_hair_strategy(strategy, cuda, data, mesh_path, hair3D_path)
            else:
                print(f'  [skip] mesh not generated for {item}, skipping strand growth.')

    print(f'\n[recon3D_strategy] Done. Results saved to:')
    print(f'  Meshes:  {output_mesh_dir}')
    print(f'  Strands: {output_hair3D_dir}')


if __name__ == '__main__':
    opt = BaseOptions().parse()
    recon3D_strategy(opt)
