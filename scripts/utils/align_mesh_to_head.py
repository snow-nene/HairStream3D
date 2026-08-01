import sys, os
import numpy as np
import cv2
import trimesh
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lib.mesh_util import load_obj_mesh
from lib.util.opt_lmk import load_point_ids
from scripts.utils.align_glb_lmk import get_lmk, backproject_render_lmk

def umeyama(P, Q):
    assert P.shape == Q.shape
    n, m = P.shape
    mean_P = P.mean(axis=0)
    mean_Q = Q.mean(axis=0)
    P_centered = P - mean_P
    Q_centered = Q - mean_Q
    var_P = np.var(P, axis=0).sum()
    cov_matrix = P_centered.T @ Q_centered / n
    U, D, Vt = np.linalg.svd(cov_matrix)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m-1, m-1] = -1
    R = U @ S @ Vt
    c = np.trace(np.diag(D) @ S) / var_P
    t = mean_Q - c * R @ mean_P
    return c, R, t

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_name", default="0d285f5be7fa09c3dbbf1c9334047888")
    parser.add_argument("--mesh_in", default="results/test_pixal3d/output_mesh.ply")
    parser.add_argument("--mesh_out", default="results/test_pixal3d/hair_mesh_aligned_to_head.obj")
    args = parser.parse_args()

    print("[1/5] Extracting 3D landmarks of Pixal3D mesh...")
    render_bgr = cv2.imread("results/test_pixal3d/render_front.png")
    cam_npz = np.load("results/test_pixal3d/render_camera.npz")
    pixal_lmk_2d = get_lmk(render_bgr)
    if pixal_lmk_2d is None: raise ValueError("No face detected in render_front.png")
    pixal_lmk = backproject_render_lmk(pixal_lmk_2d[:, :2], cam_npz)

    print("[2/5] Loading 3D landmarks of head_model.obj...")
    head_verts, _ = load_obj_mesh("data/head_model.obj")
    landmark_ids = load_point_ids("data/landmark_id_uschair.obj")
    head_lmk = head_verts[landmark_ids]

    print("[3/5] Computing Procrustes alignment (Pixal3D -> head_model.obj)...")
    c, R, t = umeyama(pixal_lmk, head_lmk)
    
    print("[4/5] Applying transform to Pixal3D mesh...")
    mesh = trimesh.load(args.mesh_in, process=False)
    mesh.vertices = c * (mesh.vertices @ R.T) + t
    mesh.export(args.mesh_out)

    print("[5/5] Generating debug projection image...")
    img = cv2.imread(f"results/real_imgs/resized_img/{args.img_name}.png")
    strand = cv2.imread(f"results/real_imgs/strand_map/{args.img_name}.png")
    calib_dict = np.load(f"results/real_imgs/param/{args.img_name}.npy", allow_pickle=True).item()
    
    ortho_ratio = calib_dict['ortho_ratio']
    scale = calib_dict['scale']
    center = calib_dict['center']
    calib_R = calib_dict['R']
    
    translate = -np.matmul(calib_R, center).reshape(3, 1)
    extrinsic = np.concatenate([calib_R, translate], axis=1)
    extrinsic = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0)
    
    scale_intrinsic = np.eye(4)
    scale_intrinsic[0, 0] =  scale / ortho_ratio
    scale_intrinsic[1, 1] = -scale / ortho_ratio
    scale_intrinsic[2, 2] =  scale / ortho_ratio
    
    uv_intrinsic = np.eye(4)
    uv_intrinsic[0, 0] = 1.0 / 256.0
    uv_intrinsic[1, 1] = 1.0 / 256.0
    uv_intrinsic[2, 2] = 1.0 / 256.0
    
    calib_mat = uv_intrinsic @ scale_intrinsic @ extrinsic
    
    verts = mesh.vertices
    verts_homo = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    proj = (calib_mat @ verts_homo.T).T
    
    # HairStep projection swaps X and Y in the final coordinate computation!
    pts2d_x = proj[:, 1] * 256.0 + 255.0
    pts2d_y = proj[:, 0] * 256.0 + 255.0
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.scatter(pts2d_x, pts2d_y, s=0.1, c='r', alpha=0.1)
    plt.title("Mesh over Original Image")
    plt.xlim(0, 512); plt.ylim(512, 0)
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(strand, cv2.COLOR_BGR2RGB))
    plt.scatter(pts2d_x, pts2d_y, s=0.1, c='r', alpha=0.1)
    plt.title("Mesh over Strand Map")
    plt.xlim(0, 512); plt.ylim(512, 0)
    
    plt.tight_layout()
    plt.savefig("results/test_pixal3d/debug_mesh_projection.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    import torch; torch.set_grad_enabled(True)
    main()
