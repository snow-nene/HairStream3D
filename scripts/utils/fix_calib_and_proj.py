import sys, os
import numpy as np
import cv2
import trimesh
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.recon_3d.recon3D import load_calib

def main():
    img_name = "0d285f5be7fa09c3dbbf1c9334047888"
    mesh_in = "results/test_pixal3d/hair_mesh_only_aligned_to_head.obj"
    
    # 1. Load correct calib matrix
    calib_path = f"results/real_imgs/param/{img_name}.npy"
    # HairStep uses loadSize=1024 by default
    calib_mat = load_calib(calib_path, loadSize=1024) 
    
    # 2. Load aligned mesh
    mesh = trimesh.load(mesh_in, process=False)
    verts = mesh.vertices
    verts_homo = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    
    # 3. Project to 2D
    proj = (calib_mat @ verts_homo.T).T
    
    # In HairStep, xy is in [-1, 1]. The image is 512x512.
    # To map [-1, 1] to [0, 512]:
    # x_pixel = (x + 1) / 2 * 512 = x * 256 + 256
    # NOTE: In HairStep's opt_lmk.py, x and y are swapped:
    # x_coor = xy[:, 1], y_coor = xy[:, 0]
    # AND y axis in image is downwards. Let's map it correctly:
    pts2d_x = proj[:, 1] * 256.0 + 255.0
    pts2d_y = proj[:, 0] * 256.0 + 255.0
    
    # 4. Generate debug image
    img = cv2.imread(f"results/real_imgs/resized_img/{img_name}.png")
    strand = cv2.imread(f"results/real_imgs/strand_map/{img_name}.png")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.scatter(pts2d_x, pts2d_y, s=0.1, c='r', alpha=0.1)
    plt.title("Correct Calib Mesh over Image")
    plt.xlim(0, 512); plt.ylim(512, 0)
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(strand, cv2.COLOR_BGR2RGB))
    plt.scatter(pts2d_x, pts2d_y, s=0.1, c='r', alpha=0.1)
    plt.title("Correct Calib Mesh over Strand")
    plt.xlim(0, 512); plt.ylim(512, 0)
    
    plt.tight_layout()
    plt.savefig("results/test_pixal3d/debug_mesh_projection_fixed.png", dpi=150)
    print("Saved fixed projection to results/test_pixal3d/debug_mesh_projection_fixed.png")

if __name__ == "__main__":
    main()
