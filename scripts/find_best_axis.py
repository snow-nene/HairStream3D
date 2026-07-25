import sys, os
import numpy as np
import itertools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.align_mesh_to_head import umeyama
from lib.mesh_util import load_obj_mesh
from lib.util.opt_lmk import load_point_ids
from scripts.align_glb_lmk import get_lmk, backproject_render_lmk
import cv2

render_bgr = cv2.imread("results/test_pixal3d/render_front.png")
cam_npz = np.load("results/test_pixal3d/render_camera.npz")
pixal_lmk_2d = get_lmk(render_bgr)
pixal_lmk_blender = backproject_render_lmk(pixal_lmk_2d[:, :2], cam_npz)

head_verts, _ = load_obj_mesh("data/head_model.obj")
landmark_ids = load_point_ids("data/landmark_id_uschair.obj")
head_lmk = head_verts[landmark_ids]

# Find best permutation
best_error = float('inf')
best_perm = None
best_signs = None
best_c, best_R, best_t = None, None, None

permutations = list(itertools.permutations([0, 1, 2]))
signs = list(itertools.product([1, -1], repeat=3))

for perm in permutations:
    for sign in signs:
        test_lmk = np.zeros_like(pixal_lmk_blender)
        test_lmk[:, 0] = pixal_lmk_blender[:, perm[0]] * sign[0]
        test_lmk[:, 1] = pixal_lmk_blender[:, perm[1]] * sign[1]
        test_lmk[:, 2] = pixal_lmk_blender[:, perm[2]] * sign[2]
        
        c, R, t = umeyama(test_lmk, head_lmk)
        
        # Check if determinant of R is > 0
        if np.linalg.det(R) < 0:
            continue
            
        aligned = c * (test_lmk @ R.T) + t
        error = np.mean(np.linalg.norm(aligned - head_lmk, axis=1))
        
        if error < best_error:
            best_error = error
            best_perm = perm
            best_signs = sign
            best_c, best_R, best_t = c, R, t

print("Best error:", best_error)
print("Best permutation:", best_perm)
print("Best signs:", best_signs)

# Save the best aligned mesh
import trimesh
mesh_in = "results/test_pixal3d/output_mesh.ply"
mesh = trimesh.load(mesh_in, process=False)
mesh.vertices = best_c * (mesh.vertices @ best_R.T) + best_t
mesh.export("results/test_pixal3d/hair_mesh_aligned_best.obj")

# Now project it using the correct camera
from scripts.recon3D import load_calib
img_name = "0d285f5be7fa09c3dbbf1c9334047888"
calib_path = f"results/real_imgs/param/{img_name}.npy"
calib_mat = load_calib(calib_path, loadSize=1024) 

verts_homo = np.concatenate([mesh.vertices, np.ones((len(mesh.vertices), 1))], axis=1)
proj = (calib_mat @ verts_homo.T).T

pts2d_x = proj[:, 1] * 256.0 + 255.0
pts2d_y = proj[:, 0] * 256.0 + 255.0

import matplotlib.pyplot as plt
img = cv2.imread(f"results/real_imgs/resized_img/{img_name}.png")
plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.scatter(pts2d_x, pts2d_y, s=0.1, c='r', alpha=0.1)
plt.title(f"Best Perm: {best_perm}, Signs: {best_signs}")
plt.xlim(0, 512); plt.ylim(512, 0)
plt.tight_layout()
plt.savefig("results/test_pixal3d/debug_mesh_projection_best.png", dpi=150)
print("Saved debug_mesh_projection_best.png")
