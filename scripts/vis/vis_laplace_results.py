"""
Render Laplace 3D Hair Strands and Mesh previews using Open3D / Matplotlib.
"""

import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def render_laplace_sample(hair_ply_path, mesh_obj_path, save_png_path):
    fig = plt.figure(figsize=(12, 6))

    # ---- Subplot 1: 3D Hair Strands ----
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    if os.path.exists(hair_ply_path):
        lineset = o3d.io.read_line_set(hair_ply_path)
        pts = np.asarray(lineset.points)
        lines = np.asarray(lineset.lines)

        if len(lines) > 0 and len(pts) > 0:
            # Subsample lines for clean plotting
            step = max(1, len(lines) // 1000)
            for line in lines[::step]:
                p1, p2 = pts[line[0]], pts[line[1]]
                ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='#8b4513', alpha=0.6, linewidth=0.8)
    
    ax1.set_title("Laplace PDE 3D Hair Strands")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=15, azim=-75)

    # ---- Subplot 2: 3D Coarse Mesh ----
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if os.path.exists(mesh_obj_path):
        mesh = o3d.io.read_triangle_mesh(mesh_obj_path)
        verts = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        if len(verts) > 0 and len(triangles) > 0:
            ax2.plot_trisurf(
                verts[:, 0], verts[:, 1], verts[:, 2],
                triangles=triangles,
                color='#e0ac69', alpha=0.7, edgecolor='none'
            )

    ax2.set_title("Laplace PDE 3D Mesh (Occupancy)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=15, azim=-75)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_png_path), exist_ok=True)
    plt.savefig(save_png_path, dpi=150)
    plt.close()
    print(f"Rendered preview saved to {save_png_path}")

if __name__ == '__main__':
    mesh_dir = "results/real_imgs/mesh_laplace"
    hair_dir = "results/real_imgs/hair3D_laplace"
    out_dir = "results/rendered_previews_laplace"

    items = [f for f in os.listdir(hair_dir) if f.endswith('.ply')]
    print(f"Found {len(items)} reconstructed samples in {hair_dir}")

    for item in items[:3]:  # Render first 3 samples
        sample_name = item[:-4]
        hair_path = os.path.join(hair_dir, sample_name + ".ply")
        mesh_path = os.path.join(mesh_dir, sample_name + ".obj")
        save_path = os.path.join(out_dir, sample_name + "_preview.png")

        render_laplace_sample(hair_path, mesh_path, save_path)
