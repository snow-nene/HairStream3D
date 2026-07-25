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
            # Group lines into contiguous strands
            strands = []
            curr_strand = [lines[0][0], lines[0][1]]
            for i in range(1, len(lines)):
                if lines[i][0] == curr_strand[-1]:
                    curr_strand.append(lines[i][1])
                else:
                    strands.append(curr_strand)
                    curr_strand = [lines[i][0], lines[i][1]]
            strands.append(curr_strand)
            
            # Plot a subset of strands for visualization
            step = max(1, len(strands) // 300)
            for strand in strands[::step]:
                p = pts[strand]
                ax1.plot(p[:, 0], p[:, 1], p[:, 2], color='#8b4513', alpha=0.6, linewidth=0.8)
    
    ax1.set_title("RK4 PDE 3D Hair Strands")
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

    ax2.set_title("Aligned Pixal3D Mesh")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=15, azim=-75)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_png_path), exist_ok=True)
    plt.savefig(save_png_path, dpi=150)
    plt.close()
    print(f"Rendered preview saved to {save_png_path}")

render_laplace_sample(
    "results/test_pde_rk4/hair.ply",
    "results/test_pixal3d/hair_mesh_only_aligned_to_head.obj",
    "results/test_pde_rk4/hair_preview.png"
)
