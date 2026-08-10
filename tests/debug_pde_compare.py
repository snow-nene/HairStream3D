"""
Debug 诊断脚本：对比 run_pde_generation.py vs run_pde_multiview.py
输出到 results/multiview_data/<img_id>/debug_log/
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.getcwd()))

import torch
import numpy as np
import cv2
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_ID = "0a1ba3dbefc8934ab60577c5c91f66a0"
DATA_DIR = f"results/multiview_data/{IMG_ID}"
DEBUG_DIR = f"{DATA_DIR}/debug_log"
os.makedirs(DEBUG_DIR, exist_ok=True)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

def save_log():
    with open(os.path.join(DEBUG_DIR, "debug_log.txt"), "w") as f:
        f.write("\n".join(log_lines))

def save_heatmap(arr2d, path, title="", cmap="viridis", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(arr2d, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

def save_quiver(dx, dy, mask, path, title="", step=20):
    H, W = dx.shape
    ys, xs = np.mgrid[step//2:H:step, step//2:W:step]
    u = dx[ys, xs]; v = dy[ys, xs]; m = mask[ys, xs]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.quiver(xs[m>0], ys[m>0], u[m>0], -v[m>0], scale=20, scale_units="inches", width=0.003, color="red")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()

# ==== STEP 1 ====
log("="*60)
log("STEP 1: Load input data")
strand_png = os.path.join(DATA_DIR, "maps", "strand_map", "front.png")
depth_npy  = os.path.join(DATA_DIR, "maps", "depth_map", "front.npy")
strand_bgr = cv2.imread(strand_png)
strand_f   = strand_bgr.astype(np.float32) / 255.0 * 2.0 - 1.0
depth_map  = np.load(depth_npy).astype(np.float32)
log(f"  strand shape={strand_bgr.shape}, depth range={depth_map.min():.3f}~{depth_map.max():.3f}")
log(f"  depth>0.05 pixels: {(depth_map>0.05).sum()}")
save_heatmap(strand_f[:,:,0], f"{DEBUG_DIR}/01_B_dx_raw.png", "B channel (dx) raw [-1,1]", "RdBu", -1, 1)
save_heatmap(strand_f[:,:,1], f"{DEBUG_DIR}/01_G_dy_raw.png", "G channel (dy) raw [-1,1]", "RdBu", -1, 1)
save_heatmap(depth_map,       f"{DEBUG_DIR}/01_depth.png",    "Depth [0,1]", "plasma")

# ==== STEP 2 ====
log("\nSTEP 2: strand_dx/dy")
strand_dx = -strand_f[:,:,0]
strand_dy = -strand_f[:,:,1]
mask = (depth_map > 0.05).astype(np.float32)
log(f"  dx range: {strand_dx.min():.3f}~{strand_dx.max():.3f}")
log(f"  dy range: {strand_dy.min():.3f}~{strand_dy.max():.3f}")
log(f"  mask px: {int(mask.sum())}")
save_quiver(strand_dx, strand_dy, mask, f"{DEBUG_DIR}/02_direction_quiver.png", "2D strand directions")

# ==== STEP 3 ====
log("\nSTEP 3: Divergence")
dx_dx = cv2.Sobel(strand_dx, cv2.CV_32F, 1, 0, ksize=3)
dy_dy = cv2.Sobel(strand_dy, cv2.CV_32F, 0, 1, ksize=3)
divergence = cv2.GaussianBlur((dx_dx + dy_dy) * mask, (5,5), 0)
d_std = divergence[mask>0].std() if (mask>0).any() else 1.0
log(f"  div range: {divergence.min():.4f}~{divergence.max():.4f}, std={d_std:.4f}")
save_heatmap(divergence, f"{DEBUG_DIR}/03_divergence.png", "Divergence (red=spread, blue=clump)", "RdBu", -d_std*3, d_std*3)

# ==== STEP 4 ====
log("\nSTEP 4: Calibration")
calib_path = os.path.join(DATA_DIR, "maps", "param", "front.npy")
has_real = os.path.exists(calib_path)
log(f"  Real calib exists: {has_real}")
from lib.multiview_fusion import build_blender_calib
hair_mesh = o3d.io.read_triangle_mesh(f"{DATA_DIR}/pixal3d/hair_mesh_sdf.obj")
b_center = hair_mesh.get_axis_aligned_bounding_box().get_center()
calib_fb, _ = build_blender_calib("front", b_center, 0.3)
log(f"  Fallback center: {b_center}")
log(f"  Fallback calib:\n{calib_fb}")
if has_real:
    from scripts.recon_3d.recon3D import load_calib
    calib_real = load_calib(calib_path, loadSize=1024)
    cr = calib_real.numpy() if hasattr(calib_real, "numpy") else calib_real
    log(f"  Real calib:\n{cr}")
    calib_use = cr
else:
    calib_use = calib_fb

# ==== STEP 5 ====
log("\nSTEP 5: Root projection")
from lib.hair_util import get_hair_root
cuda = torch.device("cuda:0")
roots_np = get_hair_root("data/roots10k.obj")
N = roots_np.shape[1]
root_t  = torch.from_numpy(roots_np).float().unsqueeze(0).to(cuda)
calib_t = torch.from_numpy(calib_use).float().unsqueeze(0).to(cuda)
r3d = root_t.squeeze(0)
homo = torch.cat([r3d, torch.ones(1, N, device=cuda)], dim=0)
uv = torch.matmul(calib_t.squeeze(0), homo)
uv_ndc = uv[:2] / (uv[3:4] + 1e-8)
log(f"  uv_ndc X: {uv_ndc[0].min().item():.3f}~{uv_ndc[0].max().item():.3f}")
log(f"  uv_ndc Y: {uv_ndc[1].min().item():.3f}~{uv_ndc[1].max().item():.3f}")
uv_px = ((uv_ndc + 1.0) * 0.5 * 511).long().clamp(0, 511).cpu().numpy()
in_mask = mask[uv_px[1], uv_px[0]]
n_in = int((in_mask > 0).sum())
log(f"  Roots inside mask: {n_in}/{N}")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(strand_bgr[:,:,::-1])
xs_on = uv_px[0][in_mask>0]; ys_on = uv_px[1][in_mask>0]
xs_off = uv_px[0][in_mask==0]; ys_off = uv_px[1][in_mask==0]
axes[0].scatter(xs_on,  ys_on,  c="lime", s=0.5, alpha=0.5, label=f"in-mask ({len(xs_on)})")
axes[0].scatter(xs_off, ys_off, c="red",  s=0.3, alpha=0.3, label=f"out-mask ({len(xs_off)})")
axes[0].set_title(f"Root projection (real_calib={has_real})"); axes[0].legend(markerscale=5)
axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Hair mask (depth>0.05)")
plt.tight_layout(); plt.savefig(f"{DEBUG_DIR}/05_root_projection.png", dpi=120); plt.close()

# ==== STEP 6 ====
log("\nSTEP 6: Orientation volume comparison")
paths = {
    "generation": "results/test_pde_rk4/debug_orien_vol.npy",
    "multiview":  os.path.join(DATA_DIR, "pde_reconstruction", "debug_orien_vol.npy"),
}
vols = {}
for name, p in paths.items():
    if os.path.exists(p):
        v = np.load(p)
        vols[name] = v
        mag = np.linalg.norm(v, axis=0)
        log(f"  {name}: shape={v.shape}, mean_mag={mag.mean():.4f}, nonzero(>0.01)={(mag>0.01).sum()}")
        R = v.shape[1]; mid = R//2
        fig, axes = plt.subplots(1, 3, figsize=(15,5))
        for ci, ch in enumerate(["X","Y","Z"]):
            im = axes[ci].imshow(v[ci,:,:,mid], cmap="RdBu", vmin=-1, vmax=1, origin="upper")
            axes[ci].set_title(f"{name} {ch} (Z mid)"); plt.colorbar(im, ax=axes[ci])
        plt.tight_layout(); plt.savefig(f"{DEBUG_DIR}/06_orien_{name}_Zmid.png", dpi=120); plt.close()
        save_heatmap(mag[:,:,mid], f"{DEBUG_DIR}/06_orien_{name}_mag.png", f"{name} magnitude (Z mid)", "hot", 0, 1)
    else:
        log(f"  {name}: NOT FOUND at {p}")

if "generation" in vols and "multiview" in vols:
    g, m_ = vols["generation"], vols["multiview"]
    if g.shape == m_.shape:
        diff = np.abs(g - m_)
        log(f"  Max diff: {diff.max():.4f}, Mean diff: {diff.mean():.6f}")
        R = g.shape[1]; mid = R//2
        fig, axes = plt.subplots(1, 3, figsize=(15,5))
        for ci, ch in enumerate(["X","Y","Z"]):
            im = axes[ci].imshow(diff[ci,:,:,mid], cmap="hot", origin="upper")
            axes[ci].set_title(f"ABS DIFF {ch}"); plt.colorbar(im, ax=axes[ci])
        plt.tight_layout(); plt.savefig(f"{DEBUG_DIR}/06_orien_DIFF.png", dpi=120); plt.close()
    else:
        log(f"  Shape mismatch: gen={g.shape}, mv={m_.shape}, cannot diff")

# ==== STEP 7 ====
log("\nSTEP 7: KEY DIFFERENCES FOUND")
log("-"*60)
log("  run_pde_generation:")
log("    Collision SDF resolution: R=384 (hardcoded)")
log("    Kept 1576/10000 strands")
log("")
log("  run_pde_multiview:")
log("    Collision SDF resolution: R=args.pde_resolution=256")
log("    Kept 339/10000 strands")
log("")
log("  >>> SUSPECT #1: SDF resolution 256 vs 384")
log("     Coarser collision SDF => wrong penetration pushback => strands shortened => pruned")
log("  >>> SUSPECT #2: Calib fallback reads wrong strand_map path (out_dir vs maps/)")
log(f"     out_dir path tried: {DATA_DIR}/pde_reconstruction/strand_map/front.png")
log(f"     actual path:        {DATA_DIR}/maps/strand_map/front.png")
log("     => fallback falls through to bbox center => wrong calib => roots off-image")
log(f"     => roots in mask: {n_in}/{N} (should be ~7000+)")
log("-"*60)

# ==== STEP 8 ====
log("\nSTEP 8: Root SDF vs meshes")
for mp, mn in [(f"{DATA_DIR}/pixal3d/hair_mesh_sdf.obj", "hair_mesh_sdf"),
               ("data/head_model.obj", "head_model")]:
    mo = o3d.io.read_triangle_mesh(mp)
    tm = o3d.t.geometry.TriangleMesh.from_legacy(mo)
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(tm)
    sdf_v = sc.compute_signed_distance(
        o3d.core.Tensor(roots_np.T.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    ).numpy()
    n_inside = (sdf_v < 0).sum()
    log(f"  {mn}: inside={n_inside}/{N}, |sdf|<0.005={(np.abs(sdf_v)<0.005).sum()}")

save_log()
print(f"\nAll debug files saved to: {DEBUG_DIR}/")
