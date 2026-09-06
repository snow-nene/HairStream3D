"""固定可见曲线与锚点，比较网格平滑及 Depth Pro 两种辅助方式。"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import map_coordinates, distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.recon3D import load_calib
from lib.hair_util import save_polyline_strands


def compare_sparse_depth_methods():
    base = Path("results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888")
    root = base / "pde_governance/sparse_guides"
    out = root / "step_09_method_comparison"
    out.mkdir(parents=True, exist_ok=True)
    guides = np.load(root / "step_01_front_curves/guides_pixel_depth.npz")
    fragments = np.load(root / "step_03_mesh_lift/mesh_guides_world.npz")
    evidence = np.load(root / "step_03_mesh_lift/raycast_evidence.npz")
    calibration = json.loads((root / "step_06_depth_fusion/report.json").read_text())
    depth = np.load(root / "step_05_matted_depth_pro/corrected_hair_depth.npy")
    confmap = np.load(root / "step_05_matted_depth_pro/depth_confidence.npy")
    seg = cv2.imread(str(base / "maps/seg/front.png"), 0) > 127
    border = distance_transform_edt(seg)
    camera = load_calib(str(base / "maps/param/front_dense_silhouette.npy")).numpy().astype(float)
    inverse = np.linalg.inv(camera)
    factor = np.linalg.norm(inverse[:3, 2])
    names = ("mesh_raw", "mesh_smooth", "mesh_depthpro_absolute", "mesh_depthpro_gradient")
    results = {name: {} for name in names}
    diagnostics = {name: {"hidden_errors_mm": [], "anchor_errors_mm": [], "depth_second_difference_mm": [], "steps_mm": [], "projection_errors_px": []} for name in names}
    hidden_count = 0
    excluded = 0
    for key in fragments.files:
        parent = key.split("_segment_")[0]
        ids = evidence[key + "_source_indices"]
        xy = guides[parent][ids, :2]
        world = fragments[key].astype(float)
        n = len(world)
        if n < 12:
            excluded += 1
            continue
        mono = map_coordinates(depth, [xy[:,1],xy[:,0]],order=1,mode="nearest")
        confidence = map_coordinates(confmap,[xy[:,1],xy[:,0]],order=1,mode="nearest")
        safe = (evidence[parent + "_incidence"][ids] > .3) & (map_coordinates(border,[xy[:,1],xy[:,0]],order=1) > 5)
        if safe.sum() < 8 or not np.isfinite(mono).all():
            excluded += 1
            continue
        z = (np.column_stack([world,np.ones(n)]) @ camera.T)[:,2]
        prior = mono * calibration["affine_scale_offset"][0] + calibration["affine_scale_offset"][1]
        # 仅原来整曲线留出集用于补全验证；每25点留出中间5点，防止逐点交错过于容易。
        hidden = safe & (np.arange(n)%25 >= 10) & (np.arange(n)%25 < 15)
        hidden[:3] = False; hidden[-3:] = False
        if parent not in calibration["validation_guides"]:
            hidden[:] = False
        if (safe & ~hidden).sum() < 4:
            hidden[:] = False
        hidden_count += int(hidden.sum())
        d1 = sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format="csc")
        d2 = sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format="csc")
        for name in names:
            if name == "mesh_raw":
                solved = z
            else:
                for validation_run in (True,False):
                    anchors = safe & ~hidden if validation_run else safe
                    weights = anchors.astype(float)*10
                    matrix = sparse.diags(weights+.000001) + 20*d2.T@d2
                    rhs = weights*z + .000001*np.median(z[anchors])
                    if name == "mesh_depthpro_absolute":
                        w = .2*confidence
                        matrix += sparse.diags(w)
                        rhs += w*prior
                    if name == "mesh_depthpro_gradient":
                        w = sparse.diags(.2*np.minimum(confidence[:-1],confidence[1:]))
                        matrix += d1.T@w@d1
                        rhs += d1.T@w@np.diff(prior)
                    solved = spsolve(matrix.tocsc(),rhs)
                    if validation_run:
                        diagnostics[name]["hidden_errors_mm"].extend((np.abs(solved[hidden]-z[hidden])*factor*1000).tolist())
            clip = np.column_stack([xy / [seg.shape[1]-1,seg.shape[0]-1]*2-1,solved,np.ones(n)])
            homogeneous = clip@inverse.T
            points = homogeneous[:,:3]/homogeneous[:,3:4]
            results[name][key] = points.astype(np.float32)
            projected = np.column_stack([points,np.ones(n)])@camera.T
            pixels = (projected[:,:2]/projected[:,3:4]+1)*[seg.shape[1]-1,seg.shape[0]-1]/2
            diagnostics[name]["projection_errors_px"].extend(np.linalg.norm(pixels-xy,axis=1).tolist())
            diagnostics[name]["anchor_errors_mm"].extend((np.abs(solved[safe]-z[safe])*factor*1000).tolist())
            diagnostics[name]["depth_second_difference_mm"].extend((np.abs(np.diff(solved,n=2))*factor*1000).tolist())
            diagnostics[name]["steps_mm"].extend((np.linalg.norm(np.diff(points,axis=0),axis=1)*1000).tolist())
    head = o3d.io.read_triangle_mesh("data/head_model.obj")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(head))
    report = {"hidden_samples":hidden_count,"excluded_fragments":excluded,"methods":{},
              "settings":{"anchor_weight":10,"smoothness_weight":20,"depthpro_weight":.2},
              "limitations":"仅当前样例固定可见片段的深度消融；未填长缺口、未连接发根。网格为参考非真值，隐藏验证只在原留出曲线中进行。隐式场及全量PDE未纳入同条件实跑，不能据此排名整管线。"}
    for name in names:
        assert results[name]
        np.savez_compressed(out / (name + ".npz"),**results[name])
        save_polyline_strands(list(results[name].values()),str(out/(name+".ply")))
        points = np.concatenate(list(results[name].values()))
        sdf = scene.compute_signed_distance(o3d.core.Tensor(points)).numpy()
        summary = {"fragments":len(results[name]),"points":len(points),"template_inside_gt2mm_fraction":float(np.mean(sdf<-.002))}
        for metric, values in diagnostics[name].items():
            summary[metric+"_q50_q90_q99"] = np.quantile(values,[.5,.9,.99]).tolist() if values else None
        assert np.isfinite(points).all() and max(diagnostics[name]["projection_errors_px"])<.001
        summary["segments_gt10mm"] = int(np.sum(np.asarray(diagnostics[name]["steps_mm"])>10))
        report["methods"][name] = summary
    raw = cv2.imread(str(base/"raw_img.png"))
    overlay = raw.copy()
    for key in results["mesh_smooth"]:
        xy = guides[key.split("_segment_")[0]][evidence[key+"_source_indices"],:2]
        cv2.polylines(overlay,[np.rint(xy).astype(np.int32)],False,(80,230,80),1,cv2.LINE_AA)
    cv2.imwrite(str(out/"common_curves_on_raw.png"),overlay)
    mesh = o3d.io.read_triangle_mesh(str(base/"pixal3d/hair_mesh_aligned_best.obj"))
    vertices = np.asarray(mesh.vertices)[::50]
    fig,axes = plt.subplots(1,4,figsize=(17,5))
    for ax,name in zip(axes,names):
        ax.scatter(vertices[:,2],vertices[:,1],s=.15,c="gray",alpha=.2)
        for index,curve in enumerate(results[name].values()):
            ax.plot(curve[:,2],curve[:,1],lw=.6,c=plt.cm.turbo(index/len(results[name])))
        ax.set(xlim=(-.18,.23),ylim=(1.54,1.95),title=name,xlabel="Z (m)",ylabel="Y (m)")
        ax.set_aspect("equal")
    fig.tight_layout();fig.savefig(out/"side_comparison.png",dpi=160);plt.close(fig)
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    compare_sparse_depth_methods()
