"""裁剪无锚点端段，对同一保留点比较网格与平滑曲线的头模穿插。"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt, map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.hair_util import save_polyline_strands


def audit_trimmed_guide_collisions():
    base = Path("results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888")
    root = base / "pde_governance/sparse_guides"
    out = root / "step_10_trim_and_head_audit"
    out.mkdir(parents=True, exist_ok=True)
    smooth = np.load(root / "step_09_method_comparison/mesh_smooth.npz")
    original = np.load(root / "step_09_method_comparison/mesh_raw.npz")
    evidence = np.load(root / "step_03_mesh_lift/raycast_evidence.npz")
    guides = np.load(root / "step_01_front_curves/guides_pixel_depth.npz")
    seg = cv2.imread(str(base / "maps/seg/front.png"),0)>127
    border = distance_transform_edt(seg)
    raw = cv2.imread(str(base / "raw_img.png"))
    overlay = raw.copy()
    removed_overlay = raw.copy()
    head = o3d.io.read_triangle_mesh("data/head_model.obj")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(head))
    curves = {}
    pairs = []
    removed = 0
    for key in smooth.files:
        parent = key.split("_segment_")[0]
        ids = evidence[key+"_source_indices"]
        xy = guides[parent][ids,:2]
        safe = (evidence[parent+"_incidence"][ids]>.3) & (map_coordinates(border,[xy[:,1],xy[:,0]],order=1)>5)
        anchors = np.flatnonzero(safe)
        assert len(anchors)>=8
        start,stop = int(anchors[0]),int(anchors[-1])+1
        keep = np.arange(start,stop)
        removed_ids = np.concatenate([np.arange(start),np.arange(stop,len(xy))])
        removed += len(removed_ids)
        curves[key] = smooth[key][keep]
        pairs.append((original[key][keep],smooth[key][keep],xy[keep]))
        for p in xy[removed_ids]:
            cv2.circle(removed_overlay,tuple(np.rint(p).astype(int)),2,(30,30,255),-1)
    mesh_points = np.concatenate([p[0] for p in pairs]).astype(np.float32)
    points = np.concatenate([p[1] for p in pairs]).astype(np.float32)
    xy = np.concatenate([p[2] for p in pairs])
    mesh_sdf = scene.compute_signed_distance(o3d.core.Tensor(mesh_points)).numpy()
    smooth_sdf = scene.compute_signed_distance(o3d.core.Tensor(points)).numpy()
    mesh_inside = mesh_sdf<-.002
    smooth_inside = smooth_sdf<-.002
    for index in np.argsort(-smooth_sdf):
        color = (70,210,70)
        if mesh_inside[index] and smooth_inside[index]:
            color = (30,30,255)
        elif smooth_inside[index]:
            color = (0,180,255)
        elif mesh_inside[index]:
            color = (255,190,30)
        cv2.circle(overlay,tuple(np.rint(xy[index]).astype(int)),1,color,-1)
    cv2.putText(removed_overlay,"Red: removed unsupported ends",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(overlay,"Red: both inside; orange: smooth only",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
    cv2.imwrite(str(out/"audit_on_raw.png"),np.concatenate([raw,removed_overlay,overlay],axis=1))
    np.savez_compressed(out/"trimmed_guides.npz",**curves)
    np.savez_compressed(out/"collision_evidence.npz",xy=xy,mesh_points=mesh_points,smooth_points=points,mesh_signed_distance=mesh_sdf,smooth_signed_distance=smooth_sdf)
    save_polyline_strands(list(curves.values()),str(out/"trimmed_guides.ply"))
    vertices = np.asarray(head.vertices)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,collection,title in zip(axes,(smooth,curves,curves),("Before trimming","After trimming","Head intersections")):
        ax.scatter(vertices[:,2],vertices[:,1],s=.25,c="gray",alpha=.3)
        for curve in collection.values():
            ax.plot(curve[:,2],curve[:,1],lw=.45,c="steelblue",alpha=.65)
        if title=="Head intersections":
            ax.scatter(points[smooth_inside,2],points[smooth_inside,1],c="red",s=2)
        ax.set(xlim=(-.18,.23),ylim=(1.54,1.95),title=title,xlabel="Z (m)",ylabel="Y (m)")
        ax.set_aspect("equal")
    fig.tight_layout();fig.savefig(out/"side_audit.png",dpi=150);plt.close(fig)
    report={"fragments":len(curves),"before_points":sum(len(smooth[k]) for k in smooth.files),"removed_end_points":removed,"retained_points":len(points),
            "same_retained_points_mesh_inside":int(mesh_inside.sum()),"same_retained_points_smooth_inside":int(smooth_inside.sum()),
            "both_inside":int((mesh_inside&smooth_inside).sum()),"smooth_only_inside":int((~mesh_inside&smooth_inside).sum()),
            "mesh_only_inside":int((mesh_inside&~smooth_inside).sum()),
            "smooth_penetration_mm_q50_q90":np.quantile(-smooth_sdf[smooth_inside]*1000,[.5,.9]).tolist() if smooth_inside.any() else [],
            "head_is_watertight":head.is_watertight(),
            "notes":"负有向距离依赖头模封闭性。原网格交点也穿插表示不是平滑独有问题，但不能独自证明哪套模型错误。仅裁首末可靠锚点之外端段，未连接内部长缺口、未做碰撞修正。"}
    assert report["before_points"]-removed==len(points)
    assert all(np.isfinite(c).all() for c in curves.values())
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    audit_trimmed_guide_collisions()
