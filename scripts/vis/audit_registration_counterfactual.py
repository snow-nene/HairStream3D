"""固定原图、头模及左侧像素，隔离 GLB 配准对遮挡可行域的影响。"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth, load_observation_camera
from scripts.recon_3d.regrow_subject_strands import SubjectGuards


def scene_for(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--repair-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--candidate-transforms', type=Path)
    a = parser.parse_args()
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出须按 Image ID 聚合')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    h = np.load(a.repair_dir/'geometry/template_head/template_head.npz')
    head = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(h['vertices']), o3d.utility.Vector3iVector(h['faces']))
    if not head.is_watertight():
        raise ValueError('碰撞头模必须闭合')
    seg = {v: cv2.imread(str(a.data_dir/'maps/seg'/f'{v}.png'), 0)>127 for v in ['front','left']}
    front_cam = load_observation_camera(a.data_dir, 'front')
    left_cam = load_observation_camera(a.data_dir, 'left')
    front = VisibleSurfaceGrowth(front_cam, seg['front'].astype(int), np.zeros((512,512,2)), head)
    guard = SubjectGuards({'front': front}, side_veto=False)
    evidence = np.load(a.repair_dir/'feasible_space_20260911/feasible_space.npz')
    ids = evidence['pixel_indices']
    old = np.load(a.data_dir/'pixal3d/glb_to_world.npz')['icp_T']
    new = np.load(a.repair_dir/'icp_order_audit_20260911/transforms.npz')['consistent_indices']
    variants = {'saved': np.eye(4), 'fixed_indices': new@np.linalg.inv(old), 'without_icp': np.linalg.inv(old)}
    if a.candidate_transforms:
        variants['photo_constrained'] = np.load(a.candidate_transforms)['photo_weight_1']@np.linalg.inv(old)
    glb = o3d.io.read_triangle_mesh(str(a.data_dir/'pixal3d/hair_mesh_aligned_best.obj'))
    glb_scene = scene_for(glb)
    original_left = VisibleSurfaceGrowth(left_cam, seg['left'].astype(int), np.zeros((512,512,2)), glb)
    glb_points = original_left.head_points[ids]
    glb_hit = original_left.head_hit.ravel()[ids]
    toward_front = front.inverse[:3,2].copy()
    toward_front /= np.linalg.norm(toward_front)
    offsets = np.linspace(.0008, .05, 100)
    results, arrays = {}, {}
    image = Image.open(a.data_dir/'flux_redrawn/left.png').convert('RGB').resize((512,512))
    canvas = Image.new('RGB',(512*len(variants),610),'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/TTF/LXGWWenKai-Regular.ttf',19)
    titles = {'saved':'现有错误 ICP', 'fixed_indices':'仅修正面索引', 'without_icp':'去掉 ICP（仅 Umeyama）'}
    titles['photo_constrained'] = '原图约束配准（预设权重1）'
    for col,(name,delta) in enumerate(variants.items()):
        camera = left_cam@np.linalg.inv(delta)
        chart = VisibleSurfaceGrowth(camera, seg['left'].astype(int), np.zeros((512,512,2)), head)
        surface = chart.head_points[ids]
        hit = chart.head_hit.ravel()[ids]
        toward = chart.inverse[:3,2].copy()
        toward /= np.linalg.norm(toward)
        feasible = np.zeros((len(ids),len(offsets)),bool)
        for k,offset in enumerate(offsets):
            points = surface+offset*toward
            codes = guard.points(points,collision=False)
            signed = front.scene.compute_signed_distance(o3d.core.Tensor(points.astype(np.float32)),nsamples=3).numpy()
            feasible[:,k] = hit & (codes==0) & (signed>=.0004)
        # 同一 GLB 左图首交表面，使用 GLB 自身判断原图方向的遮挡。
        gp = glb_points@delta[:3,:3].T+delta[:3,3]
        inv = np.linalg.inv(delta)
        origins = (gp+2*toward_front)@inv[:3,:3].T+inv[:3,3]
        direction = -toward_front@inv[:3,:3].T
        rays = np.c_[origins,np.broadcast_to(direction,origins.shape)]
        distance = glb_scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()
        visible = distance>=2-.0001
        pixels,_,valid = front.pixels(gp)
        positive = np.zeros(len(ids),bool)
        positive[valid] = seg['front'][pixels[valid,1],pixels[valid,0]]
        own_conflict = glb_hit & ((visible & ~positive) | ~valid)
        head_codes = guard.points(gp,collision=False)
        signed = front.scene.compute_signed_distance(o3d.core.Tensor(gp.astype(np.float32)),nsamples=3).numpy()
        possible = feasible.any(1)
        results[name] = {'fixed_pixels':len(ids),'head_ray_hits':int(hit.sum()),
            'sampled_feasible':int(possible.sum()),'hit_but_infeasible':int((hit & ~possible).sum()),
            'no_head_hit':int((~hit).sum()),
            'feasible_by_max_offset_mm':{str(mm):int(feasible[:,offsets<=mm/1000].any(1).sum()) for mm in [2,5,10,20,50]},
            'glb_ray_hits':int(glb_hit.sum()),'glb_self_visibility_front_conflicts':int(own_conflict.sum()),
            'glb_points_front_conflicts_using_head':int((glb_hit & (head_codes!=0)).sum()),
            'glb_points_inside_head':int((glb_hit & (signed<0)).sum()),
            'glb_points_signed_head_distance_mm_10_50_90':(np.quantile(signed[glb_hit],[.1,.5,.9])*1000).tolist()}
        arrays[name+'_feasible'] = feasible
        arrays[name+'_hit'] = hit
        overlay = np.asarray(image).copy().reshape(-1,3)
        colors = np.tile([255,45,65],(len(ids),1))
        colors[possible]=[0,230,150]
        colors[~hit]=[255,200,0]
        overlay[ids] = (.25*overlay[ids]+.75*colors).astype(np.uint8)
        canvas.paste(Image.fromarray(overlay.reshape(512,512,3)),(col*512,38))
        draw.text((col*512+8,8),titles[name],font=font,fill='black')
        draw.text((col*512+8,558),f'可行 {possible.sum()}；无解 {np.sum(hit & ~possible)}；未命中 {np.sum(~hit)}',font=font,fill='black')
        print(name,results[name],flush=True)
    common = np.logical_and.reduce([arrays[name+'_hit'] for name in variants])
    results['common_hit_comparison']={'pixels':int(common.sum()), **{name:int((arrays[name+'_feasible'].any(1)&common).sum()) for name in variants}}
    base = arrays['saved_feasible'].any(1)
    new_ok = arrays['fixed_indices_feasible'].any(1)
    results['fixed_vs_saved_transitions']={'recovered':int((~base & new_ok).sum()),'lost':int((base & ~new_ok).sum())}
    results['limits']='固定原左侧9285个缺口像素、原始front相机/seg及渲染头模；0.8–50mm离散射线审计，不是重新生长/渲染后的缺口率。GLB仅用于诊断自身可见性，非重建真值。'
    draw.text((8,587),'绿：采样存在合法位置；红：命中头模但采样无解；黄：未命中头模。固定同一组左图像素。',font=font,fill='black')
    canvas.save(a.output_dir/'counterfactual.png')
    np.savez_compressed(a.output_dir/'evidence.npz',pixel_indices=ids,offsets=offsets,**arrays)
    (a.output_dir/'report.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
