#!/usr/bin/env python3
"""在同一 Blender 模板下渲染所有有效根前缀，不按终止状态筛选。"""
import argparse
import json
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Matrix, Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_blender import create_curves_from_strands


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--views',nargs='+',choices=['front','left','right','back'],default=['front'])
    args=ap.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    d=np.load(args.input);strands=d['strands'];pieces=[];ids=[]
    for i,s in enumerate(strands):
        keep=np.r_[True,np.linalg.norm(np.diff(s,axis=0),axis=1)>1e-8]
        if keep.sum()>1:pieces.append(s[keep]);ids.append(i)
    bpy.ops.wm.open_mainfile(filepath=str(Path('assets/render_template.blend').resolve()))
    hair=create_curves_from_strands('all_valid_root_prefixes',pieces,bevel_depth=.0003,root_taper_points=3,tip_taper_points=4)
    mat=bpy.data.materials.get('Hair')
    if mat is None:
        mat=bpy.data.materials.new('Hair');mat.diffuse_color=(.2,.1,.05,1)
    hair.data.materials.append(mat)
    world=bpy.context.scene.world
    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type=='TEX_ENVIRONMENT':node.image=bpy.data.images.load(str(Path('assets/ENV.Environment.exr').resolve()),check_existing=True)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=32
    scene.render.resolution_x=768;scene.render.resolution_y=768;scene.render.resolution_percentage=100
    scene.render.threads_mode='FIXED';scene.render.threads=4
    cam=scene.camera;original=cam.matrix_world.copy();pivot=Vector((0,0,1.73))
    for view in args.views:
        angle={'front':0,'left':np.pi/2,'right':-np.pi/2,'back':np.pi}[view]
        cam.matrix_world=Matrix.Translation(pivot)@Matrix.Rotation(angle,4,'Z')@Matrix.Translation(-pivot)@original
        scene.render.filepath=str((args.output_dir/(view+'.png')).resolve())
        bpy.ops.render.render(write_still=True)
    cam.matrix_world=original
    bpy.ops.wm.save_as_mainfile(filepath=str((args.output_dir/'all_prefixes.blend').resolve()))
    report={'source':str(args.input.resolve()),'input_roots':len(strands),'rendered_nonzero_roots':len(ids),'zero_length_roots':len(strands)-len(ids),'selection':'all_nonzero_prefixes_no_termination_filter','views':args.views,'bevel_depth_m':.0003,'samples':32,'rendered_root_indices':ids}
    (args.output_dir/'render_report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
