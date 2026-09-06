"""固定投影线宽，比较短根连接前后覆盖率、泄漏和长度。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.recon3D import load_calib


def project_polylines(strands,camera,shape):
    flat=np.asarray(strands).reshape(-1,3)
    clip=np.c_[flat,np.ones(len(flat))]@camera.T
    if not np.isfinite(clip).all() or np.any(np.abs(clip[:,3])<1e-12):
        raise ValueError('invalid projection')
    uv=clip[:,:2]/clip[:,3:4]
    pixels=(uv+1)*np.array([shape[1]-1,shape[0]-1])/2
    return np.rint(pixels).astype(np.int32).reshape(*strands.shape[:2],2)


def compare_coverage(args):
    base,reference,candidate=args.data_dir,args.reference_dir,args.candidate_dir
    if candidate.parent != reference.parent or candidate.parent.name!='volume_partition_integration':
        raise ValueError('compare runs from the same image integration directory')
    seg=cv2.imread(str(base/'maps/seg/front.png'),0)>127
    camera=load_calib(str(base/'maps/param/front_dense_silhouette.npy')).numpy()
    paths=[reference/'diagnostic_strands.npz',candidate/'connected_strands.npz']
    strands=[np.load(p)['strands'] for p in paths]
    polylines=[project_polylines(s,camera,seg.shape) for s in strands]
    all_pixels=np.concatenate([x.reshape(-1,2) for x in polylines])
    low=np.minimum(all_pixels.min(0),[0,0])-args.line_width
    high=np.maximum(all_pixels.max(0),[seg.shape[1]-1,seg.shape[0]-1])+args.line_width
    size=high-low+1
    if size.max()>10000 or args.line_width<1:
        raise ValueError('invalid canvas or line width')
    mask=np.zeros(tuple(size[::-1]),bool)
    mask[-low[1]:-low[1]+seg.shape[0],-low[0]:-low[0]+seg.shape[1]]=seg
    reports=[]; canvases=[]
    for lines,points in zip(polylines,strands):
        canvas=np.zeros(mask.shape,np.uint8)
        cv2.polylines(canvas,list(lines-low),False,255,args.line_width,lineType=cv2.LINE_8)
        drawn=canvas>0
        lengths=np.linalg.norm(np.diff(points,axis=1),axis=2).sum(1)
        reports.append({'strands':len(points),'effective_strands_ge_1cm':int((lengths>=.01).sum()),
            'original_root_denominator':10000,'length_median_m':float(np.median(lengths)),
            'coverage':float((drawn&mask).sum()/mask.sum()),
            'leakage':float((drawn&~mask).sum()/max(1,drawn.sum()))})
        canvases.append(canvas)
    before,after=reports
    gates={name:after[name]>=.95*before[name] for name in
           ['effective_strands_ge_1cm','length_median_m','coverage']}
    gates['leakage']=after['leakage']<=before['leakage']+.01
    report={'scope':'root-connection increment against previous mesh diagnostic; not the required three-way baseline',
            'line_width_px':args.line_width,'outside_image_pixels_included_in_leakage':True,
            'reference':before,'candidate':after,'incremental_quality_gates':gates,
            'passed':all(gates.values()),'full_acceptance':False,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (candidate/'coverage_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    montage=[]
    for canvas,name in zip(canvases,['Before root connections','After root connections']):
        rgb=np.zeros((*mask.shape,3),np.uint8);rgb[mask]=[35,35,35]
        rgb[(canvas>0)&mask]=[100,210,100];rgb[(canvas>0)&~mask]=[60,60,255]
        cv2.putText(rgb,name,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
        montage.append(rgb)
    cv2.imwrite(str(candidate/'coverage_comparison.png'),np.concatenate(montage,axis=1))
    plan=np.load(candidate/'root_connection_plan.npz')
    root_pixels=project_polylines(plan['roots_world'][:,None],camera,seg.shape)[:,0]
    starts=project_polylines(plan['starts_world'][:,None],camera,seg.shape)[:,0]
    overlay=cv2.imread(str(base/'raw_img.png'))
    overlay=cv2.resize(overlay,(seg.shape[1],seg.shape[0]))
    colors={'short_connection':(0,200,0),'too_far':(0,0,255),'ambiguous_partition':(0,180,255)}
    for state,color in colors.items():
        for i in np.flatnonzero(plan['status']==state):
            cv2.circle(overlay,tuple(root_pixels[i]),2,color,-1)
            if state=='short_connection':
                cv2.line(overlay,tuple(root_pixels[i]),tuple(starts[i]),color,1)
    cv2.imwrite(str(candidate/'root_status_overlay.png'),overlay)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--candidate-dir',type=Path,required=True)
    parser.add_argument('--line-width',type=int,default=1)
    compare_coverage(parser.parse_args())
