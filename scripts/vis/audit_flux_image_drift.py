"""独立比较原始渲染与FLUX重绘的图像对应；不使用三维反投影自检。"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def identity(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def foreground(image, threshold):
    mask = (image.max(2) > threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    component = labels == (1+np.argmax(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else mask > 0
    contours,_=cv2.findContours(component.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filled=np.zeros_like(mask)
    if contours: cv2.drawContours(filled,contours,-1,1,cv2.FILLED)
    return filled>0


def audit(reference, generated, hair, view=None):
    h,w = reference.shape[:2]
    # 仅统计头发膨胀带外的主体；黑背景对应不提供配准证据。
    exclusion = cv2.dilate(hair.astype(np.uint8), np.ones((21,21), np.uint8))>0
    body_a,body_b = foreground(reference,20),foreground(generated,20)
    masks = [(body & ~exclusion).astype(np.uint8)*255 for body in [body_a,body_b]]
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=.015)
    clahe = cv2.createCLAHE(2., (8,8))
    k1,d1 = sift.detectAndCompute(clahe.apply(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)),masks[0])
    k2,d2 = sift.detectAndCompute(clahe.apply(cv2.cvtColor(generated,cv2.COLOR_BGR2GRAY)),masks[1])
    pairs=[]
    if d1 is not None and d2 is not None and min(len(d1),len(d2)) >= 2:
        matcher=cv2.BFMatcher()
        forward={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(d1,d2,k=2) if m.distance < .8*n.distance}
        backward={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(d2,d1,k=2) if m.distance < .8*n.distance}
        pairs=[(i,j) for i,j in forward.items() if backward.get(j)==i]
    a=np.array([k1[i].pt for i,j in pairs],float).reshape(-1,2)
    b=np.array([k2[j].pt for i,j in pairs],float).reshape(-1,2)
    result={'nonhair_mutual_matches':len(a), 'features':[len(k1),len(k2)],
            'match_warning':'互相匹配仍可能错误；未验证匹配的大位移不等于真实几何漂移',
            'coordinate_system':'reference image pixels; FLUX resized to reference dimensions without registration'}
    canvas=cv2.addWeighted(reference,.5,generated,.5,0)
    if len(a)>=9:
        # 按空间位置排序划分；留出对应不参与拟合，也不按拟合残差筛选。
        order=np.lexsort((a[:,0],a[:,1]));a,b=a[order],b[order]
        holdout=np.arange(len(a))%3==0
        cv2.setRNGSeed(42)
        matrix,inliers=cv2.estimateAffinePartial2D(a[~holdout],b[~holdout],method=cv2.RANSAC,
            ransacReprojThreshold=3.,maxIters=5000,confidence=.999)
        displacement=np.linalg.norm(b-a,axis=1)
        for region,mask in [('all',np.ones(len(a),bool)),('upper_body',a[:,1]<.65*h),('lower_body',a[:,1]>=.65*h)]:
            result[region]={'matches':int(mask.sum()),'displacement_px_q50_q90':np.quantile(displacement[mask],[.5,.9]).tolist() if mask.any() else None}
        if matrix is not None:
            residual=np.linalg.norm(np.c_[a,np.ones(len(a))]@matrix.T-b,axis=1)
            result['similarity_fit']={'matrix':matrix.tolist(),'training_inliers':int(inliers.sum()),
                'holdout_count':int(holdout.sum()),'holdout_residual_px_q50_q90':np.quantile(residual[holdout],[.5,.9]).tolist(),
                'scale':float(np.linalg.norm(matrix[:,0])),
                'rotation_deg':float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0]))),
                'status':'diagnostic_only_not_installed'}
        for x,y in zip(a,b):
            cv2.arrowedLine(canvas,tuple(np.rint(x).astype(int)),tuple(np.rint(y).astype(int)),(0,255,255),1,tipLength=.3)
    else:
        result['feature_status']='insufficient_matches_for_heldout_fit'
    result['nonhair_boundary_threshold_sweep']={}
    result['profile_band_threshold_sweep']={}
    for threshold in [8,16,24,32]:
        ma,mb=foreground(reference,threshold),foreground(generated,threshold)
        if view in ['left','right']:
            differences=[]
            for row in range(int(.38*h),int(.61*h)):
                xa,xb=np.flatnonzero(ma[row]),np.flatnonzero(mb[row])
                if not len(xa) or not len(xb): continue
                x1,x2=(xa[0],xb[0]) if view=='left' else (xa[-1],xb[-1])
                if min(x1,x2)<=1 or max(x1,x2)>=w-2: continue
                if exclusion[row,x1] or exclusion[row,x2]: continue
                differences.append(float(x2-x1))
            result['profile_band_threshold_sweep'][str(threshold)]={
                'band_rows':[int(.38*h),int(.61*h)],'rows':len(differences),
                'absolute_dx_px_q50_q90':np.quantile(np.abs(differences),[.5,.9]).tolist() if differences else None,
                'signed_dx_median_px':float(np.median(differences)) if differences else None,
                'limit':'按当前图像位置选取侧脸轮廓带；相同行比较，不能分离垂直位移与局部形变'}
        ea=ma & ~(cv2.erode(ma.astype(np.uint8),np.ones((3,3),np.uint8))>0)
        eb=mb & ~(cv2.erode(mb.astype(np.uint8),np.ones((3,3),np.uint8))>0)
        ea &= ~exclusion; eb &= ~exclusion
        if ea.any() and eb.any():
            distances=np.r_[distance_transform_edt(~eb)[ea],distance_transform_edt(~ea)[eb]]
            result['nonhair_boundary_threshold_sweep'][str(threshold)]={'symmetric_distance_px_q50_q90':np.quantile(distances,[.5,.9]).tolist(),'samples':len(distances),
                'touches_image_border':bool(ma[[0,-1]].any() or mb[[0,-1]].any() or ma[:,[0,-1]].any() or mb[:,[0,-1]].any()),
                'status':'diagnostic_threshold_sensitive_not_a_geometry_acceptance_metric'}
    cv2.putText(canvas,'yellow: independent image matches',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
    return result,np.hstack([reference,generated,canvas]),a,b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--views',nargs='+',default=['left','right','back'])
    a=p.parse_args()
    if any(v not in ['left','right','back','top'] for v in a.views):
        raise ValueError('此审计只处理生成的非front视角')
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出须按Image ID聚合')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    report={'scope':'图像间漂移诊断，原始front不参与重建输入替换',
        'limits':['特征对应不是人工解剖标注，低纹理区域不能据此证明无漂移',
                  '蒙版排除区来自现有FLUX hair seg并膨胀10px，不能保证完全排除渲染图头发',
                  '黑背景阈值只适用于当前图像，阈值扫参用于检查稳定性',
                  '不使用相机或mesh反投影构造对应'], 'views':{}}
    for view in a.views:
        paths=[a.data_dir/'blender_renders'/f'{view}.png',a.data_dir/'flux_redrawn'/f'{view}.png',a.data_dir/'maps/seg'/f'{view}.png']
        images=[cv2.imread(str(x)) for x in paths]
        if any(x is None for x in images): raise FileNotFoundError(paths)
        reference,generated,hair=images
        original_shape=generated.shape[:2]
        generated=cv2.resize(generated,reference.shape[1::-1],interpolation=cv2.INTER_AREA)
        hair=cv2.resize(hair[:,:,0],reference.shape[1::-1],interpolation=cv2.INTER_NEAREST)>127
        item,canvas,source,target=audit(reference,generated,hair,view)
        item.update({'inputs':[identity(x) for x in paths],'render_shape':reference.shape[:2],'flux_original_shape':original_shape})
        report['views'][view]=item
        cv2.imwrite(str(a.output_dir/f'{view}_comparison.png'),canvas)
        np.savez_compressed(a.output_dir/f'{view}_matches.npz',reference_xy=source,flux_xy=target)
    manifest=a.data_dir/'flux_redrawn/generation_manifest.json'
    report['generation_manifest_exists']=manifest.exists()
    (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
