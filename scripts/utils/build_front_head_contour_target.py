#!/usr/bin/env python3
"""从原始 front 图像构造头部轮廓目标；不读取 FLUX 侧视图。"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.utils.align_glb_lmk import get_lmk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-id', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    base = ROOT/'results/multiview_data'/args.img_id
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(base/'raw_img.png'))
    hair = cv2.imread(str(base/'maps/seg/front.png'), cv2.IMREAD_GRAYSCALE) > 127
    body = cv2.imread(str(base/'maps/body_img/front.png'), cv2.IMREAD_GRAYSCALE) > 127
    if image is None or hair is None or body is None: raise FileNotFoundError('缺少 raw_img 或 front mask')
    h,w = hair.shape
    lmk = get_lmk(cv2.resize(image,(512,512)))
    if lmk is None or len(lmk)<68: raise RuntimeError('front 关键点检测失败')
    pts = np.rint(lmk[:68,:2]).astype(np.int32)
    # Face polygon补足被头发遮挡的额头/鬓角；不把肩颈纳入目标。
    face = np.zeros((h,w), np.uint8)
    cv2.fillConvexPoly(face, cv2.convexHull(pts[0:17]), 1)
    cv2.fillConvexPoly(face, cv2.convexHull(pts[17:68]), 1)
    face = cv2.dilate(face, np.ones((9,9),np.uint8), iterations=1)>0
    y_chin = int(np.max(pts[0:17,1]) + 0.10*max(20, np.ptp(pts[0:17,1])))
    yy=np.arange(h)[:,None]
    face &= yy <= min(h-1, y_chin)
    # 头发取最大连通区域，避免孤立误分割点。
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hair.astype(np.uint8),8)
    hair_main = np.zeros_like(hair)
    if n>1: hair_main = lab == (1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    target = (hair_main | (face & body))
    target = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7,7),np.uint8), iterations=1)>0
    # 轮廓带供距离变换优化使用。
    outer=cv2.dilate(target.astype(np.uint8),np.ones((9,9),np.uint8),1)
    inner=cv2.erode(target.astype(np.uint8),np.ones((5,5),np.uint8),1)
    band=(outer>0)&~(inner>0)
    cv2.imwrite(str(out/'target_head_mask.png'), target.astype(np.uint8)*255)
    cv2.imwrite(str(out/'target_contour_band.png'), band.astype(np.uint8)*255)
    overlay=image.copy(); overlay[target]=(0.55*overlay[target]+0.45*np.array([0,220,0])).astype(np.uint8)
    cv2.polylines(overlay,[cv2.convexHull(pts)],True,(0,0,255),1)
    cv2.imwrite(str(out/'target_overlay.png'),overlay)
    report={'source':'raw_img.png + maps/seg/front.png + maps/body_img/front.png','forbidden_sources':['flux_redrawn','blender_renders'],
            'image_shape':[h,w],'hair_pixels':int(hair_main.sum()),'body_pixels':int(body.sum()),
            'target_pixels':int(target.sum()),'contour_band_pixels':int(band.sum()),'chin_cut_y':y_chin,
            'landmark_count':int(len(pts))}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
