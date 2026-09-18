#!/usr/bin/env python3
"""从原始 front hair mask 提取保守 scalp envelope，避免把发丝外扩当成头模实体。"""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--img-id',required=True); ap.add_argument('--output-dir',required=True); a=ap.parse_args()
 b=ROOT/'results/multiview_data'/a.img_id; o=Path(a.output_dir); o.mkdir(parents=True,exist_ok=True)
 h=cv2.imread(str(b/'maps/seg/front.png'),0)>127
 n,l,s,_=cv2.connectedComponentsWithStats(h.astype('uint8'),8); core=l==(1+np.argmax(s[1:,cv2.CC_STAT_AREA]))
 # 去除外扩发丝：保留距离边界至少 3 px 的核心，再闭运算填补小孔。
 dist=cv2.distanceTransform(core.astype('uint8'),cv2.DIST_L2,5)
 scalp=dist>=3.0; scalp=cv2.morphologyEx(scalp.astype('uint8'),cv2.MORPH_CLOSE,np.ones((7,7),np.uint8),1)>0
 # 外轮廓和核心分离，供遮挡损失设置不同权重。
 rim=core & ~scalp
 cv2.imwrite(str(o/'scalp_core.png'),scalp.astype('uint8')*255); cv2.imwrite(str(o/'scalp_rim.png'),rim.astype('uint8')*255)
 d={'source':'raw front maps/seg/front.png','core_pixels':int(scalp.sum()),'rim_pixels':int(rim.sum()),'hair_component_pixels':int(core.sum()),'erosion_distance_px':3.0,'use':'遮挡校准目标；rim仅弱约束'}
 (o/'report.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps(d,ensure_ascii=False))
if __name__=='__main__': main()
