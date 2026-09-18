"""将独立图像对比、固定缺口可行域和原图投影合成中文诊断图。"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import load_observation_camera


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--repair-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):raise ValueError('输出须按图像聚合')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    def read(path):return np.asarray(Image.open(path).convert('RGB').resize((512,512)))
    reference=read(a.data_dir/'blender_renders/left.png')
    generated=read(a.data_dir/'flux_redrawn/left.png')
    photo=read(a.data_dir/'raw_img.png')
    render=read(a.output_dir/'render/left.png')
    evidence=np.load(a.repair_dir/'feasible_space_20260911/feasible_space.npz')
    feasible=evidence['feasible'].any(1)
    red=np.array([255,45,65]);green=np.array([0,230,150])
    colors=np.where(feasible[:,None],green,red)
    gap_overlay=generated.copy().reshape(-1,3)
    index=evidence['pixel_indices']
    gap_overlay[index]=(.25*gap_overlay[index]+.75*colors).astype(np.uint8)
    gap_overlay=gap_overlay.reshape(512,512,3)
    # 对应固定左侧射线最靠近头模的采样点，而非声称发丝失败点。
    points=evidence['surface']+.0008*evidence['toward_camera']
    camera=load_observation_camera(a.data_dir,'front')
    q=np.c_[points,np.ones(len(points))]@camera.T
    uv=(q[:,:2]/q[:,3:]+1)*255.5
    front_overlay=photo.copy()
    for flag in [True,False]:
        selected=np.flatnonzero(feasible==flag)[::3]
        for i in selected:
            x,y=np.rint(uv[i]).astype(int)
            if 0<=x<512 and 0<=y<512:cv2.circle(front_overlay,(x,y),1,tuple(int(c) for c in colors[i]),-1)
    edges=generated.copy()
    for image,color in [(reference,(0,220,255)),(generated,(255,190,0))]:
        mask=(image.max(2)>24).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if contours:cv2.drawContours(edges,[max(contours,key=cv2.contourArea)],-1,color,1)
    font=ImageFont.truetype('/usr/share/fonts/TTF/LXGWWenKai-Regular.ttf',24)
    small=ImageFont.truetype('/usr/share/fonts/TTF/LXGWWenKai-Regular.ttf',19)
    canvas=Image.new('RGB',(1560,1275),(245,246,249));draw=ImageDraw.Draw(canvas)
    draw.text((20,10),'左侧缺口：图像漂移、可行空间与原图投影',font=font,fill=(20,25,35))
    panels=[(reference,'① FLUX 输入：原始 Blender 左图','这是生成先验的来源，不是原始 front 真值'),
            (generated,'② FLUX 输出：头发与皮肤均被重绘','侧脸外轮廓偏差约 1–3 px；局部形变未排除'),
            (render,'③ 同一审计版本的发丝渲染','full_exact_2mm：左侧仍有明显露头区域'),
            (edges,'④ 生成前后外轮廓对照','青色：输入；橙色：输出。纹理变化不等于位移'),
            (gap_overlay,'⑤ 剩余缺口叠加在 FLUX 左图上','红：采样未找到合法位置；绿：存在合法位置'),
            (front_overlay,'⑥ 同组射线近头模点投影到原始 front','颜色与⑤对应；用于定位冲突，不是失败发丝轨迹')]
    for i,(im,title,caption) in enumerate(panels):
        x=8+(i%3)*520;y=52+(i//3)*586
        draw.text((x+3,y),title,font=small,fill=(20,25,35))
        canvas.paste(Image.fromarray(im),(x,y+29))
        draw.text((x+3,y+546),caption,font=small,fill=(45,50,65))
    draw.text((15,1230),'固定相机与头模；9285 个缺口像素，6521 条射线在外侧 0.8–50 mm 的100个采样点无解。有限采样不是不可行证明。',font=small,fill=(40,45,60))
    canvas.save(a.output_dir/'left_conflict_board.png')
    Image.fromarray(front_overlay).save(a.output_dir/'front_projection.png')
    Image.fromarray(gap_overlay).save(a.output_dir/'left_feasibility_overlay.png')
    print(a.output_dir/'left_conflict_board.png')


if __name__=='__main__':main()
