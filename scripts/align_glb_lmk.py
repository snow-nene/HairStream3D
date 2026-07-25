"""
GLB Mesh LMK 对齐 -> 计算校准矩阵，然后提取头发 Mesh。

完全遵照 get_lmk.py + opt_cam.py 的模式：
  - 用 3DDFA_V2 检测渲染图和原图的 2D lmk（get_lmk.py 的逻辑）
  - 用 OptLandmark 风格的模块优化相机参数（opt_cam.py 的逻辑）
  - 用得到的 calib 投影 GLB 顶点，提取头发 Mesh

运行:
  pixi run python scripts/align_glb_lmk.py \
      --input_img  results/real_imgs/resized_img/0d285f5be7fa09c3dbbf1c9334047888.png \
      --render_img results/test_pixal3d/render_front.png \
      --render_cam results/test_pixal3d/render_camera.npz \
      --glb_ply    results/test_pixal3d/output_mesh.ply \
      --strand_map results/real_imgs/strand_map/0d285f5be7fa09c3dbbf1c9334047888.png \
      --out_dir    results/test_pixal3d/
"""
import sys, os, copy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'external/3DDFA_V2'))

import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import yaml
import trimesh

# ─── 3DDFA 必须从它自己的目录运行 ───────────────────────────────
def get_lmk(img_bgr, size=256):
    """完全照搬 get_lmk.py 的 GetPoseLms2d 逻辑，单张图"""
    orig_h, orig_w = img_bgr.shape[:2]
    img_r = cv2.resize(img_bgr, (size, size))

    os.chdir(os.path.join(ROOT, 'external/3DDFA_V2'))
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    from TDDFA import TDDFA
    from FaceBoxes import FaceBoxes
    tddfa      = TDDFA(gpu_mode=True, **cfg)
    face_boxes = FaceBoxes()
    os.chdir(ROOT)

    # TDDFA 内部可能残留 no_grad 状态，用 no_grad 显式包裹确保进出一致
    with torch.no_grad():
        boxes = face_boxes(img_r)
        if len(boxes) == 0:
            return None
        param_lst, roi_box_lst = tddfa(img_r, boxes)
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)

    pts = ver_lst[0].T   # (68, 3)

    # 还原到原图尺寸
    pts_out = pts.copy()
    pts_out[:, 0] = pts[:, 0] / size * orig_w
    pts_out[:, 1] = pts[:, 1] / size * orig_h
    return pts_out  # (68, 3)


# ─── 正交相机反投影（Blender render_camera.npz）────────────────
def backproject_render_lmk(pts2d_px, cam_npz):
    """
    渲染图的像素坐标 -> 3D 世界坐标（用 Blender 正交相机参数反投影）
    Z 坐标设为 mesh 中心的 Z（正交投影 Z 无法从 2D 还原，用 mesh 中心近似）
    """
    ortho_scale = float(cam_npz['ortho_scale'])
    cam_R       = cam_npz['cam_R']    # (3,3) world_R_cam
    cam_pos     = cam_npz['cam_pos']  # (3,)
    mesh_center = cam_npz['mesh_center']  # (3,)
    S           = int(cam_npz['img_size'])

    # pixel -> 相机局部坐标（正交，lz=0 表示在图像平面上）
    lx = (pts2d_px[:, 0] - S/2) / (S/2) * (ortho_scale / 2)
    ly = -(pts2d_px[:, 1] - S/2) / (S/2) * (ortho_scale / 2)
    # 正交投影无深度信息，用 mesh 中心的深度（相机局部 Z=0 对应相机位置平面）
    # 实际面部顶点在相机前方：cam_pos 沿 -cam_Z 方向偏移 mesh_center_z_local
    # 计算 mesh_center 在相机局部坐标系中的 z 分量
    cam_Z_world = cam_R[:, 2]  # 相机局部 Z 轴在世界坐标中的方向
    center_in_cam = cam_R.T @ (mesh_center - cam_pos)  # 转到相机坐标
    lz = np.full(len(pts2d_px), center_in_cam[2])

    local_pts = np.stack([lx, ly, lz], axis=1)   # (N, 3)
    world_pts = (cam_R @ local_pts.T).T + cam_pos
    return world_pts.astype(np.float32)

# ─── 完全对应 OptLandmark 的 camera 优化模块 ────────────────────
class OptGLBLandmark(nn.Module):
    """
    和 lib/util/opt_lmk.py 中的 OptLandmark 完全一致的结构，
    只是 lmk_3D 来自 GLB 渲染图反投影，而不是 head_model.obj。
    """
    def __init__(self, lmk_3d_np, lmk_gt_np, input_img_path, width=512):
        super().__init__()
        self.width = width
        self.rendering_load_size = 1024
        self.ortho_ratio = 0.2

        # 读入原图尺寸，计算 lmk 缩放系数（照搬 opt_lmk.py read_img 逻辑）
        from skimage.io import imread as sk_imread
        from skimage.transform import resize as sk_resize
        images_gt = sk_imread(input_img_path, as_gray=False)
        self.lmk_coeffi = [float(width) / images_gt.shape[1],
                            float(width) / images_gt.shape[0]]
        images_gt = sk_resize(images_gt, (width, width))
        self.images_gt = np.array(images_gt) * 255.0

        # 3D lmk（GLB 世界坐标）
        self.lmk_3D = torch.Tensor(lmk_3d_np.T).float().unsqueeze(0).cuda()  # (1,3,N)

        # GT 2D lmk（完全照搬 load_lmk_gt）
        lmk_gt = lmk_gt_np[:, :2].copy()
        lmk_gt[:, 0] *= self.lmk_coeffi[0]
        lmk_gt[:, 1] *= self.lmk_coeffi[1]
        self.lmk_gt = torch.from_numpy(lmk_gt).float().cuda().unsqueeze(0)  # (1,N,2)

        # 可优化参数 —— 完全对应 opt_lmk.py（必须是 Parameter，不是 buffer）
        ext = float(np.linalg.norm(lmk_3d_np.max(0) - lmk_3d_np.min(0)))
        init_scale = 363.1 / max(ext / 0.9, 1e-3)
        center_init = torch.Tensor(lmk_3d_np.mean(0)).view(3, 1)

        self.register_parameter('scale',    nn.Parameter(torch.Tensor([init_scale])))
        self.register_parameter('center',   nn.Parameter(center_init))
        self.register_parameter('rotation', nn.Parameter(torch.eye(3)))
        self.register_parameter('scale_dis',    nn.Parameter(torch.zeros(1)))
        self.register_parameter('center_dis',   nn.Parameter(torch.zeros(3, 1)))
        self.register_parameter('rotation_dis', nn.Parameter(torch.zeros(3, 3)))

        self.lmk_loss = nn.MSELoss()

    def get_camera(self):
        """与 OptLandmark.get_camera 完全一致（避免 in-place 操作破坏梯度）"""
        ortho_ratio = self.ortho_ratio
        scale  = self.scale + self.scale_dis          # (1,)
        center = self.center + self.center_dis        # (3,1)
        R      = self.rotation + self.rotation_dis    # (3,3)

        translate = -torch.mm(R, center).reshape(3, 1)
        bottom    = torch.zeros(1, 4, device=R.device)
        bottom[0, 3] = 1.0
        extrinsic = torch.cat([torch.cat([R, translate], dim=1), bottom], dim=0)  # (4,4)

        # 用 torch.diag + cat 代替 in-place 赋值，保持梯度链
        S_val = scale / ortho_ratio   # scalar tensor with grad
        uv    = 1.0 / float(self.rendering_load_size // 2)
        zero  = torch.zeros(1, device=R.device)
        # scale_intrinsic 对角: [S, -S, S, 1] * uv
        diag_vals = torch.cat([S_val * uv, -S_val * uv, S_val * uv,
                                torch.ones(1, device=R.device) * uv])
        # 构建 4x4 对角矩阵（保梯度）
        intrinsic = torch.diag(diag_vals)              # (4,4)

        calib = torch.mm(intrinsic, extrinsic).unsqueeze(0)  # (1,4,4)
        return calib

    def lmk_proj(self, calibs):
        """与 OptLandmark.lmk_proj 完全一致"""
        rot   = calibs[:, :3, :3]
        trans = calibs[:, :3, 3:4]
        xyz   = torch.baddbmm(trans, rot, self.lmk_3D)   # (1,3,N)
        xy    = xyz[:, :2, :]

        y_coor = xy.squeeze().permute(1, 0)[:, 0]
        x_coor = xy.squeeze().permute(1, 0)[:, 1]

        x_coor = x_coor * self.width/2 + self.width/2 - 1
        y_coor = y_coor * self.width/2 + self.width/2 - 1

        lmk_2D = torch.cat((y_coor.unsqueeze(1), x_coor.unsqueeze(1)), dim=1)
        return lmk_2D.unsqueeze(0)   # (1,N,2)

    def forward(self):
        calibs = self.get_camera()
        self.lmk_2D = self.lmk_proj(calibs)
        loss = torch.sum(torch.abs(self.lmk_2D - self.lmk_gt))
        return loss

    def save_param(self, output_path):
        """与 OptLandmark.save_param 完全一致"""
        dic = {
            'ortho_ratio': self.ortho_ratio,
            'scale':  (self.scale + self.scale_dis).detach().cpu().numpy(),
            'center': (self.center + self.center_dis).detach().cpu().numpy(),
            'R':      (self.rotation + self.rotation_dis).detach().cpu().numpy(),
        }
        np.save(output_path, dic)

    def get_img_lmk(self):
        """与 OptLandmark.get_img_lmk 完全一致"""
        gt_lmk   = np.floor(self.lmk_gt.detach().cpu().numpy()[0]).astype(np.int32)
        pred_lmk = np.floor(self.lmk_2D.detach().cpu().numpy()[0]).astype(np.int32)
        img = self.images_gt.copy()
        gt_lmk   = np.clip(gt_lmk,   0, self.width - 1)
        pred_lmk = np.clip(pred_lmk, 0, self.width - 1)
        img[gt_lmk[:, 1],   gt_lmk[:, 0],   :3] = [255, 0, 0]
        img[pred_lmk[:, 1], pred_lmk[:, 0], :3] = [0, 0, 255]
        return img.astype(np.uint8)

    def get_calib_matrix(self):
        """返回 (4,4) numpy calib 矩阵，供后续提取 hair mesh 用"""
        calib = self.get_camera()
        return calib.squeeze(0).detach().cpu().numpy()

# ─── 用 calib 提取头发 mesh ─────────────────────────────────────
def extract_hair_mesh(glb_ply, calib_4x4, strand_map_path, img_size=512):
    raw = trimesh.load(glb_ply)
    verts = raw.vertices.astype(np.float32)

    # 投影
    N    = len(verts)
    ones = np.ones((N, 1), 'f')
    proj = (calib_4x4 @ np.concatenate([verts, ones], 1).T).T
    u = proj[:, 0] * (img_size/2) + (img_size/2)
    v = proj[:, 1] * (img_size/2) + (img_size/2)

    # strand_map 作为头发 mask
    strand = cv2.imread(strand_map_path)
    gray   = cv2.cvtColor(strand, cv2.COLOR_BGR2GRAY)
    hair_mask = (gray > 10).astype(np.uint8) * 255
    hair_mask = cv2.resize(hair_mask, (img_size, img_size))

    uc = np.clip(u.astype(int), 0, img_size-1)
    vc = np.clip(v.astype(int), 0, img_size-1)
    in_hair = hair_mask[vc, uc] > 128

    faces    = raw.faces
    face_mask = in_hair[faces[:,0]] & in_hair[faces[:,1]] & in_hair[faces[:,2]]
    print(f"  总面: {len(faces)}  头发面: {face_mask.sum()} ({100*face_mask.sum()/len(faces):.1f}%)")

    if face_mask.sum() == 0:
        return None, u, v

    hair_faces = faces[face_mask]
    used       = np.unique(hair_faces)
    old2new    = {o: n for n, o in enumerate(used)}
    new_faces  = np.array([[old2new[f] for f in face] for face in hair_faces])
    new_verts  = verts[used]
    return (new_verts, new_faces), u, v

# ─── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img",  required=True, help="输入原图")
    parser.add_argument("--render_img", required=True, help="Blender 渲染图")
    parser.add_argument("--render_cam", required=True, help="render_camera.npz")
    parser.add_argument("--glb_ply",   required=True, help="GLB 转出的 PLY")
    parser.add_argument("--strand_map",required=True, help="strand_map 图（头发区域）")
    parser.add_argument("--out_dir",   required=True)
    parser.add_argument("--lr",        type=float, default=0.01)
    parser.add_argument("--epochs",    type=int,   default=201)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Step 1: 检测渲染图 lmk ─────────────────────────────────
    print("\n[1/4] 检测渲染图 lmk（3DDFA_V2）...")
    render_bgr = cv2.imread(args.render_img)
    render_lmk = get_lmk(render_bgr)
    if render_lmk is None:
        print("[ERROR] 渲染图无人脸！")
        sys.exit(1)
    print(f"  OK: {len(render_lmk)} lmk")

    # ── Step 2: 反投影到 3D ─────────────────────────────────────
    print("\n[2/4] 反投影渲染 lmk -> 3D (Blender 正交相机)...")
    cam_npz  = np.load(args.render_cam)
    lmk_3d   = backproject_render_lmk(render_lmk[:, :2], cam_npz)
    print(f"  3D 范围: {lmk_3d.min(0).round(4)} ~ {lmk_3d.max(0).round(4)}")

    # ── Step 3: 检测原图 lmk ────────────────────────────────────
    print("\n[3/4] 检测输入原图 lmk（3DDFA_V2）...")
    input_bgr = cv2.imread(args.input_img)
    input_lmk = get_lmk(input_bgr)
    if input_lmk is None:
        print("[ERROR] 原图无人脸！")
        sys.exit(1)
    print(f"  OK: {len(input_lmk)} lmk")

    # TDDFA 内部会通过 _C._set_grad_enabled(False) 关掉 autograd 且不恢复
    # 必须在所有 TDDFA 推理完后手动重新开启
    torch.set_grad_enabled(True)

    # ── Step 4: 优化相机（opt_cam.py 的逻辑）────────────────────
    print(f"\n[4/4] 优化相机参数 (lr={args.lr}, epochs={args.epochs})...")
    lmk_opt  = OptGLBLandmark(lmk_3d, input_lmk, args.input_img).cuda()
    optimizer = torch.optim.Adam(lmk_opt.parameters(), lr=args.lr, betas=(0.5, 0.99))
    schedule  = [100, 150]
    lr = args.lr
    from lib.train_util import adjust_learning_rate
    for i in range(args.epochs):
        loss = lmk_opt.forward()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = adjust_learning_rate(optimizer, i, lr, schedule, 0.5)
        if i % 50 == 0:
            print(f"  iter {i:3d}  loss={loss.item():.2f}")

    # 保存 param（与 opt_cam.py 完全一致格式）
    param_path = os.path.join(args.out_dir, "glb_param.npy")
    lmk_opt.save_param(param_path)
    print(f"\n  param 保存: {param_path}")

    # 保存对齐可视化（与 opt_cam.py 一致）
    vis = lmk_opt.get_img_lmk()
    vis_path = os.path.join(args.out_dir, "debug_lmk_align.png")
    import imageio.v2 as imageio
    imageio.imwrite(vis_path, vis)
    print(f"  对齐可视化: {vis_path}  (红=GT, 蓝=3D投影)")

    # ── 用对齐后的 calib 提取头发 Mesh ──────────────────────────
    print("\n[+] 提取头发 Mesh...")
    calib_4x4 = lmk_opt.get_calib_matrix()
    result, u, v = extract_hair_mesh(args.glb_ply, calib_4x4,
                                      args.strand_map, img_size=512)

    if result is None:
        print("  [警告] 未提取到头发面")
    else:
        new_verts, new_faces = result
        out_obj = os.path.join(args.out_dir, "hair_mesh_aligned.obj")
        with open(out_obj, 'w') as f:
            for v3 in new_verts:
                f.write(f"v {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
            for face in new_faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"  头发 Mesh: {out_obj}")
        print(f"  顶点: {len(new_verts)}  面: {len(new_faces)}")

if __name__ == '__main__':
    main()
