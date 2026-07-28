# 多视角 3D 发丝融合 — 实施计划

## 已有资产

```
inputs:
  results/multiview_strand_depth/
  ├── strand_map/front.png        ← 原始 2D 真值，不动
  ├── strand_map/left.png         ← FLUX→SAM→模型 预测
  ├── strand_map/right.png
  ├── strand_map/back.png
  ├── depth_map/front.npy         ← 原始 2D 真值深度
  ├── depth_map/left.npy
  ├── depth_map/right.npy
  ├── depth_map/back.npy
  └── combined_strand.npz         ← (4, 512, 512, 3)

  data/head_model.obj             ← 碰撞头皮模型
  3D hair mesh (.obj/.glb)        ← Blender 渲染用的对齐网格
```

## Blender 相机参数

正交相机，5 个视角，相机位置/方向硬编码在 `render_multiview_blender.py`：

```
center = mesh 顶点中心
dist   = extent * 2.0           (extent = mesh 包围盒对角线)
scale  = extent * 1.4           (ortho_scale)

相机位置（世界坐标，center 为原点偏移）:
  front: (0, +dist, 0)   朝向 (0, -1, 0)   rot: (π/2, 0, π)
  back:  (0, -dist, 0)   朝向 (0, +1, 0)   rot: (π/2, 0, 0)
  left:  (-dist, 0, 0)   朝向 (+1, 0, 0)   rot: (π/2, 0, -π/2)
  right: (+dist, 0, 0)   朝向 (-1, 0, 0)   rot: (π/2, 0, π/2)
```

Blender 坐标系: Y-up, Z-forward（需转换为世界坐标系 Y-up, Z-back）

## 核心思路

**单视角 PDE 管线的局限**：
- 只有 front 视角的 strand+depth
- 侧面和后面的发丝方向全靠 PDE 插值外推
- 深度也只有正面一层

**多视角融合要解决的问题**：
- 把 4 个视角的 2D 发丝方向 **反向投影** 到 3D 体积中
- Front 视角作为 **绝对真值**，权重最高
- 各视角交叠区域加权融合，互补区域直接补充
- 形成一张覆盖全头部的 **3D 发丝方向贴图**（类似 UV 纹理但存的是 3D 向量场）

## 实施步骤

### Step 1: 构建多视角校准矩阵

为每个视角构建 4×4 正交投影矩阵（格式兼容现有 `calib`）。

```
对每个视角 v ∈ {front, left, right, back}:
  R = 3×3 旋转矩阵（Blender rot → 世界坐标系）
  T = -R @ camera_position
  P = 正交投影矩阵（scale = extent * 1.4, 映射到 [-1, 1] NDC）
  calib_v = P @ [R | T]   → (4, 4)
```

### Step 2: 融合为 3D 发丝方向体积

仿照 `LaplacePDEStrategy._build_orientation_volume` 的思路，但从 **多个视角** 而不是单视角构建边界条件。

```
def build_multiview_orientation_volume(resolution, b_min, b_max,
                                        strand_maps, depth_maps, calibs):
    """
    对每个 voxel (ix, iy, iz):
      1. 计算 voxel 世界坐标
      2. 投影到 4 个相机:
         px_v, py_v = project(calib_v, world_pos)
         depth_v = world_pos 在相机 v 的深度
         surf_depth_v = depth_maps[v][py, px]
      3. 判断哪几个视角能看到这个 voxel:
         - 像素在 hair mask 内 (strand_map R > 0)
         - voxel 深度 ≈ 该视角的表面深度 (在 margin 内)
      4. 对每个能看到此 voxel 的视角:
         从 strand_map 解码 2D 方向 → 反投影到 3D
      5. 加权融合:
         front 权重 = 2.0
         left/right 权重 = 1.0
         back 权重 = 0.5
         融合公式: dir_3d = Σ(w_v * backproject(dx_v, dy_v, calib_v)) / Σ(w_v)
    """
```

**2D→3D 反投影公式**（对正交相机 v）:
```
给定: dx_2d, dy_2d (图像空间发丝方向，归一化)
     R_v (相机旋转矩阵)

相机空间方向:  d_cam = (dx_2d, dy_2d, depth_gradient)
                  depth_gradient ≈ 从 depth_map 局部梯度估算

世界空间方向:  d_world = R_v^T @ normalize(d_cam)
```

### Step 3: 表面边界条件设置

融合后的可见表面 voxels（至少被一个视角看到）→ Dirichlet 边界条件:
- 值 = 融合后的 3D 方向向量

未被任何视角看到的内部 voxels → 由 PDE 求解:
- 现有 Laplace 求解器不变，只需传入更完整的边界条件

### Step 4: PDE 求解 + 发丝合成

复用现有 `LaplacePDEStrategy`:
1. 将融合后的边界条件传入 `_build_orientation_volume`
2. 不改动 CG 求解器逻辑
3. 不改动 RK4 积分/depth/碰撞

### Step 5: 多视角深度融合

```
对每个 voxel 的深度:
  - 如果多个视角都能看到，取 front 深度优先
  - 融合后形成更完整的 3D 表面约束
```

## 文件结构

```
新增:
  lib/multiview_fusion.py       ← 核心融合逻辑
  scripts/run_pde_multiview.py  ← 多视角 PDE 生成脚本

修改:
  lib/recon_strategy/laplace_pde.py  ← 支持多视角边界条件
```

## 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `multiview_resolution` | 256 | 3D 体积分辨率 |
| `front_weight` | 2.0 | front 视角融合权重（真值锚定） |
| `side_weight` | 1.0 | left/right 视角融合权重 |
| `back_weight` | 0.5 | back 视角融合权重（发丝方向噪声大） |
| `surface_margin` | 0.015 | 表面 voxel 判定阈值 |

## 预期效果

- Front 面：发丝方向与原始 2D 真值完全一致
- Left/Right 面：从侧面预测的发丝方向补充填补
- Back 面：后脑发丝方向由 back 视角提供，PDE 插值附近区域
- 整体：形成覆盖 360° 的 3D 发丝方向场，比单视角外推更准确
