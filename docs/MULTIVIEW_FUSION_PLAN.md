# 多视角 3D 发丝融合 — 实施计划

## 已有资产

```text
输入资源:
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
  3D hair mesh (.obj/.glb)        ← 用于 Blender 渲染的对齐网格
```

## Blender 相机参数

正交相机，包含 5 个视角，相机位置/方向硬编码在 `render_multiview_blender.py` 中：

```text
center = 网格 (mesh) 顶点中心
dist   = extent * 2.0           (extent = 网格包围盒的对角线长度)
scale  = extent * 1.4           (正交缩放 ortho_scale)

相机位置（世界坐标系下，center 作为原点偏移量）:
  正面 (front): (0, +dist, 0)   朝向 (0, -1, 0)   旋转角: (π/2, 0, π)
  背面 (back):  (0, -dist, 0)   朝向 (0, +1, 0)   旋转角: (π/2, 0, 0)
  左侧 (left):  (-dist, 0, 0)   朝向 (+1, 0, 0)   旋转角: (π/2, 0, -π/2)
  右侧 (right): (+dist, 0, 0)   朝向 (-1, 0, 0)   旋转角: (π/2, 0, π/2)
```

Blender 坐标系: Y 轴向上，Z 轴向前（需要转换为世界坐标系：Y 轴向上，Z 轴向后）

## 核心思路

**单视角 PDE 管线的局限性**：
- 仅有正面 (front) 视角的 strand (发丝方向) 和 depth (深度)
- 侧面和背面的发丝方向完全依赖 PDE 插值进行外推
- 深度也只有正面一层

**多视角融合需要解决的问题**：
- 将 4 个视角的 2D 发丝方向 **反向投影 (backproject)** 到 3D 体积场中
- 正面 (front) 视角作为 **绝对真值**，分配最高权重
- 各视角重叠区域进行加权融合，互补区域直接进行补充
- 最终形成一张覆盖整个头部的 **3D 发丝方向贴图**（类似 UV 纹理，但存储的是 3D 向量场）

## 实施步骤

### 第 1 步: 构建多视角校准矩阵

为每个视角构建 4×4 正交投影矩阵（格式与现有的 `calib` 兼容）。

```text
对于每个视角 v ∈ {front, left, right, back}:
  R = 3×3 旋转矩阵（从 Blender 旋转角 → 世界坐标系）
  T = -R @ camera_position
  P = 正交投影矩阵（scale = extent * 1.4, 映射到 [-1, 1] 的 NDC 空间）
  calib_v = P @ [R | T]   → (4, 4)
```

### 第 2 步: 融合生成 3D 发丝方向体积场

参考 `LaplacePDEStrategy._build_orientation_volume` 的思路，但边界条件的构建来源于 **多个视角** 而非单一视角。

```python
def build_multiview_orientation_volume(resolution, b_min, b_max,
                                        strand_maps, depth_maps, calibs):
    """
    针对每个体素 (voxel) (ix, iy, iz):
      1. 计算体素的世界坐标
      2. 投影到 4 个相机视角:
         px_v, py_v = project(calib_v, world_pos)
         depth_v = world_pos 在相机 v 视角下的深度
         surf_depth_v = depth_maps[v][py, px]
      3. 判断哪些视角能够观测到该体素:
         - 像素落在头发掩码 (hair mask) 内 (strand_map 的 R 通道 > 0)
         - 体素深度 ≈ 该视角的表面深度 (位于设定的 margin 误差范围内)
      4. 针对每一个能观测到该体素的视角:
         从 strand_map 解码出 2D 方向 → 反向投影到 3D 空间
      5. 加权融合:
         正面 (front) 权重 = 2.0
         左/右侧 (left/right) 权重 = 1.0
         背面 (back) 权重 = 0.5
         融合公式: dir_3d = Σ(w_v * backproject(dx_v, dy_v, calib_v)) / Σ(w_v)
    """
```

**2D→3D 反向投影公式**（针对正交相机 v）:
```text
已知参数: dx_2d, dy_2d (图像空间中的发丝方向，已归一化)
        R_v (相机的旋转矩阵)

相机空间方向:  d_cam = (dx_2d, dy_2d, depth_gradient)
                   depth_gradient ≈ 通过 depth_map 局部梯度进行估算

世界空间方向:  d_world = R_v^T @ normalize(d_cam)
```

### 第 3 步: 表面边界条件设置

对于融合后的可见表面体素（即至少被一个视角观测到的体素）→ 采用狄利克雷边界条件 (Dirichlet boundary condition):
- 其值 = 融合计算后的 3D 方向向量

对于未被任何视角观测到的内部体素 → 由 PDE 求解器计算得出:
- 现有的 Laplace 求解器保持不变，只需传入更完整的边界条件即可

### 第 4 步: PDE 求解与发丝合成

复用现有的 `LaplacePDEStrategy`:
1. 将融合后得到的边界条件传入 `_build_orientation_volume` 方法中
2. 保持 CG (共轭梯度) 求解器的逻辑不变
3. 保持 RK4 积分、深度计算以及碰撞检测的逻辑不变

### 第 5 步: 多视角深度融合

```text
针对每个体素的深度计算:
  - 如果被多个视角同时观测到，优先采纳正面 (front) 视角的深度
  - 融合后形成更完整、准确的 3D 表面约束
```

## 文件结构调整

```text
新增文件:
  lib/multiview_fusion.py       ← 核心融合逻辑代码
  scripts/recon_3d/run_pde_multiview.py  ← 多视角 PDE 生成执行脚本

修改文件:
  lib/recon_strategy/laplace_pde.py  ← 增加对多视角边界条件的支持
```

## 关键参数

| 参数名 | 默认值 | 参数说明 |
|---|---|---|
| `multiview_resolution` | 256 | 3D 体积场的分辨率 |
| `front_weight` | 2.0 | 正面视角融合权重（作为真值锚定参考） |
| `side_weight` | 1.0 | 左侧/右侧视角融合权重 |
| `back_weight` | 0.5 | 背面视角融合权重（背面发丝方向往往存在较大噪声） |
| `surface_margin` | 0.015 | 表面体素判定的误差阈值 |

## 预期效果

- 正面区域：发丝方向与原始 2D 真值保持完全一致
- 左右侧区域：通过侧面预测出的发丝方向进行有效地填补和修正
- 背面区域：后脑勺发丝方向由背面视角提供，PDE 根据附近区域进行平滑插值
- 整体效果：形成一个覆盖 360° 的 3D 发丝方向场，比传统单视角外推方案拥有更高的准确度和自然度
