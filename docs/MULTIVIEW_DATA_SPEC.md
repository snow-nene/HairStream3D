# 多视图中间数据目录规范与流程设计

本文档定义了 HairStep 项目中多视角（Multi-View）头发重建管线的中间数据目录组织规范及全流程处理逻辑。

---

## 一、 目录组织规范 (按 Image ID 聚合)

所有的多视图中间数据（包含 Blender 导出的多视角渲染、FLUX 重绘结果、提取的 2D 特征 Map 以及 3D 重建结果）必须严格统一存放于 `results/multiview_data/<image_id>/` 路径下，**禁止**在 `results/` 根目录下直接平铺创建杂乱的视项目子目录。

### 标准目录树结构

```text
results/
└── multiview_data/                                 <-- 多视图中间数据总目录
    └── <image_id>/                                 <-- 单张输入图片的唯一 ID (例如 0a1ba3dbefc8934ab60577c5c91f66a0)
        ├── blender_renders/                        <-- 1. 基础 3D 模板或单视角先验出多视角 3D 渲染
        │   ├── front.png
        │   ├── left.png
        │   ├── right.png
        │   └── back.png
        │
        ├── flux_redrawn/                           <-- 2. 使用 FLUX.2 重绘增强后的各视角头发图像
        │   ├── front.png
        │   ├── left.png
        │   ├── right.png
        │   └── back.png
        │
        ├── maps/                                   <-- 3. 多视角 2D 特征图 (用于 3D PDE / 深度融合)
        │   ├── strand_map/                         <-- 各视角 2D 发丝方向切线图 (.png)
        │   │   ├── front.png
        │   │   ├── left.png
        │   │   ├── right.png
        │   │   └── back.png
        │   ├── depth_map/                          <-- 各视角 2D 深度图 (.npy)
        │   │   ├── front.npy
        │   │   ├── left.npy
        │   │   ├── right.npy
        │   │   └── back.npy
        │   ├── seg/                                <-- 各视角头发 Segmentation 遮罩 (.png)
        │   │   ├── front.png
        │   │   ├── left.png
        │   │   ├── right.png
        │   │   └── back.png
        │   ├── normal_map/                         <-- 各视角表面法线图 (.png)
        │   │   ├── front.png
        │   │   ├── left.png
        │   │   ├── right.png
        │   │   └── back.png
        │   └── param/                              <-- 各视角相机标定与内/外参矩阵 (.npy)
        │       ├── front.npy
        │       ├── left.npy
        │       ├── right.npy
        │       └── back.npy
        │
        └── pde_reconstruction/                     <-- 4. 多视角融合求解出的最终 3D 发丝模型
            ├── hair_multiview.ply
            └── hair_multiview_pruned.ply
```

---

## 二、 灵活多视角生成 Pipeline 架构设计

系统设计支持两种输入解算模式：
1. **纯单视角模式 (Single-View Mode)**：仅使用输入的正面图像 (`front`) 生成，快速解算；
2. **多视角扩展增强模式 (Multi-View Mode)**：以正面图像为基础，根据需求生成任意多视角（如 `front`, `left`, `right`, `back`）进行 3D 融合解算。

### 1. 2D 特征图与深度图生成
* **单视角提取 (`scripts/infer_2d/img2strand.py`, `img2depth.py`, `img2masks.py`)**：
  * 输入：正面 RGB 图 (`<image_id>.png`)
  * 输出：`results/multiview_data/<image_id>/maps/strand_map/front.png`, `depth_map/front.npy`, `seg/front.png`
* **多视角批量提取 (`scripts/infer_2d/compute_multiview_maps.py`)**：
  * 支持 `--views front left right back` 任意视角列表参数
  * 自动从 `flux_redrawn/` 提取多视角的 `strand_map` 与 `depth_map`，并写入 `maps/` 对应子目录。

### 2. FLUX.2 视角重绘 (`scripts/infer_2d/flux_redraw_multiview.py`)
  * 支持接收指定视角列表 `--views`（如 `left right back`）
  * 从 `blender_renders/` 读取渲染图，输出重绘增强发丝图至 `flux_redrawn/<view>.png`。

### 3. 3D 融合 PDE 解算 (`scripts/recon_3d/run_pde_multiview.py`)
  * 支持以 `--img_id <image_id>` 方式直接一键加载 `results/multiview_data/<image_id>/maps/` 下的所有有效视角。
  * 可选配置使用视角子集（例如 `--views front left right back` 或 `--views front back`）。
