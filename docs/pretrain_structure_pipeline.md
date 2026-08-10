# 结构张量预训练 Pipeline 设计方案

## 目标

利用 **48,252 张无标注头发图片**（`datasets/0002.mqset/`）通过结构张量自动生成方向场 + 置信度伪标签，对 HRNet-w32 进行预训练，再在 1,054 张 HiSa 标注数据上微调。

**核心价值**：零标注成本，结构张量提取的方向场与 Hairstep 的 strand_map 格式完全一致，网络可以无痛迁移。

---

## Pipeline 总览

```
datasets/0002.mqset/ (48,252 张原始图片)
        │
        ▼  [Step 1] SAM 抠图 → hair mask
        │
datasets/pretrain/img/   (512×512 RGB)
datasets/pretrain/seg/   (hair mask)
        │
        ▼  [Step 2] 结构张量 → 方向场 + 置信度
        │
datasets/pretrain/strand_map/     (3通道: [mask, strand_x, strand_y])
datasets/pretrain/confidence/     (coherence × mask)
        │
        ▼  [Step 3] 预训练 (HRNet-w32, 90 epochs)
        │
checkpoints/pretrain/hrnet_w32/
        │
        ▼  [Step 4] 微调 (HiSa 1,054 张标注数据)
        │
checkpoints/img2hairstep/hrnet_w32/  (最终模型)
```

---

## Step 1: 批量生成头发 mask（SAM3/SAM 抠图）

### 输入
- `datasets/0002.mqset/**/*.jpg`（48,252 张，31 个分类文件夹）

### 输出
- `datasets/pretrain/img/{hash}.png` — pad+resize 到 512×512 的 RGB 图像
- `datasets/pretrain/seg/{hash}.png` — SAM3（默认）或 SAM 生成的头发区域 mask（0/255）

### 实现
- `scripts/train/batch_generate_masks.py`
- 默认 `--seg_backend sam3`：subprocess 调用 `ext/sam3` 独立环境中的
  `scripts/infer_2d/sam3_seg_worker.py`（文本提示 + 形态学后处理）；
  `--seg_backend sam` 回退旧版 `SamPredictor` 点提示逻辑
- 多人场景默认只抠**主人物**及其头发（`--sam3_person_selection largest`）：
  person 实例先做高 IoU 去重，再综合中心位置、头发占比、面积和边界占用率选择主角；
  头发按整体重叠率和稳健距离归属，仅保留主角的头发；传 `all` 恢复"抠所有人"的旧行为
- 使用文件内容的 MD5 hash 作为文件名，避免重名

### 运行
```bash
# 冒烟测试（仅 5 张）
PYTHONPATH=. pixi run python scripts/train/batch_generate_masks.py --max_samples 5

# 全量（48,252 张）
PYTHONPATH=. pixi run python scripts/train/batch_generate_masks.py
```

---

## Step 2: 生成 0-180° 方向场 + 置信度图（结构张量）

### 方向场格式

结构张量输出 `(vx, vy)` 归一化向量，直接作为 `(cosθ, sinθ)`：

```
strand_map (3通道, 与 Hairstep 格式完全一致):
  R = mask (0 或 1)
  G = (field_x + 1) / 2 * 255    → cosθ 映射到 [0, 255]
  B = (field_y + 1) / 2 * 255    → sinθ 映射到 [0, 255]
```

### 置信度图

```
confidence = coherence × hair_mask
```
- `coherence` ∈ [0, 1]：结构张量的方向相干性，0 = 各向同性，1 = 完全方向性
- 只在 `confidence > threshold` 的像素上计算 loss

### 结构张量参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `sigma_inner` | 3.0 | 梯度平滑半径（小 → 捕捉细节） |
| `sigma_outer` | 9.0 | 结构张量积分半径（大 → 稳定方向） |
| `min_coherence` | 0.3 | 最小相干性阈值 |
| `min_brightness` | 0.1 | 最小亮度阈值 |
| `resize_factor` | 0.5 | 处理分辨率缩放（512→256，加速） |

### 实现
- `scripts/train/generate_structure_labels.py`
- 批量调用 `scripts/infer_2d/standard_pipeline.py` 的 `extract_structure_field()`
- 输出 `strand_map/` 和 `confidence/` 目录

### 运行
```bash
PYTHONPATH=. pixi run python scripts/train/generate_structure_labels.py \
  --input_dir datasets/pretrain \
  --output_dir datasets/pretrain
```

---

## Step 3: 预训练（置信度调度 + 多阶段学习）

### 核心思想

> 先让网络学习高置信度的整体结构，然后逐步引入细节信息。
> 网络学到的"线条是连续的"这个先验会自然弥补结构张量自身的断线问题。

### 三阶段置信度调度

| 阶段 | Epoch | confidence_threshold | 学习目标 |
|------|-------|:---:|------|
| Phase 1 | 0-29 | **0.8** | 只学纹理清晰、方向明确的核心区域（整体流向） |
| Phase 2 | 30-59 | **0.5** | 加入中等置信度区域（补充过渡/边缘细节） |
| Phase 3 | 60-89 | **0.2** | 全部区域，网络自行填补低置信度缝隙 |

### 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | HRNet-w32 (ImageNet 预训练) |
| 数据量 | ~48,252 张（有 mask 的有效子集） |
| Batch size | 8 |
| 学习率 | 1e-4 (Adam) |
| Epochs | 90 (每阶段 30) |
| 损失函数 | L1 (w=1.0) + Cos (w=1.0) |
| 保存频率 | 每 10 epoch |

### 实现
- `scripts/train/pretrain_structure.py`
- 新建 `PretrainDataset` 类，返回 `(rgb_img, strand_gt, mask, confidence)`
- 每 epoch 检查 `current_epoch` 动态更新 `confidence_threshold`

### 运行
```bash
PYTHONPATH=. pixi run python scripts/train/pretrain_structure.py \
  --img2strand_backbone hrnet --hrnet_variant hrnet_w32 --hrnet_pretrained \
  --num_epoch 90
```

---

## Step 4: 微调到 HiSa 标注数据集

加载预训练 checkpoint，在 1,054 张标注数据上微调：

```bash
PYTHONPATH=. pixi run python scripts/train/train_hisa.py \
  --img2strand_backbone hrnet --hrnet_variant hrnet_w32 \
  --continue_train \
  --checkpoint_img2strand ./checkpoints/pretrain/hrnet_w32/img2strand_hrnet_w32_epoch_89.pth \
  --resume_epoch -1 \
  --w_l1 1.0 --w_cos 1.0 --w_tv 0.1 --w_struct 0.05 --w_aux 0.3 \
  --num_epoch 200
```

**关键**：微调阶段恢复全部损失（TV + Struct + Aux），因为标注数据的方向场足够精确，正则化和多尺度约束才能真正发挥作用。

---

## 涉及文件清单

| 文件 | 操作 | 说明 |
|------|:---:|------|
| `docs/pretrain_structure_pipeline.md` | 新建 | 本文档 |
| `scripts/train/batch_generate_masks.py` | 新建 | 批量 SAM 抠图 |
| `scripts/train/generate_structure_labels.py` | 新建 | 批量结构张量提取 |
| `scripts/train/pretrain_structure.py` | 新建 | 预训练脚本 + 置信度调度 + PretrainDataset |
| `scripts/infer_2d/standard_pipeline.py` | 不改 | 已有 `extract_structure_field()` |
| `lib/model/img2hairstep/criterion/hairstep_losses.py` | 小改 | 新增 `confidence_mask` 参数 |
| 其他现有文件 | 不改 | 完全兼容 |

---

## 预期收益

- Epoch 0 的 L1 可能从 0.27 降至 < 0.10（方向场先验迁移）
- 微调 50 epoch 可能超越从零训练 200+ epoch 的效果
- 1,054 张小数据集过拟合风险大幅降低
- 对未见过的发型/光照更鲁棒
