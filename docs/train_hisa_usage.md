# train_hisa.py 使用说明

本文档说明如何使用 `scripts/train/train_hisa.py` 来训练 strand-map 预测模型。

```bash
# 论文原作 baseline（仅 L1）
PYTHONPATH=. pixi run python scripts/train/train_hisa.py \
  --img2strand_backbone hrnet --hrnet_variant hrnet_w32 \
  --w_l1 1.0 --w_cos 0.0 --w_tv 0.0 --w_struct 0.0 --w_aux 0.0

# L1 + 余弦 + TV + 结构 + 多尺度（全量）
PYTHONPATH=. pixi run python scripts/train/train_hisa.py \
  --img2strand_backbone hrnet --hrnet_variant hrnet_w32 \
  --w_l1 1.0 --w_cos 1.0 --w_tv 0.0 --w_struct 0.0 --w_aux 0.3
```

```bash
# 仅余弦相似度（baseline）
pixi run python scripts/train/train_hisa.py --w_cos 1.0 --w_tv 0.0 --w_struct 0.0 --w_aux 0.0

# 余弦相似度 + TV 平滑
pixi run python scripts/train/train_hisa.py --w_cos 1.0 --w_tv 0.1 --w_struct 0.0 --w_aux 0.0

# 余弦相似度 + TV 平滑 + 结构相似性
pixi run python scripts/train/train_hisa.py --w_cos 1.0 --w_tv 0.1 --w_struct 0.05 --w_aux 0.0

# 全量损失（推荐）
pixi run python scripts/train/train_hisa.py --w_cos 1.0 --w_tv 0.1 --w_struct 0.05 --w_aux 0.3
```

## 1. 基础命令

```bash
pixi run python -m scripts.train.train_hisa
```

该脚本默认使用以下配置：
- 数据集根目录: `./datasets/HiSa_HiDa`
- 训练集拆分配置: `./datasets/HiSa_HiDa/split_train.json`
- 掩蔽的 L1 损失（采用论文风格的归一化）: `sum(|pred-gt|*M)/(C*sum(M))`
- 默认开启训练数据增强（随机旋转、缩放、平移、水平翻转）

## 2. 选择主干网络 (Backbone)

### U-Net

```bash
pixi run python -m scripts.train.train_hisa --img2strand_backbone unet
```

### HRNet (timm)

```bash
pixi run python -m scripts.train.train_hisa --img2strand_backbone hrnet --hrnet_variant hrnet_w18
```

支持的 HRNet 变体：
- `hrnet_w18`
- `hrnet_w32`
- `hrnet_w48`

使用 ImageNet 预训练权重：

```bash
pixi run python -m scripts.train.train_hisa \
  --img2strand_backbone hrnet \
  --hrnet_variant hrnet_w32 \
  --hrnet_pretrained
```

## 3. 模型断点 (Checkpoint) 保存机制

断点将根据模型标签保存至不同的独立目录中：
- U-Net: `./checkpoints/img2hairstep/unet/`
- HRNet-W18: `./checkpoints/img2hairstep/hrnet_w18/`
- HRNet-W32: `./checkpoints/img2hairstep/hrnet_w32/`
- HRNet-W48: `./checkpoints/img2hairstep/hrnet_w48/`

文件命名规则:
- `img2strand_<model_tag>_epoch_<epoch>.pth`

示例:
- `img2strand_hrnet_w18_epoch_50.pth`

## 4. 恢复训练

### A) 自动恢复当前模型标签目录下的最新断点

```bash
pixi run python -m scripts.train.train_hisa \
  --img2strand_backbone hrnet \
  --hrnet_variant hrnet_w18 \
  --continue_train
```

### B) 从指定断点路径恢复

```bash
pixi run python -m scripts.train.train_hisa \
  --img2strand_backbone hrnet \
  --hrnet_variant hrnet_w18 \
  --continue_train \
  --checkpoint_img2strand ./checkpoints/img2hairstep/hrnet_w18/img2strand_hrnet_w18_epoch_200.pth \
  --resume_epoch 200
```

## 5. 常用训练参数控制

```bash
pixi run python -m scripts.train.train_hisa \
  --img2strand_backbone hrnet \
  --hrnet_variant hrnet_w18 \
  --batch_size 12 \
  --num_threads 12 \
  --pin_memory \
  --learning_rate 1e-4 \
  --num_epoch 1001 \
  --freq_save 20
```

主要参数说明:
- `--batch_size`: 训练批次大小
- `--num_threads`: DataLoader 的工作线程数
- `--pin_memory`: 启用锁页内存 (pin memory) 以加快从主机到 GPU 的拷贝速度
- `--learning_rate`: Adam 优化器学习率
- `--num_epoch`: 训练总轮数
- `--freq_save`: 断点保存频率

## 6. 高显存 (48 GB GPU) 调优技巧

起始配置：
- `--batch_size 12 --num_threads 12 --pin_memory`

如果训练稳定，可尝试增加批次大小：
- `--batch_size 16`
- `--batch_size 20`

如果发生显存不足 (OOM)，请按以下顺序减小参数：
1. 减小 `--batch_size`
2. 减小 `--num_threads`

## 7. 备注

- 首次运行时，使用 `--hrnet_pretrained` 需要从模型中心下载预训练权重。
- 如果网络不稳定，可以尝试在不添加 `--hrnet_pretrained` 的情况下运行一次，以验证训练管线是否正常。
