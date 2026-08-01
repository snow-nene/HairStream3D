#!/usr/bin/env bash
# ============================================================
# HairStep 结构张量预训练全流程脚本
# 
# 步骤:
#   Step 1: SAM 抠图 (48,252 张原始图片)
#   Step 2: 结构张量生成伪标签 (strand_map + confidence)
#   Step 3: HRNet-w32 预训练 (100 epochs, 三阶段置信度调度)
#
# Checkpoint 保存频率: 每 10 个 epoch 保存一次
# (epoch 0, 10, 20, ..., 90, 99)
# ============================================================

set -e  # 任何命令失败立即退出

cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  HairStep 结构张量预训练全流程                              ║"
echo "║  数据集: 48,252 张 → HRNet-w32 预训练 100 epoch            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "=== Step 1/3: SAM 抠图（约 25 分钟） ==="
PYTHONPATH=. pixi run python scripts/train/batch_generate_masks.py \
  --input_dir datasets/0002.mqset \
  --output_dir datasets/pretrain
echo "  ✅ Step 1 完成"
echo ""

echo "=== Step 2/3: 结构张量伪标签生成（约 45 秒） ==="
PYTHONPATH=. pixi run python scripts/train/generate_structure_labels.py
echo "  ✅ Step 2 完成"
echo ""

echo "=== Step 3/3: HRNet-w32 预训练 100 epochs（约 2 小时） ==="
echo "  Checkpoint 保存频率: 每 10 epoch"
echo "  置信度调度: epoch 0-32 → 0.8, 33-65 → 0.5, 66-99 → 0.2"
echo ""
PYTHONPATH=. pixi run python scripts/train/pretrain_structure.py \
  --img2strand_backbone hrnet \
  --hrnet_variant hrnet_w32 \
  --hrnet_pretrained \
  --num_epoch 100 \
  --batch_size 8 \
  --freq_save 10 \
  --w_l1 1.0 --w_cos 1.0

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  预训练完成!                                                ║"
echo "║  Checkpoints: checkpoints/pretrain/hrnet_w32/               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Checkpoint 列表 (每 10 epoch):"
ls -lh checkpoints/pretrain/hrnet_w32/ 2>/dev/null
echo ""
echo "下一步: 在 HiSa 数据集上微调"
echo "  PYTHONPATH=. pixi run python scripts/train/train_hisa.py \\"
echo "    --img2strand_backbone hrnet --hrnet_variant hrnet_w32 \\"
echo "    --continue_train \\"
echo "    --checkpoint_img2strand ./checkpoints/pretrain/hrnet_w32/img2strand_hrnet_w32_epoch_99.pth \\"
echo "    --w_l1 1.0 --w_cos 1.0 --w_tv 0.1 --w_struct 0.05 --w_aux 0.3 \\"
echo "    --num_epoch 200"
echo ""
echo "评估预训练效果:"
echo "  PYTHONPATH=. pixi run python scripts/test/eval_losses.py \\"
echo "    --backbone hrnet --hrnet_variant hrnet_w32 \\"
echo "    --ckpt checkpoints/pretrain/hrnet_w32/img2strand_hrnet_w32_epoch_99.pth \\"
echo "    --multi_scale false --splits train"