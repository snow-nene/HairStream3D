# Step 1: 抠图（5 张冒烟测试）
PYTHONPATH=. pixi run python scripts/train/batch_generate_masks.py \
  --input_dir datasets/0002.mqset \
  --output_dir datasets/pretrain \
  --max_samples 5

# Step 2: 伪标签生成（自动只处理已抠图的 5 张）
PYTHONPATH=. pixi run python scripts/train/generate_structure_labels.py --max_samples 5

# Step 3: 预训练（5 张图片，3 轮验证流程）
PYTHONPATH=. pixi run python scripts/train/pretrain_structure.py \
  --img2strand_backbone hrnet --hrnet_variant hrnet_w32 --hrnet_pretrained \
  --max_samples 5 --num_epoch 3 --batch_size 2

# 全量跑（去掉 --max_samples）
PYTHONPATH=. pixi run python scripts/train/batch_generate_masks.py
PYTHONPATH=. pixi run python scripts/train/generate_structure_labels.py
PYTHONPATH=. pixi run python scripts/train/pretrain_structure.py \
  --img2strand_backbone hrnet --hrnet_variant hrnet_w32 --hrnet_pretrained \
  --num_epoch 90

# ============================================================
# 评估 / 测试 loss
# ============================================================

# ---- 评估 UNet 预训练模型（HiSa 全量损失） ----
PYTHONPATH=. pixi run python scripts/test/eval_losses.py \
  --backbone unet \
  --ckpt ./checkpoints/img2hairstep/img2strand.pth \
  --splits train test

# ---- 评估 HRNet 预训练 checkpoint（在 HiSa 训练集上，关闭多尺度） ----
PYTHONPATH=. pixi run python scripts/test/eval_losses.py \
  --backbone hrnet --hrnet_variant hrnet_w32 \
  --ckpt checkpoints/pretrain/hrnet_w32/img2strand_hrnet_w32_epoch_1.pth \
  --multi_scale false --splits train

# ---- 评估 HRNet 微调后的 checkpoint（多尺度开启） ----
PYTHONPATH=. pixi run python scripts/test/eval_losses.py \
  --backbone hrnet --hrnet_variant hrnet_w32 \
  --ckpt checkpoints/img2hairstep/hrnet_w32/img2strand_hrnet_w32_epoch_49.pth \
  --multi_scale true --splits train test

# ---- 评估任意 HRNet checkpoint ----
PYTHONPATH=. pixi run python scripts/test/eval_losses.py \
  --backbone hrnet --hrnet_variant hrnet_w32 \
  --ckpt <checkpoint_path> \
  --splits train

# ---- 训练中评估 HiSa 全量损失（含 TV/Struct/Aux） ----
# 步骤：先用 continue_train 微调，再评估
PYTHONPATH=. pixi run python scripts/train/train_hisa.py \
  --img2strand_backbone hrnet --hrnet_variant hrnet_w32 \
  --continue_train \
  --checkpoint_img2strand ./checkpoints/pretrain/hrnet_w32/img2strand_hrnet_w32_epoch_89.pth \
  --w_l1 1.0 --w_cos 1.0 --w_tv 0.1 --w_struct 0.05 --w_aux 0.3 \
  --num_epoch 200
PYTHONPATH=. pixi run python scripts/test/eval_losses.py \
  --backbone hrnet --hrnet_variant hrnet_w32 \
  --ckpt checkpoints/img2hairstep/hrnet_w32/img2strand_hrnet_w32_epoch_199.pth \
  --multi_scale true --splits train test

# ---- 评估自定义损失权重 ----
PYTHONPATH=. pixi run python scripts/test/eval_losses.py \
  --backbone hrnet --hrnet_variant hrnet_w32 \
  --ckpt checkpoints/pretrain/hrnet_w32/img2strand_hrnet_w32_epoch_1.pth \
  --w_l1 1.0 --w_cos 1.0 --w_tv 0.1 --w_struct 0.05 --w_aux 0.3 \
  --splits train
