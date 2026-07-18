# train_hisa.py Usage

This document explains how to train strand-map prediction with `scripts/train_hisa.py`.

## 1. Basic command

```bash
pixi run python -m scripts.train_hisa
```

The script uses:
- dataset root: `./datasets/HiSa_HiDa`
- train split: `./datasets/HiSa_HiDa/split_train.json`
- masked L1 loss with paper-style normalization: `sum(|pred-gt|*M)/(C*sum(M))`
- training augmentation enabled by default (random rotation, scale, translation, horizontal flip)

## 2. Choose backbone

### U-Net

```bash
pixi run python -m scripts.train_hisa --img2strand_backbone unet
```

### HRNet (timm)

```bash
pixi run python -m scripts.train_hisa --img2strand_backbone hrnet --hrnet_variant hrnet_w18
```

Supported HRNet variants:
- `hrnet_w18`
- `hrnet_w32`
- `hrnet_w48`

Use ImageNet pretrained weights:

```bash
pixi run python -m scripts.train_hisa \
  --img2strand_backbone hrnet \
  --hrnet_variant hrnet_w32 \
  --hrnet_pretrained
```

## 3. Checkpoint behavior

Checkpoints are saved by model tag to separate directories:
- U-Net: `./checkpoints/img2hairstep/unet/`
- HRNet-W18: `./checkpoints/img2hairstep/hrnet_w18/`
- HRNet-W32: `./checkpoints/img2hairstep/hrnet_w32/`
- HRNet-W48: `./checkpoints/img2hairstep/hrnet_w48/`

File naming:
- `img2strand_<model_tag>_epoch_<epoch>.pth`

Example:
- `img2strand_hrnet_w18_epoch_50.pth`

## 4. Resume training

### A) Auto resume latest checkpoint in current model-tag directory

```bash
pixi run python -m scripts.train_hisa \
  --img2strand_backbone hrnet \
  --hrnet_variant hrnet_w18 \
  --continue_train
```

### B) Resume from a specific checkpoint path

```bash
pixi run python -m scripts.train_hisa \
  --img2strand_backbone hrnet \
  --hrnet_variant hrnet_w18 \
  --continue_train \
  --checkpoint_img2strand ./checkpoints/img2hairstep/hrnet_w18/img2strand_hrnet_w18_epoch_200.pth \
  --resume_epoch 200
```

## 5. Common training controls

```bash
pixi run python -m scripts.train_hisa \
  --img2strand_backbone hrnet \
  --hrnet_variant hrnet_w18 \
  --batch_size 12 \
  --num_threads 12 \
  --pin_memory \
  --learning_rate 1e-4 \
  --num_epoch 1001 \
  --freq_save 20
```

Main arguments:
- `--batch_size`: training batch size
- `--num_threads`: DataLoader workers
- `--pin_memory`: enable pin memory for faster host-to-GPU copy
- `--learning_rate`: Adam learning rate
- `--num_epoch`: total epoch count
- `--freq_save`: checkpoint save frequency

## 6. High-VRAM tuning tips (48 GB GPU)

Start from:
- `--batch_size 12 --num_threads 12 --pin_memory`

If stable, try larger batch:
- `--batch_size 16`
- `--batch_size 20`

If out-of-memory happens, reduce in this order:
1. decrease `--batch_size`
2. decrease `--num_threads`

## 7. Notes

- `--hrnet_pretrained` needs to download pretrained weights from the model hub on first run.
- If network is unstable, run once without `--hrnet_pretrained` to verify the training pipeline.
