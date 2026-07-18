#!/usr/bin/env python3
"""Evaluate HairStep model (UNet or HRNet) on train/test splits with all loss components.

Usage:
    # UNet pretrained checkpoint (full losses)
    PYTHONPATH=. pixi run python scripts/eval_losses.py \
        --backbone unet --ckpt ./checkpoints/img2hairstep/img2strand.pth

    # HRNet epoch_0
    PYTHONPATH=. pixi run python scripts/eval_losses.py \
        --backbone hrnet --hrnet_variant hrnet_w32 \
        --ckpt ./checkpoints/img2hairstep/hrnet_w32/img2strand_hrnet_w32_epoch_0.pth
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.model.img2hairstep.model_factory import create_img2strand_model
from lib.model.img2hairstep.criterion.hairstep_losses import HairStepLoss
from scripts.train_hisa import HiSaDataset


def _make_opt(backbone, hrnet_variant, multi_scale):
    """Minimal options stub for model factory."""
    opt = argparse.Namespace()
    opt.img2strand_backbone = backbone
    opt.hrnet_variant = hrnet_variant
    opt.hrnet_pretrained = False
    opt.hrnet_decoder_channels = 128
    opt.multi_scale_supervision = multi_scale
    return opt


def load_model(backbone, hrnet_variant, ckpt_path, multi_scale, device):
    opt = _make_opt(backbone, hrnet_variant, multi_scale)
    model = create_img2strand_model(opt).to(device)

    state = torch.load(ckpt_path, map_location=device)
    # Handle DataParallel / compiled wrappers
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected}")
    model.eval()
    return model


def evaluate_split(model, dataset, criterion, device, desc):
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False)

    keys = ['l1', 'cos', 'tv', 'struct', 'aux', 'total']
    sums = {k: 0.0 for k in keys}
    n_batches = 0

    with torch.no_grad():
        for rgb_img, strand_gt, mask in tqdm(loader, desc=desc, leave=False):
            rgb_img = rgb_img.to(device)
            strand_gt = strand_gt.to(device)
            mask = mask.to(device)

            out = model(rgb_img)
            if isinstance(out, tuple):
                pred, aux_preds = out
            else:
                pred, aux_preds = out, None

            _, losses = criterion(pred, aux_preds, strand_gt, mask)
            for k in keys:
                sums[k] += losses[k].item()
            n_batches += 1

    n = max(n_batches, 1)
    return {k: sums[k] / n for k in keys}


def main():
    parser = argparse.ArgumentParser(description='Evaluate HairStep losses')
    parser.add_argument('--backbone', default='unet', choices=['unet', 'hrnet'])
    parser.add_argument('--hrnet_variant', default='hrnet_w32',
                        choices=['hrnet_w18', 'hrnet_w32', 'hrnet_w48'])
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint .pth')
    parser.add_argument('--multi_scale', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--w_l1', type=float, default=1.0)
    parser.add_argument('--w_cos', type=float, default=1.0)
    parser.add_argument('--w_tv', type=float, default=0.1)
    parser.add_argument('--w_struct', type=float, default=0.05)
    parser.add_argument('--w_aux', type=float, default=0.3)
    parser.add_argument('--splits', nargs='+', default=['train', 'test'],
                        help='Which splits to evaluate (train and/or test)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {args.backbone}" + (f" ({args.hrnet_variant})" if args.backbone == 'hrnet' else ""))
    print(f"Checkpoint: {args.ckpt}")

    # ── Load model ──
    model = load_model(args.backbone, args.hrnet_variant, args.ckpt,
                       args.multi_scale, device)

    # ── Loss criterion (all enabled for evaluation; weights affect 'total' only) ──
    criterion = HairStepLoss(
        w_l1=args.w_l1, w_cos=args.w_cos, w_tv=args.w_tv,
        w_struct=args.w_struct, w_aux=args.w_aux,
    )

    data_root = './datasets/HiSa_HiDa'

    # ── Evaluate ──
    results = {}
    for split_name in args.splits:
        split_file = os.path.join(data_root, f'split_{split_name}.json')
        if not os.path.exists(split_file):
            print(f"  [skip] {split_name}: split file not found")
            continue
        ds = HiSaDataset(data_root, split_file, augment=False)
        metrics = evaluate_split(model, ds, criterion, device,
                                 desc=f"{split_name}")
        results[split_name] = {'n_samples': len(ds), **metrics}
        print(f"  {split_name}: {len(ds)} samples, "
              f"l1={metrics['l1']:.4f}, cos={metrics['cos']:.4f}, "
              f"total={metrics['total']:.4f}")

    # ── Pretty report ──
    print("\n" + "=" * 70)
    print(f"  EVALUATION REPORT")
    print(f"  Model:      {args.backbone}" + (f" ({args.hrnet_variant})" if args.backbone == 'hrnet' else ""))
    print(f"  Checkpoint: {args.ckpt}")
    if args.backbone == 'unet':
        print(f"  Multi-scale: N/A (UNet has no aux branches)")
    else:
        print(f"  Multi-scale aux: {'enabled' if args.multi_scale else 'disabled'}")
    print(f"  Loss weights (for total): l1={args.w_l1}, cos={args.w_cos}, "
          f"tv={args.w_tv}, struct={args.w_struct}, aux={args.w_aux}")
    print("-" * 70)

    header = f"  {'Split':<7} {'Samples':>8}" + \
             ''.join(f" {'L1':>9} {'Cos':>9} {'TV':>9} {'Struct':>9} {'Aux':>9} {'Total':>9}".split())
    # simpler header
    print(f"  {'Split':<7} {'Samples':>8}  {'L1':>9}  {'Cos':>9}  {'TV':>9}  {'Struct':>9}  {'Aux':>9}  {'Total':>9}")
    print("  " + "-" * 65)

    for split_name in args.splits:
        if split_name not in results:
            continue
        r = results[split_name]
        print(f"  {split_name:<7} {r['n_samples']:>8}  "
              f"{r['l1']:>9.4f}  {r['cos']:>9.4f}  {r['tv']:>9.4f}  "
              f"{r['struct']:>9.4f}  {r['aux']:>9.4f}  {r['total']:>9.4f}")

    print("=" * 70)
    print(f"\n  Note: L1 is masked MAE (HairStep Eq.2); Cos is 1-mean_cosine_sim.")
    print(f"        TV measures spatial smoothness; Struct uses gradient+laplacian.")
    print(f"        Aux is multi-scale branch supervision (HRNet only).")
    print(f"        Total = w_l1*L1 + w_cos*Cos + w_tv*TV + w_struct*Struct + w_aux*Aux")


if __name__ == '__main__':
    main()