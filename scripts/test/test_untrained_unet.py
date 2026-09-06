import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from scripts.train.train_hisa import HiSaDataset
from lib.model.img2hairstep.UNet import Model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an untrained UNet on HiSa/HiDa test split.")
    parser.add_argument("--data_root", type=str, default="./datasets/HiSa_HiDa")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Use only first N test samples. -1 means full test set.")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save a few qualitative visualization images.")
    parser.add_argument("--max_vis", type=int, default=16,
                        help="Maximum number of visualizations to save.")
    parser.add_argument("--vis_dir", type=str, default="./results/untrained_unet_test")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_vis_3ch(strand_2ch, mask_1ch):
    strand = np.clip(strand_2ch, 0.0, 1.0)
    ch0 = strand[..., 0] * mask_1ch[..., 0]
    ch1 = strand[..., 1] * mask_1ch[..., 0]
    ch2 = np.zeros_like(ch0)
    return np.stack([ch0, ch1, ch2], axis=-1)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_split = os.path.join(args.data_root, "split_test.json")
    if not os.path.exists(test_split):
        raise FileNotFoundError(f"Test split not found: {test_split}")

    dataset = HiSaDataset(args.data_root, test_split)
    if args.max_samples > 0:
        max_n = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, list(range(max_n)))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_threads,
    )

    # Randomly initialized UNet: no checkpoint loading.
    model = Model().to(device)
    model.eval()

    os.makedirs(args.vis_dir, exist_ok=True)

    total_abs_err = 0.0
    total_valid = 0.0
    saved_vis = 0

    with torch.no_grad():
        for batch_idx, (rgb_img, strand_gt, mask) in enumerate(tqdm(loader, desc="Untrained UNet testing")):
            rgb_img = rgb_img.to(device)
            strand_gt = strand_gt.to(device)
            mask = mask.to(device)

            strand_pred = model(rgb_img)

            abs_err = torch.abs((strand_pred - strand_gt) * mask)
            total_abs_err += abs_err.sum().item()
            total_valid += (mask.sum().item() * strand_pred.shape[1])

            if args.save_vis and saved_vis < args.max_vis:
                pred_np = strand_pred.detach().cpu().permute(0, 2, 3, 1).numpy()
                gt_np = strand_gt.detach().cpu().permute(0, 2, 3, 1).numpy()
                rgb_np = rgb_img.detach().cpu().permute(0, 2, 3, 1).numpy()
                mask_np = mask.detach().cpu().permute(0, 2, 3, 1).numpy()

                bsz = pred_np.shape[0]
                for i in range(bsz):
                    if saved_vis >= args.max_vis:
                        break

                    rgb = np.clip(rgb_np[i], 0.0, 1.0)
                    gt_vis = to_vis_3ch(gt_np[i], mask_np[i])
                    pred_vis = to_vis_3ch(pred_np[i], mask_np[i])

                    panel = np.concatenate([rgb, gt_vis, pred_vis], axis=1)
                    out_path = os.path.join(args.vis_dir, f"sample_{saved_vis:04d}.png")
                    imageio.imwrite(out_path, (panel * 255.0).astype(np.uint8))
                    saved_vis += 1

    mean_l1 = total_abs_err / max(total_valid, 1.0)

    print("=" * 60)
    print("Untrained UNet evaluation finished")
    print(f"Samples evaluated: {len(dataset)}")
    print(f"Masked mean L1 (2ch): {mean_l1:.6f}")
    if args.save_vis:
        print(f"Saved visualizations: {saved_vis}")
        print(f"Visualization dir: {args.vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
