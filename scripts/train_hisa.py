import os
import json
import glob
import re
import random
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio.v2 as imageio
import cv2
from tqdm import tqdm

from lib.options import BaseOptions
from lib.model.img2hairstep.model_factory import create_img2strand_model


def get_model_tag(opt):
    backbone = getattr(opt, 'img2strand_backbone', 'unet').lower()
    if backbone == 'hrnet':
        return getattr(opt, 'hrnet_variant', 'hrnet_w18')
    return 'unet'


def find_latest_checkpoint(save_dir, model_tag):
    pattern = os.path.join(save_dir, f'img2strand_{model_tag}_epoch_*.pth')
    candidates = glob.glob(pattern)
    if not candidates:
        return None, -1

    best_path = None
    best_epoch = -1
    for path in candidates:
        name = os.path.basename(path)
        match = re.search(r'_epoch_(\d+)\.pth$', name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = path
    return best_path, best_epoch

class HiSaDataset(Dataset):
    def __init__(self, data_root, split_file, augment=False):
        self.data_root = data_root
        self.augment = augment

        # Augmentation setup follows the paper description.
        self.max_rotate_deg = 25.0
        self.scale_min = 0.90
        self.scale_max = 1.10
        self.max_translate_ratio = 0.10
        self.hflip_prob = 0.5

        with open(split_file, 'r') as f:
            self.items = json.load(f)
            
        self.img_dir = os.path.join(data_root, 'img')
        self.seg_dir = os.path.join(data_root, 'seg')
        self.body_dir = os.path.join(data_root, 'body_img')
        self.strand_dir = os.path.join(data_root, 'strand_map')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Load images robustly
        rgb_raw = imageio.imread(os.path.join(self.img_dir, item))
        if len(rgb_raw.shape) == 2: rgb_raw = np.stack([rgb_raw]*3, axis=-1)
        rgb_img = rgb_raw[:, :, 0:3] / 255.
        
        mask_raw = imageio.imread(os.path.join(self.seg_dir, item))
        if len(mask_raw.shape) == 3:
            mask_raw = mask_raw[:, :, 0]
        mask = (mask_raw / 255. > 0.5).astype(np.float32)
        
        body_raw = imageio.imread(os.path.join(self.body_dir, item))
        if len(body_raw.shape) == 3:
            body_raw = body_raw[:, :, 0]
        body = (body_raw / 255. > 0.5).astype(np.float32) * (1.0 - mask)
        
        # Load ground truth strand map
        # strand_map is saved as [mask+body*0.5, strand_x*mask, strand_y*mask]
        strand_map_gt = imageio.imread(os.path.join(self.strand_dir, item)) / 255.
        
        # The ground truth strand prediction (2 channels)
        strand_gt = strand_map_gt[:, :, 1:3]

        if self.augment:
            rgb_img, mask, body, strand_gt = self.apply_geometric_augmentation(rgb_img, mask, body, strand_gt)
        
        # Apply mask
        rgb_img = rgb_img * mask[:, :, None]
        
        # Convert to tensors
        rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        strand_gt = torch.from_numpy(strand_gt).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return rgb_img, strand_gt, mask

    def apply_geometric_augmentation(self, rgb_img, mask, body, strand_gt):
        h, w = rgb_img.shape[:2]

        angle = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
        scale = random.uniform(self.scale_min, self.scale_max)
        tx = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * w
        ty = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * h

        center = (w * 0.5, h * 0.5)
        mat = cv2.getRotationMatrix2D(center, angle, scale)
        mat[0, 2] += tx
        mat[1, 2] += ty

        rgb_img = cv2.warpAffine(
            rgb_img,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        mask = cv2.warpAffine(
            mask,
            mat,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        body = cv2.warpAffine(
            body,
            mat,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        strand_gt = cv2.warpAffine(
            strand_gt,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        # Rotate strand vector directions with the same image-plane rotation.
        # Assume strand channels are normalized from [-1, 1] into [0, 1].
        strand_vec = strand_gt * 2.0 - 1.0
        rad = np.deg2rad(angle)
        c = np.cos(rad)
        s = np.sin(rad)
        rot = np.array([[c, s], [-s, c]], dtype=np.float32)
        strand_vec = strand_vec @ rot.T

        if random.random() < self.hflip_prob:
            rgb_img = np.ascontiguousarray(rgb_img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
            body = np.ascontiguousarray(body[:, ::-1])
            strand_vec = np.ascontiguousarray(strand_vec[:, ::-1, :])
            strand_vec[:, :, 0] *= -1.0

        strand_gt = (strand_vec + 1.0) * 0.5
        strand_gt = np.clip(strand_gt, 0.0, 1.0)

        mask = (mask > 0.5).astype(np.float32)
        body = (body > 0.5).astype(np.float32)

        return rgb_img, mask, body, strand_gt

def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = './datasets/HiSa_HiDa'
    train_split = os.path.join(data_root, 'split_train.json')
    
    if not os.path.exists(train_split):
        print(f"Error: Training split not found at {train_split}")
        return
        
    print("Loading dataset...")
    train_dataset = HiSaDataset(data_root, train_split, augment=True)

    loader_kwargs = {
        'batch_size': opt.batch_size,
        'shuffle': True,
        'num_workers': opt.num_threads,
        'pin_memory': opt.pin_memory,
    }
    if opt.num_threads > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    
    model = create_img2strand_model(opt).to(device)

    model_tag = get_model_tag(opt)
    save_dir = os.path.join('./checkpoints/img2hairstep', model_tag)
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = opt.resume_epoch + 1
    
    # Optionally resume from checkpoint.
    if opt.continue_train:
        resume_path = opt.checkpoint_img2strand
        resume_epoch = opt.resume_epoch

        if not os.path.exists(resume_path):
            latest_path, latest_epoch = find_latest_checkpoint(save_dir, model_tag)
            if latest_path is not None:
                resume_path = latest_path
                resume_epoch = latest_epoch
                print(f"Auto-resume from latest checkpoint: {resume_path}")

        if os.path.exists(resume_path):
            print(f"Loading checkpoint from {resume_path}")
            model.load_state_dict(torch.load(resume_path, map_location=device))
            if resume_epoch >= 0:
                start_epoch = max(start_epoch, resume_epoch + 1)
        else:
            print("Warning: continue_train is set but no checkpoint was found.")
        
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    
    print(f"Starting training for {opt.num_epoch} epochs...")
    for epoch in range(start_epoch, opt.num_epoch):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{opt.num_epoch-1}")
        for rgb_img, strand_gt, mask in pbar:
            rgb_img = rgb_img.to(device)
            strand_gt = strand_gt.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            strand_pred = model(rgb_img)
            
            # Paper-style masked L1: sum(|pred-gt|*M) / (C * sum(M))
            strand_pred_masked = strand_pred * mask
            strand_gt_masked = strand_gt * mask

            abs_err = torch.abs(strand_pred_masked - strand_gt_masked).sum()
            valid = (mask.sum() * strand_pred.shape[1]).clamp_min(1.0)
            loss = abs_err / valid
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} Average Loss: {epoch_loss:.6f}")
        
        if epoch % opt.freq_save == 0 or epoch == opt.num_epoch - 1:
            save_path = os.path.join(save_dir, f'img2strand_{model_tag}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    opt = BaseOptions().parse()
    # Raise default batch size so larger GPUs are used more effectively.
    if opt.batch_size == 1:
        opt.batch_size = 8
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    train(opt)
