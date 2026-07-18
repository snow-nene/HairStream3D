import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio
from tqdm import tqdm

from lib.options import BaseOptions
from lib.model.img2hairstep.UNet import Model

class HiSaDataset(Dataset):
    def __init__(self, data_root, split_file):
        self.data_root = data_root
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
        if len(mask_raw.shape) == 3: mask_raw = mask_raw[:, :, 0]
        mask = (mask_raw / 255. > 0.5)[:, :, None]
        
        body_raw = imageio.imread(os.path.join(self.body_dir, item))
        if len(body_raw.shape) == 3: body_raw = body_raw[:, :, 0]
        body = (body_raw / 255. > 0.5)[:, :, None] * (1 - mask)
        
        # Load ground truth strand map
        # strand_map is saved as [mask+body*0.5, strand_x*mask, strand_y*mask]
        strand_map_gt = imageio.imread(os.path.join(self.strand_dir, item)) / 255.
        
        # The ground truth strand prediction (2 channels)
        strand_gt = strand_map_gt[:, :, 1:3]
        
        # Apply mask
        rgb_img = rgb_img * mask
        
        # Convert to tensors
        rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        strand_gt = torch.from_numpy(strand_gt).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        
        return rgb_img, strand_gt, mask

def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = './datasets/HiSa_HiDa'
    train_split = os.path.join(data_root, 'split_train.json')
    
    if not os.path.exists(train_split):
        print(f"Error: Training split not found at {train_split}")
        return
        
    print("Loading dataset...")
    train_dataset = HiSaDataset(data_root, train_split)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads)
    
    model = Model().to(device)
    
    # Optionally load pretrained
    if os.path.exists(opt.checkpoint_img2strand) and opt.continue_train:
        print(f"Loading checkpoint from {opt.checkpoint_img2strand}")
        model.load_state_dict(torch.load(opt.checkpoint_img2strand, map_location=device))
        
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    criterion = nn.L1Loss() # Using L1 loss for strand maps
    
    save_dir = './checkpoints/img2hairstep'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training for {opt.num_epoch} epochs...")
    for epoch in range(opt.resume_epoch + 1, opt.num_epoch):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{opt.num_epoch-1}")
        for rgb_img, strand_gt, mask in pbar:
            rgb_img = rgb_img.to(device)
            strand_gt = strand_gt.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            strand_pred = model(rgb_img)
            
            # Apply mask to prediction for loss calculation
            strand_pred_masked = strand_pred * mask
            strand_gt_masked = strand_gt * mask
            
            loss = criterion(strand_pred_masked, strand_gt_masked)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} Average Loss: {epoch_loss:.6f}")
        
        if epoch % opt.freq_save == 0 or epoch == opt.num_epoch - 1:
            save_path = os.path.join(save_dir, f'img2strand_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    opt = BaseOptions().parse()
    # Modify default batch size if it's too small for training
    if opt.batch_size == 1:
        opt.batch_size = 4
    train(opt)
