import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Import the dataset from train_hisa so we don't have to duplicate the logic
from scripts.train_hisa import HiSaDataset
from lib.options import BaseOptions
from lib.model.img2hairstep.UNet import Model

def test(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = './datasets/HiSa_HiDa'
    test_split = os.path.join(data_root, 'split_test.json')
    
    if not os.path.exists(test_split):
        print(f"Error: Test split not found at {test_split}")
        return
        
    print("Loading test dataset...")
    test_dataset = HiSaDataset(data_root, test_split)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads)
    
    model = Model().to(device)
    
    # Evaluate with the checkpoint specified in options, which is default or the one just saved
    checkpoint_path = opt.checkpoint_img2strand
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Evaluating with random weights.")
        
    model.eval()
    criterion = nn.L1Loss()
    running_loss = 0.0
    
    print(f"Starting evaluation on {len(test_dataset)} test samples...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for rgb_img, strand_gt, mask in pbar:
            rgb_img = rgb_img.to(device)
            strand_gt = strand_gt.to(device)
            mask = mask.to(device)
            
            strand_pred = model(rgb_img)
            
            # Apply mask
            strand_pred_masked = strand_pred * mask
            strand_gt_masked = strand_gt * mask
            
            loss = criterion(strand_pred_masked, strand_gt_masked)
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': loss.item()})
            
    avg_loss = running_loss / len(test_loader)
    print(f"Test Complete. Average L1 Loss: {avg_loss:.6f}")

if __name__ == "__main__":
    opt = BaseOptions().parse()
    if opt.batch_size == 1:
        opt.batch_size = 4
    test(opt)
