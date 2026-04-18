import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OsapiensDataset
from models.fusion_net import FusionNet

def dice_loss_fn(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits).view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    dice_score = (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return 1. - dice_score

def combo_loss(logits, targets):
    """Combines BCE for pixel stability with Dice Loss for strict IoU polygon optimization."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_fn(logits, targets)
    return 0.5 * bce + 0.5 * dice

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Load and dynamically normalize raw satellite inputs to mean=0, std=1 to prevent exploding gradients
        def norm(x):
            if x.nelement() == 0: return x
            # Scrub geospatial NODATA (NaNs and infs mapping to zero or valid numbers)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            # Normalize across the batch/spatial dims
            return (x - x.mean()) / (x.std() + 1e-6)

        s1  = norm(batch["s1"].to(device))
        s2  = norm(batch["s2"].to(device))
        aef = norm(batch["aef"].to(device))
        
        target = batch["label"].to(device)  
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
        # Prevent any -9999.0 Earth Engine NoData values from causing BCE target errors by clamping to soft [0,1]
        target = torch.clamp(target, min=0.0, max=1.0)
        
        optimizer.zero_grad()
        
        logits = model(s1, s2, aef)       
        loss = combo_loss(logits, target)
            
        loss.backward()
        
        # Clip max gradients to prevent any remaining instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)


def pad_collate_fn(batch):
    """
    Custom collator that zero-pads spatial dimensions so images of natively different 
    sizes can be stacked into a single batch without stretching/resizing them.
    """
    # Find max H and W in this specific batch
    max_h = max([item["s2"].shape[-2] for item in batch])
    max_w = max([item["s2"].shape[-1] for item in batch])
    
    def pad_tensor(t):
        if t.nelement() == 0: return t
        h, w = t.shape[-2], t.shape[-1]
        pad_h = max_h - h
        pad_w = max_w - w
        # Pad right and bottom
        return F.pad(t, (0, pad_w, 0, pad_h), "constant", 0)

    collated = {
        "tile_id": [item["tile_id"] for item in batch],
        "s1": torch.stack([pad_tensor(item["s1"]) for item in batch]),
        "s2": torch.stack([pad_tensor(item["s2"]) for item in batch]),
        "aef": torch.stack([pad_tensor(item["aef"]) for item in batch]),
        "label": torch.stack([pad_tensor(item["label"]) for item in batch])
    }
    return collated

if __name__ == "__main__":
    import torch.backends.cudnn as cudnn
    
    # AMD ROCm Stability Fixes
    # Disables the MIOpen kernel autotuner that causes "Workspace ... is not allocated" warnings
    cudnn.benchmark = False
    cudnn.deterministic = True

    DATA_ROOT = "data/makeathon-challenge/" # Adjust relative path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = OsapiensDataset(data_root=DATA_ROOT, split="train", seq_len=12)
    # Scaled down batch_size for 1024x1024 time-series imagery to fit 192GB VRAM limits!
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=16, pin_memory=True, collate_fn=pad_collate_fn)
    
    print("Auto-detecting channel dimensions from the first batch...")
    first_batch = next(iter(dataloader))
    
    s1_dim = first_batch["s1"].shape[2] if first_batch["s1"].nelement() > 0 else 2
    s2_dim = first_batch["s2"].shape[2] if first_batch["s2"].nelement() > 0 else 10
    aef_dim = first_batch["aef"].shape[1] if first_batch["aef"].nelement() > 0 else 768
    
    print(f"Detected Channels -> S1: {s1_dim}, S2: {s2_dim}, AEF: {aef_dim}")
    
    model = FusionNet(s1_channels=s1_dim, s2_channels=s2_dim, aef_channels=aef_dim, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    num_epochs = 15
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        avg_loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")
