import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OsapiensDataset
from models.fusion_net import FusionNet

def confidence_weighted_bce_loss(logits, targets):
    """BCE with logits — targets are already soft [0,1] consensus values."""
    return F.binary_cross_entropy_with_logits(logits, targets)

def train_one_epoch(model, dataloader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        s1  = batch["s1"].to(device)   # (B, T, C_s1, H, W)
        s2  = batch["s2"].to(device)   # (B, T, C_s2, H, W)
        aef = batch["aef"].to(device)  # (B, C_aef, H, W)
        target = batch["label"].to(device)  # (B, 1, H, W) soft consensus
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            logits = model(s1, s2, aef)       # (B, 1, H, W)
            loss = confidence_weighted_bce_loss(logits, target)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
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
    DATA_ROOT = "data/makeathon-challenge/" # Adjust relative path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # We create the dataset with a sequence length of 12 (approx 1 year dataset)
    dataset = OsapiensDataset(data_root=DATA_ROOT, split="train", seq_len=12)
    # Tuned for 192GB VRAM and 20 vCPUs
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True, collate_fn=pad_collate_fn)
    
    model = FusionNet(s1_channels=2, s2_channels=10, aef_channels=768, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda')
    
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        avg_loss = train_one_epoch(model, dataloader, optimizer, scaler, device)
        print(f"Epoch Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")
