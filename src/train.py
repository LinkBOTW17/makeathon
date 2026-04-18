import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OsapiensDataset
from models.fusion_net import FusionNet
from label_consensus import generate_consensus_label

def confidence_weighted_bce_loss(logits, targets, confidences):
    """
    Binary Cross Entropy that weights pixel losses by the confidence of the consensus label.
    """
    bce = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
    weighted_bce = bce * confidences
    return weighted_bce.mean()

def train_one_epoch(model, dataloader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        aef = batch["aef"].to(device)
        labels = batch["labels"]
        
        # Ensure proper shapes if data is empty
        if s1.nelement() == 0: s1 = torch.zeros(1, 1, 2, 256, 256).to(device)
        if s2.nelement() == 0: s2 = torch.zeros(1, 1, 10, 256, 256).to(device)
        if aef.nelement() == 0: aef = torch.zeros(1, 768, 256, 256).to(device)
        
        target, conf = generate_consensus_label(labels)
        if target is None:
            continue # skip if no labels at all
            
        target = target.to(device)
        conf = conf.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            logits = model(s1, s2, aef)
            loss = confidence_weighted_bce_loss(logits, target, conf)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

if __name__ == "__main__":
    DATA_ROOT = "data/makeathon-challenge/" # Adjust relative path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # We create the dataset with a sequence length of 12 (approx 1 year dataset)
    dataset = OsapiensDataset(data_root=DATA_ROOT, split="train", seq_len=12)
    # Tuned for 192GB VRAM and 20 vCPUs
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
    
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
