import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OsapiensDataset
from models.fusion_net import FusionNet
from train import pad_collate_fn

def calculate_metrics(preds, targets, threshold=0.5):
    """
    Calculates PyTorch pixel-level metrics required by the Osapiens Challenge.
    """
    # Binarize outputs
    preds_binary = (preds > threshold).float()
    targets_binary = (targets >= 0.5).float() # Weak labels are soft, round up

    TP = torch.sum((preds_binary == 1) & (targets_binary == 1)).item()
    FP = torch.sum((preds_binary == 1) & (targets_binary == 0)).item()
    FN = torch.sum((preds_binary == 0) & (targets_binary == 1)).item()
    TN = torch.sum((preds_binary == 0) & (targets_binary == 0)).item()

    # Metrics
    iou = TP / (TP + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    
    # "Area Over Recall" approximation. 
    # If the model predicts massive false regions to cheat recall, precision drops.
    # We can represent the geometric accuracy roughly via F1 Score or AUPRC.
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return iou, FP, precision, recall, f1_score

if __name__ == "__main__":
    DATA_ROOT = "data/makeathon-challenge/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Evaluation on {device}...")

    # We evaluate against the training set since the true Test labels are hidden.
    # This proves how well the model learned the "Weak" GLAD/RADD labels!
    dataset = OsapiensDataset(data_root=DATA_ROOT, split="train", seq_len=12)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=pad_collate_fn)

    print("Auto-detecting channel dimensions...")
    first_sample = dataset[0]
    s1_dim = first_sample["s1"].shape[1] if first_sample["s1"].nelement() > 0 else 2
    s2_dim = first_sample["s2"].shape[1] if first_sample["s2"].nelement() > 0 else 10
    aef_dim = first_sample["aef"].shape[0] if first_sample["aef"].nelement() > 0 else 768

    model = FusionNet(s1_channels=s1_dim, s2_channels=s2_dim, aef_channels=aef_dim, num_classes=1).to(device)
    model.load_state_dict(torch.load("checkpoints/model_epoch_15.pt"))
    model.eval()

    total_iou = 0
    total_fp = 0
    total_prec = 0
    total_rec = 0
    
    pbar = tqdm(dataloader, desc="Evaluating Metrics")
    with torch.no_grad():
        for batch in pbar:
            def norm(x):
                if x.nelement() == 0: return x
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                return (x - x.mean()) / (x.std() + 1e-6)

            s1 = norm(batch["s1"].to(device))
            s2 = norm(batch["s2"].to(device))
            aef = norm(batch["aef"].to(device))
            
            target = batch["label"].to(device)
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
            target = torch.clamp(target, min=0.0, max=1.0)
            
            # Predict
            logits = model(s1, s2, aef)
            probs = torch.sigmoid(logits)
            
            iou, fp, prec, rec, f1 = calculate_metrics(probs, target, threshold=0.5)
            
            total_iou += iou
            total_fp += fp
            total_prec += prec
            total_rec += rec
            
            pbar.set_postfix({"IoU": f"{iou:.4f}", "FP": fp})

    n = len(dataloader)
    print("\n" + "="*40)
    print("FINAL LOCAL VALIDATION METRICS (vs Weak Labels)")
    print("="*40)
    print(f"Mean IoU (Intersection over Union): {total_iou / n * 100:.2f}%")
    print(f"Mean Precision:                     {total_prec / n * 100:.2f}%")
    print(f"Mean Recall:                        {total_rec / n * 100:.2f}%")
    print(f"Total False Positive Pixels:        {total_fp}")
    print("="*40)
    print("If your IoU is > 50% on training data, your model is performing phenomenally!")
