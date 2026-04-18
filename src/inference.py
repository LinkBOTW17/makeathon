import os
import torch
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

from dataset import OsapiensDataset
from models.fusion_net import FusionNet

def enable_dropout(model):
    """Function to recursively enable dropout layers during inference."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def run_mc_dropout(model, s1, s2, aef, num_samples=10):
    """
    Runs Monte Carlo Dropout to generate a mean prediction and an uncertainty estimation.
    """
    model.eval()
    enable_dropout(model) # Re-enable dropout for MC
    
    preds = []
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for _ in range(num_samples):
                logits = model(s1, s2, aef)
                probs = torch.sigmoid(logits)
                preds.append(probs)
                
    preds_stack = torch.stack(preds, dim=0)
    mean_prob = preds_stack.mean(dim=0)
    variance = preds_stack.var(dim=0) # Represents Model Uncertainty (Epistemic)
    
    return mean_prob, variance

def write_prediction_raster(tile_id, mask, output_dir, reference_tif_path):
    """Writes the binary mask to a GeoTIFF using metadata from a reference file."""
    with rasterio.open(reference_tif_path) as src:
        meta = src.meta.copy()
        meta.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw'
        )
        
        output_path = Path(output_dir) / f"{tile_id}_pred.tif"
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(mask.astype(np.uint8), 1)
            
    return output_path

if __name__ == "__main__":
    DATA_ROOT = Path("data/makeathon-challenge/")
    TEST_ROOT = DATA_ROOT / "sentinel-2" / "test"
    OUTPUT_DIR = Path("predictions")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading test dataset...")
    dataset = OsapiensDataset(data_root=DATA_ROOT, split="test", seq_len=12)
    
    # Auto-detect channels to match train.py dynamically
    print("Auto-detecting channel dimensions from test set...")
    first_sample = dataset[0]
    s1_dim = first_sample["s1"].shape[1] if first_sample["s1"].nelement() > 0 else 2
    s2_dim = first_sample["s2"].shape[1] if first_sample["s2"].nelement() > 0 else 10
    aef_dim = first_sample["aef"].shape[0] if first_sample["aef"].nelement() > 0 else 768
    
    model = FusionNet(s1_channels=s1_dim, s2_channels=s2_dim, aef_channels=aef_dim, num_classes=1).to(device)
    # model.load_state_dict(torch.load("checkpoints/model_epoch_X.pt")) # Load best checkpoint here
    
    pbar = tqdm(dataset, desc="Running Inference")
    for sample in pbar:
        tile_id = sample["tile_id"]
        s1 = sample["s1"].unsqueeze(0).to(device)
        s2 = sample["s2"].unsqueeze(0).to(device)
        aef = sample["aef"].unsqueeze(0).to(device)
        
        # MC Dropout for Prediction and Uncertainty
        mean_prob, uncertainty = run_mc_dropout(model, s1, s2, aef, num_samples=5)
        
        # Binarize output (0.5 threshold)
        binary_mask = (mean_prob > 0.5).cpu().numpy().squeeze(0) # Shape: (1, H, W) -> (H, W) if num_classes=1
        if binary_mask.ndim == 3:
            binary_mask = binary_mask[0]

        # Save to TIFF
        # We need a reference TIF to copy spatial metadata (CRS, transform). Any test S2 patch works.
        ref_files = list((TEST_ROOT / f"{tile_id}__s2_l2a").glob("*.tif"))
        if ref_files:
            write_prediction_raster(tile_id, binary_mask, OUTPUT_DIR, ref_files[0])
            
        # Optional: Save uncertainty map for visualization (Bonus 2)
        # np.save(OUTPUT_DIR / f"{tile_id}_uncertainty.npy", uncertainty.cpu().numpy())
    
    print("Inference complete. Call submission_utils.py externally to build GeoJSON.")
