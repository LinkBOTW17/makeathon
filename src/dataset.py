import json
from pathlib import Path
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import geopandas as gpd

class OsapiensDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train", seq_len: int = 12, transform=None):
        """
        Multimodal Dataset for osapiens Challenge.
        
        Args:
            data_root: Path to 'data/makeathon-challenge/'
            split: 'train' or 'test'
            seq_len: Number of temporal steps to randomly sample or sequence (default 12 for 1 year)
            transform: Optional spatial augmentations
        """
        self.data_root = Path(data_root)
        self.split = split
        self.seq_len = seq_len
        self.transform = transform
        
        # Load tiles dynamically from directory structure
        s2_dir = self.data_root / "sentinel-2" / split
        self.tiles = [p.name.split("__")[0] for p in s2_dir.glob("*") if p.is_dir()]
        
        if len(self.tiles) == 0:
            print(f"WARNING: No tiles found in {s2_dir}. Ensure 'make download_data_from_s3' was completed successfully.")
            
        self.tiles = sorted(list(set(self.tiles)))

    def _read_raster(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read()

    def _get_time_series(self, directory: Path, prefix: str):
        """Fetches a sequence of raster images and pads/clips to seq_len."""
        files = sorted(list(directory.glob("*.tif")))
        
        # Subsampling or padding to seq_len
        if len(files) == 0:
            return None
        
        step_idx = np.linspace(0, len(files)-1, self.seq_len, dtype=int)
        files = [files[i] for i in step_idx]
        
        arrays = [self._read_raster(f) for f in files]
        return np.stack(arrays, axis=0)  # Shape: (seq_len, channels, H, W)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile_id = self.tiles[idx]
        
        # Load Sentinel-2 L2A (Optical)
        s2_dir = self.data_root / "sentinel-2" / self.split / f"{tile_id}__s2_l2a"
        s2_data = self._get_time_series(s2_dir, f"{tile_id}__s2_l2a")
        
        # Load Sentinel-1 RTC (Radar)
        s1_dir = self.data_root / "sentinel-1" / self.split / f"{tile_id}__s1_rtc"
        s1_data = self._get_time_series(s1_dir, f"{tile_id}__s1_rtc")
        
        # Load AEF Embeddings (Foundation Model)
        aef_dir = self.data_root / "aef-embeddings" / self.split
        aef_files = sorted(list(aef_dir.glob(f"{tile_id}_*.tiff")))
        aef_data = np.stack([self._read_raster(f) for f in aef_files], axis=0) if aef_files else None
        
        labels = {}
        if self.split == "train":
            lbl_dir = self.data_root / "labels" / "train"
            # Attempt to find labels dynamically
            for lname in ["gladl", "glads2", "radd"]:
                spec_dir = lbl_dir / lname
                # Gather all alert files, ignoring alertDate metadata rasters
                label_files = [f for f in spec_dir.glob(f"*{tile_id}*.tif") if "alertDate" not in f.name]
                
                if label_files:
                    # Some sources have multiple years (alert21, alert22, etc.). 
                    # We compute the overall deforestation mask by taking the max across years.
                    arrays = [self._read_raster(f) for f in label_files]
                    stacked = np.stack(arrays, axis=0)
                    labels[lname] = np.max(stacked, axis=0) # Cumulative mask
                else:
                    labels[lname] = None

        # Convert to tensors
        sample = {
            "tile_id": tile_id,
            "s2": torch.tensor(s2_data, dtype=torch.float32) if s2_data is not None else torch.empty(0),
            "s1": torch.tensor(s1_data, dtype=torch.float32) if s1_data is not None else torch.empty(0),
            "aef": torch.tensor(aef_data, dtype=torch.float32) if aef_data is not None else torch.empty(0)
        }
        
        if self.split == "train":
            sample["labels"] = labels
            
        return sample
