import json
from pathlib import Path
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

class OsapiensDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train", seq_len: int = 12, transform=None):
        """
        Multimodal Dataset for osapiens Challenge.
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

    def _align_arrays(self, arrays: list[np.ndarray]) -> list[np.ndarray]:
        """
        Non-destructively enforces uniform spatial dimensions across a time series.
        No interpolation or mathematical scaling is used. It simply zero-pads or 
        crops the edge pixels to precisely match the first image's shape.
        """
        if not arrays:
            return arrays
            
        target_shape = arrays[0].shape
        channels, target_h, target_w = target_shape[0], target_shape[1], target_shape[2]
        
        aligned = []
        for arr in arrays:
            c, h, w = arr.shape
            if (c, h, w) == target_shape:
                aligned.append(arr)
            else:
                # Pad with zeros or crop 
                new_arr = np.zeros((c, target_h, target_w), dtype=arr.dtype)
                copy_h = min(h, target_h)
                copy_w = min(w, target_w)
                copy_c = min(c, channels)
                
                # Copy raw values without tweaking them
                new_arr[:copy_c, :copy_h, :copy_w] = arr[:copy_c, :copy_h, :copy_w]
                aligned.append(new_arr)
                
        return aligned

    def _read_raster(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read()

    def _get_time_series(self, directory: Path, prefix: str):
        files = sorted(list(directory.glob("*.tif")))
        if len(files) == 0:
            return None
        
        step_idx = np.linspace(0, len(files)-1, self.seq_len, dtype=int)
        files = [files[i] for i in step_idx]
        
        arrays = [self._read_raster(f) for f in files]
        aligned_arrays = self._align_arrays(arrays)
        return np.stack(aligned_arrays, axis=0)  

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile_id = self.tiles[idx]
        
        s2_dir = self.data_root / "sentinel-2" / self.split / f"{tile_id}__s2_l2a"
        s2_data = self._get_time_series(s2_dir, f"{tile_id}__s2_l2a")
        
        s1_dir = self.data_root / "sentinel-1" / self.split / f"{tile_id}__s1_rtc"
        s1_data = self._get_time_series(s1_dir, f"{tile_id}__s1_rtc")
        
        aef_dir = self.data_root / "aef-embeddings" / self.split
        aef_files = sorted(list(aef_dir.glob(f"{tile_id}_*.tiff")))
        if aef_files:
            aef_arrays = [self._read_raster(f) for f in aef_files]
            aligned_aef = self._align_arrays(aef_arrays)
            aef_data = np.stack(aligned_aef, axis=0).mean(axis=0)
        else:
            aef_data = None

        label_tensor = None
        if self.split == "train":
            lbl_dir = self.data_root / "labels" / "train"
            source_masks = []
            for lname in ["gladl", "glads2", "radd"]:
                spec_dir = lbl_dir / lname
                label_files = [f for f in spec_dir.glob(f"*{tile_id}*.tif") if "alertDate" not in f.name]
                if label_files:
                    arrays = [self._read_raster(f) for f in label_files]
                    aligned_arrays = self._align_arrays(arrays)
                    year_stack = np.stack(aligned_arrays, axis=0)
                    cumulative = np.max(year_stack, axis=0) 
                    # Alert rasters can store date-like positive values rather than binary 0/1 masks.
                    # Convert them to a boolean presence mask before combining sources.
                    source_masks.append((cumulative[0] > 0).astype(np.float32)) 
            
            if source_masks:
                aligned_sources = self._align_arrays([np.expand_dims(sm, axis=0) for sm in source_masks])
                consensus = np.stack(aligned_sources, axis=0).mean(axis=0)  
                label_tensor = torch.tensor(consensus, dtype=torch.float32) 

        # We must align s1 and aef spatially exactly with s2 so PyTorch model doesn't crash on fusion
        ref_h, ref_w = s2_data.shape[2], s2_data.shape[3]
        
        def pad_to_ref(data_tensor, h, w):
            if data_tensor is None: return None
            # Non-destructive slice/pad to exact width/height of S2
            ch_count, curr_h, curr_w = data_tensor.shape[0], data_tensor.shape[-2], data_tensor.shape[-1]
            if curr_h == h and curr_w == w: return data_tensor
            
            new_tensor = torch.zeros(data_tensor.shape[:-2] + (h, w), dtype=data_tensor.dtype)
            copy_h, copy_w = min(h, curr_h), min(w, curr_w)
            new_tensor[..., :copy_h, :copy_w] = data_tensor[..., :copy_h, :copy_w]
            return new_tensor

        s2_tensor = torch.tensor(s2_data, dtype=torch.float32)
        s1_tensor = pad_to_ref(torch.tensor(s1_data, dtype=torch.float32), ref_h, ref_w) if s1_data is not None else torch.empty(0)
        aef_tensor = pad_to_ref(torch.tensor(aef_data, dtype=torch.float32), ref_h, ref_w) if aef_data is not None else torch.empty(0)
        label_final = pad_to_ref(label_tensor, ref_h, ref_w) if label_tensor is not None else torch.empty(0)
        
        # Phase 2: Synchronized Spatio-Temporal Data Augmentation
        if self.split == "train":
            # Horizontal Flip (50% chance)
            if torch.rand(1).item() > 0.5:
                s2_tensor = torch.flip(s2_tensor, dims=[-1])
                if s1_tensor.nelement() > 0: s1_tensor = torch.flip(s1_tensor, dims=[-1])
                if aef_tensor.nelement() > 0: aef_tensor = torch.flip(aef_tensor, dims=[-1])
                if label_final.nelement() > 0: label_final = torch.flip(label_final, dims=[-1])
            # Vertical Flip (50% chance)
            if torch.rand(1).item() > 0.5:
                s2_tensor = torch.flip(s2_tensor, dims=[-2])
                if s1_tensor.nelement() > 0: s1_tensor = torch.flip(s1_tensor, dims=[-2])
                if aef_tensor.nelement() > 0: aef_tensor = torch.flip(aef_tensor, dims=[-2])
                if label_final.nelement() > 0: label_final = torch.flip(label_final, dims=[-2])

        sample = {
            "tile_id": tile_id,
            "s2": s2_tensor,
            "s1": s1_tensor,
            "aef": aef_tensor,
            "label": label_final
        }
            
        return sample

