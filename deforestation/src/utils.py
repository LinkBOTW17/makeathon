"""Shared utilities: config loading, logging, constants, metrics."""

import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# ─── Date constants ───────────────────────────────────────────────────────────
# RADD encodes alert date as days since 2014-12-31
RADD_EPOCH_DAYS_TO_2020 = 1827   # days from 2014-12-31 to 2020-01-01

# GLAD-S2 encodes alert date as days since 2019-01-01
GLADS2_EPOCH_DAYS_TO_2020 = 366  # days from 2019-01-01 to 2020-01-01


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load config.yaml and resolve all {data_root} placeholders in paths."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]

    def _resolve(obj):
        if isinstance(obj, str):
            return obj.replace("{data_root}", data_root)
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(v) for v in obj]
        return obj

    cfg["paths"] = _resolve(cfg["paths"])
    return cfg


# ─── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ─── Label decoding helpers ───────────────────────────────────────────────────

def decode_radd(arr: np.ndarray, min_days: int, min_conf: int) -> np.ndarray:
    """
    RADD encoding: leading digit = confidence (2=low, 3=high),
    remaining 4 digits = days since 2014-12-31.
    Returns binary uint8 mask for post-2020 deforestation.
    """
    conf = (arr // 10000).astype(np.int32)
    days = (arr % 10000).astype(np.int32)
    return ((arr > 0) & (conf >= min_conf) & (days >= min_days)).astype(np.uint8)


def decode_gladl(arr: np.ndarray, min_conf: int) -> np.ndarray:
    """GLAD-L: 0=no loss, 2=probable, 3=confirmed. Binary mask."""
    return (arr >= min_conf).astype(np.uint8)


def decode_glads2_alert(arr: np.ndarray, min_conf: int) -> np.ndarray:
    """GLAD-S2 alert: 0=no loss, 1-4 increasing confidence. Binary mask."""
    return (arr >= min_conf).astype(np.uint8)


def decode_glads2_date(date_arr: np.ndarray, alert_arr: np.ndarray,
                       min_days: int, min_conf: int) -> np.ndarray:
    """
    GLAD-S2 with date filtering: only keep alerts where date offset (days since
    2019-01-01) >= min_days, so we only count post-2020 events.
    """
    conf_mask = alert_arr >= min_conf
    date_mask = date_arr >= min_days
    return (conf_mask & date_mask).astype(np.uint8)


# ─── Raster helpers ───────────────────────────────────────────────────────────

def load_and_resample_to_ref(src_path, ref_shape, ref_transform, ref_crs,
                              resampling=None):
    """Load a single-band raster and reproject/resample to match reference grid."""
    import rasterio
    from rasterio.enums import Resampling as RS
    from rasterio.warp import reproject as warp_reproject

    if resampling is None:
        resampling = RS.nearest

    with rasterio.open(src_path) as src:
        src_dtype = src.dtypes[0]
        dest = np.zeros((ref_shape[0], ref_shape[1]), dtype=src_dtype)
        warp_reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling,
        )
    return dest


def find_file_multi_pattern(candidates: list) -> "Path | None":
    """Return the first existing path from a list of candidates."""
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def find_gladl_alert(gladl_dir: str, tile_id: str, yy: int) -> "Path | None":
    """Try multiple known directory layouts for GLAD-L alert files."""
    d = Path(gladl_dir)
    return find_file_multi_pattern([
        d / tile_id / f"alert{yy}.tif",
        d / f"gladl_{tile_id}_alert{yy}.tif",
        d / tile_id / f"alert{yy:02d}.tif",
    ])


def find_glads2_alert(glads2_dir: str, tile_id: str) -> "Path | None":
    """Try multiple known directory layouts for GLAD-S2 alert files."""
    d = Path(glads2_dir)
    return find_file_multi_pattern([
        d / f"glads2_{tile_id}_alert.tif",
        d / tile_id / "alert.tif",
    ])


def find_glads2_date(glads2_dir: str, tile_id: str) -> "Path | None":
    """Try multiple known directory layouts for GLAD-S2 alertDate files."""
    d = Path(glads2_dir)
    return find_file_multi_pattern([
        d / f"glads2_{tile_id}_alertDate.tif",
        d / tile_id / "alertDate.tif",
    ])


# ─── Coordinate & raster sampling helpers (Phase 2+) ─────────────────────────

def get_pixel_xy_coords(radd_path: str, rows: np.ndarray, cols: np.ndarray):
    """
    Convert RADD grid (row, col) indices to (x, y) coordinates in the RADD CRS.
    Returns xs, ys arrays and the CRS object.
    """
    import rasterio
    import rasterio.transform as rt

    with rasterio.open(radd_path) as src:
        transform = src.transform
        crs = src.crs

    # rasterio.transform.xy returns centres of pixels
    xs, ys = rt.xy(transform, rows, cols)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64), crs


def transform_coords(xs: np.ndarray, ys: np.ndarray, src_crs, dst_crs) -> tuple:
    """Reproject (xs, ys) from src_crs to dst_crs using rasterio.warp."""
    from rasterio.warp import transform as warp_transform

    xs_dst, ys_dst = warp_transform(src_crs, dst_crs, xs.tolist(), ys.tolist())
    return np.array(xs_dst, dtype=np.float64), np.array(ys_dst, dtype=np.float64)


def read_raster_at_coords(raster_path: str, xs: np.ndarray, ys: np.ndarray,
                           src_crs, band_indices: list | None = None) -> np.ndarray:
    """
    Read a raster at a list of (x, y) geographic coordinates (in src_crs).
    Returns array of shape (n_pixels, n_bands).
    Pixels outside the raster extent are filled with NaN.

    Uses read-all-then-index for speed — tiles are assumed to fit in RAM.
    """
    import rasterio
    import rasterio.transform as rt

    with rasterio.open(raster_path) as src:
        if band_indices is None:
            band_indices = list(range(1, src.count + 1))

        # Transform coords into this raster's CRS if different
        if src.crs != src_crs:
            xs_r, ys_r = transform_coords(xs, ys, src_crs, src.crs)
        else:
            xs_r, ys_r = xs, ys

        # Convert geographic coords → pixel row/col
        rows_r, cols_r = rt.rowcol(src.transform, xs_r, ys_r)
        rows_r = np.asarray(rows_r, dtype=np.int64)
        cols_r = np.asarray(cols_r, dtype=np.int64)

        H, W = src.height, src.width
        valid = (rows_r >= 0) & (rows_r < H) & (cols_r >= 0) & (cols_r < W)

        data = src.read(band_indices).astype(np.float32)   # (n_bands, H, W)

    n_pixels = len(xs)
    n_bands  = len(band_indices)
    result   = np.full((n_pixels, n_bands), np.nan, dtype=np.float32)
    if valid.any():
        result[valid] = data[:, rows_r[valid], cols_r[valid]].T

    return result


# ─── Spectral indices ─────────────────────────────────────────────────────────

def spectral_index(b1: np.ndarray, b2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Generic normalized difference: (b1 - b2) / (b1 + b2)."""
    denom = b1 + b2
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = np.where(np.abs(denom) > eps, (b1 - b2) / denom, np.nan)
    return idx.astype(np.float32)


# ─── Temporal statistics ──────────────────────────────────────────────────────

def compute_ts_stats(ts: np.ndarray, prefix: str) -> dict:
    """
    Compute summary statistics over a (n_pixels, T) time series with possible NaN.

    Returns dict of {prefix_stat: array of shape (n_pixels,)}.
    Stats: mean, std, min, max, slope (linear trend), max_drop, valid_frac.
    """
    n, T = ts.shape

    with np.errstate(all="ignore"):
        mean_v  = np.nanmean(ts, axis=1)
        std_v   = np.nanstd(ts, axis=1)
        min_v   = np.nanmin(ts, axis=1)
        max_v   = np.nanmax(ts, axis=1)

    # Valid fraction
    valid_frac = np.isfinite(ts).sum(axis=1) / T

    # Linear trend slope via least-squares on valid timesteps
    t_idx = np.arange(T, dtype=np.float32)
    slope = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        mask = np.isfinite(ts[i])
        if mask.sum() >= 3:
            try:
                slope[i] = np.polyfit(t_idx[mask], ts[i, mask], 1)[0]
            except Exception:
                pass

    # Max single-step drop (largest decrease between consecutive timesteps)
    diffs    = np.diff(ts, axis=1)  # (n, T-1)
    max_drop = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        valid_d = diffs[i][np.isfinite(diffs[i])]
        if len(valid_d) > 0:
            max_drop[i] = float(np.min(valid_d))  # most negative diff = biggest drop

    return {
        f"{prefix}_mean":       mean_v.astype(np.float32),
        f"{prefix}_std":        std_v.astype(np.float32),
        f"{prefix}_min":        min_v.astype(np.float32),
        f"{prefix}_max":        max_v.astype(np.float32),
        f"{prefix}_slope":      slope,
        f"{prefix}_max_drop":   max_drop,
        f"{prefix}_valid_frac": valid_frac.astype(np.float32),
    }


# ─── File discovery helpers ───────────────────────────────────────────────────

def discover_s2_files(s2_tile_dir: str) -> list:
    """
    Discover available Sentinel-2 files for one tile.
    Pattern: <tile>__s2_l2a_<year>_<month>.tif
    Returns list of (year, month, Path) sorted chronologically.
    """
    result = []
    for f in Path(s2_tile_dir).glob("*__s2_l2a_*.tif"):
        parts = f.stem.split("_")
        try:
            # Last token = month, second-to-last = year
            month = int(parts[-1])
            year  = int(parts[-2])
            result.append((year, month, f))
        except (ValueError, IndexError):
            continue
    return sorted(result)


def discover_s1_files(s1_tile_dir: str, orbit: str = "ascending") -> list:
    """
    Discover available Sentinel-1 files for one tile.
    Pattern: <tile>__s1_rtc_<year>_<month>_<orbit>.tif
    Returns list of (year, month, Path) sorted chronologically.
    """
    result = []
    for f in Path(s1_tile_dir).glob(f"*__s1_rtc_*_{orbit}.tif"):
        parts = f.stem.split("_")
        try:
            # e.g. ['18NWG', '6', '6', '', 's1', 'rtc', '2020', '10', 'ascending']
            # orbit = parts[-1], month = parts[-2], year = parts[-3]
            month = int(parts[-2])
            year  = int(parts[-3])
            result.append((year, month, f))
        except (ValueError, IndexError):
            continue
    return sorted(result)


# ─── AEF helpers ─────────────────────────────────────────────────────────────

def find_aef_file(aef_dir: str, tile_id: str, year: int) -> "Path | None":
    """Locate AEF embedding file for a tile/year. Tries .tiff and .tif."""
    d = Path(aef_dir)
    return find_file_multi_pattern([
        d / f"{tile_id}_{year}.tiff",
        d / f"{tile_id}_{year}.tif",
    ])


def normalize_aef(data: np.ndarray) -> np.ndarray:
    """
    If AEF embeddings are stored as integers, normalize per-band to [0, 1].
    Float data is returned as-is.
    """
    if not np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32)

    out = np.empty_like(data, dtype=np.float32)
    for b in range(data.shape[0]):
        band  = data[b].astype(np.float32)
        b_min = float(np.nanmin(band))
        b_max = float(np.nanmax(band))
        out[b] = (band - b_min) / (b_max - b_min + 1e-8)
    return out


# ─── PyTorch model & dataset (Phase 3b+) ────────────────────────────────────

def get_device() -> "torch.device":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


class DeforestationDataset:
    """
    PyTorch Dataset for pixel-level deforestation detection.

    Stores time series and AEF arrays entirely in RAM for fast batch access.
    NaN values (cloud/no-data) are replaced with 0 at fetch time.
    """

    def __init__(self,
                 indices: np.ndarray,
                 s2_ts:   np.ndarray,   # (N_total, T, 2)  NDVI+NBR
                 s1_ts:   np.ndarray,   # (N_total, T, 1)  VV_dB
                 aef:     np.ndarray,   # (N_total, 64)
                 labels:  np.ndarray,   # (N_total,)
                 ts_mean: np.ndarray | None = None,  # (3,) global mean per channel
                 ts_std:  np.ndarray | None = None,  # (3,) global std per channel
                 aef_mean: np.ndarray | None = None,
                 aef_std:  np.ndarray | None = None):
        self.indices  = indices
        self.s2_ts    = s2_ts
        self.s1_ts    = s1_ts
        self.aef      = aef
        self.labels   = labels
        self.ts_mean  = ts_mean if ts_mean is not None else np.zeros(3, dtype=np.float32)
        self.ts_std   = ts_std  if ts_std  is not None else np.ones(3, dtype=np.float32)
        self.aef_mean = aef_mean if aef_mean is not None else np.zeros(64, dtype=np.float32)
        self.aef_std  = aef_std  if aef_std  is not None else np.ones(64, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        import torch

        i  = self.indices[idx]
        s2 = self.s2_ts[i]           # (T, 2)
        s1 = self.s1_ts[i]           # (T, 1)
        ts = np.concatenate([s2, s1], axis=1).astype(np.float32)  # (T, 3)

        # Replace NaN (cloud / no-data) with 0 before normalisation
        ts = np.nan_to_num(ts, nan=0.0)

        # Z-normalise per channel using training statistics
        ts = (ts - self.ts_mean[None, :]) / (self.ts_std[None, :] + 1e-8)

        aef = self.aef[i].astype(np.float32)
        aef = (aef - self.aef_mean) / (self.aef_std + 1e-8)

        return (
            torch.tensor(ts,                       dtype=torch.float32),
            torch.tensor(aef,                      dtype=torch.float32),
            torch.tensor([self.labels[i]],         dtype=torch.float32),
        )


class DeforestationModel:
    """
    Wrapper to make the nn.Module importable without top-level torch import.
    Actual nn.Module is returned by DeforestationModel.build().
    """

    @staticmethod
    def build(n_ts_features: int = 3,
              aef_dim:       int = 64,
              lstm_hidden:   int = 64,
              lstm_layers:   int = 2,
              lstm_dropout:  float = 0.2,
              mlp_layers:    list  = None):
        import torch.nn as nn

        if mlp_layers is None:
            mlp_layers = [128, 64, 32]

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()

                # Branch A: temporal encoder
                # VV slope is dominant signal (Phase 3a showed #1 gain) —
                # LSTM over full 72-step trajectory captures this far better
                # than a single summary statistic.
                self.lstm = nn.LSTM(
                    input_size  = n_ts_features,
                    hidden_size = lstm_hidden,
                    num_layers  = lstm_layers,
                    batch_first = True,
                    dropout     = lstm_dropout if lstm_layers > 1 else 0.0,
                )

                # Branch B: AEF semantic projection
                self.aef_proj = nn.Sequential(
                    nn.Linear(aef_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                )

                # Fusion head
                in_dim = lstm_hidden + 64
                layers = []
                for out_dim in mlp_layers:
                    layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.2)]
                    in_dim = out_dim
                layers.append(nn.Linear(in_dim, 1))
                self.head = nn.Sequential(*layers)

            def forward(self, ts, aef):
                # ts:  (B, T, n_ts_features)
                # aef: (B, aef_dim)
                _, (h_n, _) = self.lstm(ts)
                temporal_emb = h_n[-1]        # (B, lstm_hidden)
                aef_emb      = self.aef_proj(aef)   # (B, 64)
                fused        = torch.cat([temporal_emb, aef_emb], dim=1)
                return self.head(fused)        # (B, 1) logits

        import torch  # noqa: F401 — ensures torch is importable before returning
        return _Model()


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision, recall, F1, and confusion matrix for binary classification."""
    from sklearn.metrics import (precision_recall_fscore_support,
                                 roc_auc_score, average_precision_score,
                                 confusion_matrix)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    # AUC metrics only if y_pred has probabilities (float) and both classes present
    if y_true.sum() > 0 and (1 - y_true).sum() > 0:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred))
            metrics["auc_pr"] = float(average_precision_score(y_true, y_pred))
        except Exception:
            pass

    return metrics
