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
