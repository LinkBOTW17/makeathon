"""
Pixel-Level LightGBM v3 — Full Multimodal Deforestation Classifier
===================================================================

Features per pixel (151 total):
  AEF change signals  (78):
    aef_l2_delta_{2021-2024}       — L2 between consecutive year embeddings
    aef_cos_delta_{2021-2024}      — cosine distance between consecutive years
    aef_from2020_l2_{2021-2024}    — L2 drift from 2020 baseline
    aef_max_delta_l2 + year_idx    — peak change magnitude and timing
    aef_delta_vec_{0-63}           — 64-dim change vector (emb_2024 − emb_2020)

  NDVI signals  (4):
    ndvi_pre, ndvi_post            — mean NDVI in 2020-21 vs 2022-24
    ndvi_change                    — pre − post (positive = vegetation loss)
    ndvi_std                       — temporal variance across full series

  SAR signals  (3):
    sar_pre, sar_post              — mean SAR dB in 2020-21 vs 2022-24
    sar_change                     — pre − post in dB

  Harmonic physics  (2):  ← NEW — was top-2 feature at polygon level
    harmonic_t_stat                — vectorised per-pixel NDVI breakpoint t-stat
    harmonic_change_mag            — mean absolute post-cutoff NDVI residual

  AEF baseline  (64):
    aef_2020_{0-63}                — 2020 embedding (forest type baseline)

Post-processing:
  Gaussian smoothing (σ=1.5) — kills isolated-pixel false positives.

Threshold: dual-target per region from LOTO-CV.
  IoU-optimal threshold OR min-recall threshold (whichever is lower),
  biased toward Recall to close the 53% → 70%+ gap.

Usage (from makeathon root):
    python deforestation/pixel_model.py --mode all
    python deforestation/pixel_model.py --mode predict --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from lightgbm import LGBMClassifier
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter
from shapely.geometry import shape
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from mislabel_detection_v2 import DATA_ROOT, _ndvi, _s1_db  # noqa: E402
from validate_spatial import build_gt_pixel_map               # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────

AEF_YEARS         = [2020, 2021, 2022, 2023, 2024]
DELTA_YEARS       = [2021, 2022, 2023, 2024]
N_FEATURES        = 151
N_SAMPLE_PER_TILE = 80_000   # increased; positives uncapped
GAUSS_SIGMA       = 1.5
RECALL_TARGET     = 0.65     # minimum recall to enforce at threshold selection

REGIONS = {"SEA": ("47", "48"), "SAM": ("18", "19")}

LGBM_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=32,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

FEATURE_NAMES = (
    [f"aef_l2_delta_{yr}"     for yr in DELTA_YEARS] +   # 4
    [f"aef_cos_delta_{yr}"    for yr in DELTA_YEARS] +   # 4
    [f"aef_from2020_l2_{yr}"  for yr in DELTA_YEARS] +  # 4
    ["aef_max_delta_l2", "aef_change_year_idx"] +        # 2
    [f"aef_delta_vec_{i:02d}" for i in range(64)] +      # 64
    ["ndvi_pre", "ndvi_post", "ndvi_change", "ndvi_std"] +  # 4
    ["sar_pre", "sar_post", "sar_change"] +              # 3
    ["harmonic_t_stat", "harmonic_change_mag"] +         # 2
    [f"aef_2020_{i:02d}"      for i in range(64)]        # 64 — forest-type baseline
)   # total = 151
assert len(FEATURE_NAMES) == N_FEATURES


# ── helpers ───────────────────────────────────────────────────────────────────

def tile_normalize(feats: np.ndarray) -> np.ndarray:
    """
    Robust z-score each feature column within the tile's own distribution.
    Converts absolute embedding values into "deviation from this tile's median" —
    making the model invariant to inter-tile baseline differences (forest type, region).
    """
    med = np.nanmedian(feats, axis=0)
    p25 = np.nanpercentile(feats, 25, axis=0)
    p75 = np.nanpercentile(feats, 75, axis=0)
    iqr = p75 - p25
    iqr = np.where(iqr < 1e-6, 1.0, iqr)
    return ((feats - med) / iqr).astype(np.float32)


def get_region(tile_id: str) -> str:
    prefix = tile_id[:2]
    for region, prefixes in REGIONS.items():
        if prefix in prefixes:
            return region
    return "SAM"


def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fp + fn + 1e-9)


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn + 1e-9)


def compute_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return fp / (tp + fp + 1e-9)


# ── AEF loading ───────────────────────────────────────────────────────────────

def _load_aef_reprojected(
    path: Path, ref_transform, ref_crs, ref_shape: tuple,
) -> np.ndarray:
    with rasterio.open(path) as src:
        n = src.count
        out = np.zeros((n, *ref_shape), dtype=np.float32)
        for b in range(1, n + 1):
            reproject(
                source=src.read(b).astype(np.float32), destination=out[b - 1],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
    return out


def load_aef_all_years(
    tile_id: str, split: str, ref_transform, ref_crs, ref_shape: tuple,
) -> dict[int, np.ndarray]:
    aef_dir = DATA_ROOT / "aef-embeddings" / split
    result: dict[int, np.ndarray] = {}
    for f in aef_dir.glob(f"{tile_id}_*.tiff"):
        try:
            yr = int(f.stem.split("_")[-1])
        except ValueError:
            continue
        if yr in AEF_YEARS:
            result[yr] = _load_aef_reprojected(f, ref_transform, ref_crs, ref_shape)
    return result


# ── NDVI + SAR pixel maps ─────────────────────────────────────────────────────

def _month_index(year: int, month: int) -> int:
    return (year - 2020) * 12 + (month - 1)


def load_ndvi_sar_maps(
    tile_id: str,
    split: str,
    ref_transform,
    ref_crs,
    ref_shape: tuple,
) -> tuple[dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
    """
    Returns (maps, ndvi_ts, ndvi_months) where:
      maps       — pixel maps: ndvi_pre/post/change/std, sar_pre/post/change (H,W each)
      ndvi_ts    — raw NDVI time series (T, H, W) float32, or None
      ndvi_months— integer month indices (T,), or None
    """
    H, W = ref_shape
    out: dict[str, np.ndarray] = {}

    # ── NDVI from Sentinel-2 ──────────────────────────────────────────────────
    s2_dir   = DATA_ROOT / "sentinel-2" / split / f"{tile_id}__s2_l2a"
    s2_files = sorted(s2_dir.glob("*.tif"))
    ndvi_stack, ndvi_months = [], []

    for f in s2_files:
        parts = f.stem.split("_")
        try:
            yr, mo = int(parts[-2]), int(parts[-1])
        except (ValueError, IndexError):
            continue
        nd = _ndvi(f)
        if nd.shape != (H, W):
            reproj = np.full((H, W), np.nan, dtype=np.float32)
            with rasterio.open(f) as src:
                reproject(
                    source=nd, destination=reproj,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=ref_transform, dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                )
            nd = reproj
        ndvi_stack.append(nd)
        ndvi_months.append(_month_index(yr, mo))

    ndvi_ts_arr    = None
    ndvi_months_arr = None
    if ndvi_stack:
        ndvi_arr = np.stack(ndvi_stack, axis=0)            # (T, H, W)
        ndvi_ts_arr     = ndvi_arr
        ndvi_months_arr = np.array(ndvi_months, dtype=np.int32)
        months   = ndvi_months_arr
        pre_sel  = months < 24
        post_sel = months >= 24

        with np.errstate(all="ignore"):
            ndvi_pre  = np.nanmean(ndvi_arr[pre_sel],  axis=0) if pre_sel.any()  else np.full((H, W), np.nan)
            ndvi_post = np.nanmean(ndvi_arr[post_sel], axis=0) if post_sel.any() else np.full((H, W), np.nan)
            ndvi_std  = np.nanstd(ndvi_arr,             axis=0)

        out["ndvi_pre"]    = ndvi_pre.astype(np.float32)
        out["ndvi_post"]   = ndvi_post.astype(np.float32)
        out["ndvi_change"] = (ndvi_pre - ndvi_post).astype(np.float32)
        out["ndvi_std"]    = ndvi_std.astype(np.float32)

        # Per-year NDVI means
        step_years = np.array([m // 12 + 2020 for m in ndvi_months], dtype=np.int32)
        yr_means: dict[int, np.ndarray] = {}
        for yr in AEF_YEARS:
            sel = step_years == yr
            if sel.any():
                yr_means[yr] = np.nanmean(ndvi_arr[sel], axis=0).astype(np.float32)
                out[f"ndvi_yr_{yr}"] = yr_means[yr]
            else:
                out[f"ndvi_yr_{yr}"] = np.full((H, W), np.nan, dtype=np.float32)

        # NDVI change year: which year had the largest consecutive NDVI drop
        best_drop = np.full((H, W), -np.inf, dtype=np.float32)
        best_yr_idx = np.zeros((H, W), dtype=np.float32)
        for i, yr in enumerate(DELTA_YEARS):  # [2021,2022,2023,2024]
            if yr in yr_means and (yr - 1) in yr_means:
                drop = yr_means[yr - 1] - yr_means[yr]  # positive = NDVI dropped
                update = np.isfinite(drop) & (drop > best_drop)
                best_yr_idx = np.where(update, float(i), best_yr_idx)
                best_drop   = np.where(update, drop, best_drop)
        out["ndvi_change_year_idx"] = best_yr_idx.astype(np.float32)
        out["ndvi_max_drop"]        = np.where(np.isfinite(best_drop), best_drop,
                                               np.nan).astype(np.float32)

    # ── SAR from Sentinel-1 ───────────────────────────────────────────────────
    s1_dir   = DATA_ROOT / "sentinel-1" / split / f"{tile_id}__s1_rtc"
    sar_stack, sar_months = [], []

    for f in sorted(s1_dir.glob("*_ascending.tif")):
        parts = f.stem.split("_")
        try:
            yr, mo = int(parts[-3]), int(parts[-2])
        except (ValueError, IndexError):
            continue
        with rasterio.open(f) as src:
            raw    = src.read(1).astype(np.float32)
            reproj = np.zeros((H, W), dtype=np.float32)
            reproject(
                source=raw, destination=reproj,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
        db = _s1_db(reproj)
        sar_stack.append(db)
        sar_months.append(_month_index(yr, mo))

    if sar_stack:
        sar_arr  = np.stack(sar_stack, axis=0)
        months_s = np.array(sar_months)
        pre_s    = months_s < 24
        post_s   = months_s >= 24

        with np.errstate(all="ignore"):
            sar_pre  = np.nanmean(sar_arr[pre_s],  axis=0) if pre_s.any()  else np.full((H, W), np.nan)
            sar_post = np.nanmean(sar_arr[post_s], axis=0) if post_s.any() else np.full((H, W), np.nan)

        out["sar_pre"]    = sar_pre.astype(np.float32)
        out["sar_post"]   = sar_post.astype(np.float32)
        out["sar_change"] = (sar_pre - sar_post).astype(np.float32)

    return out, ndvi_ts_arr, ndvi_months_arr


# ── vectorised harmonic NDVI breakpoint ──────────────────────────────────────

def compute_harmonic_maps(
    ndvi_ts: np.ndarray,        # (T, H, W) float32
    ndvi_months: np.ndarray,    # (T,) int
    train_cutoff: int = 24,
    n_harmonics: int = 2,
    min_train_obs: int = 8,
    min_test_obs:  int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised harmonic regression across all pixels simultaneously.
    Returns (t_stat_map, change_mag_map), each (H, W) float32.

    t_stat > 0 means NDVI dropped below the harmonic baseline (deforestation signal).
    """
    T, H, W = ndvi_ts.shape
    N = H * W
    K = 2 + 2 * n_harmonics    # intercept + trend + harmonics

    t = ndvi_months.astype(np.float32)
    cols = [np.ones(T, dtype=np.float32), t]
    for k in range(1, n_harmonics + 1):
        cols += [np.cos(2 * np.pi * k * t / 12), np.sin(2 * np.pi * k * t / 12)]
    X = np.column_stack(cols).astype(np.float32)   # (T, K)

    ndvi_flat  = ndvi_ts.reshape(T, N).astype(np.float32)
    valid_mask = np.isfinite(ndvi_flat)             # (T, N)
    ndvi_clean = np.where(valid_mask, ndvi_flat, 0.0)

    tr_sel = ndvi_months < train_cutoff
    te_sel = ndvi_months >= train_cutoff

    X_tr = X[tr_sel];  X_te = X[te_sel]
    y_tr = ndvi_clean[tr_sel];  y_te = ndvi_clean[te_sel]
    vm_tr = valid_mask[tr_sel]; vm_te = valid_mask[te_sel]

    n_tr = vm_tr.sum(axis=0).astype(np.float32)    # (N,)
    n_te = vm_te.sum(axis=0).astype(np.float32)

    # OLS: β = (X_tr^T X_tr)^{-1} X_tr^T y  — single shared XtX (approx for NaN)
    XtX     = (X_tr.T @ X_tr).astype(np.float64)
    XtX_inv = np.linalg.pinv(XtX + 1e-6 * np.eye(K)).astype(np.float32)
    beta    = XtX_inv @ (X_tr.T @ y_tr)            # (K, N)

    # Baseline residual std
    resid_tr = np.where(vm_tr, y_tr - X_tr @ beta, np.nan)   # (T_tr, N)
    std_tr   = np.nanstd(resid_tr, axis=0).clip(1e-6)         # (N,)

    # Post-cutoff residuals: positive = NDVI fell below prediction
    resid_te    = np.where(vm_te, (X_te @ beta) - y_te, np.nan)
    mean_resid  = np.nanmean(resid_te, axis=0)
    change_mag  = np.nanmean(np.abs(resid_te), axis=0)

    t_stat = mean_resid / (std_tr / np.sqrt(n_te.clip(1)))

    # Zero-out pixels with insufficient observations
    bad = (n_tr < min_train_obs) | (n_te < min_test_obs)
    t_stat[bad]    = 0.0
    change_mag[bad] = 0.0

    return (
        t_stat.reshape(H, W).astype(np.float32),
        change_mag.reshape(H, W).astype(np.float32),
    )


# ── pixel feature matrix ──────────────────────────────────────────────────────

def compute_pixel_features(
    aef_by_year: dict[int, np.ndarray],
    ndvi_sar:    dict[str, np.ndarray] | None,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    harmonic_maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Build (N, 149) feature matrix.
    If rows/cols given, extracts only sampled pixels; otherwise flattens all.
    """
    any_emb  = next(iter(aef_by_year.values()))
    _, H, W  = any_emb.shape
    N        = len(rows) if rows is not None else H * W

    def _px(arr2d: np.ndarray) -> np.ndarray:
        """Extract sampled pixels or flatten."""
        return arr2d[rows, cols] if rows is not None else arr2d.ravel()

    def _px3(arr3d: np.ndarray) -> np.ndarray:
        """(64, H, W) → (64, N)."""
        return arr3d[:, rows, cols] if rows is not None else arr3d.reshape(64, -1)

    emb: dict[int, np.ndarray] = {yr: _px3(cube) for yr, cube in aef_by_year.items()}

    feats = np.full((N, N_FEATURES), np.nan, dtype=np.float32)
    c = 0

    # ── AEF L2 deltas ─────────────────────────────────────────────────────────
    l2_deltas: dict[int, np.ndarray] = {}
    for yr in DELTA_YEARS:
        if yr in emb and (yr - 1) in emb:
            l2 = np.linalg.norm(emb[yr] - emb[yr - 1], axis=0)
        else:
            l2 = np.zeros(N, dtype=np.float32)
        feats[:, c] = l2;  l2_deltas[yr] = l2;  c += 1

    # ── AEF cosine deltas ─────────────────────────────────────────────────────
    for yr in DELTA_YEARS:
        if yr in emb and (yr - 1) in emb:
            a = emb[yr];   na = np.linalg.norm(a, axis=0) + 1e-9
            b = emb[yr-1]; nb = np.linalg.norm(b, axis=0) + 1e-9
            feats[:, c] = (1.0 - (a * b).sum(axis=0) / (na * nb)).clip(0, 2)
        c += 1

    # ── cumulative L2 from 2020 ───────────────────────────────────────────────
    for yr in DELTA_YEARS:
        if yr in emb and 2020 in emb:
            feats[:, c] = np.linalg.norm(emb[yr] - emb[2020], axis=0)
        c += 1

    # ── max delta + argmax year ───────────────────────────────────────────────
    l2_stack = np.stack([l2_deltas[yr] for yr in DELTA_YEARS], axis=0)
    feats[:, c] = l2_stack.max(axis=0);   c += 1
    feats[:, c] = l2_stack.argmax(axis=0).astype(np.float32); c += 1

    # ── AEF change vector: emb_latest − emb_2020 (64-dim) ────────────────────
    latest = max((yr for yr in aef_by_year if yr > 2020), default=None)
    if latest and 2020 in emb:
        feats[:, c:c + 64] = (emb[latest] - emb[2020]).T
    c += 64

    # ── NDVI features ─────────────────────────────────────────────────────────
    for key in ("ndvi_pre", "ndvi_post", "ndvi_change", "ndvi_std"):
        if ndvi_sar and key in ndvi_sar:
            feats[:, c] = _px(ndvi_sar[key])
        c += 1

    # ── SAR features ──────────────────────────────────────────────────────────
    for key in ("sar_pre", "sar_post", "sar_change"):
        if ndvi_sar and key in ndvi_sar:
            feats[:, c] = _px(ndvi_sar[key])
        c += 1

    # ── Harmonic t-stat + change magnitude ───────────────────────────────────
    if harmonic_maps is not None:
        feats[:, c]     = _px(harmonic_maps[0])   # t_stat
        feats[:, c + 1] = _px(harmonic_maps[1])   # change_mag
    c += 2

    # ── AEF 2020 baseline embedding (64-dim) ─────────────────────────────────
    if 2020 in emb:
        feats[:, c:c + 64] = emb[2020].T
    c += 64

    assert c == N_FEATURES, f"Feature count mismatch: {c} != {N_FEATURES}"
    return feats


# ── tile pixel sampling ───────────────────────────────────────────────────────

def sample_tile(
    tile_id: str,
    mislabels: pd.DataFrame | None,
    n_sample: int = N_SAMPLE_PER_TILE,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:

    from mislabel_detection_v2 import load_tile
    rng = np.random.default_rng(seed)

    try:
        td = load_tile(tile_id)
    except FileNotFoundError as e:
        if verbose: print(f"  {tile_id}: SKIP — {e}")
        return None

    H, W = td.ref_shape
    tile_rows = (
        mislabels[mislabels["tile_id"] == tile_id].reset_index(drop=True)
        if mislabels is not None else None
    )
    gt_binary, gt_year = build_gt_pixel_map(td, tile_rows)

    flat_gt   = gt_binary.ravel()
    flat_year = gt_year.ravel()

    pos_idx = np.where(flat_gt == 1)[0]
    neg_idx = np.where(flat_gt == 0)[0]

    if len(pos_idx) == 0:
        if verbose: print(f"  {tile_id}: no positive pixels — SKIP")
        return None

    # Use ALL positives; match with 2× negatives (uncapped positive class)
    n_pos = len(pos_idx)
    n_neg = min(len(neg_idx), 2 * n_pos)
    # Hard cap to keep memory sane
    if n_pos > n_sample:
        n_pos = n_sample
        pos_idx = rng.choice(pos_idx, n_pos, replace=False)

    idx = np.concatenate([
        pos_idx,
        rng.choice(neg_idx, n_neg, replace=False),
    ])
    rows     = (idx // W).astype(np.int32)
    cols_arr = (idx %  W).astype(np.int32)

    # AEF
    aef_by_year = load_aef_all_years(
        tile_id, "train", td.ref_transform, td.ref_crs, td.ref_shape
    )
    if not aef_by_year or 2020 not in aef_by_year:
        if verbose: print(f"  {tile_id}: missing 2020 AEF — SKIP")
        return None

    # NDVI + SAR pixel maps (also returns raw time series for harmonic maps)
    ndvi_sar, ndvi_ts, ndvi_months = load_ndvi_sar_maps(
        tile_id, "train", td.ref_transform, td.ref_crs, td.ref_shape
    )

    # Harmonic t-stat maps (vectorised)
    harmonic_maps = None
    ts_src  = ndvi_ts     if ndvi_ts     is not None else td.ndvi_ts
    mo_src  = ndvi_months if ndvi_months is not None else (td.ndvi_months if hasattr(td, "ndvi_months") else None)
    if ts_src is not None and mo_src is not None and len(mo_src) > 0:
        harmonic_maps = compute_harmonic_maps(ts_src, mo_src)

    feats  = compute_pixel_features(aef_by_year, ndvi_sar, rows, cols_arr, harmonic_maps)
    labels = flat_gt[idx].astype(np.int32)
    yrs    = flat_year[idx].astype(np.int32)
    return feats, labels, yrs


# ── training ──────────────────────────────────────────────────────────────────

def train(mislabels_csv: Path, out_dir: Path, verbose: bool = True) -> dict:
    mislabels = pd.read_csv(mislabels_csv)
    mislabels["flagged"] = mislabels["flagged"].astype(bool)

    meta_path = DATA_ROOT / "metadata" / "train_tiles.geojson"
    with open(meta_path) as f:
        gj = json.load(f)
    props = gj["features"][0]["properties"]
    cands = ["tile_id", "id", "name", "tile", "TILE_ID"]
    key   = next((k for k in cands if k in props), list(props.keys())[0])
    all_tiles = [feat["properties"][key] for feat in gj["features"]]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Pixel LightGBM v2 — {len(all_tiles)} tiles, {N_FEATURES} features")
        print(f"{'='*60}\n")

    tile_feats:  dict[str, np.ndarray] = {}
    tile_labels: dict[str, np.ndarray] = {}
    tile_years:  dict[str, np.ndarray] = {}

    for tile_id in all_tiles:
        if verbose:
            print(f"  Sampling {tile_id} …", end=" ", flush=True)
        result = sample_tile(tile_id, mislabels, verbose=verbose)
        if result is None:
            continue
        feats, labels, years = result
        tile_feats[tile_id]  = feats
        tile_labels[tile_id] = labels
        tile_years[tile_id]  = years
        if verbose:
            print(f"{len(labels)} px  (pos={labels.sum()}  neg={(labels==0).sum()})")

    sampled_tiles = list(tile_feats.keys())
    regions = {t: get_region(t) for t in sampled_tiles}

    models:  dict[str, LGBMClassifier] = {}
    scalers: dict[str, RobustScaler]   = {}
    loto_rows: list[dict] = []

    for region in ["SEA", "SAM"]:
        rtiles = [t for t in sampled_tiles if regions[t] == region]
        if not rtiles:
            continue

        if verbose:
            print(f"\n  [{region}] LOTO-CV on {len(rtiles)} tiles …")

        oof_preds: dict[str, dict] = {}
        for held in rtiles:
            train_t = [t for t in rtiles if t != held]
            if not train_t:
                continue

            X_tr = np.vstack([tile_feats[t]  for t in train_t])
            y_tr = np.concatenate([tile_labels[t] for t in train_t])
            X_va = tile_feats[held]
            y_va = tile_labels[held]

            med  = np.nanmedian(X_tr, axis=0)
            med  = np.where(np.isfinite(med), med, 0.0)
            X_tr = np.where(np.isfinite(X_tr), X_tr, med)
            X_va = np.where(np.isfinite(X_va), X_va, med)

            sc   = RobustScaler()
            clf  = LGBMClassifier(**LGBM_PARAMS)
            clf.fit(sc.fit_transform(X_tr), y_tr)
            p_va = clf.predict_proba(sc.transform(X_va))[:, 1]

            # Dual-target threshold: IoU-optimal OR recall-floor, whichever is lower
            best_iou, best_t = 0.0, 0.5
            recall_t = 0.15   # most aggressive fallback
            for t in np.arange(0.15, 0.85, 0.01):
                pred_t = (p_va >= t).astype(int)
                iou = compute_iou(y_va, pred_t)
                rec = compute_recall(y_va, pred_t)
                if iou > best_iou:
                    best_iou, best_t = iou, float(t)
                if rec >= RECALL_TARGET:
                    recall_t = float(t)   # keep updating → highest t with recall≥target

            final_t = min(best_t, recall_t)   # lower = more recall
            pred = (p_va >= final_t).astype(int)
            oof_preds[held] = {
                "iou": compute_iou(y_va, pred),
                "recall": compute_recall(y_va, pred),
                "fpr": compute_fpr(y_va, pred),
                "thresh": final_t,
            }
            loto_rows.append({"tile": held, "region": region, **oof_preds[held]})
            if verbose:
                r = oof_preds[held]
                print(f"    {held}: IoU={r['iou']:.3f}  Recall={r['recall']:.3f}  "
                      f"FPR={r['fpr']:.3f}  thresh={r['thresh']:.2f}"
                      f"  (iou_t={best_t:.2f} recall_t={recall_t:.2f})")

        # Median threshold for region
        region_thresh = float(np.median([v["thresh"] for v in oof_preds.values()])) \
            if oof_preds else 0.5

        # Final model on all region data
        X_all = np.vstack([tile_feats[t]  for t in rtiles])
        y_all = np.concatenate([tile_labels[t] for t in rtiles])
        med   = np.nanmedian(X_all, axis=0)
        med   = np.where(np.isfinite(med), med, 0.0)
        X_all = np.where(np.isfinite(X_all), X_all, med)

        sc_f  = RobustScaler()
        clf_f = LGBMClassifier(**LGBM_PARAMS)
        clf_f.fit(sc_f.fit_transform(X_all), y_all)

        sc_f.nan_fill_   = med
        sc_f.threshold_  = region_thresh
        models[region]   = clf_f
        scalers[region]  = sc_f

        if verbose and oof_preds:
            iou_vals = [v["iou"] for v in oof_preds.values()]
            print(f"  [{region}] mean LOTO IoU={np.mean(iou_vals):.3f}  "
                  f"thresh={region_thresh:.2f}")

    # Global fallback model
    X_g   = np.vstack(list(tile_feats.values()))
    y_g   = np.concatenate(list(tile_labels.values()))
    med_g = np.nanmedian(X_g, axis=0)
    med_g = np.where(np.isfinite(med_g), med_g, 0.0)
    X_g   = np.where(np.isfinite(X_g), X_g, med_g)
    sc_g  = RobustScaler()
    clf_g = LGBMClassifier(**LGBM_PARAMS)
    clf_g.fit(sc_g.fit_transform(X_g), y_g)
    sc_g.nan_fill_  = med_g
    sc_g.threshold_ = 0.4

    # Summary
    if verbose and loto_rows:
        ldf = pd.DataFrame(loto_rows)
        print(f"\n{'='*60}")
        print("LOTO Pixel Summary")
        print(f"{'='*60}")
        print(f"  IoU    mean={ldf['iou'].mean():.3f}  min={ldf['iou'].min():.3f}  max={ldf['iou'].max():.3f}")
        print(f"  Recall mean={ldf['recall'].mean():.3f}")
        print(f"  FPR    mean={ldf['fpr'].mean():.3f}")
        for region in ldf["region"].unique():
            sub = ldf[ldf["region"] == region]
            print(f"  [{region}] IoU={sub['iou'].mean():.3f}  "
                  f"Recall={sub['recall'].mean():.3f}  FPR={sub['fpr'].mean():.3f}")

    # Feature importance
    if verbose:
        print(f"\n{'='*60}")
        print("Top-15 Features by Region")
        print(f"{'='*60}")
        for region, model in models.items():
            imp = pd.Series(model.feature_importances_, index=FEATURE_NAMES)
            print(f"  [{region}]")
            for feat, val in imp.nlargest(15).items():
                print(f"    {feat:35s} {val:6.0f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "models": models, "scalers": scalers,
        "global_model": clf_g, "global_scaler": sc_g,
        "feature_names": FEATURE_NAMES,
    }
    model_path = out_dir / "pixel_model.pkl"
    joblib.dump(artifacts, model_path)
    if verbose:
        print(f"\nSaved → {model_path}")
    return artifacts


# ── prediction ────────────────────────────────────────────────────────────────

def predict_tile(
    tile_id: str, split: str, artifacts: dict, verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple, object, object] | None:

    region = get_region(tile_id)
    model  = artifacts["models"].get(region, artifacts["global_model"])
    scaler = artifacts["scalers"].get(region, artifacts["global_scaler"])
    thresh = getattr(scaler, "threshold_", 0.4)
    nan_fill = getattr(scaler, "nan_fill_", None)

    s2_dir   = DATA_ROOT / "sentinel-2" / split / f"{tile_id}__s2_l2a"
    s2_files = sorted(s2_dir.glob("*.tif"))
    if not s2_files:
        if verbose: print(f"  {tile_id}: no S2 — SKIP")
        return None

    with rasterio.open(s2_files[0]) as src:
        ref_transform = src.transform
        ref_crs       = src.crs
        ref_shape     = src.shape

    H, W = ref_shape

    aef_by_year = load_aef_all_years(tile_id, split, ref_transform, ref_crs, ref_shape)
    if not aef_by_year or 2020 not in aef_by_year:
        if verbose: print(f"  {tile_id}: missing AEF — SKIP")
        return None

    ndvi_sar, ndvi_ts, ndvi_months = load_ndvi_sar_maps(tile_id, split, ref_transform, ref_crs, ref_shape)

    harmonic_maps = None
    if ndvi_ts is not None and ndvi_months is not None and len(ndvi_months) > 0:
        harmonic_maps = compute_harmonic_maps(ndvi_ts, ndvi_months)

    feats = compute_pixel_features(aef_by_year, ndvi_sar, harmonic_maps=harmonic_maps)   # (H*W, N_FEATURES)

    if nan_fill is not None:
        feats = np.where(np.isfinite(feats), feats, nan_fill)
    else:
        feats = np.where(np.isfinite(feats), feats, 0.0)

    # Predict in chunks
    CHUNK = 200_000
    proba = np.zeros(H * W, dtype=np.float32)
    for s in range(0, H * W, CHUNK):
        e = min(s + CHUNK, H * W)
        proba[s:e] = model.predict_proba(scaler.transform(feats[s:e]))[:, 1]

    # Gaussian spatial smoothing — kills isolated-pixel FP
    prob_map    = gaussian_filter(proba.reshape(H, W), sigma=GAUSS_SIGMA)
    binary_map  = (prob_map >= thresh).astype(np.uint8)

    # SAR confirmation filter — remove optical-only FP (cloud shadows, agri cycles)
    # Real forest clearing: SAR backscatter decreases (forest → bare ground)
    # sar_change = sar_pre − sar_post; positive ≡ SAR decreased ≡ real structural loss
    if ndvi_sar and "sar_change" in ndvi_sar:
        sar_confirmed = (ndvi_sar["sar_change"] > 1.0).astype(np.uint8)
        binary_map    = binary_map & sar_confirmed

    # Year: prefer NDVI change year (interpretable), fallback to AEF argmax
    if ndvi_sar and "ndvi_change_year_idx" in ndvi_sar:
        # ndvi_change_year_idx: 0=2021,1=2022,2=2023,3=2024
        yr_idx_map = ndvi_sar["ndvi_change_year_idx"].astype(np.int32).clip(0, 3)
    else:
        yr_idx_map = feats[:, 14].astype(np.int32).clip(0, 3).reshape(H, W)

    year_map = (yr_idx_map + 2021).astype(np.int16)
    year_map[binary_map == 0] = 0

    if verbose:
        n_pos = binary_map.sum()
        print(f"  {tile_id} [{region}]: thresh={thresh:.2f}  "
              f"defor={n_pos:,}  ({100*n_pos/(H*W):.2f}%)")

    return binary_map, year_map, ref_shape, ref_transform, ref_crs


def predict_all(
    artifacts: dict, out_dir: Path, split: str = "test", verbose: bool = True,
) -> list[Path]:
    meta = DATA_ROOT / "metadata" / f"{split}_tiles.geojson"
    with open(meta) as f:
        gj = json.load(f)
    props = gj["features"][0]["properties"]
    cands = ["tile_id", "id", "name", "tile", "TILE_ID"]
    key   = next((k for k in cands if k in props), list(props.keys())[0])
    tiles = [feat["properties"][key] for feat in gj["features"]]

    raster_dir = out_dir / "pixel_predictions"
    raster_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Predicting {len(tiles)} {split} tile(s)")
        print(f"{'='*60}\n")

    written: list[Path] = []
    for tile_id in tiles:
        result = predict_tile(tile_id, split, artifacts, verbose=verbose)
        if result is None:
            continue
        binary_map, year_map, ref_shape, ref_transform, ref_crs = result
        out_path = raster_dir / f"{tile_id}_prediction.tif"
        yr_off = (year_map - 2020).clip(0, 255).astype(np.uint8)
        with rasterio.open(
            out_path, "w", driver="GTiff",
            height=ref_shape[0], width=ref_shape[1],
            count=2, dtype="uint8",
            crs=ref_crs, transform=ref_transform, compress="lzw",
        ) as dst:
            dst.write(binary_map, 1)
            dst.write(yr_off, 2)
        written.append(out_path)

    if written:
        _build_submission(written, out_dir, verbose=verbose)
    return written


def _build_submission(raster_paths: list[Path], out_dir: Path, verbose: bool) -> None:
    pieces = []
    for rpath in raster_paths:
        with rasterio.open(rpath) as src:
            binary = src.read(1).astype(np.uint8)
            yr_off = src.read(2).astype(np.uint8)
            transform, crs = src.transform, src.crs

        polygons, timesteps = [], []
        for geom_dict, val in shapes(binary, mask=binary, transform=transform):
            if val != 1:
                continue
            geom = shape(geom_dict)
            cx, cy = geom.centroid.x, geom.centroid.y
            col = max(0, min(int((cx - transform.c) / transform.a), binary.shape[1] - 1))
            row = max(0, min(int((cy - transform.f) / transform.e), binary.shape[0] - 1))
            yr  = max(2021, min(2024, int(yr_off[row, col]) + 2020))
            polygons.append(geom)
            timesteps.append(f"{yr % 100:02d}06")

        if not polygons:
            continue
        gdf = gpd.GeoDataFrame({"time_step": timesteps}, geometry=polygons, crs=crs)
        gdf = gdf.to_crs("EPSG:4326")
        utm = gdf.estimate_utm_crs()
        gdf = gdf[(gdf.to_crs(utm).area / 10_000) >= 0.5].reset_index(drop=True)
        if not gdf.empty:
            pieces.append(gdf)

    if not pieces:
        print("  No polygons above 0.5 ha.")
        return

    sub = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs="EPSG:4326")
    out = out_dir / "submission.geojson"
    sub.to_file(out, driver="GeoJSON")
    if verbose:
        print(f"\n  Submission → {out}  ({len(sub)} polygons)")
        for ts, cnt in sub["time_step"].value_counts().sort_index().items():
            print(f"    20{ts[:2]}-{ts[2:]}: {cnt}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode",      choices=["train", "predict", "all"], default="all")
    p.add_argument("--mislabels", default="results/mislabels_v2.csv")
    p.add_argument("--out-dir",   default="results")
    p.add_argument("--split",     default="test")
    p.add_argument("--quiet",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    out_dir = Path(args.out_dir)
    verbose = not args.quiet

    if args.mode in ("train", "all"):
        artifacts = train(Path(args.mislabels), out_dir, verbose)

    if args.mode in ("predict", "all"):
        if args.mode == "predict":
            mp = out_dir / "pixel_model.pkl"
            if not mp.exists():
                sys.exit(f"No model at {mp}. Run --mode train first.")
            artifacts = joblib.load(mp)
        predict_all(artifacts, out_dir, args.split, verbose)

    print("\nDone.")
