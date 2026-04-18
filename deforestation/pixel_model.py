"""
Pixel-Level LightGBM — Deforestation Classifier
================================================

Trains directly on pixels from training tiles, not on polygon aggregations.

Features per pixel (all change-based → generalises across regions):
  aef_l2_delta_{2021-2024}      — L2 distance between consecutive year embeddings
  aef_cos_delta_{2021-2024}     — cosine distance between consecutive year embeddings
  aef_from2020_l2_{2021-2024}   — L2 drift from 2020 baseline
  aef_max_delta_l2              — peak L2 change across all year pairs
  aef_change_year_idx           — which year had the biggest change (0=2021 … 3=2024)
  aef_delta_vec_{0-63}          — 64-dim change vector (emb_2024 − emb_2020)
                                   captures semantic direction of land-cover change

Target:
  Pixel-level majority vote of RADD + GLAD-S2 + GLAD-L weak labels,
  with CL corrections from mislabels_v2.csv applied at polygon level.

Training strategy:
  Stratified pixel sampling (balanced pos/neg) per tile.
  Leave-one-tile-out CV per region for honest metric reporting.
  Separate models for SEA and SAM; falls back to global for unknown regions.

Usage (from makeathon root):
    python deforestation/pixel_model.py --mode all
    python deforestation/pixel_model.py --mode train
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
from shapely.geometry import mapping, shape
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from mislabel_detection_v2 import DATA_ROOT, load_tile  # noqa: E402
from validate_spatial import build_gt_pixel_map          # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────

AEF_YEARS   = [2020, 2021, 2022, 2023, 2024]
DELTA_YEARS = [2021, 2022, 2023, 2024]
N_SAMPLE_PER_TILE = 60_000   # pixels per tile (balanced pos/neg)

REGIONS = {"SEA": ("47", "48"), "SAM": ("18", "19")}

LGBM_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=24,
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


# ── helpers ───────────────────────────────────────────────────────────────────

def get_region(tile_id: str) -> str:
    prefix = tile_id[:2]
    for region, prefixes in REGIONS.items():
        if prefix in prefixes:
            return region
    return "SAM"  # conservative fallback for unknown regions


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
    path: Path,
    ref_transform,
    ref_crs,
    ref_shape: tuple[int, int],
) -> np.ndarray:
    """Load AEF tiff and reproject to S2 reference grid. Returns (bands, H, W)."""
    with rasterio.open(path) as src:
        n_bands = src.count
        out = np.zeros((n_bands, *ref_shape), dtype=np.float32)
        for b in range(1, n_bands + 1):
            reproject(
                source=src.read(b).astype(np.float32),
                destination=out[b - 1],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
    return out   # (64, H, W)


def load_aef_all_years(
    tile_id: str,
    split: str,
    ref_transform,
    ref_crs,
    ref_shape: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Load all annual AEF files for a tile. Returns {year: (64, H, W)}."""
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


# ── pixel feature extraction ──────────────────────────────────────────────────

def compute_pixel_features(
    aef_by_year: dict[int, np.ndarray],
    sample_rows: np.ndarray | None = None,
    sample_cols: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute per-pixel change features from annual AEF embeddings.

    If sample_rows/sample_cols provided, only compute for those pixels.
    Otherwise computes for all pixels and returns (H*W, n_feat) flattened.

    Features (all change-based):
      4  aef_l2_delta per year pair
      4  aef_cos_delta per year pair
      4  aef_from2020_l2 per year
      1  aef_max_delta_l2
      1  aef_change_year_idx  (0=2021, 1=2022, 2=2023, 3=2024)
      64 aef_delta_vec  (emb_2024 - emb_2020, or nearest available)
    ─────────────────────────────────────────
      78 total features
    """
    # Determine shape
    any_emb = next(iter(aef_by_year.values()))   # (64, H, W)
    _, H, W = any_emb.shape

    if sample_rows is not None:
        # Extract sampled pixels from each year: (64, N)
        def _px(arr): return arr[:, sample_rows, sample_cols]
        N = len(sample_rows)
    else:
        # Flatten all pixels: (64, H*W)
        def _px(arr): return arr.reshape(64, -1)
        N = H * W

    emb: dict[int, np.ndarray] = {
        yr: _px(cube) for yr, cube in aef_by_year.items()
    }  # yr → (64, N)

    feats = np.full((N, 78), np.nan, dtype=np.float32)
    col = 0

    # ── year-over-year L2 delta ───────────────────────────────────────────────
    l2_deltas = {}
    for i, yr in enumerate(DELTA_YEARS):
        if yr in emb and (yr - 1) in emb:
            diff = emb[yr] - emb[yr - 1]           # (64, N)
            l2   = np.linalg.norm(diff, axis=0)    # (N,)
        else:
            l2 = np.zeros(N, dtype=np.float32)
        feats[:, col] = l2
        l2_deltas[yr] = l2
        col += 1

    # ── year-over-year cosine delta ───────────────────────────────────────────
    for yr in DELTA_YEARS:
        if yr in emb and (yr - 1) in emb:
            a  = emb[yr];    na = np.linalg.norm(a, axis=0) + 1e-9
            b  = emb[yr-1];  nb = np.linalg.norm(b, axis=0) + 1e-9
            cos_sim = (a * b).sum(axis=0) / (na * nb)
            feats[:, col] = (1.0 - cos_sim).clip(0, 2)
        col += 1

    # ── cumulative L2 from 2020 baseline ─────────────────────────────────────
    for yr in DELTA_YEARS:
        if yr in emb and 2020 in emb:
            feats[:, col] = np.linalg.norm(emb[yr] - emb[2020], axis=0)
        col += 1

    # ── max delta and argmax year ─────────────────────────────────────────────
    l2_stack = np.stack([l2_deltas[yr] for yr in DELTA_YEARS], axis=0)  # (4, N)
    feats[:, col]     = l2_stack.max(axis=0)
    col += 1
    feats[:, col]     = l2_stack.argmax(axis=0).astype(np.float32)
    col += 1

    # ── 64-dim change vector: emb_latest - emb_2020 ──────────────────────────
    latest_yr = max(yr for yr in aef_by_year if yr > 2020) if any(yr > 2020 for yr in aef_by_year) else None
    if latest_yr is not None and 2020 in emb:
        delta_vec = (emb[latest_yr] - emb[2020]).T   # (N, 64)
        feats[:, col:col + 64] = delta_vec
    col += 64

    assert col == 78, f"Feature count mismatch: {col}"
    return feats


FEATURE_NAMES = (
    [f"aef_l2_delta_{yr}"      for yr in DELTA_YEARS] +
    [f"aef_cos_delta_{yr}"     for yr in DELTA_YEARS] +
    [f"aef_from2020_l2_{yr}"   for yr in DELTA_YEARS] +
    ["aef_max_delta_l2", "aef_change_year_idx"] +
    [f"aef_delta_vec_{i:02d}"  for i in range(64)]
)


# ── tile pixel sampling ───────────────────────────────────────────────────────

def sample_tile(
    tile_id: str,
    mislabels: pd.DataFrame | None,
    n_sample: int = N_SAMPLE_PER_TILE,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Sample pixels from one training tile.
    Returns (features, labels, year_labels) or None on failure.

    Stratified: up to n_sample//2 positives + n_sample//2 negatives.
    """
    rng = np.random.default_rng(seed)

    try:
        td = load_tile(tile_id)
    except FileNotFoundError as e:
        print(f"  {tile_id}: SKIP — {e}")
        return None

    H, W = td.ref_shape

    # GT pixel labels
    tile_rows = (
        mislabels[mislabels["tile_id"] == tile_id].reset_index(drop=True)
        if mislabels is not None else None
    )
    gt_binary, gt_year = build_gt_pixel_map(td, tile_rows)

    flat_gt   = gt_binary.ravel()
    flat_year = gt_year.ravel()

    pos_idx = np.where(flat_gt == 1)[0]
    neg_idx = np.where(flat_gt == 0)[0]

    n_pos = min(len(pos_idx), n_sample // 2)
    n_neg = min(len(neg_idx), n_sample - n_pos)

    if n_pos == 0:
        print(f"  {tile_id}: no positive pixels — SKIP")
        return None

    chosen_pos = rng.choice(pos_idx, n_pos, replace=False)
    chosen_neg = rng.choice(neg_idx, n_neg, replace=False)
    idx        = np.concatenate([chosen_pos, chosen_neg])

    rows = (idx // W).astype(np.int32)
    cols = (idx %  W).astype(np.int32)

    # Load AEF all years
    aef_by_year = load_aef_all_years(
        tile_id, "train", td.ref_transform, td.ref_crs, td.ref_shape
    )
    if not aef_by_year or 2020 not in aef_by_year:
        print(f"  {tile_id}: missing 2020 AEF — SKIP")
        return None

    feats      = compute_pixel_features(aef_by_year, rows, cols)
    labels     = flat_gt[idx].astype(np.int32)
    year_labels = flat_year[idx].astype(np.int32)

    return feats, labels, year_labels


# ── training ──────────────────────────────────────────────────────────────────

def train(
    mislabels_csv: Path,
    out_dir: Path,
    verbose: bool = True,
) -> dict:
    mislabels = pd.read_csv(mislabels_csv)
    mislabels["flagged"] = mislabels["flagged"].astype(bool)

    # Get training tiles
    meta_path = DATA_ROOT / "metadata" / "train_tiles.geojson"
    with open(meta_path) as f:
        gj = json.load(f)
    props = gj["features"][0]["properties"]
    candidates = ["tile_id", "id", "name", "tile", "TILE_ID"]
    key = next((k for k in candidates if k in props), list(props.keys())[0])
    all_tiles = [feat["properties"][key] for feat in gj["features"]]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Pixel-Level LightGBM — {len(all_tiles)} training tile(s)")
        print(f"{'='*60}\n")

    # ── sample pixels from all training tiles ─────────────────────────────────
    tile_feats:  dict[str, np.ndarray] = {}
    tile_labels: dict[str, np.ndarray] = {}
    tile_years:  dict[str, np.ndarray] = {}

    for tile_id in all_tiles:
        if verbose:
            print(f"  Sampling {tile_id} …", end=" ", flush=True)
        result = sample_tile(tile_id, mislabels)
        if result is None:
            continue
        feats, labels, years = result
        tile_feats[tile_id]  = feats
        tile_labels[tile_id] = labels
        tile_years[tile_id]  = years
        if verbose:
            print(f"{len(labels)} pixels  (pos={labels.sum()}  neg={(labels==0).sum()})")

    sampled_tiles = list(tile_feats.keys())
    regions = {t: get_region(t) for t in sampled_tiles}

    # ── per-region LOTO-CV then final model ───────────────────────────────────
    models:  dict[str, LGBMClassifier] = {}
    scalers: dict[str, RobustScaler]   = {}
    global_model:  LGBMClassifier | None = None
    global_scaler: RobustScaler   | None = None

    loto_results: list[dict] = []

    for region in ["SEA", "SAM"]:
        region_tiles = [t for t in sampled_tiles if regions[t] == region]
        if not region_tiles:
            continue

        if verbose:
            print(f"\n  [{region}] {len(region_tiles)} tiles — LOTO-CV …")

        # LOTO cross-validation
        oof_preds = {}
        for held in region_tiles:
            train_tiles = [t for t in region_tiles if t != held]
            if not train_tiles:
                continue

            X_tr = np.vstack([tile_feats[t]  for t in train_tiles])
            y_tr = np.concatenate([tile_labels[t] for t in train_tiles])
            X_va = tile_feats[held]
            y_va = tile_labels[held]

            # Impute NaN with column median from training data
            col_medians = np.nanmedian(X_tr, axis=0)
            col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
            X_tr = np.where(np.isfinite(X_tr), X_tr, col_medians)
            X_va = np.where(np.isfinite(X_va), X_va, col_medians)

            scaler = RobustScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            clf = LGBMClassifier(**LGBM_PARAMS)
            clf.fit(X_tr_s, y_tr)
            p_va = clf.predict_proba(X_va_s)[:, 1]

            # IoU-optimal threshold on validation fold
            best_iou, best_thresh = 0.0, 0.5
            for thresh in np.arange(0.15, 0.85, 0.01):
                pred = (p_va >= thresh).astype(int)
                iou  = compute_iou(y_va, pred)
                if iou > best_iou:
                    best_iou, best_thresh = iou, thresh

            pred_best = (p_va >= best_thresh).astype(int)
            oof_preds[held] = {
                "iou":    compute_iou(y_va, pred_best),
                "recall": compute_recall(y_va, pred_best),
                "fpr":    compute_fpr(y_va, pred_best),
                "thresh": best_thresh,
            }
            loto_results.append({"tile": held, "region": region, **oof_preds[held]})

            if verbose:
                r = oof_preds[held]
                print(f"    {held}: IoU={r['iou']:.3f}  Recall={r['recall']:.3f}  "
                      f"FPR={r['fpr']:.3f}  thresh={r['thresh']:.2f}")

        # Region summary
        if oof_preds and verbose:
            iou_vals = [v["iou"] for v in oof_preds.values()]
            thresh_vals = [v["thresh"] for v in oof_preds.values()]
            print(f"  [{region}] mean LOTO IoU: {np.mean(iou_vals):.3f}  "
                  f"median thresh: {np.median(thresh_vals):.2f}")

        # Final model on all region data
        X_all = np.vstack([tile_feats[t]  for t in region_tiles])
        y_all = np.concatenate([tile_labels[t] for t in region_tiles])

        col_medians = np.nanmedian(X_all, axis=0)
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
        X_all = np.where(np.isfinite(X_all), X_all, col_medians)

        scaler_final = RobustScaler()
        X_all_s = scaler_final.fit_transform(X_all)

        clf_final = LGBMClassifier(**LGBM_PARAMS)
        clf_final.fit(X_all_s, y_all)

        # Use median LOTO threshold for this region
        region_thresholds = [v["thresh"] for v in oof_preds.values()] if oof_preds else [0.5]
        region_thresh = float(np.median(region_thresholds))

        models[region]  = clf_final
        scalers[region] = scaler_final

        # Store NaN-fill values (medians) with scaler
        scaler_final.nan_fill_ = col_medians
        scaler_final.threshold_ = region_thresh

    # ── global fallback model (all regions) ───────────────────────────────────
    X_g = np.vstack(list(tile_feats.values()))
    y_g = np.concatenate(list(tile_labels.values()))
    med_g = np.nanmedian(X_g, axis=0)
    med_g = np.where(np.isfinite(med_g), med_g, 0.0)
    X_g   = np.where(np.isfinite(X_g), X_g, med_g)

    global_scaler = RobustScaler()
    X_g_s = global_scaler.fit_transform(X_g)
    global_model = LGBMClassifier(**LGBM_PARAMS)
    global_model.fit(X_g_s, y_g)
    global_scaler.nan_fill_ = med_g
    global_scaler.threshold_ = 0.4   # conservative default

    # ── LOTO summary ──────────────────────────────────────────────────────────
    if verbose and loto_results:
        ldf = pd.DataFrame(loto_results)
        print(f"\n{'='*60}")
        print("LOTO Pixel-Level Summary")
        print(f"{'='*60}")
        print(f"  Overall mean IoU   : {ldf['iou'].mean():.3f}")
        print(f"  Overall mean Recall: {ldf['recall'].mean():.3f}")
        print(f"  Overall mean FPR   : {ldf['fpr'].mean():.3f}")
        for region in ldf["region"].unique():
            sub = ldf[ldf["region"] == region]
            print(f"  [{region}] IoU={sub['iou'].mean():.3f}  "
                  f"Recall={sub['recall'].mean():.3f}  FPR={sub['fpr'].mean():.3f}")

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "models":        models,
        "scalers":       scalers,
        "global_model":  global_model,
        "global_scaler": global_scaler,
        "feature_names": FEATURE_NAMES,
    }
    model_path = out_dir / "pixel_model.pkl"
    joblib.dump(artifacts, model_path)
    if verbose:
        print(f"\nSaved → {model_path}")

    return artifacts


# ── prediction ────────────────────────────────────────────────────────────────

def predict_tile(
    tile_id: str,
    split: str,
    artifacts: dict,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple, object, object] | None:
    """
    Predict deforestation for one tile at pixel level.
    Returns (binary_map, year_map, ref_shape, ref_transform, ref_crs) or None.
    """
    region = get_region(tile_id)
    model  = artifacts["models"].get(region, artifacts["global_model"])
    scaler = artifacts["scalers"].get(region, artifacts["global_scaler"])
    thresh = getattr(scaler, "threshold_", 0.4)
    nan_fill = getattr(scaler, "nan_fill_", None)

    # Reference grid from S2
    s2_dir = DATA_ROOT / "sentinel-2" / split / f"{tile_id}__s2_l2a"
    s2_files = sorted(s2_dir.glob("*.tif"))
    if not s2_files:
        if verbose:
            print(f"  {tile_id}: no S2 data — SKIP")
        return None

    with rasterio.open(s2_files[0]) as src:
        ref_transform = src.transform
        ref_crs       = src.crs
        ref_shape     = src.shape

    H, W = ref_shape

    # Load AEF all years
    aef_by_year = load_aef_all_years(
        tile_id, split, ref_transform, ref_crs, ref_shape
    )
    if not aef_by_year or 2020 not in aef_by_year:
        if verbose:
            print(f"  {tile_id}: missing AEF data — SKIP")
        return None

    # Compute pixel features for entire tile (flat)
    feats = compute_pixel_features(aef_by_year)   # (H*W, 78)

    if nan_fill is not None:
        mask_nan = ~np.isfinite(feats)
        feats[mask_nan] = np.broadcast_to(nan_fill, feats.shape)[mask_nan]
    else:
        feats = np.where(np.isfinite(feats), feats, 0.0)

    # Predict in chunks to avoid memory spikes
    CHUNK = 200_000
    proba = np.zeros(H * W, dtype=np.float32)
    for start in range(0, H * W, CHUNK):
        end = min(start + CHUNK, H * W)
        chunk_s = scaler.transform(feats[start:end])
        proba[start:end] = model.predict_proba(chunk_s)[:, 1]

    binary_map = (proba >= thresh).astype(np.uint8).reshape(H, W)

    # Year map: argmax AEF L2 delta (feature index 0-3 → year 2021-2024)
    year_idx_flat = feats[:, 14].astype(np.int32).clip(0, 3)  # aef_change_year_idx
    year_map = (year_idx_flat + 2021).astype(np.int16).reshape(H, W)
    year_map[binary_map == 0] = 0

    if verbose:
        n_pos = binary_map.sum()
        print(f"  {tile_id} [{region}]: thresh={thresh:.2f}  "
              f"defor_pixels={n_pos:,}  ({100*n_pos/(H*W):.2f}%)")

    return binary_map, year_map, ref_shape, ref_transform, ref_crs


def predict_all(
    artifacts: dict,
    out_dir: Path,
    split: str = "test",
    verbose: bool = True,
) -> list[Path]:
    # Get tile list
    meta = DATA_ROOT / "metadata" / f"{split}_tiles.geojson"
    with open(meta) as f:
        gj = json.load(f)
    props = gj["features"][0]["properties"]
    candidates = ["tile_id", "id", "name", "tile", "TILE_ID"]
    key = next((k for k in candidates if k in props), list(props.keys())[0])
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
        with rasterio.open(
            out_path, "w", driver="GTiff",
            height=ref_shape[0], width=ref_shape[1],
            count=2, dtype="uint8",
            crs=ref_crs, transform=ref_transform,
            compress="lzw",
        ) as dst:
            dst.write(binary_map, 1)
            year_offset = (year_map - 2020).clip(0, 255).astype(np.uint8)
            dst.write(year_offset, 2)
            dst.update_tags(band_1="deforestation", band_2="year_offset_from_2020")

        written.append(out_path)

    if written:
        _build_submission(written, out_dir, verbose=verbose)

    return written


def _build_submission(raster_paths: list[Path], out_dir: Path, verbose: bool) -> None:
    pieces = []
    for rpath in raster_paths:
        with rasterio.open(rpath) as src:
            binary    = src.read(1).astype(np.uint8)
            yr_offset = src.read(2).astype(np.uint8)
            transform = src.transform
            crs       = src.crs

        polygons, timesteps = [], []
        for geom_dict, val in shapes(binary, mask=binary, transform=transform):
            if val != 1:
                continue
            geom = shape(geom_dict)
            cx, cy = geom.centroid.x, geom.centroid.y
            col = max(0, min(int((cx - transform.c) / transform.a), binary.shape[1] - 1))
            row = max(0, min(int((cy - transform.f) / transform.e), binary.shape[0] - 1))
            yr  = int(yr_offset[row, col]) + 2020
            yr  = max(2021, min(2024, yr))
            polygons.append(geom)
            timesteps.append(f"{yr % 100:02d}06")

        if not polygons:
            continue

        gdf = gpd.GeoDataFrame({"time_step": timesteps}, geometry=polygons, crs=crs)
        gdf = gdf.to_crs("EPSG:4326")

        utm_crs = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)
        gdf     = gdf[gdf_utm.area / 10_000 >= 0.5].reset_index(drop=True)
        if not gdf.empty:
            pieces.append(gdf)

    if not pieces:
        print("  No polygons above 0.5 ha.")
        return

    submission = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs="EPSG:4326")
    out_path = out_dir / "submission.geojson"
    submission.to_file(out_path, driver="GeoJSON")

    if verbose:
        print(f"\n  Submission → {out_path}  ({len(submission)} polygons)")
        for ts, cnt in submission["time_step"].value_counts().sort_index().items():
            print(f"    20{ts[:2]}-{ts[2:]}: {cnt}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pixel-level LightGBM deforestation classifier")
    p.add_argument("--mode",      choices=["train", "predict", "all"], default="all")
    p.add_argument("--mislabels", default="results/mislabels_v2.csv")
    p.add_argument("--out-dir",   default="results")
    p.add_argument("--split",     default="test")
    p.add_argument("--quiet",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    verbose = not args.quiet
    out_dir = Path(args.out_dir)

    if args.mode in ("train", "all"):
        artifacts = train(
            mislabels_csv=Path(args.mislabels),
            out_dir=out_dir,
            verbose=verbose,
        )

    if args.mode in ("predict", "all"):
        if args.mode == "predict":
            model_path = out_dir / "pixel_model.pkl"
            if not model_path.exists():
                sys.exit(f"No model at {model_path}. Run --mode train first.")
            artifacts = joblib.load(model_path)
        predict_all(artifacts, out_dir, split=args.split, verbose=verbose)

    print("\nDone.")
