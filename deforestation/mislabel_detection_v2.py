"""
Mislabel Detection v2 — Confident Learning + Harmonic Breakpoint Detection

Pipeline
--------
1. Load all training tiles; for each polygon extract:
     - Harmonic regression on NDVI time series → physics-based change score
     - NDVI pre/post means + temporal std  (Sentinel-2)
     - SAR pre/post means                  (Sentinel-1)
     - AEF mean embeddings (64 dims)

2. Four noisy label sources per polygon:
     RADD (S1-based) · GLAD-S2 (S2-based) · GLAD-L (Landsat) · Physics label

3. Majority vote of available sources → initial noisy label for the classifier.

4. RandomForestClassifier with 5-fold stratified cross-validation
   → calibrated out-of-fold predicted probabilities.

5. Confident Learning (Northcutt et al. 2021, implemented from scratch):
     - Per-class adaptive thresholds from OOF probabilities
     - Confident joint matrix  C[given, true_est]
     - Mislabel score = 1 - p̂(given label)
     - Flag = confident estimate differs from given label

6. Output: ranked polygon list with mislabel scores + suggested corrections.

Usage (from makeathon root):
    python deforestation/mislabel_detection_v2.py --output results/mislabels_v2.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes, geometry_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape, mapping
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_ROOT = Path("data/makeathon-challenge")

# ── helpers ───────────────────────────────────────────────────────────────────

def _ndvi(s2_path: Path) -> np.ndarray:
    with rasterio.open(s2_path) as src:
        nir = src.read(8).astype(np.float32)
        red = src.read(4).astype(np.float32)
    denom = nir + red
    with np.errstate(invalid="ignore"):
        return np.where(denom > 0, (nir - red) / denom, np.nan)


def _s1_db(arr: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(arr > 0, 10.0 * np.log10(arr), np.nan)


def _reproject_onto(src_path: Path, dst_transform, dst_crs, dst_shape,
                    band: int = 1, resampling=Resampling.nearest) -> np.ndarray:
    out = np.zeros(dst_shape, dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source=src.read(band).astype(np.float32),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )
    return out


# ── data container ────────────────────────────────────────────────────────────

@dataclass
class TileData:
    tile_id: str
    ref_transform: object = None
    ref_crs: object = None
    ref_shape: tuple = None
    ndvi_ts: np.ndarray = field(default=None, repr=False)   # (T, H, W)
    ndvi_months: np.ndarray = field(default=None, repr=False) # (T,) months since Jan-2020
    sar_ts: np.ndarray = field(default=None, repr=False)    # (T, H, W)
    sar_months: np.ndarray = field(default=None, repr=False)  # (T,)
    aef: np.ndarray = field(default=None, repr=False)       # (64, H, W)
    radd_binary: np.ndarray = field(default=None, repr=False)
    glads2_binary: np.ndarray = field(default=None, repr=False)
    gladl_binary: np.ndarray = field(default=None, repr=False)


def _month_index(year: int, month: int) -> int:
    """Months since 2020-01 (0-indexed)."""
    return (year - 2020) * 12 + (month - 1)


def load_tile(tile_id: str, years: tuple[int, int] = (2020, 2024)) -> TileData:
    """Load all data for one training tile."""
    td = TileData(tile_id=tile_id)
    year_range = range(years[0], years[1] + 1)

    # ── S2 reference grid + NDVI time series ─────────────────────────────────
    s2_dir = DATA_ROOT / "sentinel-2" / "train" / f"{tile_id}__s2_l2a"
    s2_files = sorted(s2_dir.glob("*.tif"))
    if not s2_files:
        raise FileNotFoundError(f"No S2 data for {tile_id}")

    with rasterio.open(s2_files[0]) as src:
        td.ref_transform = src.transform
        td.ref_crs = src.crs
        td.ref_shape = src.shape

    ndvi_stack, ndvi_months = [], []
    for f in s2_files:
        parts = f.stem.split("_")
        try:
            yr, mo = int(parts[-2]), int(parts[-1])
        except (ValueError, IndexError):
            continue
        if yr not in year_range:
            continue
        ndvi = _ndvi(f)
        if ndvi.shape != td.ref_shape:
            reproj = np.full(td.ref_shape, np.nan, dtype=np.float32)
            with rasterio.open(f) as src:
                reproject(
                    source=ndvi, destination=reproj,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=td.ref_transform, dst_crs=td.ref_crs,
                    resampling=Resampling.bilinear,
                )
            ndvi = reproj
        ndvi_stack.append(ndvi)
        ndvi_months.append(_month_index(yr, mo))

    if ndvi_stack:
        # Sort by time
        order = np.argsort(ndvi_months)
        td.ndvi_ts = np.stack(ndvi_stack, axis=0)[order]
        td.ndvi_months = np.array(ndvi_months)[order]

    # ── S1 SAR time series ────────────────────────────────────────────────────
    s1_dir = DATA_ROOT / "sentinel-1" / "train" / f"{tile_id}__s1_rtc"
    sar_stack, sar_months = [], []
    for f in sorted(s1_dir.glob("*_ascending.tif")):
        parts = f.stem.split("_")
        try:
            yr, mo = int(parts[-3]), int(parts[-2])
        except (ValueError, IndexError):
            continue
        if yr not in year_range:
            continue
        with rasterio.open(f) as src:
            raw = src.read(1).astype(np.float32)
            reproj = np.zeros(td.ref_shape, dtype=np.float32)
            reproject(
                source=raw, destination=reproj,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=td.ref_transform, dst_crs=td.ref_crs,
                resampling=Resampling.bilinear,
            )
        sar_stack.append(_s1_db(reproj))
        sar_months.append(_month_index(yr, mo))

    if sar_stack:
        order = np.argsort(sar_months)
        td.sar_ts = np.stack(sar_stack, axis=0)[order]
        td.sar_months = np.array(sar_months)[order]

    # ── AEF embeddings ────────────────────────────────────────────────────────
    aef_dir = DATA_ROOT / "aef-embeddings" / "train"
    aef_files = sorted(aef_dir.glob(f"{tile_id}_*.tiff"), reverse=True)
    if aef_files:
        with rasterio.open(aef_files[0]) as src:
            n_bands = src.count
            aef = np.zeros((n_bands, *td.ref_shape), dtype=np.float32)
            for b in range(1, n_bands + 1):
                reproject(
                    source=src.read(b).astype(np.float32),
                    destination=aef[b - 1],
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=td.ref_transform, dst_crs=td.ref_crs,
                    resampling=Resampling.bilinear,
                )
        td.aef = aef

    # ── Weak labels (reprojected to S2 grid) ─────────────────────────────────
    labels_root = DATA_ROOT / "labels" / "train"

    def _load_label(path: Path) -> np.ndarray:
        return _reproject_onto(path, td.ref_transform, td.ref_crs,
                               td.ref_shape, resampling=Resampling.nearest)

    radd_path = labels_root / "radd" / f"radd_{tile_id}_labels.tif"
    if radd_path.exists():
        raw = _load_label(radd_path)
        conf = (raw // 10000).astype(np.int32)
        days = (raw % 10000).astype(np.int32)
        td.radd_binary = ((conf >= 2) & (days >= 1827)).astype(np.uint8)

    gs2_alert = labels_root / "glads2" / f"glads2_{tile_id}_alert.tif"
    gs2_date  = labels_root / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    if gs2_alert.exists():
        a = _load_label(gs2_alert)
        d = _load_label(gs2_date) if gs2_date.exists() else np.ones_like(a) * 999
        td.glads2_binary = ((a >= 2) & (d > 365)).astype(np.uint8)

    gladl_dir = labels_root / "gladl"
    layers = []
    for yy in range(21, 25):
        for cand in [gladl_dir / f"gladl_{tile_id}_alert{yy}.tif",
                     gladl_dir / tile_id / f"gladl_{tile_id}_alert{yy}.tif"]:
            if cand.exists():
                raw = _load_label(cand)
                layers.append((raw >= 2).astype(np.uint8))
                break
    if layers:
        td.gladl_binary = np.any(np.stack(layers, axis=0), axis=0).astype(np.uint8)

    return td


# ── harmonic regression + CUSUM ───────────────────────────────────────────────

def _design_matrix(t: np.ndarray, n_harmonics: int = 2) -> np.ndarray:
    """Build harmonic regression design matrix for time vector t (months)."""
    cols = [np.ones(len(t)), t.astype(float)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(2 * np.pi * k * t / 12.0))
        cols.append(np.sin(2 * np.pi * k * t / 12.0))
    return np.column_stack(cols)


def harmonic_anomaly(
    months: np.ndarray,
    series: np.ndarray,
    train_cutoff: int = 24,   # months 0-23 = 2020+2021 as baseline
    n_harmonics: int = 2,
    min_train_obs: int = 8,
    min_test_obs: int = 3,
) -> tuple[int, float, float]:
    """
    Fit a harmonic + trend model on the baseline period and measure how much
    the post-cutoff observations deviate from the model prediction.

    Parameters
    ----------
    months      : 1-D array of month indices (0 = Jan-2020).
    series      : 1-D array of NDVI values (NaN where missing).
    train_cutoff: last month index included in training (exclusive).
    n_harmonics : number of sine/cosine harmonics.
    min_train_obs: minimum valid training observations required.
    min_test_obs : minimum valid test observations required.

    Returns
    -------
    physics_label : 0 or 1 (1 = significant vegetation loss detected)
    t_statistic   : signed t-stat of post-cutoff residuals (positive = NDVI dropped)
    change_mag    : mean absolute post-cutoff residual (0 if not enough data)
    """
    train_mask = months < train_cutoff
    test_mask  = months >= train_cutoff

    train_valid = train_mask & np.isfinite(series)
    test_valid  = test_mask  & np.isfinite(series)

    if train_valid.sum() < min_train_obs or test_valid.sum() < min_test_obs:
        return 0, 0.0, 0.0

    # Fit OLS on training period
    X_train = _design_matrix(months[train_valid], n_harmonics)
    y_train = series[train_valid]
    try:
        beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    except np.linalg.LinAlgError:
        return 0, 0.0, 0.0

    # Baseline residuals (to estimate noise std)
    y_hat_train = X_train @ beta
    baseline_resid = y_train - y_hat_train
    baseline_std = float(np.std(baseline_resid)) + 1e-6

    # Post-cutoff residuals: positive = observed below prediction = NDVI dropped
    X_test = _design_matrix(months[test_valid], n_harmonics)
    y_test = series[test_valid]
    y_hat_test = X_test @ beta
    post_resid = y_hat_test - y_test   # positive = drop

    n_test = len(post_resid)
    mean_resid = float(post_resid.mean())
    t_stat = mean_resid / (baseline_std / np.sqrt(n_test))
    change_mag = float(np.abs(post_resid).mean())

    # Significant NDVI drop: t > 1.96 (one-sided 95 %)
    physics_label = 1 if t_stat > 1.96 else 0
    return physics_label, round(t_stat, 4), round(change_mag, 4)


# ── consensus polygon builder ─────────────────────────────────────────────────

def build_consensus_polygons(td: TileData) -> gpd.GeoDataFrame:
    """
    Majority-vote the available weak-label rasters into a binary consensus map
    and vectorise it into polygons (min area 0.5 ha).
    """
    available = [arr for arr in [td.radd_binary, td.glads2_binary, td.gladl_binary]
                 if arr is not None]
    if not available:
        raise RuntimeError(f"No weak labels for tile {td.tile_id}")

    n = len(available)
    vote_sum = np.stack(available, axis=0).sum(axis=0)   # 0..n
    consensus = (vote_sum > n / 2).astype(np.uint8)

    polys, labels, agreements = [], [], []
    for geom_dict, val in shapes(consensus, mask=None, transform=td.ref_transform):
        lbl = int(val)
        geom = shape(geom_dict)
        # Source agreement at centroid pixel
        cx, cy = geom.centroid.x, geom.centroid.y
        col = int((cx - td.ref_transform.c) / td.ref_transform.a)
        row = int((cy - td.ref_transform.f) / td.ref_transform.e)
        col = max(0, min(col, td.ref_shape[1] - 1))
        row = max(0, min(row, td.ref_shape[0] - 1))
        v = int(vote_sum[row, col])
        agree = v if lbl == 1 else (n - v)
        polys.append(geom)
        labels.append(lbl)
        agreements.append(agree)

    gdf = gpd.GeoDataFrame(
        {"label": labels, "source_agreement": agreements},
        geometry=polys, crs=td.ref_crs,
    )
    gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
    gdf = gdf[(gdf_utm.area / 10_000) >= 0.5].reset_index(drop=True)
    return gdf


# ── per-polygon feature extraction ───────────────────────────────────────────

def _polygon_mask(geom, transform, shape) -> np.ndarray:
    return ~geometry_mask([mapping(geom)], transform=transform,
                          invert=False, out_shape=shape)


def _masked_series(stack: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over polygon pixels for each time step; NaN where no valid pixel."""
    result = np.full(len(stack), np.nan)
    for i, frame in enumerate(stack):
        px = frame[mask]
        valid = px[np.isfinite(px)]
        if valid.size:
            result[i] = float(valid.mean())
    return result


def extract_features(row_geom, td: TileData) -> dict:
    """
    Extract all features for one polygon geometry from a loaded TileData.
    Returns a flat dict of feature_name → float.
    """
    mask = _polygon_mask(row_geom, td.ref_transform, td.ref_shape)
    if mask.sum() == 0:
        return {}

    feats: dict = {}

    # ── NDVI features ─────────────────────────────────────────────────────────
    if td.ndvi_ts is not None and len(td.ndvi_months) > 0:
        ndvi_series = _masked_series(td.ndvi_ts, mask)
        months = td.ndvi_months

        pre_mask  = months < 24   # 2020-2021
        post_mask = months >= 24  # 2022+

        def _finite_mean(arr, sel):
            v = arr[sel]
            v = v[np.isfinite(v)]
            return float(v.mean()) if len(v) else np.nan

        feats["ndvi_pre"]    = _finite_mean(ndvi_series, pre_mask)
        feats["ndvi_post"]   = _finite_mean(ndvi_series, post_mask)
        feats["ndvi_change"] = (feats["ndvi_pre"] - feats["ndvi_post"]
                                if np.isfinite(feats["ndvi_pre"]) and np.isfinite(feats["ndvi_post"])
                                else np.nan)
        valid_all = ndvi_series[np.isfinite(ndvi_series)]
        feats["ndvi_std"] = float(valid_all.std()) if len(valid_all) > 1 else np.nan

        # Harmonic breakpoint
        phys_lbl, t_stat, change_mag = harmonic_anomaly(months, ndvi_series)
        feats["harmonic_physics_label"] = float(phys_lbl)
        feats["harmonic_t_stat"]        = t_stat
        feats["harmonic_change_mag"]    = change_mag

    # ── SAR features ──────────────────────────────────────────────────────────
    if td.sar_ts is not None and len(td.sar_months) > 0:
        sar_series = _masked_series(td.sar_ts, mask)
        months_s1  = td.sar_months
        pre_s1  = months_s1 < 24
        post_s1 = months_s1 >= 24

        def _s1_mean(sel):
            v = sar_series[sel]
            v = v[np.isfinite(v)]
            return float(v.mean()) if len(v) else np.nan

        feats["sar_pre"]    = _s1_mean(pre_s1)
        feats["sar_post"]   = _s1_mean(post_s1)
        feats["sar_change"] = (feats["sar_pre"] - feats["sar_post"]
                               if np.isfinite(feats["sar_pre"]) and np.isfinite(feats["sar_post"])
                               else np.nan)

    # ── AEF embedding means ───────────────────────────────────────────────────
    if td.aef is not None:
        for b in range(td.aef.shape[0]):
            px = td.aef[b][mask]
            valid = px[np.isfinite(px)]
            feats[f"aef_{b:02d}"] = float(valid.mean()) if valid.size else 0.0

    return feats


# ── confident learning (manual implementation) ────────────────────────────────

def confident_learning(
    proba: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Confident Learning for binary classification (Northcutt et al., 2021).

    Parameters
    ----------
    proba  : (n, 2) out-of-fold predicted probabilities, calibrated.
    labels : (n,)  given noisy binary labels (0 or 1).

    Returns
    -------
    mislabel_scores : (n,) float in [0,1]; higher = more likely mislabelled.
    flagged         : (n,) bool; confident mislabels.
    suggested       : (n,) int; estimated true label.
    """
    n = len(labels)
    K = 2

    # Per-class threshold: average p(class c) among samples labelled c
    thresholds = np.zeros(K)
    for c in range(K):
        mask = labels == c
        if mask.sum() > 0:
            thresholds[c] = proba[mask, c].mean()
        else:
            thresholds[c] = 0.5

    # Confident joint C[given_label, estimated_true_label]
    # A sample is "confidently" class j if p(j) >= threshold[j]
    C = np.zeros((K, K), dtype=int)
    confident_true = np.full(n, -1, dtype=int)   # -1 = not confident in any class
    for i in range(n):
        j = int(np.argmax(proba[i]))
        if proba[i, j] >= thresholds[j]:
            C[labels[i], j] += 1
            confident_true[i] = j

    # Mislabel score: 1 - p(given label) — lower prob for own class = more suspicious
    mislabel_scores = 1.0 - proba[np.arange(n), labels]

    # Flag: sample is flagged if it is confidently classified as a DIFFERENT class
    flagged = np.array([
        confident_true[i] != -1 and confident_true[i] != labels[i]
        for i in range(n)
    ])

    # Suggested label: argmax of predicted probability
    suggested = np.argmax(proba, axis=1)

    return mislabel_scores, flagged, suggested.astype(int)


# ── main pipeline ─────────────────────────────────────────────────────────────

def _get_train_tiles() -> list[str]:
    meta = DATA_ROOT / "metadata" / "train_tiles.geojson"
    with open(meta) as f:
        gj = json.load(f)
    props = gj["features"][0]["properties"]
    # Auto-detect key
    candidates = ["tile_id", "id", "name", "tile", "TILE_ID", "tileid", "tile_name"]
    key = next((k for k in candidates if k in props), None)
    if key is None:
        import re
        pat = re.compile(r"^\d{2}[A-Z]{3}_\d+_\d+$")
        key = next((k for k, v in props.items() if isinstance(v, str) and pat.match(v)), None)
    if key is None:
        raise KeyError(f"Cannot find tile ID key. Properties: {list(props.keys())}")
    return [feat["properties"][key] for feat in gj["features"]]


def run_pipeline(
    tiles: list[str],
    contamination: float = 0.1,
    n_cv_splits: int = 5,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Full mislabel detection pipeline across a list of training tiles.

    Steps:
      1. Load each tile, extract polygon features.
      2. Build 4-source majority-vote labels.
      3. Train RF with cross-validation → OOF probabilities.
      4. Confident Learning → mislabel scores + flags.

    Returns a GeoDataFrame (EPSG:4326) with all polygons annotated.
    """
    all_records = []
    all_geoms   = []
    all_tile_ids = []

    # ── Step 1 & 2: Load tiles, extract features ─────────────────────────────
    for tile_id in tiles:
        if verbose:
            print(f"  Loading {tile_id} …", end=" ", flush=True)
        try:
            td = load_tile(tile_id)
        except FileNotFoundError as e:
            if verbose:
                print(f"SKIP ({e})")
            continue

        try:
            gdf = build_consensus_polygons(td)
        except RuntimeError as e:
            if verbose:
                print(f"SKIP ({e})")
            continue

        if verbose:
            print(f"{len(gdf)} polygons")

        for _, row in gdf.iterrows():
            feats = extract_features(row.geometry, td)
            if not feats:
                continue

            # 4th label source: harmonic physics
            physics_lbl = int(feats.get("harmonic_physics_label", 0))

            # Majority vote: RADD, GLAD-S2, GLAD-L, physics
            source_votes = []
            # pixel vote from weak label rasters at polygon centroid
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            col = max(0, min(int((cx - td.ref_transform.c) / td.ref_transform.a),
                             td.ref_shape[1] - 1))
            row_px = max(0, min(int((cy - td.ref_transform.f) / td.ref_transform.e),
                                td.ref_shape[0] - 1))
            for arr in [td.radd_binary, td.glads2_binary, td.gladl_binary]:
                if arr is not None:
                    source_votes.append(int(arr[row_px, col]))
            source_votes.append(physics_lbl)

            majority_label = int(round(np.mean(source_votes))) if source_votes else int(row.label)

            record = {
                "tile_id": tile_id,
                "given_label": int(row.label),           # original consensus
                "majority_label": majority_label,        # 4-source majority (used for CL)
                "source_agreement": int(row.source_agreement),
                "n_sources": len(source_votes),
            }
            record.update(feats)
            all_records.append(record)
            all_geoms.append(row.geometry)
            all_tile_ids.append(tile_id)

    if not all_records:
        raise RuntimeError("No polygons extracted from any tile.")

    df = pd.DataFrame(all_records)
    if verbose:
        print(f"\nTotal polygons: {len(df)}")
        print(f"  label=1 (deforestation):    {(df.majority_label==1).sum()}")
        print(f"  label=0 (no deforestation): {(df.majority_label==0).sum()}")

    # ── Step 3: Build feature matrix ──────────────────────────────────────────
    # Columns to use as ML features (drop metadata & label columns)
    exclude = {"tile_id", "given_label", "majority_label", "source_agreement",
               "n_sources", "harmonic_physics_label"}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    # Impute NaN with column median (robust to outliers)
    for col in X.columns:
        median = X[col].median()
        X[col] = X[col].fillna(median if np.isfinite(median) else 0.0)
    X = X.values.astype(np.float32)

    y = df["majority_label"].values.astype(int)

    # ── Step 4: Out-of-fold probabilities via stratified k-fold ───────────────
    if verbose:
        print(f"\nFitting RandomForest with {n_cv_splits}-fold CV …")

    proba_oof = np.zeros((len(y), 2), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        clf = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        proba_oof[val_idx] = clf.predict_proba(X[val_idx])

    # ── Step 5: Confident Learning ────────────────────────────────────────────
    if verbose:
        print("Running Confident Learning …")

    mislabel_scores, flagged, suggested = confident_learning(proba_oof, y)

    # ── Step 6: Build output GeoDataFrame ─────────────────────────────────────
    # Collect CRS from first geometry's tile (they may differ across UTM zones)
    # → reproject everything to EPSG:4326 for a unified output
    first_td_crs = None
    for tile_id in all_tile_ids:
        try:
            s2_dir = DATA_ROOT / "sentinel-2" / "train" / f"{tile_id}__s2_l2a"
            f = next(iter(sorted(s2_dir.glob("*.tif"))))
            with rasterio.open(f) as src:
                first_td_crs = src.crs
            break
        except StopIteration:
            continue

    # Build per-tile CRS map for correct geometry handling
    tile_crs_map: dict = {}
    for tile_id in set(all_tile_ids):
        try:
            s2_dir = DATA_ROOT / "sentinel-2" / "train" / f"{tile_id}__s2_l2a"
            f = next(iter(sorted(s2_dir.glob("*.tif"))))
            with rasterio.open(f) as src:
                tile_crs_map[tile_id] = src.crs
        except StopIteration:
            pass

    # Create per-tile GeoDataFrames then reproject each to 4326 before concat
    records_by_tile: dict[str, list] = {}
    geoms_by_tile: dict[str, list] = {}
    for i, (tile_id, geom) in enumerate(zip(all_tile_ids, all_geoms)):
        records_by_tile.setdefault(tile_id, []).append(i)
        geoms_by_tile.setdefault(tile_id, []).append(geom)

    pieces = []
    for tile_id, idxs in records_by_tile.items():
        crs = tile_crs_map.get(tile_id, "EPSG:32618")
        sub_df = df.iloc[idxs].copy().reset_index(drop=True)
        sub_df["mislabel_score"]  = mislabel_scores[idxs]
        sub_df["flagged"]         = flagged[idxs]
        sub_df["suggested_label"] = suggested[idxs]
        sub_df["p_deforestation"] = proba_oof[idxs, 1]
        sub_gdf = gpd.GeoDataFrame(
            sub_df,
            geometry=geoms_by_tile[tile_id],
            crs=crs,
        ).to_crs("EPSG:4326")
        pieces.append(sub_gdf)

    result = pd.concat(pieces, ignore_index=True)

    # Summary
    if verbose:
        flagged_total = result["flagged"].sum()
        print(f"\n{'='*60}")
        print(f"Flagged as mislabelled: {flagged_total} / {len(result)} polygons")
        print(f"  Given label=1, suggested=0: {((result.flagged) & (result.given_label==1) & (result.suggested_label==0)).sum()}")
        print(f"  Given label=0, suggested=1: {((result.flagged) & (result.given_label==0) & (result.suggested_label==1)).sum()}")
        print(f"\nTop 15 most suspicious polygons:")
        show_cols = ["tile_id", "given_label", "suggested_label", "mislabel_score",
                     "p_deforestation", "harmonic_t_stat", "ndvi_change", "sar_change"]
        show_cols = [c for c in show_cols if c in result.columns]
        print(result.nlargest(15, "mislabel_score")[show_cols].to_string(index=True))

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Mislabel Detection v2: Confident Learning + Harmonic Breakpoint"
    )
    p.add_argument("--output", default="results/mislabels_v2.csv",
                   help="Output path (.csv or .geojson)")
    p.add_argument("--contamination", type=float, default=0.1,
                   help="Expected mislabel fraction (unused, kept for API compat)")
    p.add_argument("--cv-splits", type=int, default=5,
                   help="Number of stratified CV folds (default: 5)")
    p.add_argument("--tile", default=None,
                   help="Run on a single tile instead of all tiles")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    tiles = [args.tile] if args.tile else _get_train_tiles()

    print(f"\n{'='*60}")
    print(f"Mislabel Detection v2 — {len(tiles)} tile(s)")
    print(f"{'='*60}\n")

    result = run_pipeline(
        tiles=tiles,
        contamination=args.contamination,
        n_cv_splits=args.cv_splits,
        verbose=True,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".geojson":
        result.to_file(out, driver="GeoJSON")
    else:
        result.drop(columns="geometry").to_csv(out, index=False)

    print(f"\nSaved → {out}")
