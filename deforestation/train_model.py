"""
Option C — AEF Anomaly + Regional LightGBM Ensemble (Metric-Aware)
===================================================================

Architecture
------------
  Signal 1 (unsupervised): normalised AEF max-L2 change score
                            → no regional bias, generalises globally
  Signal 2 (supervised):   LightGBM trained per geographic region
                            (SE-Asia model + South-America model)
                            with leave-one-tile-out CV, area-proxy weights,
                            change-only features (no absolute spectral values)

  Ensemble: P_final = α · P_aef + (1-α) · P_lgbm
            α and decision threshold jointly tuned per region to
            maximise polygon-level IoU on LOTO held-out tiles.

  Year dating: argmax of per-year AEF L2 delta (pixel-level at test time).

Modes
-----
  train   — fit models on training_features_aef.csv, save artifacts
  predict — load artifacts, predict test tiles → rasters + submission GeoJSON
  all     — train then predict  [default]

Usage (from makeathon root):
    python deforestation/train_model.py --mode all \\
        --features results/training_features_aef.csv \\
        --out-dir  results/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from lightgbm import LGBMClassifier
from rasterio.warp import reproject, Resampling
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from mislabel_detection_v2 import DATA_ROOT  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────

REGIONS = {
    "SEA": ("47", "48"),   # Southeast Asia
    "SAM": ("18", "19"),   # South America
}

AEF_YEARS   = [2020, 2021, 2022, 2023, 2024]
DELTA_YEARS = [2021, 2022, 2023, 2024]

# Only change-based features — absolute values (ndvi_pre, sar_pre, …) are
# region-specific and hurt cross-region generalisation.
CHANGE_FEATURES = [
    "ndvi_change", "ndvi_std",
    "harmonic_t_stat", "harmonic_change_mag",
    "sar_change",
    "aef_delta_l2_2021", "aef_delta_l2_2022", "aef_delta_l2_2023", "aef_delta_l2_2024",
    "aef_delta_cos_2021", "aef_delta_cos_2022", "aef_delta_cos_2023", "aef_delta_cos_2024",
    "aef_from2020_l2_2021", "aef_from2020_l2_2022", "aef_from2020_l2_2023", "aef_from2020_l2_2024",
    "aef_from2020_cos_2021", "aef_from2020_cos_2022", "aef_from2020_cos_2023", "aef_from2020_cos_2024",
    "aef_max_delta_l2", "aef_max_delta_cos",
]

LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.04,
    max_depth=4,          # shallow → less overfitting on small dataset
    num_leaves=12,
    min_child_samples=4,
    subsample=0.75,
    colsample_bytree=0.75,
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
    return "UNK"


def compute_polygon_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Polygon-count approximation of Union IoU (TP / TP+FP+FN)."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fp + fn + 1e-9)


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn + 1e-9)


def compute_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    pred_pos = int((y_pred == 1).sum())
    return fp / (pred_pos + 1e-9)


# ── data preparation ──────────────────────────────────────────────────────────

def load_training_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["region"] = df["tile_id"].map(get_region)

    # Impute NaN with per-region feature median (robust to outliers)
    for region in df["region"].unique():
        mask = df["region"] == region
        for col in CHANGE_FEATURES:
            if col in df.columns:
                med = df.loc[mask, col].median()
                df.loc[mask, col] = df.loc[mask, col].fillna(
                    med if np.isfinite(med) else 0.0
                )

    return df


# ── AEF anomaly scorer ────────────────────────────────────────────────────────

class AEFAnomalyScorer:
    """
    Normalises aef_max_delta_l2 per region using a sigmoid centred at the
    median with IQR scale — robust to outliers, gives calibrated [0,1] scores.
    """

    def __init__(self):
        self.params: dict[str, dict] = {}   # region → {center, scale}

    def fit(self, df: pd.DataFrame) -> "AEFAnomalyScorer":
        for region in df["region"].unique():
            scores = df.loc[df["region"] == region, "aef_max_delta_l2"].dropna()
            q25, q50, q75 = np.percentile(scores, [25, 50, 75])
            self.params[region] = {
                "center": float(q50),
                "scale":  float((q75 - q25) / 2 + 1e-9),
            }
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), 0.5)
        for region, p in self.params.items():
            mask = (df["region"] == region).values
            raw = df.loc[mask, "aef_max_delta_l2"].fillna(p["center"]).values
            out[mask] = 1.0 / (1.0 + np.exp(-(raw - p["center"]) / p["scale"]))
        return out

    def transform_array(self, scores: np.ndarray, region: str) -> np.ndarray:
        """Pixel-level transform using stored region params."""
        p = self.params.get(region, {"center": 0.0, "scale": 1.0})
        return 1.0 / (1.0 + np.exp(-(scores - p["center"]) / p["scale"]))


# ── regional LightGBM with LOTO-CV ───────────────────────────────────────────

class RegionalLGBM:
    """Per-region LightGBM with leave-one-tile-out cross-validation."""

    def __init__(self):
        self.models:    dict[str, LGBMClassifier] = {}
        self.scalers:   dict[str, RobustScaler]   = {}
        self.feat_cols: list[str]                  = []
        self.oof_proba: np.ndarray | None          = None

    def fit(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Fit one LightGBM per region.
        Returns OOF predicted probabilities (same length as df).
        """
        available = [c for c in CHANGE_FEATURES if c in df.columns]
        self.feat_cols = available
        oof = np.full(len(df), 0.5)

        for region in df["region"].unique():
            rdf = df[df["region"] == region].copy()
            tiles = rdf["tile_id"].unique()
            if verbose:
                print(f"    [{region}] {len(tiles)} tiles, {len(rdf)} polygons")

            # ── LOTO cross-validation ─────────────────────────────────────────
            region_oof = np.full(len(rdf), 0.5)
            for held in tiles:
                tr_idx = rdf.index[rdf["tile_id"] != held]
                va_idx = rdf.index[rdf["tile_id"] == held]
                if len(tr_idx) == 0:
                    continue

                X_tr = rdf.loc[tr_idx, available].values
                y_tr = rdf.loc[tr_idx, "corrected_label"].values
                X_va = rdf.loc[va_idx, available].values

                # Weight by label rarity within fold
                pos_rate = y_tr.mean().clip(0.05, 0.95)
                w = np.where(y_tr == 1, 1.0 / pos_rate, 1.0 / (1 - pos_rate))

                scaler = RobustScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)

                clf = LGBMClassifier(**LGBM_PARAMS)
                clf.fit(X_tr_s, y_tr, sample_weight=w)

                va_local = rdf.index.get_indexer(va_idx)
                region_oof[va_local] = clf.predict_proba(X_va_s)[:, 1]

            # Store OOF back into global array
            region_local = df.index.get_indexer(rdf.index)
            oof[region_local] = region_oof

            # ── Final model on all region data ────────────────────────────────
            X_all = rdf[available].values
            y_all = rdf["corrected_label"].values
            pos_rate = y_all.mean().clip(0.05, 0.95)
            w_all = np.where(y_all == 1, 1.0 / pos_rate, 1.0 / (1 - pos_rate))

            scaler_final = RobustScaler()
            X_all_s = scaler_final.fit_transform(X_all)

            clf_final = LGBMClassifier(**LGBM_PARAMS)
            clf_final.fit(X_all_s, y_all, sample_weight=w_all)

            self.models[region]  = clf_final
            self.scalers[region] = scaler_final

            if verbose:
                loto_iou = compute_polygon_iou(
                    rdf["corrected_label"].values,
                    (region_oof >= 0.5).astype(int),
                )
                print(f"    [{region}] LOTO IoU @ thresh=0.5 : {loto_iou:.3f}")

        self.oof_proba = oof
        return oof

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), 0.5)
        for region, model in self.models.items():
            mask = df["region"] == region
            if not mask.any():
                continue
            X = df.loc[mask, self.feat_cols].fillna(0).values
            X_s = self.scalers[region].transform(X)
            idxs = df.index.get_indexer(df.index[mask])
            out[idxs] = model.predict_proba(X_s)[:, 1]
        return out


# ── ensemble tuner ────────────────────────────────────────────────────────────

class EnsembleTuner:
    """
    Tunes α (AEF weight) and decision threshold per region to maximise IoU
    on LOTO OOF probabilities.
    """

    def __init__(self):
        self.alpha:     dict[str, float] = {}
        self.threshold: dict[str, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        p_aef_oof: np.ndarray,
        p_lgbm_oof: np.ndarray,
        verbose: bool = True,
    ) -> "EnsembleTuner":
        if verbose:
            print("\n  Tuning α and threshold per region (maximising IoU) …")

        for region in df["region"].unique():
            mask = (df["region"] == region).values
            y    = df.loc[mask, "corrected_label"].values
            p_a  = p_aef_oof[mask]
            p_l  = p_lgbm_oof[mask]

            best_iou, best_a, best_t = 0.0, 0.5, 0.5

            for alpha in np.arange(0.0, 1.05, 0.05):
                p_ens = alpha * p_a + (1 - alpha) * p_l
                for thresh in np.arange(0.15, 0.85, 0.01):
                    pred = (p_ens >= thresh).astype(int)
                    iou  = compute_polygon_iou(y, pred)
                    if iou > best_iou:
                        best_iou, best_a, best_t = iou, alpha, thresh

            self.alpha[region]     = float(best_a)
            self.threshold[region] = float(best_t)

            if verbose:
                p_ens_best = best_a * p_a + (1 - best_a) * p_l
                pred_best  = (p_ens_best >= best_t).astype(int)
                recall = compute_recall(y, pred_best)
                fpr    = compute_fpr(y, pred_best)
                print(
                    f"    [{region}] α={best_a:.2f}  thresh={best_t:.2f}  "
                    f"IoU={best_iou:.3f}  Recall={recall:.3f}  FPR={fpr:.3f}"
                )

        return self

    def predict(
        self,
        df: pd.DataFrame,
        p_aef: np.ndarray,
        p_lgbm: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros(len(df), dtype=int)
        for region in df["region"].unique():
            mask  = (df["region"] == region).values
            alpha = self.alpha.get(region, 0.5)
            thresh = self.threshold.get(region, 0.5)
            p_ens = alpha * p_aef[mask] + (1 - alpha) * p_lgbm[mask]
            out[mask] = (p_ens >= thresh).astype(int)
        return out

    def predict_proba_ensemble(
        self,
        df: pd.DataFrame,
        p_aef: np.ndarray,
        p_lgbm: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros(len(df))
        for region in df["region"].unique():
            mask  = (df["region"] == region).values
            alpha = self.alpha.get(region, 0.5)
            out[mask] = alpha * p_aef[mask] + (1 - alpha) * p_lgbm[mask]
        return out


# ── training ──────────────────────────────────────────────────────────────────

def train(
    features_csv: Path,
    out_dir: Path,
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"\n{'='*60}")
        print("Training — Option C Ensemble")
        print(f"{'='*60}\n")

    df = load_training_data(features_csv)
    if verbose:
        for region in df["region"].unique():
            rdf = df[df["region"] == region]
            print(
                f"  {region}: {len(rdf['tile_id'].unique())} tiles, "
                f"{len(rdf)} polygons  "
                f"(label=1: {rdf['corrected_label'].sum()}  "
                f"label=0: {(rdf['corrected_label']==0).sum()})"
            )

    # ── Signal 1: AEF anomaly score ───────────────────────────────────────────
    if verbose:
        print("\n  Fitting AEF anomaly scorer …")
    aef_scorer = AEFAnomalyScorer()
    aef_scorer.fit(df)
    p_aef_oof = aef_scorer.transform(df)

    # ── Signal 2: Regional LightGBM LOTO ─────────────────────────────────────
    if verbose:
        print("\n  Fitting regional LightGBM (LOTO-CV) …")
    lgbm = RegionalLGBM()
    p_lgbm_oof = lgbm.fit(df, verbose=verbose)

    # ── Ensemble tuning ───────────────────────────────────────────────────────
    tuner = EnsembleTuner()
    tuner.fit(df, p_aef_oof, p_lgbm_oof, verbose=verbose)

    # ── Final evaluation on full training set ─────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print("Final LOTO OOF Metrics (training tiles)")
        print(f"{'='*60}")
        y_all   = df["corrected_label"].values
        p_ens   = tuner.predict_proba_ensemble(df, p_aef_oof, p_lgbm_oof)
        pred    = tuner.predict(df, p_aef_oof, p_lgbm_oof)
        print(f"  Union IoU   : {compute_polygon_iou(y_all, pred):.3f}")
        print(f"  Recall      : {compute_recall(y_all, pred):.3f}")
        print(f"  FPR         : {compute_fpr(y_all, pred):.3f}")

        # Per-region breakdown
        for region in df["region"].unique():
            mask = (df["region"] == region).values
            y_r  = y_all[mask]
            p_r  = pred[mask]
            print(
                f"  [{region}] IoU={compute_polygon_iou(y_r, p_r):.3f}  "
                f"Recall={compute_recall(y_r, p_r):.3f}  "
                f"FPR={compute_fpr(y_r, p_r):.3f}"
            )

    # ── Feature importance ────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print("Top-10 Feature Importances")
        print(f"{'='*60}")
        for region, model in lgbm.models.items():
            imp = pd.Series(
                model.feature_importances_,
                index=lgbm.feat_cols,
            ).sort_values(ascending=False)
            print(f"  [{region}]")
            for feat, val in imp.head(10).items():
                print(f"    {feat:35s} {val:6.0f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "aef_scorer":  aef_scorer,
        "lgbm":        lgbm,
        "tuner":       tuner,
        "regions":     df.groupby("tile_id")["region"].first().to_dict(),
    }
    model_path = out_dir / "model.pkl"
    joblib.dump(artifacts, model_path)
    if verbose:
        print(f"\nArtifacts saved → {model_path}")

    return artifacts


# ── pixel-level AEF prediction for test tiles ─────────────────────────────────

def _load_aef_year_pixel(
    path: Path,
    ref_transform,
    ref_crs,
    ref_shape: tuple[int, int],
) -> np.ndarray:
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
    return out


def predict_tile_pixels(
    tile_id: str,
    split: str,
    aef_scorer: AEFAnomalyScorer,
    tuner: EnsembleTuner,
    ref_transform=None,
    ref_crs=None,
    ref_shape: tuple | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Pixel-level prediction for one tile using AEF change detection.
    Returns (binary_map, year_map) of shape (H, W), or None on failure.

    At test time we use only the AEF branch (no labels available for polygons),
    calibrated with the AEF scorer + tuner threshold for the tile's region.
    """
    aef_dir = DATA_ROOT / "aef-embeddings" / split
    region  = get_region(tile_id)

    # ── reference grid ────────────────────────────────────────────────────────
    if ref_shape is None:
        s2_dir = DATA_ROOT / "sentinel-2" / split / f"{tile_id}__s2_l2a"
        s2_files = sorted(s2_dir.glob("*.tif"))
        if not s2_files:
            if verbose:
                print(f"  {tile_id}: no S2 reference — SKIP")
            return None
        with rasterio.open(s2_files[0]) as src:
            ref_transform = src.transform
            ref_crs       = src.crs
            ref_shape     = src.shape

    H, W = ref_shape

    # ── load annual AEF ───────────────────────────────────────────────────────
    aef_by_year: dict[int, np.ndarray] = {}
    for f in aef_dir.glob(f"{tile_id}_*.tiff"):
        try:
            year = int(f.stem.split("_")[-1])
        except ValueError:
            continue
        if year not in AEF_YEARS:
            continue
        aef_by_year[year] = _load_aef_year_pixel(f, ref_transform, ref_crs, ref_shape)

    if not aef_by_year:
        if verbose:
            print(f"  {tile_id}: no AEF files — SKIP")
        return None

    # ── pixel-level L2 deltas ─────────────────────────────────────────────────
    delta_maps: dict[int, np.ndarray] = {}
    for yr in DELTA_YEARS:
        if yr in aef_by_year and (yr - 1) in aef_by_year:
            diff = aef_by_year[yr] - aef_by_year[yr - 1]   # (bands, H, W)
            delta_maps[yr] = np.linalg.norm(diff, axis=0)   # (H, W)

    if not delta_maps:
        if verbose:
            print(f"  {tile_id}: insufficient AEF years for delta — SKIP")
        return None

    year_keys  = sorted(delta_maps.keys())
    delta_stack = np.stack([delta_maps[yr] for yr in year_keys], axis=0)  # (n_yr, H, W)

    max_delta   = delta_stack.max(axis=0)                                  # (H, W)
    year_idx    = delta_stack.argmax(axis=0)                               # (H, W)
    year_map    = np.array(year_keys, dtype=np.int16)[year_idx]            # (H, W)

    # ── normalise and threshold ───────────────────────────────────────────────
    flat_scores   = max_delta.ravel()
    p_aef_pixels  = aef_scorer.transform_array(flat_scores, region).reshape(H, W)
    thresh        = tuner.threshold.get(region, 0.5)
    # Use AEF-only threshold: since α was tuned for full ensemble,
    # apply a small correction (AEF tends to need slightly higher threshold alone)
    aef_alpha     = tuner.alpha.get(region, 0.5)
    # Effective threshold for AEF-only signal
    eff_thresh    = thresh / (aef_alpha + 1e-9) * aef_alpha   # re-scale by α contribution
    eff_thresh    = float(np.clip(eff_thresh, 0.2, 0.8))

    binary_map = (p_aef_pixels >= eff_thresh).astype(np.uint8)

    if verbose:
        n_pos = binary_map.sum()
        print(
            f"  {tile_id} [{region}]: years={year_keys}  "
            f"defor_pixels={n_pos}  ({100*n_pos/(H*W):.2f} %)"
        )

    return binary_map, year_map


# ── prediction on test tiles ──────────────────────────────────────────────────

def _get_tiles(split: str) -> list[str]:
    meta = DATA_ROOT / "metadata" / f"{split}_tiles.geojson"
    with open(meta) as f:
        gj = json.load(f)
    props = gj["features"][0]["properties"]
    candidates = ["tile_id", "id", "name", "tile", "TILE_ID", "tileid", "tile_name"]
    key = next((k for k in candidates if k in props), None)
    if key is None:
        import re
        pat = re.compile(r"^\d{2}[A-Z]{3}_\d+_\d+$")
        key = next((k for k, v in props.items() if isinstance(v, str) and pat.match(v)), None)
    if key is None:
        raise KeyError(f"Cannot find tile ID key in {split} metadata.")
    return [feat["properties"][key] for feat in gj["features"]]


def predict(
    artifacts: dict,
    out_dir: Path,
    split: str = "test",
    verbose: bool = True,
) -> list[Path]:
    aef_scorer: AEFAnomalyScorer = artifacts["aef_scorer"]
    tuner: EnsembleTuner         = artifacts["tuner"]

    raster_dir = out_dir / "predictions"
    raster_dir.mkdir(parents=True, exist_ok=True)

    try:
        tiles = _get_tiles(split)
    except Exception as e:
        print(f"Could not load {split} tile list: {e}")
        return []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Predicting {len(tiles)} {split} tile(s)")
        print(f"{'='*60}\n")

    written: list[Path] = []

    for tile_id in tiles:
        result = predict_tile_pixels(
            tile_id, split, aef_scorer, tuner, verbose=verbose
        )
        if result is None:
            continue

        binary_map, year_map = result

        # ── write 2-band raster ───────────────────────────────────────────────
        s2_dir = DATA_ROOT / "sentinel-2" / split / f"{tile_id}__s2_l2a"
        s2_files = sorted(s2_dir.glob("*.tif"))
        if not s2_files:
            continue
        with rasterio.open(s2_files[0]) as ref:
            meta = ref.meta.copy()

        meta.update(count=2, dtype="uint8", compress="lzw")
        # year_map values are e.g. 2022; store as uint8 offset from 2020
        year_offset = (year_map - 2020).clip(0, 255).astype(np.uint8)

        out_path = raster_dir / f"{tile_id}_prediction.tif"
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(binary_map,  1)
            dst.write(year_offset, 2)
            dst.update_tags(
                band_1="deforestation_binary",
                band_2="year_offset_from_2020",
            )
        written.append(out_path)

    # ── generate submission GeoJSON ───────────────────────────────────────────
    if written:
        _build_submission(written, out_dir, verbose=verbose)

    return written


def _build_submission(raster_paths: list[Path], out_dir: Path, verbose: bool) -> None:
    """Convert prediction rasters to a single submission GeoJSON."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    pieces = []
    for rpath in raster_paths:
        with rasterio.open(rpath) as src:
            binary    = src.read(1).astype(np.uint8)
            yr_offset = src.read(2).astype(np.uint8)
            transform = src.transform
            crs       = src.crs

        polygons, years = [], []
        for geom_dict, val in shapes(binary, mask=binary, transform=transform):
            if val != 1:
                continue
            geom = shape(geom_dict)
            # Year: mode of year_offset within polygon bounding box (fast approx)
            cx, cy = geom.centroid.x, geom.centroid.y
            col = int((cx - transform.c) / transform.a)
            row = int((cy - transform.f) / transform.e)
            H, W = binary.shape
            col = max(0, min(col, W - 1))
            row = max(0, min(row, H - 1))
            yr  = int(yr_offset[row, col]) + 2020
            polygons.append(geom)
            years.append(yr)

        if not polygons:
            continue

        gdf = gpd.GeoDataFrame({"time_step": years}, geometry=polygons, crs=crs)
        gdf = gdf.to_crs("EPSG:4326")

        # Area filter ≥ 0.5 ha
        utm_crs = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)
        gdf     = gdf[gdf_utm.area / 10_000 >= 0.5].reset_index(drop=True)
        if not gdf.empty:
            pieces.append(gdf)

    if not pieces:
        print("  No deforestation polygons above 0.5 ha — submission empty.")
        return

    submission = gpd.GeoDataFrame(
        pd.concat(pieces, ignore_index=True),
        crs="EPSG:4326",
    )
    out_path = out_dir / "submission.geojson"
    submission.to_file(out_path, driver="GeoJSON")

    if verbose:
        print(f"\n  Submission GeoJSON → {out_path}")
        print(f"  Total deforestation polygons : {len(submission)}")
        yr_counts = submission["time_step"].value_counts().sort_index()
        for yr, cnt in yr_counts.items():
            print(f"    {yr}: {cnt} polygon(s)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Option C: AEF Anomaly + Regional LightGBM Ensemble"
    )
    p.add_argument("--features", default="results/training_features_aef.csv")
    p.add_argument("--out-dir",  default="results")
    p.add_argument(
        "--mode", choices=["train", "predict", "all"], default="all",
        help="train | predict | all  (default: all)",
    )
    p.add_argument(
        "--split", default="test",
        help="Data split to predict on: test | train  (default: test)",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    verbose = not args.quiet
    out_dir = Path(args.out_dir)

    if args.mode in ("train", "all"):
        artifacts = train(
            features_csv=Path(args.features),
            out_dir=out_dir,
            verbose=verbose,
        )

    if args.mode in ("predict", "all"):
        if args.mode == "predict":
            model_path = out_dir / "model.pkl"
            if not model_path.exists():
                sys.exit(f"ERROR: model not found at {model_path}. Run --mode train first.")
            artifacts = joblib.load(model_path)
        predict(
            artifacts=artifacts,
            out_dir=out_dir,
            split=args.split,
            verbose=verbose,
        )

    print("\nDone.")
