"""
Mislabel detection for deforestation annotation polygons.

Works in two modes:

  TRAIN MODE (default): uses weak labels (RADD / GLAD-L / GLAD-S2) to form
    consensus polygons, then checks each polygon against satellite signals.

  TEST MODE (--predictions): takes a submitted prediction GeoJSON (no weak labels
    available), uses only Sentinel-1, Sentinel-2, and AEF embeddings to flag
    polygons whose spectral/SAR signatures contradict their predicted label.

Signals (all modes):
  1. Cross-source disagreement OR S1-vs-S2 cross-modal consistency (test mode)
  2. NDVI change consistency     – true deforestation must show a measurable NDVI drop
  3. SAR backscatter change      – deforestation causes a VV dB decrease
  4. AEF embedding anomaly       – Isolation Forest on per-polygon mean embeddings

Bonus: suggested correct label via signal majority vote + KNN in embedding space.

Usage (from the makeathon root, after data is on disk):
    # Training tile — uses weak labels for Signal 1:
    python deforestation/mislabel_detection.py --tile 18NWG_6_6 --output results/mislabels.geojson

    # Test tile — uses prediction GeoJSON + S1/S2/AEF only:
    python deforestation/mislabel_detection.py --tile 18NWH_3_4 --split test \\
        --predictions submission/pred_18NWH_3_4.geojson --output results/mislabels_test.geojson
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

DATA_ROOT = Path("data/makeathon-challenge")

# ── helpers ───────────────────────────────────────────────────────────────────

def _read_band(path: Path, band: int = 1) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        meta = {"transform": src.transform, "crs": src.crs, "shape": src.shape}
    return data, meta


def _reproject_to(
    src_path: Path, dst_transform, dst_crs, dst_shape: tuple[int, int], band: int = 1
) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_data = src.read(band).astype(np.float32)
        out = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=src_data,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    return out


def _ndvi(s2_path: Path) -> np.ndarray:
    """Compute NDVI from a Sentinel-2 tile (NIR=band8, Red=band4)."""
    with rasterio.open(s2_path) as src:
        nir = src.read(8).astype(np.float32)
        red = src.read(4).astype(np.float32)
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, np.nan)
    return ndvi


def _s1_db(s1_path: Path) -> np.ndarray:
    data, _ = _read_band(s1_path)
    return np.where(data > 0, 10.0 * np.log10(data), np.nan)


# ── data loading ──────────────────────────────────────────────────────────────

@dataclass
class TileData:
    tile_id: str
    ref_transform: object = None
    ref_crs: object = None
    ref_shape: tuple = None

    # NDVI time-series: shape (T, H, W)
    ndvi_ts: np.ndarray = field(default=None, repr=False)
    ndvi_dates: list[tuple[int, int]] = field(default_factory=list)

    # SAR time-series: shape (T, H, W)
    sar_ts: np.ndarray = field(default=None, repr=False)
    sar_dates: list[tuple[int, int]] = field(default_factory=list)

    # AEF embeddings: shape (64, H, W) reprojected to ref grid
    aef: np.ndarray = field(default=None, repr=False)

    # Weak label rasters (on ref grid), all uint8 binary (1=deforestation)
    radd_binary: np.ndarray = field(default=None, repr=False)
    gladl_binary: np.ndarray = field(default=None, repr=False)
    glads2_binary: np.ndarray = field(default=None, repr=False)

    # Vectorized polygons GeoDataFrame
    polygons: gpd.GeoDataFrame = field(default=None, repr=False)


def load_tile(tile_id: str, split: str = "train", years: list[int] | None = None) -> TileData:
    """Load all satellite data for a tile.

    Args:
        tile_id: Tile identifier, e.g. ``18NWG_6_6``.
        split:   ``"train"`` (loads weak labels too) or ``"test"`` (S1/S2/AEF only).
        years:   Years to load (default 2020-2024).
    """
    if years is None:
        years = list(range(2020, 2025))

    td = TileData(tile_id=tile_id)

    # ── reference grid from first available S2 ────────────────────────────────
    s2_dir = DATA_ROOT / "sentinel-2" / split / f"{tile_id}__s2_l2a"
    s2_files = sorted(s2_dir.glob("*.tif"))
    if not s2_files:
        raise FileNotFoundError(f"No Sentinel-2 files found for tile {tile_id} (split={split})")

    with rasterio.open(s2_files[0]) as src:
        td.ref_transform = src.transform
        td.ref_crs = src.crs
        td.ref_shape = src.shape

    # ── NDVI time-series ──────────────────────────────────────────────────────
    ndvi_stack, ndvi_dates = [], []
    for f in s2_files:
        stem_parts = f.stem.split("_")
        try:
            yr, mo = int(stem_parts[-2]), int(stem_parts[-1])
        except (ValueError, IndexError):
            continue
        if yr not in years:
            continue
        ndvi_stack.append(_ndvi(f))
        ndvi_dates.append((yr, mo))

    if ndvi_stack:
        td.ndvi_ts = np.stack(ndvi_stack, axis=0)
        td.ndvi_dates = ndvi_dates

    # ── SAR time-series (ascending only) ─────────────────────────────────────
    s1_dir = DATA_ROOT / "sentinel-1" / split / f"{tile_id}__s1_rtc"
    sar_stack, sar_dates = [], []
    for f in sorted(s1_dir.glob("*_ascending.tif")):
        stem_parts = f.stem.split("_")
        try:
            yr, mo = int(stem_parts[-3]), int(stem_parts[-2])
        except (ValueError, IndexError):
            continue
        if yr not in years:
            continue
        raw = _s1_db(f)
        # reproject S1 onto S2 grid
        reproj = np.zeros(td.ref_shape, dtype=np.float32)
        with rasterio.open(f) as src:
            reproject(
                source=src.read(1).astype(np.float32),
                destination=reproj,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=td.ref_transform,
                dst_crs=td.ref_crs,
                resampling=Resampling.bilinear,
            )
        sar_stack.append(np.where(reproj > 0, 10.0 * np.log10(reproj), np.nan))
        sar_dates.append((yr, mo))

    if sar_stack:
        td.sar_ts = np.stack(sar_stack, axis=0)
        td.sar_dates = sar_dates

    # ── AEF embeddings (use most recent available year) ───────────────────────
    aef_dir = DATA_ROOT / "aef-embeddings" / split
    aef_files = sorted(aef_dir.glob(f"{tile_id}_*.tiff"), reverse=True)
    if aef_files:
        aef_path = aef_files[0]
        with rasterio.open(aef_path) as src:
            n_bands = src.count
            aef_reproj = np.zeros((n_bands, *td.ref_shape), dtype=np.float32)
            for b in range(1, n_bands + 1):
                reproject(
                    source=src.read(b).astype(np.float32),
                    destination=aef_reproj[b - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=td.ref_transform,
                    dst_crs=td.ref_crs,
                    resampling=Resampling.bilinear,
                )
        td.aef = aef_reproj

    # ── Weak labels → binary rasters (train split only) ─────────────────────
    if split != "train":
        return td

    labels_root = DATA_ROOT / "labels" / "train"

    def _reproject_label(src_path: Path, band: int = 1) -> np.ndarray:
        """Read a label band and reproject it onto the reference (S2) grid."""
        out = np.zeros(td.ref_shape, dtype=np.float32)
        with rasterio.open(src_path) as src:
            reproject(
                source=src.read(band).astype(np.float32),
                destination=out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=td.ref_transform,
                dst_crs=td.ref_crs,
                resampling=Resampling.nearest,
            )
        return out

    # RADD: leading digit 2/3 = alert; post-2020 = days >= 1827
    radd_path = labels_root / "radd" / f"radd_{tile_id}_labels.tif"
    if radd_path.exists():
        raw = _reproject_label(radd_path)
        conf = (raw // 10000).astype(np.int32)
        days = (raw % 10000).astype(np.int32)
        td.radd_binary = ((conf >= 2) & (days >= 1827)).astype(np.uint8)

    # GLAD-S2: confidence 2-4 = alert; alertDate > 365 = post-2020
    gs2_alert_path = labels_root / "glads2" / f"glads2_{tile_id}_alert.tif"
    gs2_date_path  = labels_root / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    if gs2_alert_path.exists():
        raw_alert = _reproject_label(gs2_alert_path)
        if gs2_date_path.exists():
            raw_date = _reproject_label(gs2_date_path)
            td.glads2_binary = ((raw_alert >= 2) & (raw_date > 365)).astype(np.uint8)
        else:
            td.glads2_binary = (raw_alert >= 2).astype(np.uint8)

    # GLAD-L: years 21-24 (post-2020); confidence 2-3
    gladl_dir = labels_root / "gladl"
    gladl_layers = []
    for yy in range(21, 25):
        for candidate in [
            gladl_dir / f"gladl_{tile_id}_alert{yy}.tif",
            gladl_dir / tile_id / f"gladl_{tile_id}_alert{yy}.tif",
        ]:
            if candidate.exists():
                raw = _reproject_label(candidate)
                gladl_layers.append((raw >= 2).astype(np.uint8))
                break

    if gladl_layers:
        td.gladl_binary = np.any(np.stack(gladl_layers, axis=0), axis=0).astype(np.uint8)

    return td


# ── polygon feature extraction ────────────────────────────────────────────────

def load_predictions_as_polygons(pred_geojson: str | Path, td: TileData) -> gpd.GeoDataFrame:
    """
    Load a submitted prediction GeoJSON and reproject it to the tile's reference CRS.
    Used in test mode where no weak labels are available.

    The GeoJSON must contain polygon features; any ``label`` property is used when
    present (1 = deforestation).  If absent, all polygons are assumed to be
    deforestation predictions (the typical submission format).

    Args:
        pred_geojson: Path to the prediction ``.geojson`` file.
        td:           TileData for the tile (already loaded).
    """
    gdf = gpd.read_file(pred_geojson)
    gdf = gdf.to_crs(td.ref_crs)

    if "label" not in gdf.columns:
        gdf["label"] = 1          # submitted polygons are all deforestation
    else:
        gdf["label"] = gdf["label"].fillna(1).astype(int)

    gdf["source_agreement"] = -1  # sentinel: no weak labels available

    # UTM area filter: keep >= 0.5 ha
    gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
    gdf = gdf[(gdf_utm.area / 10_000) >= 0.5].reset_index(drop=True)
    return gdf[["label", "source_agreement", "geometry"]]


def compute_cross_modal_agreement(polygon, td: TileData) -> float:
    """
    Test-mode substitute for weak-label agreement.

    Computes whether Sentinel-2 (NDVI change) and Sentinel-1 (SAR change) both
    independently agree on whether deforestation occurred.

    Returns a disagreement score 0-1 (1 = strong cross-modal contradiction).
    """
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping

    mask = ~geometry_mask(
        [mapping(polygon)],
        transform=td.ref_transform,
        invert=False,
        out_shape=td.ref_shape,
    )
    if mask.sum() == 0:
        return 0.0

    signals = []

    if td.ndvi_ts is not None and len(td.ndvi_dates) >= 2:
        pre_idx  = [i for i, (y, _) in enumerate(td.ndvi_dates) if y <= 2021]
        post_idx = [i for i, (y, _) in enumerate(td.ndvi_dates) if y >= 2022]

        def _mean(idx_list, stack):
            vals = []
            for i in idx_list:
                px = stack[i][mask]
                px = px[np.isfinite(px)]
                if px.size:
                    vals.append(float(px.mean()))
            return float(np.mean(vals)) if vals else np.nan

        ndvi_change = _mean(pre_idx, td.ndvi_ts) - _mean(post_idx, td.ndvi_ts)
        if np.isfinite(ndvi_change):
            # S2 says deforestation if NDVI dropped > 0.1
            signals.append(1 if ndvi_change > 0.10 else 0)

    if td.sar_ts is not None and len(td.sar_dates) >= 2:
        pre_idx  = [i for i, (y, _) in enumerate(td.sar_dates) if y <= 2021]
        post_idx = [i for i, (y, _) in enumerate(td.sar_dates) if y >= 2022]

        def _sar_mean(idx_list):
            vals = []
            for i in idx_list:
                px = td.sar_ts[i][mask]
                px = px[np.isfinite(px)]
                if px.size:
                    vals.append(float(px.mean()))
            return float(np.mean(vals)) if vals else np.nan

        sar_change = _sar_mean(pre_idx) - _sar_mean(post_idx)
        if np.isfinite(sar_change):
            # S1 says deforestation if VV dB dropped > 0.5
            signals.append(1 if sar_change > 0.5 else 0)

    if len(signals) < 2:
        return 0.0  # not enough data to compute disagreement

    # disagreement: S2 and S1 give opposite answers
    return float(signals[0] != signals[1])


def vectorize_consensus_labels(td: TileData) -> gpd.GeoDataFrame:
    """
    Build a set of labelled polygons by majority vote across the three weak-label
    sources. Each polygon gets a `label` (1=deforestation, 0=no deforestation) and
    a `source_agreement` score (0-3 sources agreeing).
    """
    available = []
    if td.radd_binary is not None:
        available.append(td.radd_binary)
    if td.glads2_binary is not None:
        available.append(td.glads2_binary)
    if td.gladl_binary is not None:
        available.append(td.gladl_binary)

    if not available:
        raise RuntimeError("No weak labels available for this tile.")

    stack = np.stack(available, axis=0)  # (N_sources, H, W)
    vote_sum = stack.sum(axis=0)  # 0..N_sources
    n = len(available)
    consensus = (vote_sum > n / 2).astype(np.uint8)

    # Vectorise consensus raster
    polygons, labels, agreements = [], [], []
    for geom, val in shapes(consensus, mask=None, transform=td.ref_transform):
        lbl = int(val)
        geom_shape = shape(geom)
        # count how many sources agree with consensus label
        # (for deforestation polys: how many say 1; for non-def: how many say 0)
        agreement = int(vote_sum[
            max(0, int((geom_shape.centroid.y - td.ref_transform.f) // td.ref_transform.e)),
            max(0, int((geom_shape.centroid.x - td.ref_transform.c) // td.ref_transform.a)),
        ]) if lbl == 1 else n - int(vote_sum[
            max(0, int((geom_shape.centroid.y - td.ref_transform.f) // td.ref_transform.e)),
            max(0, int((geom_shape.centroid.x - td.ref_transform.c) // td.ref_transform.a)),
        ])
        polygons.append(geom_shape)
        labels.append(lbl)
        agreements.append(min(agreement, n))

    gdf = gpd.GeoDataFrame(
        {"label": labels, "source_agreement": agreements},
        geometry=polygons,
        crs=td.ref_crs,
    )
    # UTM area filter: keep polygons >= 0.5 ha
    gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
    gdf = gdf[(gdf_utm.area / 10_000) >= 0.5].reset_index(drop=True)
    return gdf


def extract_polygon_features(polygon, td: TileData) -> dict:
    """
    Extract per-polygon features from S2 NDVI, S1 SAR, and AEF embeddings.
    All values are means over the polygon's pixels in the reference grid.
    """
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping

    mask = ~geometry_mask(
        [mapping(polygon)],
        transform=td.ref_transform,
        invert=False,
        out_shape=td.ref_shape,
    )
    if mask.sum() == 0:
        return None

    feats: dict = {}

    # NDVI: split into pre-2022 baseline and post-2022 (post-deforestation window)
    if td.ndvi_ts is not None and len(td.ndvi_dates) > 0:
        pre_idx  = [i for i, (y, _) in enumerate(td.ndvi_dates) if y <= 2021]
        post_idx = [i for i, (y, _) in enumerate(td.ndvi_dates) if y >= 2022]

        def _masked_mean(stack_idx):
            vals = []
            for i in stack_idx:
                px = td.ndvi_ts[i][mask]
                px = px[np.isfinite(px)]
                if px.size:
                    vals.append(float(px.mean()))
            return float(np.mean(vals)) if vals else np.nan

        feats["ndvi_pre"]    = _masked_mean(pre_idx)
        feats["ndvi_post"]   = _masked_mean(post_idx)
        feats["ndvi_change"] = feats["ndvi_pre"] - feats["ndvi_post"]   # positive = drop = deforestation signal
        # temporal variance (high variance can indicate noise or cloud artefacts)
        all_means = [_masked_mean([i]) for i in range(len(td.ndvi_dates))]
        valid_means = [v for v in all_means if np.isfinite(v)]
        feats["ndvi_std"] = float(np.std(valid_means)) if valid_means else np.nan

    # SAR: same split
    if td.sar_ts is not None and len(td.sar_dates) > 0:
        pre_idx  = [i for i, (y, _) in enumerate(td.sar_dates) if y <= 2021]
        post_idx = [i for i, (y, _) in enumerate(td.sar_dates) if y >= 2022]

        def _sar_mean(stack_idx):
            vals = []
            for i in stack_idx:
                px = td.sar_ts[i][mask]
                px = px[np.isfinite(px)]
                if px.size:
                    vals.append(float(px.mean()))
            return float(np.mean(vals)) if vals else np.nan

        feats["sar_pre"]    = _sar_mean(pre_idx)
        feats["sar_post"]   = _sar_mean(post_idx)
        feats["sar_change"] = feats["sar_pre"] - feats["sar_post"]  # positive = dB drop = deforestation signal

    # AEF embeddings: mean over polygon pixels
    if td.aef is not None:
        emb_means = []
        for b in range(td.aef.shape[0]):
            px = td.aef[b][mask]
            px = px[np.isfinite(px)]
            emb_means.append(float(px.mean()) if px.size else 0.0)
        for i, v in enumerate(emb_means):
            feats[f"aef_{i:02d}"] = v

    return feats


# ── mislabel scoring ──────────────────────────────────────────────────────────

def compute_suspicion_scores(
    gdf: gpd.GeoDataFrame,
    td: TileData,
    contamination: float = 0.1,
) -> gpd.GeoDataFrame:
    """
    Add suspicion scores and flags to each polygon:

    - `score_label_disagreement`: fraction of sources that DISAGREE with label (0-1)
    - `score_spectral`:  how much the spectral change contradicts the label
    - `score_anomaly`:   Isolation Forest anomaly score on AEF embeddings
    - `suspicion_total`: weighted sum (higher = more suspicious)
    - `flagged`:         True if polygon is likely mislabelled
    - `suggested_label`: most probable correct class (bonus)
    """
    print(f"Extracting features for {len(gdf)} polygons …")
    n_sources = 3  # RADD, GLAD-L, GLAD-S2

    records = []
    for idx, row in gdf.iterrows():
        feats = extract_polygon_features(row.geometry, td)
        if feats is None:
            feats = {}
        feats["poly_idx"] = idx
        feats["label"] = row.label
        feats["source_agreement"] = row.source_agreement
        records.append(feats)

    df = pd.DataFrame(records).set_index("poly_idx")

    # ── Signal 1: disagreement (two modes) ───────────────────────────────────
    avail_sources = sum([
        td.radd_binary is not None,
        td.glads2_binary is not None,
        td.gladl_binary is not None,
    ])

    if avail_sources > 0:
        # Train mode: fraction of weak-label sources that disagree with polygon's label
        df["score_label_disagreement"] = (
            1.0 - df["source_agreement"].clip(0, avail_sources) / avail_sources
        )
    else:
        # Test mode: S1-vs-S2 cross-modal disagreement computed per polygon
        cross_modal = []
        for idx, row in gdf.iterrows():
            cross_modal.append(compute_cross_modal_agreement(row.geometry, td))
        df["score_label_disagreement"] = cross_modal

    # ── Signal 2: spectral contradiction ─────────────────────────────────────
    # For deforestation (label=1): expect ndvi_change > 0.1, sar_change > 1 dB
    # For non-deforestation (label=0): expect small changes
    def spectral_contradiction(row):
        lbl = row.label
        ndvi_ok = np.nan
        sar_ok  = np.nan
        if "ndvi_change" in row and np.isfinite(row.ndvi_change):
            if lbl == 1:
                ndvi_ok = float(row.ndvi_change < 0.05)   # expects drop; absence = suspicious
            else:
                ndvi_ok = float(row.ndvi_change > 0.15)   # no drop expected; large drop = suspicious
        if "sar_change" in row and np.isfinite(row.sar_change):
            if lbl == 1:
                sar_ok = float(row.sar_change < 0.5)   # expects dB decrease
            else:
                sar_ok = float(row.sar_change > 1.0)   # no decrease expected
        valid = [v for v in [ndvi_ok, sar_ok] if np.isfinite(v)]
        return float(np.mean(valid)) if valid else 0.0

    df["score_spectral"] = df.apply(spectral_contradiction, axis=1)

    # ── Signal 3: AEF embedding anomaly (Isolation Forest per class) ──────────
    aef_cols = [c for c in df.columns if c.startswith("aef_")]
    df["score_anomaly"] = 0.0

    if aef_cols:
        aef_matrix = df[aef_cols].fillna(0.0).values
        scaler = StandardScaler()
        aef_scaled = scaler.fit_transform(aef_matrix)

        iso = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        iso.fit(aef_scaled)
        # anomaly_score: more negative = more anomalous → invert so higher = worse
        raw_scores = iso.score_samples(aef_scaled)
        # normalise to [0, 1]
        lo, hi = raw_scores.min(), raw_scores.max()
        norm = 1.0 - (raw_scores - lo) / (hi - lo + 1e-9)
        df["score_anomaly"] = norm

        # KNN for suggested label correction (bonus)
        knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        knn.fit(aef_scaled, df["label"].values)
        df["knn_label"] = knn.predict(aef_scaled)

    # ── Combined suspicion score ──────────────────────────────────────────────
    w_disagree  = 0.50   # strongest signal: sources disagree
    w_spectral  = 0.30   # spectral signature contradicts label
    w_anomaly   = 0.20   # embedding is anomalous for this class

    df["suspicion_total"] = (
        w_disagree * df["score_label_disagreement"]
        + w_spectral  * df["score_spectral"]
        + w_anomaly   * df["score_anomaly"]
    )

    # ── Flag threshold ────────────────────────────────────────────────────────
    # Flag polygons where suspicion_total > 0.5 AND at least two signals are elevated
    df["n_elevated"] = (
        (df["score_label_disagreement"] > 0.4).astype(int)
        + (df["score_spectral"] > 0.5).astype(int)
        + (df["score_anomaly"] > 0.6).astype(int)
    )
    df["flagged"] = (df["suspicion_total"] > 0.5) & (df["n_elevated"] >= 2)

    # ── Suggested correction (bonus) ─────────────────────────────────────────
    def suggest_label(row):
        if not row.flagged:
            return row.label
        votes = []
        # Vote 1: inverted current label (since we believe it's wrong)
        votes.append(1 - int(row.label))
        # Vote 2: KNN prediction from embedding space
        if "knn_label" in row and np.isfinite(row.knn_label):
            votes.append(int(row.knn_label))
        # Vote 3: spectral-based guess
        if "ndvi_change" in row and np.isfinite(row.ndvi_change):
            votes.append(1 if row.ndvi_change > 0.1 else 0)
        if not votes:
            return 1 - int(row.label)
        return int(round(np.mean(votes)))

    df["suggested_label"] = df.apply(suggest_label, axis=1)

    # ── Merge back to GeoDataFrame ────────────────────────────────────────────
    keep_cols = [
        "score_label_disagreement", "score_spectral", "score_anomaly",
        "suspicion_total", "n_elevated", "flagged", "suggested_label",
        "ndvi_pre", "ndvi_post", "ndvi_change", "sar_pre", "sar_post", "sar_change",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    result = gdf.copy()
    for col in keep_cols:
        result[col] = df[col].values

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def detect_mislabels(
    tile_id: str,
    split: str = "train",
    predictions: Optional[str] = None,
    output_path: Optional[str] = None,
    contamination: float = 0.1,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Run mislabel detection on a tile.

    Args:
        tile_id:      Tile identifier.
        split:        ``"train"`` or ``"test"``.
        predictions:  Path to prediction GeoJSON (required when ``split="test"``).
        output_path:  Where to write results (.geojson or .csv).
        contamination: Expected fraction of anomalies for Isolation Forest.
        verbose:      Print progress and summary.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Mislabel Detection — tile {tile_id}  [{split.upper()}]")
        print(f"{'='*60}")

    td = load_tile(tile_id, split=split)
    if verbose:
        print(f"Loaded tile data — S2 frames: {len(td.ndvi_dates)}, S1 frames: {len(td.sar_dates)}")
        if split == "train":
            sources = {"RADD": td.radd_binary, "GLAD-S2": td.glads2_binary, "GLAD-L": td.gladl_binary}
            for name, arr in sources.items():
                print(f"  {name}: {'loaded' if arr is not None else 'missing'}")
        else:
            print("  Test mode: S1 / S2 / AEF only (no weak labels)")

    # Build polygon GeoDataFrame
    if split == "test":
        if predictions is None:
            raise ValueError("--predictions is required for test-split tiles.")
        gdf = load_predictions_as_polygons(predictions, td)
        if verbose:
            print(f"Prediction polygons loaded: {len(gdf)}")
    else:
        gdf = vectorize_consensus_labels(td)
        if verbose:
            n_def = (gdf.label == 1).sum()
            n_no  = (gdf.label == 0).sum()
            print(f"Consensus polygons — deforestation: {n_def}, no-deforestation: {n_no}")

    result = compute_suspicion_scores(gdf, td, contamination=contamination)

    flagged = result[result.flagged]
    if verbose:
        print(f"\nFlagged polygons: {len(flagged)} / {len(result)}")
        if not flagged.empty:
            print("\nTop 10 most suspicious polygons:")
            cols = ["label", "suggested_label", "suspicion_total",
                    "score_label_disagreement", "score_spectral", "score_anomaly"]
            cols = [c for c in cols if c in flagged.columns]
            print(flagged.nlargest(10, "suspicion_total")[cols].to_string(index=True))

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".csv":
            result.drop(columns="geometry").to_csv(out, index=True)
        else:
            result.to_crs("EPSG:4326").to_file(out, driver="GeoJSON")
        if verbose:
            print(f"\nResults saved to: {out}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Mislabel detection for deforestation polygons")
    p.add_argument("--tile", default="18NWG_6_6", help="Tile ID (default: 18NWG_6_6)")
    p.add_argument("--split", default="train", choices=["train", "test"],
                   help="Dataset split (default: train)")
    p.add_argument("--predictions", default=None,
                   help="Path to prediction GeoJSON (required for --split test)")
    p.add_argument("--output", default="results/mislabels.geojson",
                   help="Output path (.geojson or .csv)")
    p.add_argument("--contamination", type=float, default=0.1,
                   help="Expected fraction of mislabelled polygons (Isolation Forest)")
    p.add_argument("--all-tiles", action="store_true",
                   help="Run on all training tiles and concatenate results")
    return p.parse_args()


def _get_train_tiles() -> list[str]:
    import json
    meta_path = DATA_ROOT / "metadata" / "train_tiles.geojson"
    with open(meta_path) as f:
        gj = json.load(f)
    return [feat["properties"]["tile_id"] for feat in gj["features"]]


if __name__ == "__main__":
    args = _parse_args()

    if args.all_tiles:
        tiles = _get_train_tiles()
        all_results = []
        for tile in tiles:
            try:
                r = detect_mislabels(tile, split="train",
                                     contamination=args.contamination, verbose=True)
                all_results.append(r)
            except Exception as e:
                print(f"  SKIP {tile}: {e}")
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            combined.drop(columns="geometry").to_csv(out.with_suffix(".csv"), index=True)
            print(f"\nCombined results saved.")
    else:
        detect_mislabels(
            tile_id=args.tile,
            split=args.split,
            predictions=args.predictions,
            output_path=args.output,
            contamination=args.contamination,
            verbose=True,
        )
