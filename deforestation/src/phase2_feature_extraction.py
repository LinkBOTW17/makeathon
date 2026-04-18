"""
Phase 2 — Feature Extraction

For every pixel in fused_labels.parquet:
  - AEF embedding: 64 dims (sampled from EPSG:4326 raster)
  - Sentinel-2 temporal stats: NDVI and NBR over all available months
    (7 stats each: mean, std, min, max, slope, max_drop, valid_frac)
  - Sentinel-1 temporal stats: VV backscatter in dB
    (6 stats: mean, std, min, max, max_change, valid_frac)

Outputs:
  data/processed/features_static.parquet   — (n_pixels, 64+7+7+6) feature matrix
  data/processed/labels.parquet            — (n_pixels,) labels + metadata
  data/processed/s2_timeseries.npy         — (n_pixels, T_s2, 2)  NDVI, NBR
  data/processed/s1_timeseries.npy         — (n_pixels, T_s1, 1)  VV dB
  data/processed/s2_timestamps.json        — list of "YYYY_MM" for T_s2 axis
  data/processed/s1_timestamps.json        — list of "YYYY_MM" for T_s1 axis

Run from the deforestation/ directory:
    python src/phase2_feature_extraction.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_logger,
    load_config,
    get_pixel_xy_coords,
    read_raster_at_coords,
    spectral_index,
    compute_ts_stats,
    discover_s2_files,
    discover_s1_files,
    find_aef_file,
    normalize_aef,
)

logger = get_logger(__name__)


# ─── Time-step discovery ──────────────────────────────────────────────────────

def build_global_timestep_index(tile_ids: list, cfg: dict) -> tuple[list, list]:
    """
    Scan all tiles to find the union of available (year, month) pairs for S2 and S1.
    Returns two sorted lists of "YYYY_MM" strings.
    """
    fe  = cfg["feature_extraction"]
    s2_steps: set = set()
    s1_steps: set = set()

    for tile_id in tile_ids:
        s2_dir = Path(cfg["paths"]["s2"]) / f"{tile_id}__s2_l2a"
        s1_dir = Path(cfg["paths"]["s1"]) / f"{tile_id}__s1_rtc"

        for year, month, _ in discover_s2_files(str(s2_dir)):
            s2_steps.add(f"{year}_{month:02d}")
        for year, month, _ in discover_s1_files(str(s1_dir), fe["s1_orbit"]):
            s1_steps.add(f"{year}_{month:02d}")

    return sorted(s2_steps), sorted(s1_steps)


# ─── Per-tile feature extraction ─────────────────────────────────────────────

def extract_tile_features(
    tile_id: str,
    tile_df: pd.DataFrame,
    s2_ts_index: list,
    s1_ts_index: list,
    cfg: dict,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
    """
    Extract AEF embeddings + S2/S1 temporal stats for all pixels in one tile.

    Returns:
        feat_df  — DataFrame with pixel_id + all feature columns
        s2_ts    — np.ndarray shape (n_pixels, T_s2, 2)
        s1_ts    — np.ndarray shape (n_pixels, T_s1, 1)
    """
    fe = cfg["feature_extraction"]
    n  = len(tile_df)

    rows = tile_df["row"].values
    cols = tile_df["col"].values

    # ── Get RADD pixel centres in RADD CRS ────────────────────────────────
    radd_path = Path(cfg["paths"]["labels"]["radd"]) / f"radd_{tile_id}_labels.tif"
    if not radd_path.exists():
        logger.warning(f"[{tile_id}] RADD reference raster missing — skip")
        return None

    xs_radd, ys_radd, radd_crs = get_pixel_xy_coords(str(radd_path), rows, cols)

    # ── AEF embeddings (EPSG:4326 raster) ─────────────────────────────────
    aef_path = find_aef_file(cfg["paths"]["aef"], tile_id, cfg["aef"]["year"])
    if aef_path is None:
        logger.warning(f"[{tile_id}] AEF file for year {cfg['aef']['year']} not found — filling NaN")
        aef_feats = np.full((n, 64), np.nan, dtype=np.float32)
    else:
        with rasterio.open(aef_path) as src:
            aef_crs   = src.crs
            aef_data  = normalize_aef(src.read())   # (64, H, W)
            aef_trans = src.transform
            H, W      = src.height, src.width

        from rasterio.warp import transform as warp_transform
        import rasterio.transform as rtrans

        xs_aef, ys_aef = warp_transform(radd_crs, aef_crs,
                                         xs_radd.tolist(), ys_radd.tolist())
        xs_aef = np.asarray(xs_aef, dtype=np.float64)
        ys_aef = np.asarray(ys_aef, dtype=np.float64)

        r_aef, c_aef = rtrans.rowcol(aef_trans, xs_aef, ys_aef)
        r_aef = np.asarray(r_aef, dtype=np.int64)
        c_aef = np.asarray(c_aef, dtype=np.int64)
        valid = (r_aef >= 0) & (r_aef < H) & (c_aef >= 0) & (c_aef < W)

        aef_feats = np.full((n, 64), np.nan, dtype=np.float32)
        if valid.any():
            aef_feats[valid] = aef_data[:, r_aef[valid], c_aef[valid]].T

    logger.info(f"[{tile_id}] AEF done — NaN pixels: {np.isnan(aef_feats).any(axis=1).sum()}/{n}")

    # ── Sentinel-2 time series ─────────────────────────────────────────────
    T_s2   = len(s2_ts_index)
    s2_dir = Path(cfg["paths"]["s2"]) / f"{tile_id}__s2_l2a"

    # s2_ts: (n_pixels, T_s2, 2)  channels: [NDVI, NBR]
    s2_ts  = np.full((n, T_s2, 2), np.nan, dtype=np.float32)

    # Build lookup: "YYYY_MM" → index in s2_ts_index
    s2_lookup = {ts: i for i, ts in enumerate(s2_ts_index)}

    # Discover available files for this tile
    available_s2 = discover_s2_files(str(s2_dir))
    if not available_s2:
        logger.warning(f"[{tile_id}] No S2 files found in {s2_dir}")

    for year, month, s2_path in available_s2:
        ts_key = f"{year}_{month:02d}"
        if ts_key not in s2_lookup:
            continue
        t_idx = s2_lookup[ts_key]

        try:
            vals = read_raster_at_coords(
                str(s2_path), xs_radd, ys_radd, radd_crs,
                band_indices=[fe["s2_red_band"], fe["s2_nir_band"],
                              fe["s2_swir2_band"], fe["s2_scl_band"]],
            )  # (n, 4)
        except Exception as e:
            logger.warning(f"[{tile_id}] S2 {ts_key} read error: {e}")
            continue

        red   = vals[:, 0] / fe["s2_scale_factor"]
        nir   = vals[:, 1] / fe["s2_scale_factor"]
        swir2 = vals[:, 2] / fe["s2_scale_factor"]
        scl   = vals[:, 3]

        # Cloud mask: keep only valid SCL classes
        valid_scl = np.isin(scl.astype(np.int32), fe["scl_valid_values"])

        ndvi = spectral_index(nir, red)
        nbr  = spectral_index(nir, swir2)

        # Apply cloud mask
        ndvi = np.where(valid_scl, ndvi, np.nan)
        nbr  = np.where(valid_scl, nbr,  np.nan)

        s2_ts[:, t_idx, 0] = ndvi
        s2_ts[:, t_idx, 1] = nbr

    valid_s2_count = np.isfinite(s2_ts).any(axis=2).sum(axis=1).mean()
    logger.info(f"[{tile_id}] S2 done — avg valid timesteps per pixel: {valid_s2_count:.1f}/{T_s2}")

    # ── Sentinel-1 time series ─────────────────────────────────────────────
    T_s1   = len(s1_ts_index)
    s1_dir = Path(cfg["paths"]["s1"]) / f"{tile_id}__s1_rtc"

    # s1_ts: (n_pixels, T_s1, 1)  channel: [VV dB]
    s1_ts  = np.full((n, T_s1, 1), np.nan, dtype=np.float32)
    s1_lookup = {ts: i for i, ts in enumerate(s1_ts_index)}

    available_s1 = discover_s1_files(str(s1_dir), fe["s1_orbit"])
    if not available_s1:
        logger.warning(f"[{tile_id}] No S1 files found in {s1_dir}")

    for year, month, s1_path in available_s1:
        ts_key = f"{year}_{month:02d}"
        if ts_key not in s1_lookup:
            continue
        t_idx = s1_lookup[ts_key]

        try:
            vv_vals = read_raster_at_coords(
                str(s1_path), xs_radd, ys_radd, radd_crs, band_indices=[1],
            )  # (n, 1)
        except Exception as e:
            logger.warning(f"[{tile_id}] S1 {ts_key} read error: {e}")
            continue

        vv = vv_vals[:, 0]
        if fe["s1_db_conversion"]:
            # Convert linear VV → dB; mask invalid (zero or negative) values
            valid_vv = vv > fe["s1_db_eps"]
            vv_db    = np.where(valid_vv, 10.0 * np.log10(vv + fe["s1_db_eps"]), np.nan)
            s1_ts[:, t_idx, 0] = vv_db
        else:
            s1_ts[:, t_idx, 0] = np.where(np.isfinite(vv), vv, np.nan)

    valid_s1_count = np.isfinite(s1_ts[:, :, 0]).sum(axis=1).mean()
    logger.info(f"[{tile_id}] S1 done — avg valid timesteps per pixel: {valid_s1_count:.1f}/{T_s1}")

    # ── Compute temporal stats ────────────────────────────────────────────
    ndvi_stats = compute_ts_stats(s2_ts[:, :, 0], "ndvi")
    nbr_stats  = compute_ts_stats(s2_ts[:, :, 1], "nbr")
    vv_stats   = compute_ts_stats(s1_ts[:, :, 0], "vv")

    # ── Impute remaining NaN with feature-group means ─────────────────────
    aef_nan_count = np.isnan(aef_feats).sum()
    aef_col_means = np.nanmean(aef_feats, axis=0)
    aef_col_means = np.where(np.isnan(aef_col_means), 0.0, aef_col_means)
    nan_mask      = np.isnan(aef_feats)
    aef_feats     = np.where(nan_mask, aef_col_means[None, :], aef_feats)

    if aef_nan_count > 0:
        logger.info(f"[{tile_id}] Imputed {aef_nan_count} NaN AEF values with column means")

    # ── Build feature DataFrame ───────────────────────────────────────────
    aef_cols  = {f"aef_{i}": aef_feats[:, i] for i in range(64)}
    all_stats = {**aef_cols, **ndvi_stats, **nbr_stats, **vv_stats}

    # Impute NaN in stat columns with 0 (global fallback)
    stat_nan_counts = {k: int(np.isnan(v).sum()) for k, v in all_stats.items() if np.isnan(v).any()}
    if stat_nan_counts:
        logger.info(f"[{tile_id}] Stat NaN counts (→ filled with 0): {stat_nan_counts}")
        all_stats = {k: np.nan_to_num(v, nan=0.0) for k, v in all_stats.items()}

    feat_df = pd.DataFrame(all_stats)
    feat_df.insert(0, "pixel_id", tile_df["pixel_id"].values)
    feat_df.insert(1, "tile_id",  tile_id)

    return feat_df, s2_ts, s1_ts


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config("configs/config.yaml")
    processed_dir = Path(cfg["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Load fused labels ─────────────────────────────────────────────────
    labels_path = processed_dir / "fused_labels.parquet"
    logger.info(f"Loading {labels_path} …")
    fused_df = pd.read_parquet(labels_path)
    logger.info(f"fused_labels shape: {fused_df.shape}  dtypes:\n{fused_df.dtypes.to_string()}")

    tile_ids = sorted(fused_df["tile_id"].unique())
    logger.info(f"Tiles to process: {tile_ids}")

    # ── Build global timestep index ───────────────────────────────────────
    logger.info("Scanning tiles to build global timestep index …")
    s2_ts_index, s1_ts_index = build_global_timestep_index(tile_ids, cfg)
    logger.info(f"S2 timesteps ({len(s2_ts_index)}): {s2_ts_index[:6]} … {s2_ts_index[-3:]}")
    logger.info(f"S1 timesteps ({len(s1_ts_index)}): {s1_ts_index[:6]} … {s1_ts_index[-3:]}")

    T_s2 = len(s2_ts_index)
    T_s1 = len(s1_ts_index)

    # ── Process every tile ────────────────────────────────────────────────
    all_feat_dfs = []
    all_s2_ts    = []
    all_s1_ts    = []

    for tile_id in tqdm(tile_ids, desc="Extracting features"):
        tile_df = fused_df[fused_df["tile_id"] == tile_id].reset_index(drop=True)
        logger.info(f"\n[{tile_id}] {len(tile_df):,} pixels …")

        result = extract_tile_features(tile_id, tile_df, s2_ts_index, s1_ts_index, cfg)
        if result is None:
            logger.warning(f"[{tile_id}] Skipped — no result")
            continue

        feat_df, s2_ts, s1_ts = result
        all_feat_dfs.append(feat_df)
        all_s2_ts.append(s2_ts)
        all_s1_ts.append(s1_ts)

    if not all_feat_dfs:
        logger.error("No tiles processed successfully.")
        sys.exit(1)

    # ── Concatenate ───────────────────────────────────────────────────────
    feat_df_all = pd.concat(all_feat_dfs, ignore_index=True)
    s2_ts_all   = np.concatenate(all_s2_ts, axis=0)    # (N, T_s2, 2)
    s1_ts_all   = np.concatenate(all_s1_ts, axis=0)    # (N, T_s1, 1)

    # ── Feature breakdown summary ─────────────────────────────────────────
    feature_cols = [c for c in feat_df_all.columns if c not in ("pixel_id", "tile_id")]
    aef_cols  = [c for c in feature_cols if c.startswith("aef_")]
    s2_cols   = [c for c in feature_cols if c.startswith("ndvi_") or c.startswith("nbr_")]
    s1_cols   = [c for c in feature_cols if c.startswith("vv_")]

    print("\n── Feature matrix breakdown ─────────────────────────────────────")
    print(f"  AEF features      : {len(aef_cols)}")
    print(f"  S2 stats (NDVI+NBR): {len(s2_cols)}")
    print(f"  S1 stats (VV)     : {len(s1_cols)}")
    print(f"  Total features    : {len(feature_cols)}")
    print(f"  Feature matrix    : {feat_df_all.shape}")
    print(f"  S2 time series    : {s2_ts_all.shape}  (pixels × T × [NDVI,NBR])")
    print(f"  S1 time series    : {s1_ts_all.shape}  (pixels × T × [VV_dB])")

    # ── NaN report ────────────────────────────────────────────────────────
    nan_pixels = feat_df_all[feature_cols].isna().any(axis=1).sum()
    print(f"  Pixels with any NaN feature: {nan_pixels:,} ({100*nan_pixels/len(feat_df_all):.1f}%)")

    # ── Build and save labels parquet ─────────────────────────────────────
    label_cols = ["pixel_id", "tile_id", "fused_label", "confidence", "uncertain_flag",
                  "votes", "radd_label", "gladl_label", "glads2_label"]
    labels_out = fused_df[label_cols].copy()

    # Align order to match feat_df_all (inner join on pixel_id to be safe)
    labels_out = labels_out.set_index("pixel_id").loc[feat_df_all["pixel_id"]].reset_index()

    # ── Save ──────────────────────────────────────────────────────────────
    feat_path   = processed_dir / "features_static.parquet"
    label_path  = processed_dir / "labels.parquet"
    s2_ts_path  = processed_dir / "s2_timeseries.npy"
    s1_ts_path  = processed_dir / "s1_timeseries.npy"
    s2_meta_path = processed_dir / "s2_timestamps.json"
    s1_meta_path = processed_dir / "s1_timestamps.json"

    feat_df_all.to_parquet(feat_path, index=False)
    labels_out.to_parquet(label_path, index=False)
    np.save(s2_ts_path, s2_ts_all)
    np.save(s1_ts_path, s1_ts_all)

    with open(s2_meta_path, "w") as f:
        json.dump(s2_ts_index, f)
    with open(s1_meta_path, "w") as f:
        json.dump(s1_ts_index, f)

    print(f"\n✓ Saved → {feat_path}  ({feat_df_all.shape})")
    print(f"✓ Saved → {label_path}  ({labels_out.shape})")
    print(f"✓ Saved → {s2_ts_path}  {s2_ts_all.shape}")
    print(f"✓ Saved → {s1_ts_path}  {s1_ts_all.shape}")
    print(f"✓ Saved → {s2_meta_path}  ({len(s2_ts_index)} timesteps)")
    print(f"✓ Saved → {s1_meta_path}  ({len(s1_ts_index)} timesteps)")
    print("\nDONE — Phase 2 complete.")


if __name__ == "__main__":
    main()
