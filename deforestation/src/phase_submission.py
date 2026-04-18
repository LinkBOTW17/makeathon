"""
Submission Generation

For every test tile:
  1. Extracts static features for ALL pixels (vectorized, no sampling)
  2. Runs XGBoost inference with the optimised probability threshold
  3. Applies forest pre-filter (ndvi_max > 0.35) and VV change signal filter
  4. Predicts deforestation YEAR per pixel from max VV-dB drop after 2020
  5. Writes binary prediction raster + year raster as GeoTIFFs
  6. Converts to GeoJSON polygons (min_area_ha=0.5) with a "time_step" property

Run from the deforestation/ directory:
    python src/phase_submission.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features
from rasterio.enums import Resampling
import shapely.geometry
import geopandas as gpd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # for submission_utils

from utils import (
    get_logger, load_config,
    discover_s2_files, discover_s1_files, find_aef_file, normalize_aef,
    spectral_index,
)

logger = get_logger(__name__)

SCL_VALID = {2, 4, 5, 6, 7, 11}


# ─── Vectorized feature extraction for a full tile ────────────────────────────

def _ts_stats_vectorized(ts_hw: np.ndarray, prefix: str) -> dict:
    """Compute temporal statistics for a (T, H, W) time-series cube."""
    T, H, W = ts_hw.shape
    valid    = np.isfinite(ts_hw)
    n_valid  = valid.sum(axis=0).astype(np.float32)

    with np.errstate(all="ignore"):
        mean_v  = np.nanmean(ts_hw, axis=0)
        std_v   = np.nanstd(ts_hw,  axis=0)
        min_v   = np.nanmin(ts_hw,  axis=0)
        max_v   = np.nanmax(ts_hw,  axis=0)

    valid_frac = n_valid / T

    diffs    = np.diff(ts_hw, axis=0)
    max_drop = np.nanmin(diffs, axis=0)

    t_vals   = np.arange(T, dtype=np.float32)[:, None, None]
    t_w      = np.where(valid, t_vals, np.nan)
    v_w      = np.where(valid, ts_hw,  np.nan)
    n_safe   = np.where(n_valid >= 3, n_valid, np.nan)

    t_mean   = np.nansum(t_w, axis=0) / n_safe
    v_mean   = np.nansum(v_w, axis=0) / n_safe
    cov      = np.nansum((t_w - t_mean[None]) * (v_w - v_mean[None]), axis=0) / n_safe
    var_t    = np.nansum((t_w - t_mean[None]) ** 2, axis=0) / n_safe + 1e-8
    slope    = np.where(n_valid >= 3, cov / var_t, np.nan)

    def _fill(a):
        return np.nan_to_num(a, nan=0.0).astype(np.float32)

    return {
        f"{prefix}_mean":       _fill(mean_v),
        f"{prefix}_std":        _fill(std_v),
        f"{prefix}_min":        _fill(min_v),
        f"{prefix}_max":        _fill(max_v),
        f"{prefix}_slope":      _fill(slope),
        f"{prefix}_max_drop":   _fill(max_drop),
        f"{prefix}_valid_frac": _fill(valid_frac),
    }


def extract_full_tile_features(tile_id: str, cfg: dict,
                                is_test: bool = True):
    """Build (H*W, 85) feature matrix for an entire tile."""
    fe      = cfg["feature_extraction"]
    aef_cfg = cfg["aef"]

    base_s2  = cfg["paths_test"]["s2"]  if is_test else cfg["paths"]["s2"]
    base_s1  = cfg["paths_test"]["s1"]  if is_test else cfg["paths"]["s1"]
    base_aef = cfg["paths_test"]["aef"] if is_test else cfg["paths"]["aef"]

    s2_dir = Path(base_s2) / f"{tile_id}__s2_l2a"
    s1_dir = Path(base_s1) / f"{tile_id}__s1_rtc"

    aef_path = find_aef_file(base_aef, tile_id, aef_cfg["year"])
    if aef_path is None:
        logger.warning(f"[{tile_id}] AEF not found — skipping")
        return None

    with rasterio.open(aef_path) as src:
        aef_raw   = src.read()
        transform = src.transform
        crs       = src.crs
        H, W      = src.height, src.width

    aef_data = normalize_aef(aef_raw)
    logger.info(f"[{tile_id}] Grid: {H}×{W} = {H*W:,} pixels")

    # ── S2 NDVI + NBR time series cube ────────────────────────────────────
    s2_files  = discover_s2_files(str(s2_dir))
    T_s2      = len(s2_files)
    ndvi_cube = np.full((T_s2, H, W), np.nan, dtype=np.float32)
    nbr_cube  = np.full((T_s2, H, W), np.nan, dtype=np.float32)

    for t_idx, (year, month, s2_path) in enumerate(s2_files):
        try:
            with rasterio.open(s2_path) as src:
                if (src.height, src.width) != (H, W):
                    data = src.read(
                        [fe["s2_red_band"], fe["s2_nir_band"],
                         fe["s2_swir2_band"], fe["s2_scl_band"]],
                        out_shape=(4, H, W),
                        resampling=Resampling.bilinear,
                    ).astype(np.float32)
                else:
                    data = src.read(
                        [fe["s2_red_band"], fe["s2_nir_band"],
                         fe["s2_swir2_band"], fe["s2_scl_band"]]
                    ).astype(np.float32)
        except Exception as e:
            logger.warning(f"[{tile_id}] S2 {year}-{month:02d} read error: {e}")
            continue

        red   = data[0] / fe["s2_scale_factor"]
        nir   = data[1] / fe["s2_scale_factor"]
        swir2 = data[2] / fe["s2_scale_factor"]
        scl   = data[3]
        valid_scl = np.isin(scl.astype(np.int32), fe["scl_valid_values"])

        ndvi = spectral_index(nir, red)
        nbr  = spectral_index(nir, swir2)
        ndvi_cube[t_idx] = np.where(valid_scl, ndvi, np.nan)
        nbr_cube[t_idx]  = np.where(valid_scl, nbr,  np.nan)

    ndvi_stats = _ts_stats_vectorized(ndvi_cube, "ndvi")
    nbr_stats  = _ts_stats_vectorized(nbr_cube,  "nbr")
    logger.info(f"[{tile_id}] S2 done ({T_s2} months)")

    # ── S1 VV-dB time series cube ─────────────────────────────────────────
    s1_files = discover_s1_files(str(s1_dir), fe["s1_orbit"])
    T_s1     = len(s1_files)
    vv_cube  = np.full((T_s1, H, W), np.nan, dtype=np.float32)

    for t_idx, (year, month, s1_path) in enumerate(s1_files):
        try:
            with rasterio.open(s1_path) as src:
                if (src.height, src.width) != (H, W):
                    vv = src.read(1, out_shape=(H, W),
                                  resampling=Resampling.bilinear).astype(np.float32)
                else:
                    vv = src.read(1).astype(np.float32)
        except Exception as e:
            logger.warning(f"[{tile_id}] S1 {year}-{month:02d} read error: {e}")
            continue

        if fe["s1_db_conversion"]:
            vv_cube[t_idx] = np.where(vv > fe["s1_db_eps"],
                                      10.0 * np.log10(vv + fe["s1_db_eps"]), np.nan)
        else:
            vv_cube[t_idx] = np.where(vv > 0, vv, np.nan)

    vv_stats = _ts_stats_vectorized(vv_cube, "vv")
    logger.info(f"[{tile_id}] S1 done ({T_s1} months)")

    # ── Assemble feature matrix ───────────────────────────────────────────
    feature_parts = {}
    for i in range(64):
        feature_parts[f"aef_{i}"] = aef_data[i].ravel()
    for d in [ndvi_stats, nbr_stats, vv_stats]:
        for k, v in d.items():
            feature_parts[k] = v.ravel()

    feat_matrix  = np.stack(list(feature_parts.values()), axis=1)
    feature_cols = list(feature_parts.keys())

    return feat_matrix, transform, crs, H, W, feature_cols, s1_files, vv_cube, ndvi_stats, vv_stats


def predict_year_vectorized(vv_cube: np.ndarray, s1_files: list,
                             ndvi_stats: dict,
                             start_year: int, fallback_year: int,
                             max_year: int = 2024) -> np.ndarray:
    """
    For each pixel, find the year of the largest single-step VV-dB drop
    after start_year, capped at max_year.
    """
    T, H, W = vv_cube.shape
    years   = np.array([yr for yr, _, _ in s1_files], dtype=np.int32)

    post2020_mask = (years >= start_year) & (years <= max_year)
    post2020_idx  = np.where(post2020_mask)[0]

    if len(post2020_idx) < 2:
        return np.full((H, W), fallback_year, dtype=np.int32)

    diff_cube  = np.diff(vv_cube[post2020_idx], axis=0)
    diff_years = years[post2020_idx[1:]]

    diff_cube_filled = np.where(np.isfinite(diff_cube), diff_cube, np.inf)
    drop_t_idx = np.argmin(diff_cube_filled, axis=0)

    year_map  = diff_years[drop_t_idx]
    any_valid = np.isfinite(diff_cube).any(axis=0)
    year_map  = np.where(any_valid, year_map, fallback_year)

    # Cap at max_year
    year_map = np.clip(year_map, start_year, max_year)

    return year_map.astype(np.int32)


def discover_test_tiles(cfg: dict) -> list:
    """Find test tile IDs from the AEF test directory."""
    aef_test_dir = Path(cfg["paths_test"]["aef"])
    year = cfg["aef"]["year"]
    tiles = []
    for f in sorted(aef_test_dir.glob(f"*_{year}.tiff")) + sorted(aef_test_dir.glob(f"*_{year}.tif")):
        stem    = f.stem
        tile_id = stem[:-(len(str(year)) + 1)]
        tiles.append(tile_id)
    return sorted(set(tiles))


def min_area_ha_filter(binary: np.ndarray, transform, crs,
                       min_area_ha: float) -> np.ndarray:
    """Zero out connected components smaller than min_area_ha hectares."""
    from scipy.ndimage import label as nd_label
    labeled, n_feat = nd_label(binary)
    if n_feat == 0:
        return binary

    # Pixel area in m² (assumes projected CRS in metres)
    px_area_m2 = abs(transform.a * transform.e)
    min_px = int(np.ceil((min_area_ha * 10_000) / px_area_m2))

    out = binary.copy()
    for comp_id in range(1, n_feat + 1):
        mask = labeled == comp_id
        if mask.sum() < min_px:
            out[mask] = 0
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg       = load_config("configs/config.yaml")
    sub_cfg   = cfg["submission"]
    threshold = float(sub_cfg["probability_threshold"])
    out_dir   = Path(sub_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    min_area_ha    = float(sub_cfg.get("min_area_ha", 0.5))
    forest_ndvi_th = float(sub_cfg.get("forest_ndvi_threshold", 0.35))
    vv_drop_th     = float(sub_cfg.get("vv_change_threshold", -1.5))
    max_year       = int(sub_cfg.get("max_year", 2024))

    ckpt_dir       = Path(cfg["paths"]["checkpoints"])
    model_art      = joblib.load(ckpt_dir / "xgboost_baseline.joblib")
    model          = model_art["model"]
    train_feat_cols = model_art["feature_cols"]

    logger.info(f"Loaded XGBoost model | threshold={threshold:.2f}")
    logger.info(f"Forest NDVI threshold : {forest_ndvi_th}")
    logger.info(f"VV change threshold   : {vv_drop_th} dB")
    logger.info(f"Min area              : {min_area_ha} ha")

    test_tiles = discover_test_tiles(cfg)
    if not test_tiles:
        logger.error("No test tiles found. Check paths_test.aef in config.yaml.")
        sys.exit(1)
    logger.info(f"Test tiles ({len(test_tiles)}): {test_tiles}")

    all_geojson_features = []

    for tile_id in tqdm(test_tiles, desc="Generating submissions"):
        logger.info(f"\n[{tile_id}] Extracting features …")
        result = extract_full_tile_features(tile_id, cfg, is_test=True)
        if result is None:
            continue

        feat_matrix, transform, crs, H, W, feat_cols, s1_files, vv_cube, ndvi_stats, vv_stats = result

        # ── Align feature columns to training order ────────────────────────
        col_to_idx = {c: i for i, c in enumerate(feat_cols)}
        try:
            ordered_idx = [col_to_idx[c] for c in train_feat_cols]
        except KeyError as e:
            logger.error(f"[{tile_id}] Feature column mismatch: {e}")
            continue
        X = feat_matrix[:, ordered_idx].astype(np.float32)

        # ── Inference ──────────────────────────────────────────────────────
        logger.info(f"[{tile_id}] Running XGBoost on {H*W:,} pixels …")
        probs  = model.predict_proba(X)[:, 1]
        binary = (probs >= threshold).astype(np.uint8).reshape(H, W)

        # ── Forest pre-filter: only predict in pixels that were forested ────
        ndvi_max_map = ndvi_stats["ndvi_max"].reshape(H, W)
        forest_mask  = ndvi_max_map > forest_ndvi_th
        binary       = binary & forest_mask.astype(np.uint8)

        # ── VV change-signal filter: require meaningful SAR backscatter drop ─
        vv_drop_map  = vv_stats["vv_max_drop"].reshape(H, W)
        change_mask  = vv_drop_map < vv_drop_th
        binary       = binary & change_mask.astype(np.uint8)

        # ── Remove small polygons (< min_area_ha) ─────────────────────────
        binary = min_area_ha_filter(binary, transform, crs, min_area_ha)

        n_def = int(binary.sum())
        logger.info(f"[{tile_id}] After filters: {n_def:,} deforestation pixels "
                    f"({100*n_def/(H*W):.2f}%)")

        if n_def == 0:
            logger.warning(f"[{tile_id}] No deforestation predicted — skipping")
            continue

        # ── Year prediction ────────────────────────────────────────────────
        year_map = predict_year_vectorized(
            vv_cube, s1_files, ndvi_stats,
            start_year=sub_cfg["year_pred_start_year"],
            fallback_year=sub_cfg["fallback_year"],
            max_year=max_year,
        )
        year_map = np.where(binary, year_map, 0).astype(np.int32)

        # ── Write rasters ──────────────────────────────────────────────────
        pred_raster_path = out_dir / f"pred_{tile_id}.tif"
        year_raster_path = out_dir / f"year_{tile_id}.tif"

        meta = dict(driver="GTiff", dtype="uint8", count=1,
                    height=H, width=W, crs=crs, transform=transform,
                    compress="lzw")

        with rasterio.open(pred_raster_path, "w", **meta) as dst:
            dst.write(binary, 1)

        meta_year = {**meta, "dtype": "int32"}
        with rasterio.open(year_raster_path, "w", **meta_year) as dst:
            dst.write(year_map, 1)

        # ── Polygon vectorisation + per-polygon time_step assignment ────────
        shapes_list = [(gd, int(v))
                       for gd, v in rio_features.shapes(binary, mask=binary,
                                                         transform=transform)
                       if int(v) > 0]

        if not shapes_list:
            logger.warning(f"[{tile_id}] shapes() returned no polygons")
            continue

        # Burn unique polygon ID for vectorised year aggregation
        poly_id_raster = np.zeros((H, W), dtype=np.int32)
        for pid, (gd, _) in enumerate(shapes_list, start=1):
            rio_features.rasterize([(gd, pid)], out_shape=(H, W),
                                   transform=transform, out=poly_id_raster,
                                   merge_alg=rio_features.MergeAlg.replace)

        flat_ids   = poly_id_raster[binary == 1].ravel()
        flat_years = year_map[binary == 1].ravel()
        valid_mask = flat_years > 0
        if valid_mask.any():
            yr_df      = pd.DataFrame({"pid": flat_ids[valid_mask],
                                       "year": flat_years[valid_mask]})
            mode_years = (yr_df.groupby("pid")["year"]
                          .agg(lambda x: int(x.mode().iloc[0]))
                          .to_dict())
        else:
            mode_years = {}

        fallback = int(sub_cfg["fallback_year"])
        for pid, (gd, _) in enumerate(shapes_list, start=1):
            geom = shapely.geometry.shape(gd)
            predicted_year = mode_years.get(pid, fallback)
            all_geojson_features.append({
                "type": "Feature",
                "geometry": shapely.geometry.mapping(geom),
                "properties": {
                    "tile_id":   tile_id,
                    "time_step": predicted_year,   # scorer expects "time_step"
                },
            })

        logger.info(f"[{tile_id}] Polygons: {len(shapes_list)}")

    if not all_geojson_features:
        logger.error("No predictions generated.")
        sys.exit(1)

    gdf = gpd.GeoDataFrame.from_features(all_geojson_features)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    gdf_4326 = gdf.to_crs("EPSG:4326")

    submission_path = out_dir / "submission.geojson"
    gdf_4326.to_file(submission_path, driver="GeoJSON")

    area_ha = 0.0
    try:
        gdf_utm = gdf_4326.to_crs(gdf_4326.estimate_utm_crs())
        area_ha = float(gdf_utm.area.sum() / 10_000)
    except Exception:
        pass

    year_counts = gdf_4326["time_step"].value_counts().sort_index()

    print("\n── Submission summary ───────────────────────────────────────────")
    print(f"  Tiles processed      : {len(test_tiles)}")
    print(f"  Total polygons       : {len(gdf_4326):,}")
    print(f"  Total predicted area : {area_ha:.1f} ha")
    print(f"  Threshold            : {threshold:.2f}")
    print(f"  Forest NDVI filter   : ndvi_max > {forest_ndvi_th}")
    print(f"  VV change filter     : vv_max_drop < {vv_drop_th} dB")
    print(f"  Min area             : {min_area_ha} ha")
    print(f"  Year distribution (time_step):")
    for yr, cnt in year_counts.items():
        print(f"    {yr}: {cnt:,} polygons")
    print(f"\n✓ Saved → {submission_path}")
    print("\nDONE — Submission generation complete.")


if __name__ == "__main__":
    main()
