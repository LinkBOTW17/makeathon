"""
Submission Generation

For every test tile:
  1. Extracts static features for ALL pixels (vectorized, no sampling)
  2. Runs XGBoost inference with the optimised probability threshold
  3. Applies post-processing filters:
       - Forest pre-filter: AEF tree-cover dims (A16/A23) + ndvi_max > 0.35
       - AEF validity mask: exclude all-zero AEF pixels (ocean/NoData)
       - AEF non-forest exclusion: water (A64) and bare (A63) pixels
       - VV change signal: vv_max_drop < vv_change_threshold
  4. Filters polygons < 0.5 ha (official submission_utils default)
  5. Predicts deforestation YEAR (time_step) per polygon from max VV-dB drop
  6. Writes binary prediction + year rasters and combined submission.geojson

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
from utils import (
    get_logger, load_config,
    discover_s2_files, discover_s1_files, find_aef_file, normalize_aef,
    spectral_index,
)

logger = get_logger(__name__)

SCL_VALID = {2, 4, 5, 6, 7, 11}

# AEF dimension indices (0-based) from alphaearth_workshop.ipynb cell 25
# Notation is 1-based (A16 = index 15), values are from proper dequantization
AEF_TREE_COVER_DIMS = [15, 22]       # A16, A23 → Tree cover (forest)
AEF_WATER_DIM       = 63             # A64      → Permanent water bodies
AEF_BARE_DIM        = 62             # A63      → Bare/sparse vegetation
AEF_BUILDUP_DIMS    = [8, 34]        # A09, A35 → Built-up
AEF_CROPLAND_DIMS   = [11, 49]       # A12, A50 → Cropland

# Dequantization constants from alphaearth_workshop.ipynb
_AEF_OFFSET = 127.0
_AEF_SCALE  = 127.5


def dequantize_aef(raw: np.ndarray) -> np.ndarray:
    """
    Proper AEF dequantization: signed_square((raw - 127) / 127.5)
    Only applied if the data is integer (uint8). Float data is returned as-is
    (meaning it was already dequantized by the data provider).

    Also marks all-zero 64-band groups as NaN (NoData/ocean pixels).
    raw : (64, H, W)
    Returns (64, H, W) float32.
    """
    # Mark NoData pixels (all 64 bands == 0 or NaN)
    if np.issubdtype(raw.dtype, np.integer):
        nodata = np.all(raw == 0, axis=0)               # (H, W)
        out    = raw.astype(np.float32)
        x      = (out - _AEF_OFFSET) / _AEF_SCALE
        neg    = x < 0
        out    = np.abs(x) ** 2
        out    = np.where(neg, -out, out)                # signed square ∈ [-1, 1]
        out[:, nodata] = np.nan
    else:
        # Already float (challenge AEF may ship pre-dequantized)
        out    = raw.astype(np.float32)
        nodata = np.all(~np.isfinite(out), axis=0)
        out[:, nodata] = np.nan

    return out


def build_forest_and_exclusion_masks(aef_raw: np.ndarray,
                                      ndvi_max_map: np.ndarray,
                                      ndvi_valid_frac_map: np.ndarray,
                                      forest_ndvi_th: float) -> tuple:
    """
    Returns (forest_mask, exclusion_mask) both shape (H, W) bool.

    forest_mask    : pixel was forested → eligible for deforestation detection
    exclusion_mask : pixel is water/bare/NoData → NEVER deforestation

    Critical: pixels with no cloud-free S2 observations (ndvi_valid_frac == 0)
    have ndvi_max filled with 0.0 (not a real measurement). They must NOT be
    filtered by the NDVI threshold — in cloudy regions (Amazon, SE Asia) this
    would eliminate all genuine deforestation. For those pixels we fall through
    to AEF or pass the mask to let XGBoost decide.
    """
    aef_deq = dequantize_aef(aef_raw)          # (64, H, W) float32

    # NoData pixels
    nodata_mask = np.all(~np.isfinite(aef_deq), axis=0)   # (H, W)

    # Tree cover: either A16 or A23 > 0 after dequantization
    with np.errstate(all="ignore"):
        tree_dim_vals = np.nanmax(aef_deq[AEF_TREE_COVER_DIMS], axis=0)
    aef_tree = np.where(np.isfinite(tree_dim_vals), tree_dim_vals > 0.0, False)

    # Hard exclusions: permanent water, bare land, built-up, NoData
    with np.errstate(all="ignore"):
        water_val   = aef_deq[AEF_WATER_DIM]
        bare_val    = aef_deq[AEF_BARE_DIM]
        buildup_val = np.nanmax(aef_deq[AEF_BUILDUP_DIMS], axis=0)

    water_excl   = np.where(np.isfinite(water_val),   water_val   > 0.10, False)
    bare_excl    = np.where(np.isfinite(bare_val),     bare_val    > 0.10, False)
    buildup_excl = np.where(np.isfinite(buildup_val),  buildup_val > 0.15, False)
    exclusion_mask = water_excl | bare_excl | buildup_excl | nodata_mask

    # NDVI forest signal — only valid when we have actual observations
    has_ndvi_data = ndvi_valid_frac_map > 0.05   # ≥ 5% cloud-free observations
    ndvi_forest   = has_ndvi_data & (ndvi_max_map > forest_ndvi_th)

    # No NDVI data (fully cloud-masked): pass through to XGBoost result
    no_ndvi_data  = ~has_ndvi_data

    # Forest mask: AEF tree-cover OR NDVI evidence OR no NDVI data (cloudy region)
    forest_mask = (aef_tree | ndvi_forest | no_ndvi_data) & ~exclusion_mask

    return forest_mask, exclusion_mask


# ─── Vectorized feature extraction ───────────────────────────────────────────

def _ts_stats_vectorized(ts_hw: np.ndarray, prefix: str) -> dict:
    """Temporal statistics for a (T, H, W) cube. Returns {prefix_stat: (H, W)}."""
    T, H, W = ts_hw.shape
    valid    = np.isfinite(ts_hw)
    n_valid  = valid.sum(axis=0).astype(np.float32)

    with np.errstate(all="ignore"):
        mean_v = np.nanmean(ts_hw, axis=0)
        std_v  = np.nanstd(ts_hw,  axis=0)
        min_v  = np.nanmin(ts_hw,  axis=0)
        max_v  = np.nanmax(ts_hw,  axis=0)

    valid_frac = n_valid / T
    diffs      = np.diff(ts_hw, axis=0)
    max_drop   = np.nanmin(diffs, axis=0)

    t_vals = np.arange(T, dtype=np.float32)[:, None, None]
    t_w    = np.where(valid, t_vals, np.nan)
    v_w    = np.where(valid, ts_hw,  np.nan)
    n_safe = np.where(n_valid >= 3, n_valid, np.nan)

    t_mean = np.nansum(t_w, axis=0) / n_safe
    v_mean = np.nansum(v_w, axis=0) / n_safe
    cov    = np.nansum((t_w - t_mean[None]) * (v_w - v_mean[None]), axis=0) / n_safe
    var_t  = np.nansum((t_w - t_mean[None]) ** 2, axis=0) / n_safe + 1e-8
    slope  = np.where(n_valid >= 3, cov / var_t, np.nan)

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


def extract_full_tile_features(tile_id: str, cfg: dict, is_test: bool = True):
    """
    Build (H*W, 85) feature matrix for an entire tile.
    Returns: (feat_matrix, transform, crs, H, W, feat_cols,
              s1_files, vv_cube, ndvi_stats, vv_stats, aef_raw)
    """
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
        aef_raw   = src.read()          # (64, H, W) uint8 — keep raw for dequantize
        transform = src.transform
        crs       = src.crs
        H, W      = src.height, src.width

    aef_data = normalize_aef(aef_raw)   # (64, H, W) float32 — used for XGBoost features
    logger.info(f"[{tile_id}] Grid: {H}×{W} = {H*W:,} pixels")

    # ── S2 NDVI + NBR cubes ───────────────────────────────────────────────
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
                        out_shape=(4, H, W), resampling=Resampling.bilinear,
                    ).astype(np.float32)
                else:
                    data = src.read(
                        [fe["s2_red_band"], fe["s2_nir_band"],
                         fe["s2_swir2_band"], fe["s2_scl_band"]]
                    ).astype(np.float32)
        except Exception as e:
            logger.warning(f"[{tile_id}] S2 {year}-{month:02d} error: {e}")
            continue

        red   = data[0] / fe["s2_scale_factor"]
        nir   = data[1] / fe["s2_scale_factor"]
        swir2 = data[2] / fe["s2_scale_factor"]
        scl   = data[3]
        valid_scl = np.isin(scl.astype(np.int32), fe["scl_valid_values"])

        ndvi_cube[t_idx] = np.where(valid_scl, spectral_index(nir, red),  np.nan)
        nbr_cube[t_idx]  = np.where(valid_scl, spectral_index(nir, swir2), np.nan)

    ndvi_stats = _ts_stats_vectorized(ndvi_cube, "ndvi")
    nbr_stats  = _ts_stats_vectorized(nbr_cube,  "nbr")
    logger.info(f"[{tile_id}] S2 done ({T_s2} months)")

    # ── S1 VV-dB cube ─────────────────────────────────────────────────────
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
            logger.warning(f"[{tile_id}] S1 {year}-{month:02d} error: {e}")
            continue

        if fe["s1_db_conversion"]:
            vv_cube[t_idx] = np.where(vv > fe["s1_db_eps"],
                                      10.0 * np.log10(vv + fe["s1_db_eps"]), np.nan)
        else:
            vv_cube[t_idx] = np.where(vv > 0, vv, np.nan)

    vv_stats = _ts_stats_vectorized(vv_cube, "vv")
    logger.info(f"[{tile_id}] S1 done ({T_s1} months)")

    # ── Feature matrix (H*W, 85) ──────────────────────────────────────────
    feature_parts = {}
    for i in range(64):
        feature_parts[f"aef_{i}"] = aef_data[i].ravel()
    for d in [ndvi_stats, nbr_stats, vv_stats]:
        for k, v in d.items():
            feature_parts[k] = v.ravel()

    feat_matrix  = np.stack(list(feature_parts.values()), axis=1)
    feature_cols = list(feature_parts.keys())

    return (feat_matrix, transform, crs, H, W, feature_cols,
            s1_files, vv_cube, ndvi_stats, vv_stats, aef_raw)


def predict_year_vectorized(vv_cube: np.ndarray, s1_files: list,
                             start_year: int, fallback_year: int,
                             max_year: int = 2024) -> np.ndarray:
    """
    Per-pixel year of largest VV-dB single-step drop in [start_year, max_year].
    Returns (H, W) int32 array.
    """
    T, H, W = vv_cube.shape
    years   = np.array([yr for yr, _, _ in s1_files], dtype=np.int32)

    post_mask = (years >= start_year) & (years <= max_year)
    post_idx  = np.where(post_mask)[0]

    if len(post_idx) < 2:
        return np.full((H, W), fallback_year, dtype=np.int32)

    diff_cube  = np.diff(vv_cube[post_idx], axis=0)         # (K-1, H, W)
    diff_years = years[post_idx[1:]]

    filled     = np.where(np.isfinite(diff_cube), diff_cube, np.inf)
    drop_t_idx = np.argmin(filled, axis=0)                   # (H, W)

    year_map   = diff_years[drop_t_idx]
    any_valid  = np.isfinite(diff_cube).any(axis=0)
    year_map   = np.where(any_valid, year_map, fallback_year)
    year_map   = np.clip(year_map, start_year, max_year)

    return year_map.astype(np.int32)


def filter_polygons_by_area(shapes_list: list, crs,
                             min_area_ha: float) -> list:
    """
    Filter out polygons with UTM area < min_area_ha.
    Returns filtered list of (geom_dict, value) tuples.
    """
    if not shapes_list:
        return []

    geoms = [shapely.geometry.shape(gd) for gd, _ in shapes_list]
    gdf   = gpd.GeoDataFrame(geometry=geoms, crs=crs)

    # Reproject to UTM for metric-accurate area
    utm_crs  = gdf.estimate_utm_crs()
    gdf_utm  = gdf.to_crs(utm_crs)
    keep_idx = (gdf_utm.area / 10_000 >= min_area_ha)

    return [shapes_list[i] for i, keep in enumerate(keep_idx) if keep]


def discover_test_tiles(cfg: dict) -> list:
    aef_test_dir = Path(cfg["paths_test"]["aef"])
    year  = cfg["aef"]["year"]
    tiles = []
    for f in sorted(aef_test_dir.glob(f"*_{year}.tiff")) + \
             sorted(aef_test_dir.glob(f"*_{year}.tif")):
        stem    = f.stem
        tile_id = stem[:-(len(str(year)) + 1)]
        tiles.append(tile_id)
    return sorted(set(tiles))


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

    ckpt_dir        = Path(cfg["paths"]["checkpoints"])
    model_art       = joblib.load(ckpt_dir / "xgboost_baseline.joblib")
    model           = model_art["model"]
    train_feat_cols = model_art["feature_cols"]

    logger.info(f"Loaded XGBoost model | threshold={threshold:.2f}")
    logger.info(f"Forest NDVI threshold : ndvi_max > {forest_ndvi_th}")
    logger.info(f"VV change threshold   : vv_max_drop < {vv_drop_th} dB")
    logger.info(f"Min area filter       : {min_area_ha} ha (UTM-projected)")
    logger.info(f"Year cap              : {max_year}")
    logger.info(f"AEF tree-cover dims   : {[d+1 for d in AEF_TREE_COVER_DIMS]} (A-notation)")

    test_tiles = discover_test_tiles(cfg)
    if not test_tiles:
        logger.error("No test tiles found — check paths_test.aef in config.yaml.")
        sys.exit(1)
    logger.info(f"Test tiles ({len(test_tiles)}): {test_tiles}")

    all_geojson_features = []

    for tile_id in tqdm(test_tiles, desc="Generating submissions"):
        logger.info(f"\n[{tile_id}] Extracting features …")
        result = extract_full_tile_features(tile_id, cfg, is_test=True)
        if result is None:
            continue

        (feat_matrix, transform, crs, H, W, feat_cols,
         s1_files, vv_cube, ndvi_stats, vv_stats, aef_raw) = result

        # ── Align feature columns to training order ────────────────────────
        col_to_idx = {c: i for i, c in enumerate(feat_cols)}
        try:
            ordered_idx = [col_to_idx[c] for c in train_feat_cols]
        except KeyError as e:
            logger.error(f"[{tile_id}] Feature column mismatch: {e}")
            continue
        X = feat_matrix[:, ordered_idx].astype(np.float32)

        # ── XGBoost inference ─────────────────────────────────────────────
        logger.info(f"[{tile_id}] Running XGBoost on {H*W:,} pixels …")
        probs  = model.predict_proba(X)[:, 1]
        binary = (probs >= threshold).astype(np.uint8).reshape(H, W)
        logger.info(f"[{tile_id}] Pre-filter positives: {int(binary.sum()):,}")

        # ── Forest + exclusion masks using AEF (globally valid) ───────────
        ndvi_max_map       = ndvi_stats["ndvi_max"].reshape(H, W)
        ndvi_valid_frac_map = ndvi_stats["ndvi_valid_frac"].reshape(H, W)
        forest_mask, exclusion_mask = build_forest_and_exclusion_masks(
            aef_raw, ndvi_max_map, ndvi_valid_frac_map, forest_ndvi_th
        )
        binary = (binary.astype(bool) & forest_mask & ~exclusion_mask).astype(np.uint8)
        logger.info(f"[{tile_id}] After forest+exclusion mask: {int(binary.sum()):,}")

        # ── VV change-signal filter ────────────────────────────────────────
        vv_drop_map = vv_stats["vv_max_drop"].reshape(H, W)
        binary      = binary & (vv_drop_map < vv_drop_th).astype(np.uint8)
        logger.info(f"[{tile_id}] After VV change filter: {int(binary.sum()):,}")

        if int(binary.sum()) == 0:
            logger.warning(f"[{tile_id}] No deforestation after filters — skipping")
            continue

        # ── Year prediction ────────────────────────────────────────────────
        year_map = predict_year_vectorized(
            vv_cube, s1_files,
            start_year=sub_cfg["year_pred_start_year"],
            fallback_year=sub_cfg["fallback_year"],
            max_year=max_year,
        )
        year_map = np.where(binary, year_map, 0).astype(np.int32)

        # ── Write per-tile rasters ─────────────────────────────────────────
        pred_raster_path = out_dir / f"pred_{tile_id}.tif"
        year_raster_path = out_dir / f"year_{tile_id}.tif"

        meta = dict(driver="GTiff", dtype="uint8", count=1,
                    height=H, width=W, crs=crs, transform=transform,
                    compress="lzw")
        with rasterio.open(pred_raster_path, "w", **meta) as dst:
            dst.write(binary, 1)

        with rasterio.open(year_raster_path, "w", **{**meta, "dtype": "int32"}) as dst:
            dst.write(year_map, 1)

        # ── Vectorise → area filter → per-polygon year ────────────────────
        shapes_list = [
            (gd, int(v))
            for gd, v in rio_features.shapes(binary, mask=binary, transform=transform)
            if int(v) > 0
        ]
        if not shapes_list:
            logger.warning(f"[{tile_id}] shapes() returned nothing")
            continue

        # Official area filter: remove polygons < min_area_ha (uses UTM projection)
        shapes_list = filter_polygons_by_area(shapes_list, crs, min_area_ha)
        logger.info(f"[{tile_id}] After {min_area_ha} ha area filter: {len(shapes_list)} polygons")

        if not shapes_list:
            logger.warning(f"[{tile_id}] All polygons < {min_area_ha} ha — skipping")
            continue

        # Vectorised mode-year per polygon
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
            all_geojson_features.append({
                "type": "Feature",
                "geometry": shapely.geometry.mapping(shapely.geometry.shape(gd)),
                "properties": {
                    "tile_id":   tile_id,
                    "time_step": mode_years.get(pid, fallback),
                },
            })

        logger.info(f"[{tile_id}] Final polygons: {len(shapes_list)}")

    if not all_geojson_features:
        logger.error("No predictions generated — check data paths and thresholds.")
        sys.exit(1)

    gdf = gpd.GeoDataFrame.from_features(all_geojson_features)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    gdf_4326 = gdf.to_crs("EPSG:4326")

    submission_path = out_dir / "submission.geojson"
    gdf_4326.to_file(submission_path, driver="GeoJSON")

    # ── Summary ───────────────────────────────────────────────────────────
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
    print(f"  Forest mask          : AEF tree-cover (A16/A23) | ndvi_max > {forest_ndvi_th}")
    print(f"  VV change filter     : vv_max_drop < {vv_drop_th} dB")
    print(f"  Min area             : {min_area_ha} ha (UTM)")
    print(f"  Year distribution (time_step):")
    for yr, cnt in year_counts.items():
        print(f"    {yr}: {cnt:,} polygons")
    print(f"\n✓ Saved → {submission_path}")
    print("\nDONE — Submission generation complete.")


if __name__ == "__main__":
    main()
