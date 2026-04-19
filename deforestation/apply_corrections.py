"""
Data Manipulation — Apply Confident Learning Corrections
=========================================================

Reads results/mislabels_v2.csv (output of mislabel_detection_v2.py) and:

  rasters   : Burns corrected labels into per-tile GeoTIFF rasters.
              Each raster has two bands:
                Band 1 — original consensus label  (0/1)
                Band 2 — corrected label           (0/1)
              Saved under results/corrected_labels/{tile_id}_corrected.tif

  features  : Re-extracts per-polygon features and merges with corrected
              labels; exports a ML-ready CSV (and optional GeoJSON) under
              results/training_features.{csv|geojson}

  all       : Both of the above.

Usage (from makeathon root):
    python deforestation/apply_corrections.py \\
        --mislabels results/mislabels_v2.csv \\
        --mode all \\
        --score-threshold 0.5 \\
        --out-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize, shapes
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from shapely.geometry import mapping, shape

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── resolve sibling module ────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from mislabel_detection_v2 import (  # noqa: E402
    DATA_ROOT,
    TileData,
    build_consensus_polygons,
    extract_features,
    load_tile,
)

RESULTS_DIR = Path("results")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_mislabels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"tile_id", "given_label", "suggested_label", "flagged", "mislabel_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mislabels CSV is missing columns: {missing}")
    df["flagged"] = df["flagged"].astype(bool)
    return df


def _match_polygons_to_rows(
    gdf: gpd.GeoDataFrame,
    tile_rows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match rebuild GeoDataFrame rows to mislabels CSV rows by position.
    If counts differ, align on the smaller side and warn.
    """
    n_poly = len(gdf)
    n_rows = len(tile_rows)
    if n_poly != n_rows:
        print(
            f"    [warn] polygon count mismatch: rebuilt={n_poly}, csv={n_rows}. "
            "Aligning on min — some polygons will be unmatched."
        )
    n = min(n_poly, n_rows)
    matched = tile_rows.iloc[:n].copy().reset_index(drop=True)
    matched_geom = gdf.iloc[:n].copy().reset_index(drop=True)
    matched["geometry"] = matched_geom.geometry.values
    matched["original_crs"] = gdf.crs.to_string()
    return matched


# ── mode: rasters ─────────────────────────────────────────────────────────────

def burn_corrected_rasters(
    mislabels: pd.DataFrame,
    score_threshold: float,
    out_dir: Path,
    verbose: bool = True,
) -> list[Path]:
    """
    For each tile: rebuild consensus polygons, apply CL corrections above the
    score threshold, burn both the original and corrected labels into a 2-band
    GeoTIFF.

    Band 1 = original consensus label
    Band 2 = corrected label (flipped where flagged AND score >= threshold)

    Returns list of written raster paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    tiles = mislabels["tile_id"].unique()
    if verbose:
        print(f"\n[rasters] Processing {len(tiles)} tile(s) …")

    for tile_id in tiles:
        tile_rows = mislabels[mislabels["tile_id"] == tile_id].reset_index(drop=True)

        # ── load tile ─────────────────────────────────────────────────────────
        try:
            td = load_tile(tile_id)
        except FileNotFoundError as e:
            print(f"  {tile_id}: SKIP — {e}")
            continue

        # ── rebuild consensus polygons ────────────────────────────────────────
        try:
            gdf = build_consensus_polygons(td)
        except RuntimeError as e:
            print(f"  {tile_id}: SKIP — {e}")
            continue

        matched = _match_polygons_to_rows(gdf, tile_rows)

        # ── decide final label per polygon ────────────────────────────────────
        def _final_label(row) -> int:
            if row["flagged"] and row["mislabel_score"] >= score_threshold:
                return int(row["suggested_label"])
            return int(row["given_label"])

        matched["final_label"] = matched.apply(_final_label, axis=1)

        n_flipped = (matched["final_label"] != matched["given_label"]).sum()

        # ── burn to raster ────────────────────────────────────────────────────
        H, W = td.ref_shape
        original_raster  = np.zeros((H, W), dtype=np.uint8)
        corrected_raster = np.zeros((H, W), dtype=np.uint8)

        for _, row in matched.iterrows():
            geom = row["geometry"]
            orig = int(row["given_label"])
            corr = int(row["final_label"])

            mask = ~geometry_mask(
                [mapping(geom)],
                transform=td.ref_transform,
                invert=False,
                out_shape=(H, W),
            )
            original_raster[mask]  = orig
            corrected_raster[mask] = corr

        out_path = out_dir / f"{tile_id}_corrected.tif"
        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=H, width=W,
            count=2,
            dtype=np.uint8,
            crs=td.ref_crs,
            transform=td.ref_transform,
            compress="lzw",
        ) as dst:
            dst.write(original_raster,  1)
            dst.write(corrected_raster, 2)
            dst.update_tags(
                band_1="original_consensus_label",
                band_2="corrected_label",
                score_threshold=str(score_threshold),
                n_polygons=str(len(matched)),
                n_flipped=str(n_flipped),
            )

        written.append(out_path)
        if verbose:
            print(f"  {tile_id}: {len(matched)} polygons, {n_flipped} flipped → {out_path.name}")

    return written


# ── mode: features ────────────────────────────────────────────────────────────

def export_training_features(
    mislabels: pd.DataFrame,
    score_threshold: float,
    out_dir: Path,
    geojson: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Re-extract per-polygon features for every tile, merge with corrected labels,
    and export a ML-ready CSV (and optionally GeoJSON) to out_dir.

    The exported table has:
      - tile_id, geometry (GeoJSON only)
      - given_label          — original consensus label
      - suggested_label      — CL suggestion
      - corrected_label      — label to use for training
      - flagged, mislabel_score, p_deforestation
      - all harmonic / NDVI / SAR / AEF features
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = mislabels["tile_id"].unique()

    if verbose:
        print(f"\n[features] Extracting features for {len(tiles)} tile(s) …")

    all_records: list[dict] = []
    all_geoms: list = []
    all_crs: list[str] = []

    for tile_id in tiles:
        tile_rows = mislabels[mislabels["tile_id"] == tile_id].reset_index(drop=True)

        try:
            td = load_tile(tile_id)
        except FileNotFoundError as e:
            print(f"  {tile_id}: SKIP — {e}")
            continue

        try:
            gdf = build_consensus_polygons(td)
        except RuntimeError as e:
            print(f"  {tile_id}: SKIP — {e}")
            continue

        matched = _match_polygons_to_rows(gdf, tile_rows)

        for _, row in matched.iterrows():
            feats = extract_features(row["geometry"], td)
            if not feats:
                continue

            # Decide corrected label
            use_correction = row["flagged"] and row["mislabel_score"] >= score_threshold
            corrected = int(row["suggested_label"]) if use_correction else int(row["given_label"])

            record: dict = {
                "tile_id":          tile_id,
                "given_label":      int(row["given_label"]),
                "suggested_label":  int(row["suggested_label"]),
                "corrected_label":  corrected,
                "flagged":          bool(row["flagged"]),
                "mislabel_score":   float(row["mislabel_score"]),
            }
            # Carry through extra CL columns if present
            for col in ("p_deforestation", "source_agreement", "n_sources"):
                if col in row.index:
                    record[col] = row[col]

            record.update(feats)
            all_records.append(record)
            # Store geometry in tile CRS; reproject to 4326 at the end
            all_geoms.append(row["geometry"])
            all_crs.append(td.ref_crs.to_string())

        if verbose:
            n_flip = ((matched["flagged"]) & (matched["mislabel_score"] >= score_threshold)).sum()
            print(f"  {tile_id}: {len(matched)} polygons, {n_flip} corrections applied")

    if not all_records:
        raise RuntimeError("No features extracted — check tile data and mislabels CSV.")

    df = pd.DataFrame(all_records)

    # ── summary stats ──────────────────────────────────────────────────────────
    if verbose:
        total = len(df)
        n_flipped = (df["corrected_label"] != df["given_label"]).sum()
        print(f"\n  Total polygons : {total}")
        print(f"  Labels flipped : {n_flipped} ({100*n_flipped/total:.1f} %)")
        print(f"  corrected_label=1 : {(df['corrected_label']==1).sum()}")
        print(f"  corrected_label=0 : {(df['corrected_label']==0).sum()}")

    # ── write CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / "training_features.csv"
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\n  CSV  → {csv_path}")

    # ── optionally write GeoJSON ───────────────────────────────────────────────
    if geojson and all_geoms:
        # Build per-tile GDFs then concat in EPSG:4326
        pieces: list[gpd.GeoDataFrame] = []
        # Group records by tile for correct CRS assignment
        tile_indices: dict[str, list[int]] = {}
        for i, rec in enumerate(all_records):
            tile_indices.setdefault(rec["tile_id"], []).append(i)

        for tile_id, idxs in tile_indices.items():
            try:
                td = load_tile(tile_id)
                crs = td.ref_crs
            except Exception:
                crs = "EPSG:4326"
            sub = gpd.GeoDataFrame(
                [all_records[i] for i in idxs],
                geometry=[all_geoms[i] for i in idxs],
                crs=crs,
            ).to_crs("EPSG:4326")
            pieces.append(sub)

        gdf_out = pd.concat(pieces, ignore_index=True)
        geojson_path = out_dir / "training_features.geojson"
        gdf_out.to_file(geojson_path, driver="GeoJSON")
        if verbose:
            print(f"  GeoJSON → {geojson_path}")

    return csv_path


# ── mode: stats ───────────────────────────────────────────────────────────────

def print_stats(mislabels: pd.DataFrame, score_threshold: float) -> None:
    total = len(mislabels)
    flagged = mislabels["flagged"].sum()
    above_thresh = (mislabels["flagged"] & (mislabels["mislabel_score"] >= score_threshold)).sum()

    print(f"\n{'='*60}")
    print("Mislabel Summary")
    print(f"{'='*60}")
    print(f"  Total polygons        : {total}")
    print(f"  Flagged (CL)          : {flagged}  ({100*flagged/total:.1f} %)")
    print(f"  Will be corrected     : {above_thresh}  (score >= {score_threshold})")

    for tile_id in sorted(mislabels["tile_id"].unique()):
        sub = mislabels[mislabels["tile_id"] == tile_id]
        n_flag  = sub["flagged"].sum()
        n_apply = (sub["flagged"] & (sub["mislabel_score"] >= score_threshold)).sum()
        print(f"    {tile_id:20s}  polys={len(sub):4d}  flagged={n_flag:3d}  will_correct={n_apply:3d}")

    print(f"{'='*60}")

    by_type = mislabels[mislabels["flagged"] & (mislabels["mislabel_score"] >= score_threshold)]
    n_1to0 = ((by_type["given_label"] == 1) & (by_type["suggested_label"] == 0)).sum()
    n_0to1 = ((by_type["given_label"] == 0) & (by_type["suggested_label"] == 1)).sum()
    print(f"  Corrections breakdown:")
    print(f"    label 1 → 0 (false positives removed) : {n_1to0}")
    print(f"    label 0 → 1 (missed events added)     : {n_0to1}")

    print(f"\n  Score distribution of flagged polygons:")
    if flagged > 0:
        scores = mislabels.loc[mislabels["flagged"], "mislabel_score"]
        for q in [0.25, 0.50, 0.75, 0.90, 1.0]:
            print(f"    p{int(q*100):3d} = {np.quantile(scores, q):.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply Confident Learning corrections to training label data."
    )
    p.add_argument(
        "--mislabels", default="results/mislabels_v2.csv",
        help="Path to the mislabels CSV from mislabel_detection_v2.py",
    )
    p.add_argument(
        "--mode", choices=["rasters", "features", "all", "stats"],
        default="all",
        help=(
            "rasters  — burn corrected label rasters per tile\n"
            "features — export ML-ready feature CSV (+GeoJSON)\n"
            "stats    — print summary only (no files written)\n"
            "all      — rasters + features  [default]"
        ),
    )
    p.add_argument(
        "--score-threshold", type=float, default=0.5,
        help="Only apply a correction when mislabel_score >= this value (default: 0.5)",
    )
    p.add_argument(
        "--out-dir", default="results",
        help="Directory for output files (default: results/)",
    )
    p.add_argument(
        "--geojson", action="store_true",
        help="Also write a GeoJSON alongside the feature CSV (features / all mode)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-tile progress output",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    verbose = not args.quiet
    out_dir = Path(args.out_dir)

    mislabels_path = Path(args.mislabels)
    if not mislabels_path.exists():
        sys.exit(f"ERROR: mislabels file not found: {mislabels_path}")

    mislabels = _load_mislabels(mislabels_path)
    print_stats(mislabels, args.score_threshold)

    if args.mode in ("rasters", "all"):
        raster_dir = out_dir / "corrected_labels"
        written = burn_corrected_rasters(
            mislabels,
            score_threshold=args.score_threshold,
            out_dir=raster_dir,
            verbose=verbose,
        )
        print(f"\n[rasters] {len(written)} raster(s) written to {raster_dir}/")

    if args.mode in ("features", "all"):
        csv_path = export_training_features(
            mislabels,
            score_threshold=args.score_threshold,
            out_dir=out_dir,
            geojson=args.geojson,
            verbose=verbose,
        )
        print(f"\n[features] Done → {csv_path}")

    print("\nDone.")
