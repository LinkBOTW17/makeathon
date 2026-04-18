"""
Augment training_features.csv with AEF year-over-year change scores.

The existing CSV has aef_00..aef_63 from only the LATEST year's embedding.
This script loads ALL annual AEF files per tile, computes per-polygon mean
embeddings for each year, then adds:

  aef_delta_l2_{Y}        — L2 distance between year Y and Y-1 embeddings
  aef_delta_cos_{Y}       — cosine distance between year Y and Y-1
  aef_from2020_l2_{Y}     — L2 distance from 2020 baseline to year Y
  aef_from2020_cos_{Y}    — cosine distance from 2020 baseline to year Y
  aef_max_delta_l2        — max L2 delta across all post-2020 year pairs
  aef_max_delta_cos       — max cosine delta across all post-2020 year pairs
  aef_change_year         — year with the largest L2 delta (deforestation year estimate)

Usage (from makeathon root):
    python deforestation/augment_aef_changes.py \\
        --input  results/training_features.csv \\
        --output results/training_features_aef.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import mapping

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from mislabel_detection_v2 import (  # noqa: E402
    DATA_ROOT,
    build_consensus_polygons,
    load_tile,
)

AEF_YEARS = [2020, 2021, 2022, 2023, 2024]
DELTA_YEARS = [2021, 2022, 2023, 2024]   # pairs: Y vs Y-1


# ── AEF loading ───────────────────────────────────────────────────────────────

def _load_aef_year(
    path: Path,
    ref_transform,
    ref_crs,
    ref_shape: tuple[int, int],
) -> np.ndarray:
    """Load one annual AEF file and reproject to S2 reference grid.
    Returns array of shape (n_bands, H, W)."""
    with rasterio.open(path) as src:
        n_bands = src.count
        out = np.zeros((n_bands, *ref_shape), dtype=np.float32)
        for b in range(1, n_bands + 1):
            reproject(
                source=src.read(b).astype(np.float32),
                destination=out[b - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
    return out


def load_aef_all_years(
    tile_id: str,
    ref_transform,
    ref_crs,
    ref_shape: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Return {year: (n_bands, H, W)} for all available annual AEF files."""
    aef_dir = DATA_ROOT / "aef-embeddings" / "train"
    aef_by_year: dict[int, np.ndarray] = {}

    for f in aef_dir.glob(f"{tile_id}_*.tiff"):
        stem = f.stem                     # e.g. "48PWV_7_8_2022"
        # Year is always the last underscore-separated token
        try:
            year = int(stem.split("_")[-1])
        except ValueError:
            continue
        if year not in AEF_YEARS:
            continue
        aef_by_year[year] = _load_aef_year(f, ref_transform, ref_crs, ref_shape)

    return aef_by_year


# ── per-polygon embedding computation ────────────────────────────────────────

def _polygon_mask(geom, transform, shape) -> np.ndarray:
    return ~geometry_mask(
        [mapping(geom)],
        transform=transform,
        invert=False,
        out_shape=shape,
    )


def _mean_embedding(aef: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Mean of valid pixels within polygon for each band. Returns (n_bands,) or None."""
    # aef: (n_bands, H, W)
    pixels = aef[:, mask]                     # (n_bands, n_pixels)
    finite = np.isfinite(pixels).all(axis=0)  # pixels where all bands are finite
    valid = pixels[:, finite]
    if valid.shape[1] == 0:
        return None
    return valid.mean(axis=1)                 # (n_bands,)


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(1.0 - np.dot(a, b) / (na * nb))


def compute_aef_change_features(
    geom,
    aef_by_year: dict[int, np.ndarray],
    ref_transform,
    ref_shape: tuple[int, int],
) -> dict[str, float]:
    """
    For one polygon geometry, compute all AEF change features.
    Returns a flat dict of feature_name → float (NaN where data missing).
    """
    mask = _polygon_mask(geom, ref_transform, ref_shape)
    if mask.sum() == 0:
        return {}

    # Mean embedding per available year
    emb: dict[int, np.ndarray] = {}
    for yr, aef in aef_by_year.items():
        e = _mean_embedding(aef, mask)
        if e is not None:
            emb[yr] = e

    feats: dict[str, float] = {}

    # Year-over-year deltas
    for yr in DELTA_YEARS:
        prev = yr - 1
        if yr in emb and prev in emb:
            feats[f"aef_delta_l2_{yr}"]  = _l2(emb[yr], emb[prev])
            feats[f"aef_delta_cos_{yr}"] = _cosine_dist(emb[yr], emb[prev])
        else:
            feats[f"aef_delta_l2_{yr}"]  = np.nan
            feats[f"aef_delta_cos_{yr}"] = np.nan

    # Delta from 2020 baseline
    if 2020 in emb:
        for yr in DELTA_YEARS:
            if yr in emb:
                feats[f"aef_from2020_l2_{yr}"]  = _l2(emb[yr], emb[2020])
                feats[f"aef_from2020_cos_{yr}"] = _cosine_dist(emb[yr], emb[2020])
            else:
                feats[f"aef_from2020_l2_{yr}"]  = np.nan
                feats[f"aef_from2020_cos_{yr}"] = np.nan
    else:
        for yr in DELTA_YEARS:
            feats[f"aef_from2020_l2_{yr}"]  = np.nan
            feats[f"aef_from2020_cos_{yr}"] = np.nan

    # Summary: max delta and change year
    l2_deltas = {yr: feats[f"aef_delta_l2_{yr}"] for yr in DELTA_YEARS}
    valid_deltas = {yr: v for yr, v in l2_deltas.items() if np.isfinite(v)}

    if valid_deltas:
        feats["aef_max_delta_l2"]  = max(valid_deltas.values())
        feats["aef_change_year"]   = float(max(valid_deltas, key=valid_deltas.get))
    else:
        feats["aef_max_delta_l2"] = np.nan
        feats["aef_change_year"]  = np.nan

    cos_deltas = {yr: feats[f"aef_delta_cos_{yr}"] for yr in DELTA_YEARS}
    valid_cos  = {yr: v for yr, v in cos_deltas.items() if np.isfinite(v)}
    feats["aef_max_delta_cos"] = max(valid_cos.values()) if valid_cos else np.nan

    return feats


# ── main augmentation ─────────────────────────────────────────────────────────

def augment(
    input_csv: Path,
    output_csv: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    if "tile_id" not in df.columns:
        raise ValueError("CSV must have a 'tile_id' column.")

    tiles = df["tile_id"].unique()
    if verbose:
        print(f"\nAugmenting {len(df)} polygons across {len(tiles)} tile(s) …\n")

    all_new_feats: list[dict] = [{}] * len(df)  # placeholder per row

    for tile_id in tiles:
        tile_mask = df["tile_id"] == tile_id
        tile_indices = df.index[tile_mask].tolist()

        # Load tile for reference grid
        try:
            td = load_tile(tile_id)
        except FileNotFoundError as e:
            print(f"  {tile_id}: SKIP load_tile — {e}")
            continue

        # Load all annual AEF files
        aef_by_year = load_aef_all_years(
            tile_id, td.ref_transform, td.ref_crs, td.ref_shape
        )
        if not aef_by_year:
            print(f"  {tile_id}: SKIP — no AEF files found")
            continue

        years_found = sorted(aef_by_year.keys())
        if verbose:
            print(f"  {tile_id}: AEF years {years_found}", end=" … ")

        # Rebuild consensus polygons to get geometries (same order as CSV)
        try:
            gdf = build_consensus_polygons(td)
        except RuntimeError as e:
            print(f"SKIP build_consensus — {e}")
            continue

        n_poly = len(gdf)
        n_rows = len(tile_indices)
        if n_poly != n_rows and verbose:
            print(
                f"\n    [warn] {tile_id}: polygon count mismatch "
                f"rebuilt={n_poly} csv={n_rows}, aligning on min"
            )
        n = min(n_poly, n_rows)

        n_ok = 0
        for i in range(n):
            geom = gdf.iloc[i].geometry
            feats = compute_aef_change_features(
                geom, aef_by_year, td.ref_transform, td.ref_shape
            )
            all_new_feats[tile_indices[i]] = feats
            if feats:
                n_ok += 1

        if verbose:
            print(f"{n_ok}/{n} polygons augmented")

    # Build new-feature DataFrame aligned to original index
    new_cols = pd.DataFrame(all_new_feats, index=df.index)

    # Drop stale single-year AEF columns (aef_00 … aef_63) — they were from
    # the latest year only and are now superseded by the change features
    stale = [c for c in df.columns if c.startswith("aef_") and c[4:].isdigit()]
    if stale and verbose:
        print(f"\nDropping {len(stale)} stale single-year AEF columns.")
    df = df.drop(columns=stale)

    df = pd.concat([df, new_cols], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if verbose:
        new_feature_names = [c for c in new_cols.columns if new_cols[c].notna().any()]
        print(f"\nNew AEF features added  : {len(new_feature_names)}")
        print(f"Total features in output: {len(df.columns)}")
        print(f"Saved → {output_csv}")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Augment training CSV with AEF year-over-year change features."
    )
    p.add_argument(
        "--input", default="results/training_features.csv",
        help="Input CSV from apply_corrections.py (default: results/training_features.csv)",
    )
    p.add_argument(
        "--output", default="results/training_features_aef.csv",
        help="Output augmented CSV (default: results/training_features_aef.csv)",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    augment(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        verbose=not args.quiet,
    )
