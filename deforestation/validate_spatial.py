"""
Spatial Validation — True pixel-level metrics on training tiles
===============================================================

Loads pixel-level predictions from results/predictions/{tile_id}_prediction.tif
and compares against the weak-label consensus GT (majority vote of RADD, GLAD-S2,
GLAD-L) with CL corrections applied from mislabels_v2.csv.

Metrics computed match the leaderboard definitions exactly:

  Union IoU     = TP / (TP + FP + FN)          [pixel area]
  Recall        = TP / (TP + FN)
  FPR           = FP / (TP + FP)               ← leaderboard definition
  Year Accuracy = correctly-dated TP / (TP+FP+FN) where year is from RADD/GLAD

Usage (from makeathon root):
    python deforestation/validate_spatial.py
    python deforestation/validate_spatial.py --pred-dir results/predictions --mislabels results/mislabels_v2.csv
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

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from mislabel_detection_v2 import (  # noqa: E402
    DATA_ROOT,
    TileData,
    build_consensus_polygons,
    load_tile,
    _reproject_onto,
)

# Days since 2015-01-01 → year (RADD encoding)
# 2020-01-01 = day 1827, 2021-01-01 = 2192, 2022-01-01 = 2557,
# 2023-01-01 = 2922, 2024-01-01 = 3287
_RADD_YEAR_EDGES = {2020: 1827, 2021: 2192, 2022: 2557, 2023: 2922, 2024: 3287, 2025: 9999}

# GLAD-S2 alertDate: days since 2014-12-31
# 2020-01-01 = day 1827, same offsets as RADD (both from ~same epoch)
_GLADS2_YEAR_EDGES = {2020: 1827, 2021: 2192, 2022: 2557, 2023: 2922, 2024: 3287, 2025: 9999}


def _days_to_year(days: np.ndarray, edges: dict) -> np.ndarray:
    """Convert integer day array to year array using threshold edges."""
    years = np.zeros_like(days, dtype=np.int16)
    for yr in sorted(edges.keys())[:-1]:
        lo = edges[yr]
        hi = edges[yr + 1]
        years[(days >= lo) & (days < hi)] = yr
    return years


# ── GT builder ────────────────────────────────────────────────────────────────

def build_gt_pixel_map(
    td: TileData,
    tile_rows: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build pixel-level GT binary map and year map from weak labels.
    Applies CL corrections from mislabels CSV if tile_rows is provided.

    Returns
    -------
    gt_binary : (H, W) uint8  — 1 = deforestation
    gt_year   : (H, W) int16  — estimated deforestation year (0 = unknown)
    """
    H, W = td.ref_shape
    labels_root = DATA_ROOT / "labels" / "train"

    def _load(path: Path) -> np.ndarray:
        return _reproject_onto(
            path, td.ref_transform, td.ref_crs, td.ref_shape,
            resampling=Resampling.nearest,
        )

    # ── raw weak label arrays ─────────────────────────────────────────────────
    radd_raw = np.zeros((H, W), dtype=np.float32)
    radd_days = np.zeros((H, W), dtype=np.int32)
    radd_path = labels_root / "radd" / f"radd_{td.tile_id}_labels.tif"
    if radd_path.exists():
        raw = _load(radd_path)
        conf = (raw // 10000).astype(np.int32)
        days = (raw % 10000).astype(np.int32)
        radd_raw  = ((conf >= 2) & (days >= 1827)).astype(np.float32)
        radd_days = np.where(radd_raw == 1, days, 0).astype(np.int32)

    glads2_raw  = np.zeros((H, W), dtype=np.float32)
    glads2_days = np.zeros((H, W), dtype=np.int32)
    gs2_alert = labels_root / "glads2" / f"glads2_{td.tile_id}_alert.tif"
    gs2_date  = labels_root / "glads2" / f"glads2_{td.tile_id}_alertDate.tif"
    if gs2_alert.exists():
        a = _load(gs2_alert)
        d = _load(gs2_date).astype(np.int32) if gs2_date.exists() else np.ones((H, W), dtype=np.int32) * 999
        glads2_raw  = ((a >= 2) & (d > 365)).astype(np.float32)
        glads2_days = np.where(glads2_raw == 1, d, 0).astype(np.int32)

    gladl_raw = np.zeros((H, W), dtype=np.float32)
    gladl_dir = labels_root / "gladl"
    layers = []
    for yy in range(21, 25):
        for cand in [gladl_dir / f"gladl_{td.tile_id}_alert{yy}.tif",
                     gladl_dir / td.tile_id / f"gladl_{td.tile_id}_alert{yy}.tif"]:
            if cand.exists():
                raw = _load(cand)
                layers.append(((raw >= 2).astype(np.float32), 2000 + yy))
                break
    if layers:
        gladl_raw = np.any(
            np.stack([l[0] for l in layers], axis=0), axis=0
        ).astype(np.float32)

    # ── GT year map from RADD/GLAD-S2 day fields (for year accuracy only) ────
    radd_year   = _days_to_year(radd_days,   _RADD_YEAR_EDGES)
    glads2_year = _days_to_year(glads2_days, _GLADS2_YEAR_EDGES)
    year_filled = np.where(radd_year > 0, radd_year, glads2_year)

    # ── Primary GT: burn corrected polygon labels ─────────────────────────────
    # The competition evaluates against human-annotated polygon labels, not
    # RADD/GLAD alerts. Using the polygon labels as GT aligns training with
    # the leaderboard evaluation.
    gt_binary = np.zeros((H, W), dtype=np.uint8)
    use_polygon_gt = False

    if tile_rows is not None and len(tile_rows) > 0:
        try:
            gdf = build_consensus_polygons(td)
            n_match = min(len(gdf), len(tile_rows))
            for i in range(n_match):
                row  = tile_rows.iloc[i]
                # Apply CL correction: use suggested_label if flagged + score >= 0.5
                use_corr  = bool(row.get("flagged", False)) and float(row.get("mislabel_score", 0.0)) >= 0.5
                corrected = int(row["suggested_label"]) if use_corr else int(row["given_label"])
                geom = gdf.iloc[i].geometry
                mask = ~geometry_mask(
                    [mapping(geom)],
                    transform=td.ref_transform,
                    invert=False,
                    out_shape=(H, W),
                )
                gt_binary[mask] = corrected
            use_polygon_gt = True
        except Exception:
            pass

    # Fallback: RADD/GLAD majority vote (when no polygon labels are available)
    if not use_polygon_gt:
        available = [arr for arr in [radd_raw, glads2_raw, gladl_raw] if arr.sum() > 0]
        if not available:
            return np.zeros((H, W), dtype=np.uint8), np.zeros((H, W), dtype=np.int16)
        n = len(available)
        vote_sum  = np.stack(available, axis=0).sum(axis=0)
        gt_binary = (vote_sum > n / 2).astype(np.uint8)

    gt_year = np.where(gt_binary == 1, year_filled, 0).astype(np.int16)
    return gt_binary, gt_year


# ── metrics ───────────────────────────────────────────────────────────────────

def spatial_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
) -> dict[str, float]:
    """
    Pixel-level spatial metrics matching leaderboard definitions.
    pred, gt: (H, W) uint8 binary arrays.
    """
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())

    iou    = tp / (tp + fp + fn + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    fpr    = fp / (tp + fp + 1e-9)   # leaderboard: FP / total predicted positive

    return {"iou": iou, "recall": recall, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn}


def year_accuracy(
    pred: np.ndarray,
    pred_year: np.ndarray,
    gt: np.ndarray,
    gt_year: np.ndarray,
) -> float:
    """
    Pixel-level approximation of Year Accuracy:
      correctly_dated_TP / (TP + FP + FN)

    A TP pixel is 'correctly dated' if pred_year == gt_year (and gt_year > 0).
    """
    tp_mask  = (pred == 1) & (gt == 1)
    year_ok  = tp_mask & (pred_year == gt_year) & (gt_year > 0)

    tp = int(tp_mask.sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    denom = tp + fp + fn

    return float(year_ok.sum()) / (denom + 1e-9)


# ── main ──────────────────────────────────────────────────────────────────────

def validate(
    pred_dir: Path,
    mislabels_csv: Path | None,
    verbose: bool = True,
) -> pd.DataFrame:

    pred_files = sorted(pred_dir.glob("*_prediction.tif"))
    if not pred_files:
        sys.exit(f"No prediction rasters found in {pred_dir}")

    mislabels = None
    if mislabels_csv and mislabels_csv.exists():
        mislabels = pd.read_csv(mislabels_csv)
        mislabels["flagged"] = mislabels["flagged"].astype(bool)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Spatial Validation — {len(pred_files)} tile(s)")
        print(f"{'='*60}\n")
        print(f"  {'Tile':<22} {'IoU':>6} {'Recall':>8} {'FPR':>6} {'YearAcc':>9} "
              f"{'GT_px':>8} {'Pred_px':>8} {'TP':>7} {'FP':>7} {'FN':>7}")
        print("  " + "-" * 90)

    rows = []
    total_tp = total_fp = total_fn = 0
    total_yr_ok = total_denom = 0

    for pred_path in pred_files:
        tile_id = pred_path.stem.replace("_prediction", "")

        # ── load prediction raster ────────────────────────────────────────────
        with rasterio.open(pred_path) as src:
            pred_binary  = src.read(1).astype(np.uint8)
            pred_yr_off  = src.read(2).astype(np.int16)
        pred_year = (pred_yr_off + 2020).astype(np.int16)
        pred_year[pred_binary == 0] = 0

        # ── load tile data + build GT ─────────────────────────────────────────
        try:
            td = load_tile(tile_id)
        except FileNotFoundError:
            if verbose:
                print(f"  {tile_id:<22} SKIP (no tile data)")
            continue

        # Reproject prediction to match ref grid if shapes differ
        H, W = td.ref_shape
        if pred_binary.shape != (H, W):
            reproj_bin  = np.zeros((H, W), dtype=np.uint8)
            reproj_year = np.zeros((H, W), dtype=np.int16)
            with rasterio.open(pred_path) as src:
                reproject(
                    source=src.read(1).astype(np.float32), destination=reproj_bin.astype(np.float32),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=td.ref_transform, dst_crs=td.ref_crs,
                    resampling=Resampling.nearest,
                )
                reproject(
                    source=src.read(2).astype(np.float32), destination=reproj_year.astype(np.float32),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=td.ref_transform, dst_crs=td.ref_crs,
                    resampling=Resampling.nearest,
                )
            pred_binary = (reproj_bin > 0).astype(np.uint8)
            pred_year   = (reproj_year + 2020).astype(np.int16)
            pred_year[pred_binary == 0] = 0

        tile_rows = None
        if mislabels is not None:
            tile_rows = mislabels[mislabels["tile_id"] == tile_id].reset_index(drop=True)

        gt_binary, gt_year = build_gt_pixel_map(td, tile_rows)

        # ── compute metrics ───────────────────────────────────────────────────
        m = spatial_metrics(pred_binary, gt_binary)
        yr_acc = year_accuracy(pred_binary, pred_year, gt_binary, gt_year)

        total_tp    += m["tp"]
        total_fp    += m["fp"]
        total_fn    += m["fn"]

        # Year accuracy numerator/denominator accumulation
        tp_mask  = (pred_binary == 1) & (gt_binary == 1)
        year_ok  = (tp_mask & (pred_year == gt_year) & (gt_year > 0)).sum()
        total_yr_ok  += int(year_ok)
        total_denom  += m["tp"] + m["fp"] + m["fn"]

        row = {
            "tile_id": tile_id,
            "iou":      round(m["iou"],    3),
            "recall":   round(m["recall"], 3),
            "fpr":      round(m["fpr"],    3),
            "year_acc": round(yr_acc,      3),
            "gt_px":    int(gt_binary.sum()),
            "pred_px":  int(pred_binary.sum()),
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
        }
        rows.append(row)

        if verbose:
            print(
                f"  {tile_id:<22} {m['iou']:>6.3f} {m['recall']:>8.3f} "
                f"{m['fpr']:>6.3f} {yr_acc:>9.3f} "
                f"{gt_binary.sum():>8,} {pred_binary.sum():>8,} "
                f"{m['tp']:>7,} {m['fp']:>7,} {m['fn']:>7,}"
            )

    # ── aggregate ─────────────────────────────────────────────────────────────
    agg_iou    = total_tp / (total_tp + total_fp + total_fn + 1e-9)
    agg_recall = total_tp / (total_tp + total_fn + 1e-9)
    agg_fpr    = total_fp / (total_tp + total_fp + 1e-9)
    agg_yr_acc = total_yr_ok / (total_denom + 1e-9)

    if verbose:
        print("  " + "-" * 90)
        print(
            f"  {'AGGREGATE':<22} {agg_iou:>6.3f} {agg_recall:>8.3f} "
            f"{agg_fpr:>6.3f} {agg_yr_acc:>9.3f} "
            f"{total_tp+total_fn:>8,} {total_tp+total_fp:>8,} "
            f"{total_tp:>7,} {total_fp:>7,} {total_fn:>7,}"
        )
        print(f"\n{'='*60}")
        print("Leaderboard Metric Estimates")
        print(f"{'='*60}")
        print(f"  Union IoU      : {agg_iou:.3f}   ← primary metric")
        print(f"  Recall         : {agg_recall:.3f}")
        print(f"  FPR            : {agg_fpr:.3f}   ← lower is better")
        print(f"  Year Accuracy  : {agg_yr_acc:.3f}  "
              f"(only where RADD/GLAD day-fields available)")
        print()

        # ── diagnose worst tiles ───────────────────────────────────────────────
        df = pd.DataFrame(rows)
        if len(df) > 0:
            worst_iou = df.nsmallest(3, "iou")[["tile_id", "iou", "recall", "fpr"]]
            print("  Worst IoU tiles:")
            print(worst_iou.to_string(index=False))
            print()

            high_fpr = df[df["fpr"] > 0.30][["tile_id", "fpr", "pred_px", "gt_px"]]
            if not high_fpr.empty:
                print("  High FPR tiles (> 0.30) — over-predicting:")
                print(high_fpr.to_string(index=False))
                print()

            low_recall = df[df["recall"] < 0.50][["tile_id", "recall", "gt_px", "pred_px"]]
            if not low_recall.empty:
                print("  Low Recall tiles (< 0.50) — under-detecting:")
                print(low_recall.to_string(index=False))

    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spatial pixel-level validation on training tiles")
    p.add_argument("--pred-dir",  default="results/predictions")
    p.add_argument("--mislabels", default="results/mislabels_v2.csv")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    validate(
        pred_dir=Path(args.pred_dir),
        mislabels_csv=Path(args.mislabels),
        verbose=not args.quiet,
    )
