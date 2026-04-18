"""
Phase 1 — Label Fusion

Loads the three weak label sources (RADD, GLAD-L, GLAD-S2) for every training
tile, decodes each to a binary post-2020 deforestation mask, computes per-pixel
agreement, and fuses them via majority voting.

Run from the deforestation/ directory:
    python src/phase1_label_fusion.py
"""

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
    decode_radd,
    decode_gladl,
    decode_glads2_date,
    load_and_resample_to_ref,
    find_gladl_alert,
    find_glads2_alert,
    find_glads2_date,
    RADD_EPOCH_DAYS_TO_2020,
    GLADS2_EPOCH_DAYS_TO_2020,
)

logger = get_logger(__name__)


# ─── Tile discovery ───────────────────────────────────────────────────────────

def discover_tiles(radd_dir: str) -> list[str]:
    """
    Find all tile IDs from RADD label files.
    Naming convention: radd_<tile_id>_labels.tif
    """
    tiles = []
    for f in sorted(Path(radd_dir).glob("radd_*_labels.tif")):
        # "radd_18NWG_6_6_labels" → strip prefix "radd_" and suffix "_labels"
        stem = f.stem                          # radd_18NWG_6_6_labels
        tile_id = stem[len("radd_"):-len("_labels")]
        tiles.append(tile_id)

    if not tiles:
        logger.warning(f"No RADD label files found in {radd_dir}")
    return tiles


# ─── Per-tile processing ──────────────────────────────────────────────────────

def process_tile(tile_id: str, cfg: dict) -> tuple[pd.DataFrame, dict] | None:
    """
    Load all three label sources for one tile, decode, fuse, and return
    a DataFrame of sampled pixels plus a stats dict.
    """
    lbl = cfg["label_fusion"]
    paths = cfg["paths"]["labels"]

    # ── RADD (reference grid) ──────────────────────────────────────────────
    radd_path = Path(paths["radd"]) / f"radd_{tile_id}_labels.tif"
    if not radd_path.exists():
        logger.warning(f"[{tile_id}] RADD file missing — skipping tile")
        return None

    with rasterio.open(radd_path) as src:
        ref_shape     = (src.height, src.width)
        ref_transform = src.transform
        ref_crs       = src.crs
        radd_raw      = src.read(1)

    radd_bin = decode_radd(
        radd_raw,
        min_days=lbl["radd_min_days_post2020"],
        min_conf=lbl["radd_min_confidence"],
    )

    # ── GLAD-L (logical OR across all configured years) ────────────────────
    gladl_bin   = np.zeros(ref_shape, dtype=np.uint8)
    gladl_found = False
    for yy in lbl["gladl_years"]:
        alert_path = find_gladl_alert(paths["gladl"], tile_id, yy)
        if alert_path is None:
            continue
        gladl_found = True
        arr = load_and_resample_to_ref(alert_path, ref_shape, ref_transform, ref_crs)
        gladl_bin = np.maximum(gladl_bin,
                               decode_gladl(arr, lbl["gladl_min_confidence"]))

    if not gladl_found:
        logger.warning(f"[{tile_id}] No GLAD-L files found for years {lbl['gladl_years']}")

    # ── GLAD-S2 ────────────────────────────────────────────────────────────
    glads2_alert_path = find_glads2_alert(paths["glads2"], tile_id)
    glads2_date_path  = find_glads2_date(paths["glads2"], tile_id)

    if glads2_alert_path is not None:
        gs2_alert = load_and_resample_to_ref(
            glads2_alert_path, ref_shape, ref_transform, ref_crs)

        if glads2_date_path is not None:
            # Prefer date-filtered decoding to restrict to post-2020
            gs2_date = load_and_resample_to_ref(
                glads2_date_path, ref_shape, ref_transform, ref_crs)
            glads2_bin = decode_glads2_date(
                date_arr=gs2_date,
                alert_arr=gs2_alert,
                min_days=lbl["glads2_min_days_post2020"],
                min_conf=lbl["glads2_min_confidence"],
            )
        else:
            # No date raster — fall back to confidence-only decoding
            from utils import decode_glads2_alert
            glads2_bin = decode_glads2_alert(gs2_alert, lbl["glads2_min_confidence"])
            logger.warning(f"[{tile_id}] GLAD-S2 alertDate missing — using confidence-only filter")
    else:
        glads2_bin = np.zeros(ref_shape, dtype=np.uint8)
        logger.warning(f"[{tile_id}] GLAD-S2 alert file missing")

    # ── Majority-vote fusion ───────────────────────────────────────────────
    # votes in {0,1,2,3}: how many of the 3 sources flag each pixel
    label_stack = np.stack([radd_bin, gladl_bin, glads2_bin], axis=0)
    votes       = label_stack.sum(axis=0).astype(np.uint8)

    fused      = (votes >= lbl["majority_threshold"]).astype(np.uint8)
    confidence = (votes / 3.0).astype(np.float32)
    # Uncertain = exactly 1 source agrees (no majority either way)
    uncertain  = (votes == 1)

    # ── Pixel sampling ────────────────────────────────────────────────────
    # Keep ALL pixels flagged by at least one source (positives + uncertain)
    # plus a balanced sample of definite negatives (votes == 0)
    pos_r, pos_c = np.where(votes > 0)
    neg_r, neg_c = np.where(votes == 0)

    n_pos        = len(pos_r)
    n_neg_target = min(len(neg_r), n_pos * lbl["neg_sample_ratio"])

    if n_neg_target > 0 and len(neg_r) > 0:
        rng     = np.random.default_rng(lbl["neg_sample_seed"])
        neg_idx = rng.choice(len(neg_r), size=n_neg_target, replace=False)
        neg_r   = neg_r[neg_idx]
        neg_c   = neg_c[neg_idx]

    all_r = np.concatenate([pos_r, neg_r])
    all_c = np.concatenate([pos_c, neg_c])

    df = pd.DataFrame({
        "pixel_id":      [f"{tile_id}__{r}_{c}" for r, c in zip(all_r, all_c)],
        "tile_id":       tile_id,
        "row":           all_r.astype(np.int32),
        "col":           all_c.astype(np.int32),
        "radd_label":    radd_bin[all_r, all_c].astype(np.int8),
        "gladl_label":   gladl_bin[all_r, all_c].astype(np.int8),
        "glads2_label":  glads2_bin[all_r, all_c].astype(np.int8),
        "votes":         votes[all_r, all_c].astype(np.int8),
        "fused_label":   fused[all_r, all_c].astype(np.int8),
        "confidence":    confidence[all_r, all_c],
        "uncertain_flag": uncertain[all_r, all_c],
    })

    stats = {
        "tile_id":       tile_id,
        "height":        ref_shape[0],
        "width":         ref_shape[1],
        "total_pixels":  ref_shape[0] * ref_shape[1],
        "radd_pos":      int(radd_bin.sum()),
        "gladl_pos":     int(gladl_bin.sum()),
        "glads2_pos":    int(glads2_bin.sum()),
        "fused_pos":     int(fused.sum()),
        "uncertain_px":  int(uncertain.sum()),
        "gladl_found":   gladl_found,
        "glads2_found":  glads2_alert_path is not None,
        "sampled_rows":  len(df),
    }

    return df, stats


# ─── Source-level agreement table ────────────────────────────────────────────

def compute_source_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    For pixels flagged by the fused majority label, compute how often each
    individual source agrees with the fused label.
    """
    rows = []
    for src_col, name in [
        ("radd_label",   "RADD"),
        ("gladl_label",  "GLAD-L"),
        ("glads2_label", "GLAD-S2"),
    ]:
        total     = len(df)
        agree     = (df[src_col] == df["fused_label"]).sum()
        rows.append({"source": name, "agreement_pct": 100.0 * agree / total,
                     "positive_px": int(df[src_col].sum()),
                     "total_px": total})
    return pd.DataFrame(rows).set_index("source")


# ─── Cleanlab (optional) ─────────────────────────────────────────────────────

def run_cleanlab(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use cleanlab confident learning to refine labels among uncertain pixels.
    Requires a feature matrix — we use vote fractions as a proxy here.
    Only called when uncertain pixel fraction > uncertain_threshold.
    """
    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        logger.warning("cleanlab not installed — skipping confident learning")
        return df

    logger.info("Running cleanlab confident learning on uncertain pixels …")

    uncertain_mask = df["uncertain_flag"].values
    if uncertain_mask.sum() < 10:
        logger.info("Too few uncertain pixels for cleanlab — skipping")
        return df

    # Proxy feature: probability vector per class using vote fractions
    prob_pos  = df.loc[uncertain_mask, "confidence"].values.reshape(-1, 1)
    prob_neg  = 1.0 - prob_pos
    probs     = np.hstack([prob_neg, prob_pos]).astype(np.float32)
    labels    = df.loc[uncertain_mask, "fused_label"].values.astype(int)

    try:
        issue_idx = find_label_issues(labels=labels, pred_probs=probs,
                                      return_indices_ranked_by="self_confidence")
        # Flag cleanlab-identified label issues as still uncertain
        uncertain_rows = df.index[uncertain_mask]
        df.loc[uncertain_rows[issue_idx], "uncertain_flag"] = True
        logger.info(f"cleanlab flagged {len(issue_idx)} additional uncertain pixels")
    except Exception as e:
        logger.warning(f"cleanlab failed: {e}")

    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config("configs/config.yaml")
    lbl = cfg["label_fusion"]

    processed_dir = Path(cfg["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover tiles ────────────────────────────────────────────────────
    tiles = discover_tiles(cfg["paths"]["labels"]["radd"])
    logger.info(f"Discovered {len(tiles)} tiles: {tiles}")

    if not tiles:
        logger.error("No tiles found. Check data_root in config.yaml.")
        sys.exit(1)

    # ── Process every tile ────────────────────────────────────────────────
    all_dfs   = []
    all_stats = []

    for tile_id in tqdm(tiles, desc="Processing tiles"):
        result = process_tile(tile_id, cfg)
        if result is None:
            continue
        df_tile, stats = result
        all_dfs.append(df_tile)
        all_stats.append(stats)
        logger.info(
            f"[{tile_id}] RADD={stats['radd_pos']:,}  GLAD-L={stats['gladl_pos']:,}  "
            f"GLAD-S2={stats['glads2_pos']:,}  fused={stats['fused_pos']:,}  "
            f"uncertain={stats['uncertain_px']:,}  sampled_rows={stats['sampled_rows']:,}"
        )

    if not all_dfs:
        logger.error("No tiles successfully processed.")
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"\nMerged dataset shape: {df.shape}  dtypes:\n{df.dtypes}")

    # ── Per-source agreement summary ─────────────────────────────────────
    agree_df = compute_source_agreement(df)
    print("\n── Source agreement with fused majority label ──────────────────")
    print(agree_df.to_string())

    # ── Uncertain pixel check — optionally run cleanlab ──────────────────
    n_uncertain   = df["uncertain_flag"].sum()
    uncertain_pct = n_uncertain / len(df)
    logger.info(f"\nUncertain pixels: {n_uncertain:,} ({100*uncertain_pct:.1f}% of sampled rows)")

    if uncertain_pct > lbl["uncertain_threshold"]:
        logger.info(f"Uncertain > {100*lbl['uncertain_threshold']:.0f}% threshold — running cleanlab")
        df = run_cleanlab(df)

    # ── Final summary ─────────────────────────────────────────────────────
    n_deforest   = int(df["fused_label"].sum())
    n_background = int((df["fused_label"] == 0).sum())
    imbalance    = n_background / max(n_deforest, 1)

    print("\n── Fused label distribution ─────────────────────────────────────")
    print(f"  Deforestation (1) : {n_deforest:>10,}")
    print(f"  Background    (0) : {n_background:>10,}")
    print(f"  Imbalance ratio   : {imbalance:.1f} : 1  (neg:pos)")
    print(f"  Uncertain pixels  : {n_uncertain:>10,}  ({100*uncertain_pct:.1f}%)")

    print("\n── Per-tile stats ────────────────────────────────────────────────")
    stats_df = pd.DataFrame(all_stats)
    print(stats_df[["tile_id", "radd_pos", "gladl_pos", "glads2_pos",
                     "fused_pos", "uncertain_px"]].to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = processed_dir / "fused_labels.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\n✓ Saved → {out_path}  ({len(df):,} rows × {len(df.columns)} cols)")
    print("\nDONE — Phase 1 complete.")
    print(f"  Files saved:")
    print(f"    {out_path}")


if __name__ == "__main__":
    main()
