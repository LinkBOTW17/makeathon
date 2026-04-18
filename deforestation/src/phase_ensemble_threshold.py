"""
Ensemble & Threshold Optimisation

Loads validation predictions from XGBoost (and optionally LSTM), ensembles them,
then sweeps probability thresholds to find the value that maximises pixel-level
IoU — the best proxy for the Union IoU primary scoring metric.

Writes the optimal threshold back to configs/config.yaml under
submission.probability_threshold.

Run from the deforestation/ directory:
    python src/phase_ensemble_threshold.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, load_config

logger = get_logger(__name__)


def pixel_iou(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    """
    Pixel-level Intersection-over-Union — proxy for the Union IoU scoring metric.
    IoU = |pred ∩ gt| / |pred ∪ gt|
    """
    intersection = int((y_pred_bin & y_true.astype(bool)).sum())
    union        = int((y_pred_bin | y_true.astype(bool)).sum())
    return intersection / union if union > 0 else 0.0


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                     thresholds: list) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        y_pred  = (y_prob >= t).astype(int)
        n_pred  = int(y_pred.sum())
        n_pos   = int(y_true.sum())
        tp      = int((y_pred & y_true.astype(bool)).sum())
        fp      = int((y_pred & ~y_true.astype(bool)).sum())
        fn      = int((~y_pred.astype(bool) & y_true.astype(bool)).sum())

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        iou       = pixel_iou(y_true, y_pred.astype(bool))

        rows.append(dict(threshold=t, precision=precision, recall=recall,
                         f1=f1, iou=iou, n_pred=n_pred))
    return pd.DataFrame(rows)


def main():
    cfg       = load_config("configs/config.yaml")
    processed = Path(cfg["paths"]["processed"])

    # ── Load val predictions ──────────────────────────────────────────────
    xgb_path  = processed / "baseline_val_predictions.parquet"
    lstm_path = processed / "temporal_val_predictions.parquet"

    if not xgb_path.exists():
        logger.error("XGBoost val predictions not found. Run phase3a first.")
        sys.exit(1)

    xgb_df = pd.read_parquet(xgb_path)
    y_true  = xgb_df["y_true"].values.astype(bool)
    p_xgb   = xgb_df["y_prob"].values

    # Ensemble with LSTM if available (weighted average)
    if lstm_path.exists():
        lstm_df = pd.read_parquet(lstm_path)
        # Align on pixel_id
        merged  = xgb_df[["pixel_id", "y_true", "y_prob"]].merge(
            lstm_df[["pixel_id", "y_prob"]].rename(columns={"y_prob": "p_lstm"}),
            on="pixel_id", how="inner",
        )
        y_true = merged["y_true"].values.astype(bool)
        p_xgb  = merged["y_prob"].values
        p_lstm = merged["p_lstm"].values
        # Weight by val AUC-PR: XGBoost=0.546, LSTM=0.519 → ~51% / 49% weight
        p_ensemble = 0.55 * p_xgb + 0.45 * p_lstm
        logger.info("Ensembling XGBoost (55%) + LSTM (45%)")
    else:
        p_ensemble = p_xgb
        logger.info("LSTM predictions not found — using XGBoost only")

    thresholds = [round(t, 2) for t in np.arange(0.10, 0.71, 0.05)]

    print("\n── XGBoost only — threshold sweep ──────────────────────────────")
    df_xgb = sweep_thresholds(y_true, p_xgb, thresholds)
    print(df_xgb.to_string(index=False, float_format="%.4f"))

    if p_ensemble is not p_xgb:
        print("\n── Ensemble — threshold sweep ───────────────────────────────────")
        df_ens = sweep_thresholds(y_true, p_ensemble, thresholds)
        print(df_ens.to_string(index=False, float_format="%.4f"))
        best_row = df_ens.loc[df_ens["iou"].idxmax()]
        best_prob_arr = p_ensemble
        label = "ensemble"
    else:
        best_row = df_xgb.loc[df_xgb["iou"].idxmax()]
        best_prob_arr = p_xgb
        label = "xgboost"

    best_threshold = float(best_row["threshold"])
    print(f"\n── Best threshold ({label}) ──────────────────────────────────────")
    print(f"  Threshold : {best_threshold:.2f}")
    print(f"  IoU       : {best_row['iou']:.4f}   ← proxy for Union IoU")
    print(f"  F1 (def)  : {best_row['f1']:.4f}")
    print(f"  Precision : {best_row['precision']:.4f}")
    print(f"  Recall    : {best_row['recall']:.4f}")
    print(f"  N predicted deforestation pixels: {int(best_row['n_pred']):,}")

    # ── Save optimal threshold to config ─────────────────────────────────
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)

    raw_cfg["submission"]["probability_threshold"] = best_threshold

    with open(config_path, "w") as f:
        yaml.dump(raw_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Updated configs/config.yaml → submission.probability_threshold = {best_threshold}")
    print("\nDONE — Ensemble & threshold optimisation complete.")


if __name__ == "__main__":
    main()
