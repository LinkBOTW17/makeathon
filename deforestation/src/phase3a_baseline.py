"""
Phase 3a — XGBoost Baseline

Trains a gradient-boosted tree classifier on the static feature matrix
(AEF 64-dim + S2 NDVI/NBR stats + S1 VV stats) using a region-based tile split.
Uses GPU-accelerated training when CUDA is available.

Outputs:
  data/checkpoints/xgboost_baseline.joblib
  data/processed/baseline_val_predictions.parquet

Run from the deforestation/ directory:
    python src/phase3a_baseline.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, load_config

logger = get_logger(__name__)


# ─── Feature group labelling (for importance table) ───────────────────────────

def label_feature(name: str) -> str:
    if name.startswith("aef_"):
        return f"AEF dim {name[4:]}"
    parts = name.split("_", 1)
    group = parts[0].upper()
    stat  = parts[1] if len(parts) > 1 else ""
    return f"{group} {stat}"


# ─── Data loading & splitting ─────────────────────────────────────────────────

def load_and_split(cfg: dict) -> tuple:
    """
    Load features and labels, split by tile_id (never split a tile across sets).
    Returns X_train, y_train, X_val, y_val, feature_cols, val_pixel_ids.
    """
    processed = Path(cfg["paths"]["processed"])

    feat_df   = pd.read_parquet(processed / "features_static.parquet")
    labels_df = pd.read_parquet(processed / "labels.parquet")

    logger.info(f"Features shape : {feat_df.shape}")
    logger.info(f"Labels shape   : {labels_df.shape}")

    # Merge on pixel_id to ensure alignment
    df = feat_df.merge(
        labels_df[["pixel_id", "fused_label"]], on="pixel_id", how="inner"
    )

    val_tiles  = cfg["split"]["val_tiles"]
    test_tiles = cfg["split"]["test_tiles"]

    # Exclude test tiles from train/val entirely — they're reserved for Phase 4
    exclude = set(val_tiles) | set(test_tiles)
    train_df = df[~df["tile_id"].isin(exclude)].reset_index(drop=True)
    val_df   = df[df["tile_id"].isin(val_tiles)].reset_index(drop=True)

    feature_cols = [c for c in feat_df.columns if c not in ("pixel_id", "tile_id")]

    logger.info(f"Train tiles : {sorted(train_df['tile_id'].unique())}")
    logger.info(f"Val tiles   : {sorted(val_df['tile_id'].unique())}")
    logger.info(f"Test tiles  : {test_tiles} (held out for Phase 4)")
    logger.info(f"Train size  : {len(train_df):,}  |  Val size: {len(val_df):,}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["fused_label"].values.astype(np.int32)
    X_val   = val_df[feature_cols].values.astype(np.float32)
    y_val   = val_df["fused_label"].values.astype(np.int32)
    val_pids = val_df["pixel_id"].values

    return X_train, y_train, X_val, y_val, feature_cols, val_pids


# ─── Training ─────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val, cfg: dict) -> XGBClassifier:
    xgb_cfg = cfg["xgboost"]

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    logger.info(f"Train  pos={n_pos:,}  neg={n_neg:,}  scale_pos_weight={scale_pos_weight:.2f}")

    # Prefer GPU; fall back to CPU automatically if CUDA unavailable
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    logger.info(f"XGBoost device: {device}")

    model = XGBClassifier(
        n_estimators        = xgb_cfg["n_estimators"],
        early_stopping_rounds = xgb_cfg["early_stopping_rounds"],
        eval_metric         = xgb_cfg["eval_metric"],
        max_depth           = xgb_cfg["max_depth"],
        learning_rate       = xgb_cfg["learning_rate"],
        subsample           = xgb_cfg["subsample"],
        colsample_bytree    = xgb_cfg["colsample_bytree"],
        min_child_weight    = xgb_cfg["min_child_weight"],
        scale_pos_weight    = scale_pos_weight,
        tree_method         = "hist",   # histogram-based — fast on GPU and CPU
        device              = device,
        random_state        = 42,
        n_jobs              = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    return model


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model: XGBClassifier, X_val, y_val, feature_cols: list) -> dict:
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n── Validation metrics ───────────────────────────────────────────")
    print(classification_report(y_val, y_pred,
                                 target_names=["background", "deforestation"]))

    auc_roc = roc_auc_score(y_val, y_prob)
    auc_pr  = average_precision_score(y_val, y_prob)

    print(f"  AUC-ROC : {auc_roc:.4f}")
    print(f"  AUC-PR  : {auc_pr:.4f}   ← primary metric")

    cm = confusion_matrix(y_val, y_pred)
    print(f"\n  Confusion matrix (rows=true, cols=pred):\n{cm}")

    # Top-20 feature importances by gain
    importance = model.get_booster().get_score(importance_type="gain")
    # XGBoost uses f0, f1, … names internally
    named = {}
    for key, val_imp in importance.items():
        idx = int(key[1:]) if key.startswith("f") else -1
        if 0 <= idx < len(feature_cols):
            named[label_feature(feature_cols[idx])] = val_imp

    top20 = sorted(named.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\n── Top 20 features by gain ──────────────────────────────────────")
    for rank, (feat, imp) in enumerate(top20, 1):
        print(f"  {rank:>2}. {feat:<30}  {imp:.1f}")

    return {
        "auc_roc": auc_roc,
        "auc_pr":  auc_pr,
        "y_prob":  y_prob,
        "y_pred":  y_pred,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config("configs/config.yaml")
    processed   = Path(cfg["paths"]["processed"])
    checkpoints = Path(cfg["paths"]["checkpoints"])
    checkpoints.mkdir(parents=True, exist_ok=True)

    # ── Load & split ──────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, feature_cols, val_pids = load_and_split(cfg)
    logger.info(f"Feature dimensionality: {X_train.shape[1]}")

    # ── Train ─────────────────────────────────────────────────────────────
    model = train_xgboost(X_train, y_train, X_val, y_val, cfg)

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = evaluate(model, X_val, y_val, feature_cols)

    # ── Save model ────────────────────────────────────────────────────────
    model_path = checkpoints / "xgboost_baseline.joblib"
    joblib.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "val_tiles":    cfg["split"]["val_tiles"],
        "auc_pr":       results["auc_pr"],
        "auc_roc":      results["auc_roc"],
    }, model_path)

    # ── Save val predictions ──────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "pixel_id": val_pids,
        "y_true":   y_val,
        "y_pred":   results["y_pred"],
        "y_prob":   results["y_prob"],
    })
    pred_path = processed / "baseline_val_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    print(f"\n✓ Saved → {model_path}")
    print(f"✓ Saved → {pred_path}")
    print("\nDONE — Phase 3a complete.")
    print(f"  AUC-PR  (primary) : {results['auc_pr']:.4f}")
    print(f"  AUC-ROC           : {results['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
