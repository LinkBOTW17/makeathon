"""
Phase 3b — Temporal Model (LSTM + AEF Fusion)

Trains a two-branch neural network that fuses:
  Branch A: 2-layer LSTM over the 72-step S1+S2 time series
  Branch B: Linear projection of the 64-dim AEF embedding
  Fusion:   MLP head → binary deforestation prediction

Key motivation: VV slope was the #1 XGBoost feature by a large margin.
The LSTM sees the full 72-step backscatter trajectory and can learn the
exact shape of a deforestation event — not just its linear trend.

Outputs:
  data/checkpoints/temporal_model_best.pt
  data/processed/temporal_val_predictions.parquet

Run from the deforestation/ directory:
    python src/phase3b_temporal_model.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DeforestationDataset,
    DeforestationModel,
    get_device,
    get_logger,
    load_config,
)

logger = get_logger(__name__)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(cfg: dict) -> dict:
    processed = Path(cfg["paths"]["processed"])

    logger.info("Loading arrays from disk …")
    s2_ts    = np.load(processed / "s2_timeseries.npy", mmap_mode="r")  # (N, T, 2)
    s1_ts    = np.load(processed / "s1_timeseries.npy", mmap_mode="r")  # (N, T, 1)
    feat_df  = pd.read_parquet(processed / "features_static.parquet")
    label_df = pd.read_parquet(processed / "labels.parquet")

    logger.info(f"s2_ts  : {s2_ts.shape}   s1_ts : {s1_ts.shape}")
    logger.info(f"feat_df: {feat_df.shape}  label_df: {label_df.shape}")

    # AEF columns only (64 dims)
    aef_cols = [c for c in feat_df.columns if c.startswith("aef_")]
    aef      = feat_df[aef_cols].values.astype(np.float32)

    # Tile-based split — never split a tile across sets
    tile_ids   = label_df["tile_id"].values
    val_set    = set(cfg["split"]["val_tiles"])
    test_set   = set(cfg["split"]["test_tiles"])

    train_mask = ~np.isin(tile_ids, list(val_set | test_set))
    val_mask   = np.isin(tile_ids, list(val_set))

    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]

    logger.info(f"Train tiles : {sorted(set(tile_ids[train_mask]))}")
    logger.info(f"Val tiles   : {sorted(set(tile_ids[val_mask]))}")
    logger.info(f"Train size  : {len(train_idx):,}  |  Val size: {len(val_idx):,}")

    labels = label_df["fused_label"].values.astype(np.float32)
    pids   = label_df["pixel_id"].values

    return dict(s2_ts=s2_ts, s1_ts=s1_ts, aef=aef, labels=labels,
                pids=pids, train_idx=train_idx, val_idx=val_idx)


# ─── Normalisation ────────────────────────────────────────────────────────────

def compute_ts_norm_stats(s2_ts, s1_ts, train_idx, sample_n=200_000) -> tuple:
    """
    Estimate per-channel (NDVI, NBR, VV_dB) mean and std from a random
    sample of training pixels.  NaN values are excluded from statistics.
    """
    rng = np.random.default_rng(42)
    sample = rng.choice(train_idx, size=min(sample_n, len(train_idx)), replace=False)

    s2_sample = s2_ts[sample]                       # (S, T, 2)
    s1_sample = s1_ts[sample]                       # (S, T, 1)
    ts_sample = np.concatenate([s2_sample, s1_sample], axis=2)  # (S, T, 3)

    means, stds = [], []
    for c in range(3):
        ch = ts_sample[:, :, c].ravel()
        ch = ch[np.isfinite(ch)]
        means.append(float(np.mean(ch)) if len(ch) else 0.0)
        stds.append(float(np.std(ch))   if len(ch) else 1.0)

    logger.info(f"TS norm  means={[f'{m:.3f}' for m in means]}  "
                f"stds={[f'{s:.3f}' for s in stds]}")
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def compute_aef_norm_stats(aef, train_idx) -> tuple:
    aef_train = aef[train_idx]
    means = np.nanmean(aef_train, axis=0).astype(np.float32)
    stds  = np.nanstd(aef_train,  axis=0).astype(np.float32)
    stds  = np.where(stds < 1e-8, 1.0, stds)
    return means, stds


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler) -> float:
    model.train()
    total_loss = 0.0
    for ts, aef, labels in loader:
        ts, aef, labels = ts.to(device), aef.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(ts, aef)            # (B, 1)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_epoch(model, loader, device) -> tuple:
    """Returns (y_true, y_prob) numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []
    for ts, aef, labels in loader:
        ts, aef = ts.to(device), aef.to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(ts, aef).squeeze(1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.squeeze(1).numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg  = load_config("configs/config.yaml")
    mcfg = cfg["model"]

    processed   = Path(cfg["paths"]["processed"])
    checkpoints = Path(cfg["paths"]["checkpoints"])
    checkpoints.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info(f"Using device: {device}")

    # ── Load data ────────────────────────────────────────────────────────
    data = load_data(cfg)

    # ── Normalisation stats (from training pixels only) ───────────────────
    ts_mean, ts_std   = compute_ts_norm_stats(data["s2_ts"], data["s1_ts"],
                                               data["train_idx"])
    aef_mean, aef_std = compute_aef_norm_stats(data["aef"], data["train_idx"])

    # ── Build datasets and loaders ────────────────────────────────────────
    # Convert mmap arrays to float32 in memory for faster access during training
    s2_np = np.array(data["s2_ts"], dtype=np.float32)
    s1_np = np.array(data["s1_ts"], dtype=np.float32)

    common_kwargs = dict(s2_ts=s2_np, s1_ts=s1_np, aef=data["aef"],
                         labels=data["labels"],
                         ts_mean=ts_mean, ts_std=ts_std,
                         aef_mean=aef_mean, aef_std=aef_std)

    train_ds = DeforestationDataset(data["train_idx"], **common_kwargs)
    val_ds   = DeforestationDataset(data["val_idx"],   **common_kwargs)

    bs = mcfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=(device.type == "cuda"),
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs * 2, shuffle=False,
                              num_workers=4, pin_memory=(device.type == "cuda"),
                              persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = DeforestationModel.build(
        n_ts_features = mcfg["n_ts_features"],
        aef_dim       = mcfg["aef_dim"],
        lstm_hidden   = mcfg["lstm_hidden"],
        lstm_layers   = mcfg["lstm_layers"],
        lstm_dropout  = mcfg["lstm_dropout"],
        mlp_layers    = mcfg["mlp_layers"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # ── Loss: weighted BCE to handle 5.7:1 imbalance ─────────────────────
    n_pos = int(data["labels"][data["train_idx"]].sum())
    n_neg = len(data["train_idx"]) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"Train pos={n_pos:,}  neg={n_neg:,}  pos_weight={pos_weight.item():.2f}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=mcfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=mcfg["lr_scheduler_patience"],
        factor=0.5, verbose=False,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # ── Training ──────────────────────────────────────────────────────────
    best_f1    = -1.0
    best_state = None
    patience_counter = 0
    best_auc_pr = 0.0

    print(f"\n{'Epoch':>5}  {'train_loss':>10}  {'val_F1':>7}  {'val_AUC-PR':>10}  {'LR':>10}")
    print("-" * 55)

    for epoch in range(1, mcfg["max_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device, scaler)
        y_true, y_prob = evaluate_epoch(model, val_loader, device)
        y_pred = (y_prob >= 0.5).astype(int)

        val_f1    = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        val_aucpr = average_precision_score(y_true, y_prob)

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_f1:>7.4f}  "
              f"{val_aucpr:>10.4f}  {current_lr:>10.2e}")

        if val_f1 > best_f1:
            best_f1      = val_f1
            best_auc_pr  = val_aucpr
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= mcfg["patience"]:
            logger.info(f"Early stopping at epoch {epoch} (patience={mcfg['patience']})")
            break

    # ── Reload best checkpoint & final eval ──────────────────────────────
    model.load_state_dict(best_state)
    y_true, y_prob = evaluate_epoch(model, val_loader, device)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n── Temporal model validation metrics ───────────────────────────")
    print(classification_report(y_true, y_pred,
                                 target_names=["background", "deforestation"]))
    final_auc_roc = roc_auc_score(y_true, y_prob)
    final_auc_pr  = average_precision_score(y_true, y_prob)
    final_f1      = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    print(f"  AUC-ROC : {final_auc_roc:.4f}")
    print(f"  AUC-PR  : {final_auc_pr:.4f}   ← primary metric")

    # ── Side-by-side comparison with Phase 3a ────────────────────────────
    xgb_art_path  = checkpoints / "xgboost_baseline.joblib"
    xgb_pred_path = processed  / "baseline_val_predictions.parquet"
    print("\n── Model comparison ─────────────────────────────────────────────")
    if xgb_art_path.exists() and xgb_pred_path.exists():
        xgb_art     = joblib.load(xgb_art_path)
        xgb_pred_df = pd.read_parquet(xgb_pred_path)
        xgb_f1      = f1_score(xgb_pred_df["y_true"], xgb_pred_df["y_pred"],
                               pos_label=1, zero_division=0)
        print(f"  {'Metric':<12}  {'XGBoost':>10}  {'LSTM+AEF':>10}  {'Delta':>8}")
        print(f"  {'-'*50}")
        print(f"  {'F1 (def)':<12}  {xgb_f1:>10.4f}  {final_f1:>10.4f}"
              f"  {final_f1 - xgb_f1:>+8.4f}")
        print(f"  {'AUC-ROC':<12}  {xgb_art['auc_roc']:>10.4f}  {final_auc_roc:>10.4f}"
              f"  {final_auc_roc - xgb_art['auc_roc']:>+8.4f}")
        print(f"  {'AUC-PR':<12}  {xgb_art['auc_pr']:>10.4f}  {final_auc_pr:>10.4f}"
              f"  {final_auc_pr - xgb_art['auc_pr']:>+8.4f}")
    else:
        logger.warning("XGBoost artifacts not found — skipping comparison table")

    # ── Save model ────────────────────────────────────────────────────────
    model_path = checkpoints / "temporal_model_best.pt"
    torch.save({
        "model_state_dict": best_state,
        "model_cfg":        mcfg,
        "ts_mean":          ts_mean,
        "ts_std":           ts_std,
        "aef_mean":         aef_mean,
        "aef_std":          aef_std,
        "val_tiles":        cfg["split"]["val_tiles"],
        "auc_pr":           final_auc_pr,
        "auc_roc":          final_auc_roc,
        "f1":               final_f1,
    }, model_path)

    # ── Save val predictions ──────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "pixel_id": data["pids"][data["val_idx"]],
        "y_true":   y_true.astype(np.int8),
        "y_pred":   y_pred.astype(np.int8),
        "y_prob":   y_prob,
    })
    pred_path = processed / "temporal_val_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    print(f"\n✓ Saved → {model_path}")
    print(f"✓ Saved → {pred_path}")
    print("\nDONE — Phase 3b complete.")
    print(f"  AUC-PR  (primary) : {final_auc_pr:.4f}")
    print(f"  AUC-ROC           : {final_auc_roc:.4f}")
    print(f"  F1 (deforestation): {final_f1:.4f}")


if __name__ == "__main__":
    main()
