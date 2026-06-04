Balu
bxlthz
Online

elias_04 — 12:04 PM
./code tunnel kill
./code tunnel user logout
./code tunnel --name my-tunnel
Balu — 12:32 PM
Image
Image
Image
Image
sri — 1:18 PM
https://x.com/shmidtqq/status/2042655558516302143
shmidt (@shmidtqq)
FREE CLAUDE CODE + GEMINI + CHAT GPT
Image

X•4/10/2026 7:26 PM
sri — 2:22 PM
https://docs.google.com/presentation/d/15DnaBILaUF3PyjNhuDYUHV1Wm_NSSHEBJuGznxis0VQ/edit?usp=sharing
Google Docs
Makeathon
omfgwhyisthissohard — 2:46 PM
The challenge notebook defines the modalities like this:

Sentinel-2: 12 spectral bands, monthly time series, upsampled to 10 m, delivered as a single multi-band TIFF in a local UTM CRS. See challenge.ipynb:115 and challenge.ipynb:135.
Sentinel-1: 1 radar backscatter band, VV polarization, RTC product, monthly time series, delivered in a local UTM CRS aligned with Sentinel-2. See challenge.ipynb:346.
AlphaEarth Foundations: 64 embedding dimensions, annual data, delivered in EPSG:4326 WGS-84, and intended to be reprojected to local UTM before fusion. See challenge.ipynb:376, challenge.ipynb:380, and challenge.ipynb:410.
So the practical model-facing shape is:

S1: time, 1 channel
S2: time, 12 channels
AEF: year, 64 channels
If you want, I can also give you the exact tensor shapes your current OsapiensDataset returns for one sample.

GPT-5.4 mini • 0.3x
zitong — 2:50 PM
Understanding the data

# 🛰️ Satellite Data Structure and Feature Analysis Guide - English Version

## 📊 Data Type Overview

| Data Source | Resolution | Bands | CRS | Frequency | Main Use |
| ----------- | ---------- | ----- | --- | --------- | -------- |

message.txt
9 KB
omfgwhyisthissohard — 3:04 PM
https://docs.google.com/presentation/d/1HkfpMm4yk_udh2tH7oqlayuTv_w5J2sH/edit?usp=sharing&ouid=105280216781445498547&rtpof=true&sd=true
Google Docs
2045487626052284416
1 AI-Driven Deforestation Detection Post-2020 Multimodal Satellite Intelligence for Global Forest Monitoring HACKATHON 2026 • AI FOR EARTH Sentinel-1 • Sentinel-2 • AEF Embeddings Generated • April 2026
Image
UwU
sri — 3:10 PM
https://osapiens-eval.vercel.app/login

gummibärenbande

CupofCoffee1234!
osapiens Terra x TUM.ai Challenge Hub
Submission dashboard for osapiens Terra x TUM.ai challenges
sri — 3:43 PM

## WORKFLOW CONSTRAINT — READ THIS FIRST

You are running locally with NO GPU access. The execution environment is a remote AMD GPU cloud machine. The workflow is:

1. You write all code and configs here locally
2. The user pushes to git
3. The user pulls and runs on AMD cloud GPU
4. The user pastes results back into this conversation

Because of this:

- NEVER run training scripts, never execute python files
- NEVER validate model output by running code — you cannot see the GPU
- DO write clean, runnable Python scripts saved to files (not notebooks)
- DO maintain requirements.txt — every new pip install must be added to it immediately
- DO print a "ready to push" checklist after each phase listing every file created/modified
- After each phase, output the exact git commands the user should run:
  git add .
  git commit -m "phase-X: <short description>"
  git push
- Then tell the user exactly: "Run this on AMD cloud: python <script_name>.py"
- Wait for the user to paste back the terminal output before proceeding to the next phase

## RESULTS HANDOFF

When the user pastes back terminal output from AMD cloud:

- Parse the metrics printed (F1, AUC, loss, shape confirmations, etc.)
- Decide if results are good enough to proceed or if code needs fixing
- If something failed (CUDA error, missing library, shape mismatch, OOM), diagnose and fix the code locally, then give new git push + run instructions
- Only move to the next phase when the user confirms the current phase ran successfully on AMD

## RESULTS HANDOFF FORMAT

When the user pastes results, respond in this structure:
STATUS: [OK / NEEDS FIX]
OBSERVED: <what the output shows>
DECISION: <proceed to next phase / fix issue X first>
ACTION: <new code fix OR "proceed to Phase N">

---

## PROJECT CONTEXT

Goal: Pixel-level deforestation detection that generalizes to unseen geographic regions.

Data available (already downloaded on AMD cloud):

- AEF foundation model embedding tiles (.tif, 64 bands per pixel)
- Sentinel-1 radar time series rasters
- Sentinel-2 optical time series rasters
- Annotation GeoJSONs with weak/noisy labels from multiple sources

Key constraints:

- Train/val split MUST be by REGION (not random) — hidden test set is in unseen geography
- Prefer spectral indices (NDVI, NBR, VV, VH) over raw band values everywhere — more geographically invariant
- AEF embeddings are globally pretrained — lean on them heavily for generalization
- All scripts must be GPU-aware: use torch.device("cuda" if torch.cuda.is_available() else "cpu") everywhere

## PROJECT STRUCTURE

Create and maintain this folder structure from the start:

deforestation/
├── data/
│ ├── raw/ # original tifs and GeoJSONs (already on AMD, don't touch)
│ ├── processed/ # parquet, numpy arrays output by each phase
│ └── checkpoints/ # model .pt files
├── src/
│ ├── phase1_label_fusion.py
│ ├── phase2_feature_extraction.py
│ ├── phase3a_baseline.py
│ ├── phase3b_temporal_model.py
│ ├── phase4_generalization.py
│ ├── phase5_self_training.py
│ ├── phase6_temporal_localization.py
│ └── utils.py # shared helpers (data loading, metrics, logging)
├── configs/
│ └── config.yaml # all paths, hyperparameters, flags — no hardcoded values in scripts
├── requirements.txt
└── README.md

All file paths, hyperparameters, and flags go in config.yaml. Scripts read from it. Never hardcode paths inside scripts.

---

## PHASE 1 — Data audit & label fusion

Script: src/phase1_label_fusion.py

Tasks:

1. Load config.yaml to get all data paths
2. Load all annotation GeoJSONs, print:
   - Number of polygons per source
   - Class distribution per source
   - CRS and geometry validity check
3. Rasterize polygons to pixel level — for each pixel inside any annotation polygon, record its label from every source
4. Compute per-pixel label agreement rate across sources — print a summary table showing per-source agreement %
5. Fuse labels using majority voting
   - If a pixel has a majority label → assign it, set confidence = agreement fraction
   - If no majority (tie) → flag as uncertain
6. Optionally run cleanlab confident learning on top of majority voting if uncertain pixel count > 10% of total
   - pip install cleanlab (add to requirements.txt)
7. Save fused labels:
   ... (370 lines left)

message.txt
20 KB

﻿

## WORKFLOW CONSTRAINT — READ THIS FIRST

You are running locally with NO GPU access. The execution environment is a remote AMD GPU cloud machine. The workflow is:

1. You write all code and configs here locally
2. The user pushes to git
3. The user pulls and runs on AMD cloud GPU
4. The user pastes results back into this conversation

Because of this:

- NEVER run training scripts, never execute python files
- NEVER validate model output by running code — you cannot see the GPU
- DO write clean, runnable Python scripts saved to files (not notebooks)
- DO maintain requirements.txt — every new pip install must be added to it immediately
- DO print a "ready to push" checklist after each phase listing every file created/modified
- After each phase, output the exact git commands the user should run:
  git add .
  git commit -m "phase-X: <short description>"
  git push
- Then tell the user exactly: "Run this on AMD cloud: python <script_name>.py"
- Wait for the user to paste back the terminal output before proceeding to the next phase

## RESULTS HANDOFF

When the user pastes back terminal output from AMD cloud:

- Parse the metrics printed (F1, AUC, loss, shape confirmations, etc.)
- Decide if results are good enough to proceed or if code needs fixing
- If something failed (CUDA error, missing library, shape mismatch, OOM), diagnose and fix the code locally, then give new git push + run instructions
- Only move to the next phase when the user confirms the current phase ran successfully on AMD

## RESULTS HANDOFF FORMAT

When the user pastes results, respond in this structure:
STATUS: [OK / NEEDS FIX]
OBSERVED: <what the output shows>
DECISION: <proceed to next phase / fix issue X first>
ACTION: <new code fix OR "proceed to Phase N">

---

## PROJECT CONTEXT

Goal: Pixel-level deforestation detection that generalizes to unseen geographic regions.

Data available (already downloaded on AMD cloud):

- AEF foundation model embedding tiles (.tif, 64 bands per pixel)
- Sentinel-1 radar time series rasters
- Sentinel-2 optical time series rasters
- Annotation GeoJSONs with weak/noisy labels from multiple sources

Key constraints:

- Train/val split MUST be by REGION (not random) — hidden test set is in unseen geography
- Prefer spectral indices (NDVI, NBR, VV, VH) over raw band values everywhere — more geographically invariant
- AEF embeddings are globally pretrained — lean on them heavily for generalization
- All scripts must be GPU-aware: use torch.device("cuda" if torch.cuda.is_available() else "cpu") everywhere

## PROJECT STRUCTURE

Create and maintain this folder structure from the start:

deforestation/
├── data/
│ ├── raw/ # original tifs and GeoJSONs (already on AMD, don't touch)
│ ├── processed/ # parquet, numpy arrays output by each phase
│ └── checkpoints/ # model .pt files
├── src/
│ ├── phase1_label_fusion.py
│ ├── phase2_feature_extraction.py
│ ├── phase3a_baseline.py
│ ├── phase3b_temporal_model.py
│ ├── phase4_generalization.py
│ ├── phase5_self_training.py
│ ├── phase6_temporal_localization.py
│ └── utils.py # shared helpers (data loading, metrics, logging)
├── configs/
│ └── config.yaml # all paths, hyperparameters, flags — no hardcoded values in scripts
├── requirements.txt
└── README.md

All file paths, hyperparameters, and flags go in config.yaml. Scripts read from it. Never hardcode paths inside scripts.

---

## PHASE 1 — Data audit & label fusion

Script: src/phase1_label_fusion.py

Tasks:

1. Load config.yaml to get all data paths
2. Load all annotation GeoJSONs, print:
   - Number of polygons per source
   - Class distribution per source
   - CRS and geometry validity check
3. Rasterize polygons to pixel level — for each pixel inside any annotation polygon, record its label from every source
4. Compute per-pixel label agreement rate across sources — print a summary table showing per-source agreement %
5. Fuse labels using majority voting
   - If a pixel has a majority label → assign it, set confidence = agreement fraction
   - If no majority (tie) → flag as uncertain
6. Optionally run cleanlab confident learning on top of majority voting if uncertain pixel count > 10% of total
   - pip install cleanlab (add to requirements.txt)
7. Save fused labels:
   - data/processed/fused_labels.parquet
   - columns: pixel_id, tile_id, row, col, geometry (WKT), fused_label, confidence, uncertain_flag
8. Print final summary:
   - Class distribution after fusion
   - Number of uncertain pixels
   - Number of deforestation vs non-deforestation pixels (imbalance ratio)

Expected output on AMD:
Loaded N polygons from M sources
Agreement table: [source | agreement %]
Fused label distribution: {class: count}
Imbalance ratio: X:1 (negative:positive)
Uncertain pixels: N (X%)
Saved → data/processed/fused_labels.parquet

After writing the script, output:
FILES CREATED:

- src/phase1_label_fusion.py
- configs/config.yaml (initial version with placeholder paths)
- requirements.txt (initial)
- src/utils.py (initial with logging setup)

GIT COMMANDS:
git add .
git commit -m "phase-1: label fusion script"
git push

RUN ON AMD:
python src/phase1_label_fusion.py

Then wait for the user to paste terminal output before writing any Phase 2 code.
After reviewing output, ask: "Phase 1 results look [good/need fixing]. Shall I proceed to Phase 2 or fix anything first?"

---

## PHASE 2 — Feature extraction

Script: src/phase2_feature_extraction.py

Tasks:

1. Load fused_labels.parquet — iterate over each pixel by tile_id, row, col
2. For each pixel, extract AEF embedding:
   - Open the corresponding .tif with rasterio
   - Read all 64 bands at that pixel location
   - Result: aef_vec shape (64,)
3. From Sentinel-2 time series, compute temporal stats per pixel:
   - Compute NDVI = (NIR - Red) / (NIR + Red) at each timestep
   - Compute NBR = (NIR - SWIR) / (NIR + SWIR) at each timestep
   - Per index: mean, std, min, max, linear trend slope, max single-step drop
   - Also store the raw time series arrays for Phase 3b (save separately)
4. From Sentinel-1 time series, compute temporal stats per pixel:
   - VV backscatter: mean, std, min, max, max change magnitude
   - VH backscatter: mean, std, min, max, max change magnitude
   - VV/VH ratio time series: mean, std
5. Concatenate into feature matrix:
   - X shape: (n_pixels, 64 + n_S2_stats + n_S1_stats)
   - Print exact feature group breakdown
6. Handle missing data:
   - If a pixel has NaN in any band (cloud mask, no-data) → fill with group mean for that feature
   - Print count of pixels affected
7. Save outputs:
   - data/processed/features_static.parquet ← X matrix + pixel_id + tile_id
   - data/processed/labels.parquet ← y (fused_label) + pixel_id + tile_id
   - data/processed/s2_timeseries.npy ← shape (n_pixels, T, S2_bands) for Phase 3b
   - data/processed/s1_timeseries.npy ← shape (n_pixels, T, S1_bands) for Phase 3b

Expected output on AMD:
Feature matrix shape: (N, D)
Feature breakdown: AEF=64, S2_stats=X, S1_stats=X, total=D
Pixels with imputed values: N (X%)
Saved → data/processed/features_static.parquet
Saved → data/processed/s2_timeseries.npy shape (N, T, B)
Saved → data/processed/s1_timeseries.npy shape (N, T, B)

After writing the script, output:
FILES CREATED/MODIFIED:

- src/phase2_feature_extraction.py
- src/utils.py (add feature extraction helpers)
- configs/config.yaml (add S1/S2 band name mappings)

GIT COMMANDS:
git add .
git commit -m "phase-2: feature extraction script"
git push

RUN ON AMD:
python src/phase2_feature_extraction.py

Then wait for the user to paste terminal output before writing any Phase 3 code.
After reviewing output, ask: "Phase 2 results look [good/need fixing]. Shall I proceed to Phase 3a or fix anything first?"

---

## PHASE 3a — Baseline model (XGBoost)

Script: src/phase3a_baseline.py

Tasks:

1. Load features_static.parquet and labels.parquet
2. Split by tile_id — put at least one full tile aside as validation (never split a tile across train/val)
   - Print which tiles are in train vs val
3. Train XGBoost classifier:
   - scale_pos_weight = (n_negative / n_positive) to handle imbalance
   - n_estimators=500, early_stopping_rounds=50, eval_metric=["logloss", "aucpr"]
   - Use val set as eval set for early stopping
4. Evaluate on val tile:
   - Classification report (precision, recall, F1 per class)
   - AUC-ROC
   - AUC-PR (more meaningful for imbalanced data)
   - Confusion matrix
5. Print top 20 feature importances by gain — label them by feature group (AEF dim X, NDVI mean, VV std, etc.)
6. Save:
   - data/checkpoints/xgboost_baseline.joblib
   - data/processed/baseline_val_predictions.parquet (pixel_id, y_true, y_pred, y_prob)

Expected output on AMD:
Train tiles: [list] Val tile: [tile_id]
Train size: N Val size: N
Best iteration: X
--- Validation metrics ---
Precision: X Recall: X F1: X
AUC-ROC: X AUC-PR: X
Top features: [feature | importance]

After writing the script, output:
FILES CREATED/MODIFIED:

- src/phase3a_baseline.py
- configs/config.yaml (add XGBoost hyperparameters)
- requirements.txt (add xgboost, joblib)

GIT COMMANDS:
git add .
git commit -m "phase-3a: xgboost baseline"
git push

RUN ON AMD:
python src/phase3a_baseline.py

Then wait for the user to paste terminal output before writing any Phase 3b code.
After reviewing output, ask: "Phase 3a results look [good/need fixing]. Shall I proceed to Phase 3b (temporal model) or fix anything first?"

---

## PHASE 3b — Temporal model (LSTM + AEF fusion)

Script: src/phase3b_temporal_model.py

Tasks:

1. Load s1_timeseries.npy, s2_timeseries.npy, features_static.parquet (AEF columns only), labels.parquet
2. Build a PyTorch Dataset class:
   - **getitem** returns (s1_ts, s2_ts, aef_vec, label)
   - s1_ts shape: (T, S1_bands), s2_ts shape: (T, S2_bands), aef_vec shape: (64,)
3. Build model architecture in src/utils.py as a reusable class:

   class DeforestationModel(nn.Module):
   def **init**(self, s1_bands, s2_bands, aef_dim=64, hidden=64): # Branch A: concat S1+S2 → LSTM(hidden) → temporal_emb(hidden) # Branch B: AEF → Linear(64) → ReLU → aef_emb(64) # Fusion: concat(temporal_emb, aef_emb) → MLP(128→64→32→1) → sigmoid

4. Training loop:
   - Loss: weighted BCEWithLogitsLoss (pos_weight = imbalance ratio tensor on GPU)
   - Optimizer: Adam lr=1e-3
   - Scheduler: ReduceLROnPlateau on val F1, patience=5
   - Early stopping: patience=10 on val F1
   - device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   - Move model and all tensors to device
   - Print epoch, train loss, val F1 every epoch
5. Same region-based tile split as Phase 3a
6. Evaluate with identical metrics to Phase 3a — print a side-by-side comparison table
7. Save:
   - data/checkpoints/temporal_model_best.pt
   - data/processed/temporal_val_predictions.parquet

Expected output on AMD:
Using device: cuda (or cpu)
Epoch 1/100 | train_loss: X | val_F1: X
...
Early stopping at epoch X
--- Temporal model vs Baseline ---
Metric | XGBoost | Temporal model
F1 | X | X
AUC-ROC | X | X
AUC-PR | X | X

After writing the script, output:
FILES CREATED/MODIFIED:

- src/phase3b_temporal_model.py
- src/utils.py (add DeforestationModel class, Dataset class, training loop)
- configs/config.yaml (add model hyperparameters)
- requirements.txt (add torch, if not already present)

GIT COMMANDS:
git add .
git commit -m "phase-3b: lstm+aef temporal model"
git push

RUN ON AMD:
python src/phase3b_temporal_model.py

Then wait for the user to paste terminal output before writing any Phase 4 code.
After reviewing output, ask: "Phase 3b results look [good/need fixing]. Shall I proceed to Phase 4 (generalization hardening) or fix anything first?"

---

## PHASE 4 — Generalization hardening

Script: src/phase4_generalization.py

Tasks:

1. Leakage audit:
   - Confirm zero overlap between train tile_ids and val tile_ids
   - Print confirmation: "No geographic leakage detected" or list any overlapping tile_ids
2. Load the best model from Phase 3a or 3b (whichever had higher AUC-PR)
3. Spatial neighbor averaging (test-time augmentation):
   - For each val pixel, also extract features for its 4 neighbors (row±1, col±1) from the static feature parquet
   - Average the 5 predictions (center + 4 neighbors) as the final prediction
   - Compare F1/AUC before and after neighbor averaging
4. Run best model on a fully held-out tile (different from val tile, never seen in any training):
   - Report performance: F1, AUC-ROC, AUC-PR
   - Compare to in-distribution val performance
   - If F1 drop > 10 points: flag as generalization failure and suggest fix
5. If generalization failure:
   - Fix A: add geographic coordinate features (normalized lat/lon) and retrain — write updated training script
   - Fix B: oversample tiles from diverse regions during training — add tile-balanced sampler to DataLoader

Expected output on AMD:
Leakage check: PASSED / FAILED
Neighbor averaging: F1 before=X after=X (delta=+X)
In-distribution val F1: X
Out-of-distribution F1: X
Generalization gap: X points [OK / WARNING]

After writing the script, output:
FILES CREATED/MODIFIED:

- src/phase4_generalization.py
- src/utils.py (add neighbor averaging helper)

GIT COMMANDS:
git add .
git commit -m "phase-4: generalization hardening"
git push

RUN ON AMD:
python src/phase4_generalization.py

Then wait for the user to paste terminal output before writing any Phase 5 code.
After reviewing output, ask: "Phase 4 results look [good/need fixing]. Shall I proceed to Phase 5 (self-training) or fix anything first?"

---

## PHASE 5 — Noise handling & self-training

Script: src/phase5_self_training.py

Tasks:

1. Load uncertain pixels flagged in Phase 1 (uncertain_flag == True from fused_labels.parquet)
2. Load best model checkpoint (from Phase 3a or 3b)
3. Run inference on uncertain pixels — get predicted probability for each
4. Assign pseudo-labels:
   - If prob > 0.85 → pseudo-label = 1 (deforestation)
   - If prob < 0.15 → pseudo-label = 0 (non-deforestation)
   - Otherwise → still uncertain, exclude from retraining
5. Add pseudo-labeled pixels to training set with confidence weight = model probability
6. Retrain the best model for one self-training round:
   - Use weighted loss where uncertain pixels are down-weighted by 0.5 vs confident pixels
   - Same train/val tile split
7. Evaluate retrained model vs original — print comparison
8. Source reliability audit:
   - For each original label source, compute how often it agreed with final model predictions
   - Print ranked table: most reliable → least reliable source
9. Save:
   - data/checkpoints/self_trained_model_best.pt (or .joblib for XGBoost)
   - data/processed/pseudolabels.parquet

Expected output on AMD:
Uncertain pixels: N total
Pseudo-labeled: N (deforestation=X, non-deforestation=X)
Still uncertain after self-training: N
--- Self-training comparison ---
Metric | Before | After
F1 | X | X
AUC-PR | X | X
--- Source reliability ranking ---
Source | Agreement %
...

After writing the script, output:
FILES CREATED/MODIFIED:

- src/phase5_self_training.py

GIT COMMANDS:
git add .
git commit -m "phase-5: noise handling and self-training"
git push

RUN ON AMD:
python src/phase5_self_training.py

Then wait for the user to paste terminal output before writing any Phase 6 code.
After reviewing output, ask: "Phase 5 results look [good/need fixing]. Shall I proceed to Phase 6 (temporal localization bonus) or fix anything first?"

---

## PHASE 6 (BONUS) — Temporal localization

Script: src/phase6_temporal_localization.py

Tasks:

1. Load best model predictions — filter to pixels where deforestation_predicted == True
2. For each predicted deforestation pixel, run PELT changepoint detection on NDVI time series:
   - pip install ruptures (add to requirements.txt)
   - model = ruptures.Pelt(model="rbf").fit(ndvi_series)
   - breakpoint_idx = model.predict(pen=3)[0]
   - Map breakpoint_idx → timestamp from the S2 time series metadata
3. Run same changepoint detection on VV backscatter from S1 time series
4. Cross-check NDVI vs S1 breakpoints:
   - If they agree within ±1 month → temporal_confidence = "high"
   - Otherwise → temporal_confidence = "low"
5. Save final output GeoJSON:
   - data/processed/deforestation_predictions.geojson
   - columns: pixel_id, geometry, deforestation_predicted, predicted_month, predicted_year, temporal_confidence
6. Print summary:
   - Total deforestation pixels predicted
   - High-confidence temporal predictions count
   - Distribution of predicted deforestation months (histogram in text)
   - 5 sample rows of the output GeoJSON

Expected output on AMD:
Deforestation pixels predicted: N
High-confidence temporal predictions: N (X%)
Month distribution:
2021-01: XX pixels
2021-02: XX pixels
...
Sample output:
pixel_id | month | year | confidence
...
Saved → data/processed/deforestation_predictions.geojson

After writing the script, output:
FILES CREATED/MODIFIED:

- src/phase6_temporal_localization.py
- requirements.txt (add ruptures)

GIT COMMANDS:
git add .
git commit -m "phase-6: temporal localization bonus"
git push

RUN ON AMD:
python src/phase6_temporal_localization.py

Then wait for the user to paste terminal output.
After reviewing output, ask: "All 6 phases complete. Would you like me to: (a) generate a final summary report, (b) build a visualization tool for the GeoJSON output, or (c) refine any specific phase?"

---

## GENERAL RULES (apply throughout all phases)

- Always print shape and dtype after loading any dataset — never assume shapes are correct
- Add to requirements.txt immediately when any new library is introduced — never let requirements drift
- All scripts must be runnable standalone: python src/phaseN_xxx.py with no extra arguments
- config.yaml is the single source of truth for all paths and hyperparameters — no hardcoded values in scripts
- Every script must end with a clear DONE message and list of files saved with their paths
- GPU awareness is mandatory: device = torch.device("cuda" if torch.cuda.is_available() else "cpu") in every PyTorch script, tensors and model always moved to device
- If AMD cloud throws an error the user pastes back, diagnose it here and fix the code — do not ask the user to fix it manually
- Never split a tile across train and val — always split by full tile_id
- Prefer AUC-PR over AUC-ROC as the primary metric — deforestation pixels are rare, AUC-PR is more informative for imbalanced detection
- Add brief inline comments explaining WHY each modeling choice was made, not just what it does
  message.txt
  20 KB
