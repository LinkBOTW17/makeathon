# Deforestation Detection — TUM AI Makeathon 2025
### osapiens Challenge: Detecting Deforestation from Space

---

## Problem

Detect deforestation events across tropical regions from multimodal satellite data, predicting **where** (polygon geometry) and **when** (year 2021–2024) each event occurred.

**Primary metric:** Union IoU = TP / (TP + FP + FN) computed over predicted polygon area vs. human-annotated ground truth polygons.  
**Secondary metrics:** Polygon Recall, False Positive Rate, Year Accuracy.

---

## Pipeline Overview

```
Raw satellite data (S1, S2, AEF)
        │
        ▼
[1] mislabel_detection_v2.py   — Confident Learning noisy-label correction
        │
        ▼
[2] apply_corrections.py       — Burn corrected polygon labels to pixel GT
        │
        ▼
[3] pixel_model.py  --mode all — Train LightGBM + predict test tiles
        │
        ▼
results/submission.geojson     — Competition submission
```

---

## Data Sources

| Modality | Source | Signal |
|----------|--------|--------|
| Sentinel-2 optical | `sentinel-2/` | NDVI time series (monthly, 2020–2024) |
| Sentinel-1 SAR | `sentinel-1/` | Backscatter dB (cloud-penetrating) |
| AEF embeddings | `aef-embeddings/` | 64-dim foundation model vectors per pixel per year |
| Weak labels | `labels/train/` | RADD, GLAD-S2, GLAD-L alert rasters |
| Human annotations | `metadata/` | Per-tile polygon shapefiles (leaderboard GT) |

---

## Key Insight: Label Alignment

**The root cause of early poor performance:** training used RADD/GLAD majority-vote pixel alerts as ground truth, while the leaderboard evaluates against human-drawn polygon annotations. These are systematically different — alerts fire only on individual alerted pixels; polygon annotations cover the entire deforested patch boundary.

**Fix:** `build_gt_pixel_map()` in `validate_spatial.py` burns the `corrected_label` from each polygon directly into a pixel raster. RADD/GLAD signals are retained only for year estimation (day fields → calendar year).

---

## Step 1 — Noisy Label Handling (`mislabel_detection_v2.py`)

Raw polygon labels carry annotator disagreement across 3–5 label sources. We apply **Confident Learning** (Northcutt et al., 2021):

1. Build consensus polygons per tile via spatial union of label sources.
2. Extract per-polygon features: AEF mean embeddings, NDVI pre/post, SAR pre/post, harmonic regression change score.
3. Train a `RandomForestClassifier` with 5-fold stratified CV → calibrated out-of-fold predicted probabilities `p_deforestation`.
4. Flag polygons where the confident estimate disagrees with the given label (`mislabel_score ≥ 0.5`).
5. Output: `results/mislabels_v2.csv` with `given_label`, `suggested_label`, `flagged`, `mislabel_score`.

```bash
python deforestation/mislabel_detection_v2.py --output results/mislabels_v2.csv
```

---

## Step 2 — Apply Corrections (`apply_corrections.py`)

Converts polygon-level corrections into pixel-level training artefacts:

- **Rasters:** Burns original + corrected labels into 2-band GeoTIFFs per tile (`results/corrected_labels/`).
- **Feature CSV:** Re-extracts per-polygon features with corrected labels → `results/training_features.csv`.

```bash
python deforestation/apply_corrections.py --mode all --score-threshold 0.5
```

---

## Step 3 — Pixel-Level Model (`pixel_model.py`)

### Feature Engineering — 157 features per pixel

| Group | Count | Signal |
|-------|-------|--------|
| AEF L2 deltas (year-over-year) | 4 | Sudden embedding shift = forest clearing |
| AEF cosine deltas | 4 | Directional change in embedding space |
| AEF cumulative drift from 2020 | 4 | Total change since observation start |
| AEF peak change + timing | 2 | Max L2 delta + which year |
| AEF change vector (emb_2024 − emb_2020) | 64 | Direction of change; discriminates deforestation from regrowth |
| NDVI pre/post/change/std | 4 | Vegetation loss signal |
| SAR pre/post/change | 3 | Cloud-penetrating structural change |
| Harmonic breakpoint t-stat + magnitude | 2 | Per-pixel BFAST-style NDVI anomaly detection |
| AEF 2020 baseline embedding | 64 | Forest type encoder; enables cross-region generalisation |
| 5×5 spatial neighborhood means | 6 | Context: isolated pixels vs. coherent patches |

**Harmonic regression detail:** OLS fitted on the pre-2022 NDVI time series using 2 harmonics (annual + semi-annual cycles). The t-statistic of post-2022 residuals is a physics-grounded deforestation signal — equivalent to per-pixel BFAST, computed in vectorised form across all H×W pixels simultaneously.

**Neighborhood features:** `uniform_filter(size=5)` applied to `aef_max_delta_l2`, `ndvi_change`, `harmonic_t_stat`, and SAR signals. A truly deforested pixel is surrounded by similarly changed neighbours; isolated noise is not. These ranked #1 and #3 in LightGBM feature importance for both regional models.

### Model Design

**Algorithm:** LightGBM gradient-boosted trees, pixel-level binary classifier.

**Why not deep learning:** With 13–16 training tiles, a CNN/U-Net would overfit. LightGBM with explicit feature engineering reaches comparable accuracy at this data scale, trains in ~5–10 minutes, and produces interpretable feature importances.

```python
LGBMClassifier(
    n_estimators=1000, learning_rate=0.03,
    max_depth=6, num_leaves=32,
    class_weight="balanced",
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.5,
)
```

**Sampling:** All positive pixels per tile (uncapped), 2× negatives. Tiles where >80% of pixels are labeled positive are excluded — such tiles (e.g. near-fully-deforested patches) distort the model's probability calibration and inflate LOTO thresholds.

### Regional Models

Two regional models are trained and selected at inference by tile prefix:

| Region | Tile prefixes | Biome |
|--------|--------------|-------|
| SEA | 47xxx, 48xxx | Southeast Asia — fragmented small-scale deforestation |
| SAM | 18xxx, 19xxx, 33xxx | South America / Africa — large-scale Amazon clearings |

The 64-dim AEF 2020 baseline embedding acts as an implicit forest-type encoder within each regional model, enabling cross-tile generalisation without hand-engineered region flags.

### Threshold Selection — LOTO Cross-Validation

**Leave-One-Tile-Out CV** is the only valid evaluation strategy given 6–7 training tiles per region.

For each held-out tile:
1. Train on remaining tiles in the region.
2. Predict the full held-out tile (not just sampled pixels).
3. Apply the complete post-processing pipeline.
4. Grid-search threshold t ∈ [0.15, 0.85] to maximise IoU on the processed binary mask.
5. Refine with 0.01-step search around the best coarse threshold.

Region threshold = **median of per-tile IoU-optimal thresholds, capped at 0.60** (prevents outlier tiles from spiking the region threshold and collapsing test recall).

### Post-Processing

```
Raw probability map  (H × W float32)
    ↓  Gaussian filter σ=1.5      remove isolated-pixel false positives
    ↓  Threshold at region t       binary mask
    ↓  Binary opening (1 iter)     remove small noise blobs
    ↓  Binary closing (2 iters)    fill interior holes in deforested patches
    ↓  Polygonize (rasterio.shapes)
    ↓  Filter ≥ 0.1 ha             remove sub-pixel fragmentation
    ↓  Assign year from NDVI change year map
```

### Year Assignment

Per-pixel year estimation uses the NDVI time series: for each year pair (2021–2024), compute the annual mean NDVI drop; assign the year with the largest drop as the deforestation year. This improved year accuracy from 0% (fixed year=2021 in early submissions) to ~10%+ on training tiles.

---

## Running the Full Pipeline

```bash
# From makeathon root

# 1. Detect and score mislabeled polygons
python deforestation/mislabel_detection_v2.py \
    --output results/mislabels_v2.csv

# 2. Apply corrections, export corrected feature CSV
python deforestation/apply_corrections.py \
    --mislabels results/mislabels_v2.csv \
    --mode all \
    --score-threshold 0.5

# 3. Train model + predict test tiles + build submission
python deforestation/pixel_model.py \
    --mode all \
    --mislabels results/mislabels_v2.csv \
    --out-dir results

# Output: results/submission.geojson
```

---

## Results

| Version | Ground Truth | Key Change | Leaderboard IoU | Rank |
|---------|-------------|-----------|----------------|------|
| Baseline | RADD/GLAD majority vote | Basic AEF features | ~33% | 21 |
| v2 | RADD/GLAD | Harmonic breakpoint + SAR | ~38% | — |
| v3 | RADD/GLAD | NDVI year estimation, dual-threshold | 41.67% | 13 |
| v4 | **Polygon annotations** | Spatial neighbourhood features, morphological post-processing | Pending | — |

---

## File Reference

| File | Purpose |
|------|---------|
| `mislabel_detection_v2.py` | Confident Learning pipeline — scores and flags mislabeled polygons |
| `mislabel_detection.py` | v1 prototype (superseded) |
| `apply_corrections.py` | Burns corrected polygon labels to pixel GT rasters + feature CSV |
| `augment_aef_changes.py` | Standalone script to add AEF year-over-year change features to CSV |
| `validate_spatial.py` | Pixel-level IoU/Recall/FPR/YearAcc metrics on training tiles; `build_gt_pixel_map()` |
| `pixel_model.py` | Main: feature extraction, LightGBM training, LOTO-CV, prediction, submission build |
| `train_model.py` | Alternative ensemble prototype (AEF anomaly + LightGBM); superseded by pixel_model.py |

---

## Engineering Notes

- All satellite data is reprojected on-the-fly to a common Sentinel-2 reference grid using `rasterio.warp.reproject` (bilinear for continuous data, nearest for label rasters).
- NaN handling: `nanmedian` fill at training time; stored in the scaler artifact for consistent inference.
- Submission format: GeoJSON with `time_step` field as `YYMM` string (e.g. `"2106"` = June 2021, fixed mid-year).
- Model artifact (`results/pixel_model.pkl`): serialized via `joblib`; contains regional models, scalers with embedded NaN-fill and threshold, and feature names for reproducibility checks.
