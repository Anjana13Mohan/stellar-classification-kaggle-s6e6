> 🔗 Competition: [Kaggle Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6)

# Stellar Classification — Kaggle Playground Series S6E6

**Final Rank: 576 / 2816 (Top 20.5%)**

## 🔭 Problem

Classify astronomical objects as **GALAXY**, **QSO (Quasar)**, or **STAR** using photometric measurements from the Sloan Digital Sky Survey (SDSS). Metric: Balanced Accuracy (equal weight to all 3 classes).

---

## 💡 Key Insight

QSOs are the hardest class to separate from STARs. The fix comes from astrophysics: stars follow a tight, predictable curve in color-color space called the **Stellar Locus** (u-g vs g-r diagram). QSOs scatter off this locus because their light comes from accretion disk emission, not stellar atmospheres. Computing each object's distance from this locus (`locus_distance`) was one of the most powerful features in the pipeline.

---

## 🔧 Feature Engineering

Physics-informed features built from raw photometric bands (u, g, r, i, z):

- **9 color indices** (u-g, g-r, r-i, i-z, etc.) — spectral shape proxies
- **Stellar locus distance** — perpendicular distance from the fitted stellar locus in u-g vs g-r space
- **Flux features** — 10^(-0.4×mag) for all 5 bands + statistics
- **Color-plane geometry** — radius and angle in u-g vs g-r and r-i vs i-z diagrams
- **Redshift transforms** — log(1+z), z², redshift zones, redshift × magnitude interactions
- **3D sky coordinates** — phys_x/y/z from right ascension, declination, and log-redshift
- **Spectral slope and curvature** — mag_slope, blue_curvature, red_curvature
- **Target encoding** — for spectral type, galaxy population, and redshift zone combinations

---

## 🏗️ Model Architecture

### RealMLP (Primary Model)
Custom PyTorch implementation of RealMLP with:
- **PBLD (Periodic + Bias + Linear + Dense) embeddings** for continuous features
- **NTP (Normalized Two-Pass) linear layers** with per-ensemble weights
- **Categorical embeddings** with one-hot encoding for low-cardinality features
- **Floor-bucketing** of 8 raw numeric columns into categorical features
- **Label smoothing** with cosine schedule
- **Dropout** with exponential decay schedule
- **10-ensemble averaging** per fold
- 3 seeds (42, 123, 456) for diversity

### Tree Models (Diversity)
- **LightGBM** — gradient boosting with balanced class weights
- **XGBoost** (GPU) — with sample weighting and target encoding
- **CatBoost** (GPU) — with native categorical feature handling

### Stacking
**PyTorch Logistic Regression Stacker** (inspired by Chris Deotte's approach):
- 5-fold cross-validation × 5 seeds (25 total fits)
- Inputs: logit-transformed probabilities from all base models
- Class-weighted cross-entropy loss
- Adam optimizer with L2 regularization

---

## 📊 Results

| Model | OOF Balanced Accuracy |
|---|---|
| RealMLP seed=42 | 0.96935 |
| RealMLP seed=123 | 0.96787 |
| XGBoost | 0.96391 |
| CatBoost | 0.96407 |
| LightGBM | 0.95728 |
| **5-model LR Stacker** | **0.96993** |

**Private LB: 0.96945 — Rank 576/2816 (Top 20.5%)**

---

## 🛡️ Validation Strategy

A key principle throughout: **trust OOF score, not public LB score.**

The public LB uses only 20% of test data. Many competing notebooks used LB-probing (submitting hundreds of variants to reverse-engineer the test labels). Our OOF score matched the public LB score almost exactly (0.96993 OOF vs 0.96995 public), which confirmed the pipeline was well-calibrated and not overfitting to the public sample. We gained 28 ranks on the private LB as a result.

---

## 🗂️ Notebooks

| Notebook | Purpose |
|---|---|
| `stellar-realmlp-seed42` | RealMLP training, seed=42 |
| `stellar-realmlp-seed123` | RealMLP training, seed=123 |
| `stellar-realmlp-seed456` | RealMLP training, seed=456 |
| `stellar-lgb-xgb-reseed` | LGB seed=123 + XGB seed=123 |
| `stellar-cat-seed123` | CatBoost seed=123 |
| `stellar-xgb-v2` | Stronger XGBoost with TE |
| `stellar-5model-lr-stacker` | Final stacker (best submission) |
| `stellar-hillclimb` | Hill climbing ensemble (comparison) |
| `stellar-qso-star-residual` | QSO→STAR residual correction |

---

## 🔑 Key Learnings

1. **Physics-informed features outperform blind feature engineering** — understanding why QSOs scatter off the stellar locus led directly to the most useful feature
2. **Architecture diversity matters more than seed diversity** — adding a second RealMLP seed helped (+0.0004) but adding duplicate tree models hurt
3. **OOF=LB match is the gold standard** — our 0.96993 OOF matched 0.96995 public LB, confirming no overfitting
4. **LR stacking > hill climbing** for this task — per-class learned weights (18 input features) outperformed scalar blend weights
5. **Resist LB-probing temptation** — many notebooks above us on public LB dropped on private; we gained 28 ranks

---

## 🛠️ Tech Stack

- Python, PyTorch, LightGBM, XGBoost, CatBoost
- scikit-learn (StratifiedKFold, TargetEncoder, KBinsDiscretizer)
- Kaggle GPU T4 environment (PyTorch cu128)

---

## 👤 Author

**Anjana Mohan** — Nutritionist turned ML Practitioner
- MS in AI & ML (Scaler/Woolf University, 2025)
- [GitHub](https://github.com/Anjana13Mohan)
- [LinkedIn](https://linkedin.com/in/anjana-mohan-21a64a65)

---
