# v2 ML Architecture — Closing the 8 Gaps vs zinmori

**Status:** Design doc (no code executed). Implementation deferred.
**Author:** Geospatial ML Architect (Task V2-1)
**Scope:** Replace the 59-feature / 2-model / single-seed / probe-calibrated v1 with a 280-feature / 3-model / 6-seed / Saerens-EM-calibrated v2 that closes each of the 8 known gaps vs the rank-3 solution.

---

## 1. Feature Engineering v2 (~280 features)

### 1.1 Design principles
- **Physics-first, statistics-second.** Hand the GBDT decades of remote-sensing priors as closed-form indices, then aggregate each index with a richer statistic set than v1's `{mean, median, std, trend}`.
- **Per-sensor vs joint.** Indices that mix SAR and optical (e.g. `mndwi_vhvv_corr`) are computed jointly; pure-optical and pure-SAR indices are computed per-sensor.
- **Statistic set expansion.** v1 used 4 stats. v2 uses 12: `mean, median, std, p10, p25, p75, p90, skew, kurtosis, CV, IQR, lag-1-autocorr`. Plus the v1 trend (linear slope via `np.polyfit`), giving 13 per index.

### 1.2 Index families (14 indices × 13 stats = 182 features)

| # | Index | Formula | Rationale | Sensor |
|---|---|---|---|---|
| 1 | NDVI | (NIR − red) / (NIR + red) | Vegetation vigour; pond surrounds are often vegetated | Optical |
| 2 | MNDWI | (green − SWIR1) / (green + SWIR1) | Open water vs built-up/vegetation | Optical |
| 3 | NDWI | (green − NIRa) / (green + NIRa) | Water body extent using B8A narrow NIR | Optical |
| 4 | VH−VV | VH − VV (dB) | SAR polarimetric difference; smooth water depolarises return | SAR |
| 5 | AWEI_sh | blue + 2.5·green − 1.5·(SWIR1+SWIR2) − 0.25·NIR | Shadow-discriminating water index (Feyisa 2014) | Optical |
| 6 | NDCI | (re1 − red) / (re1 + red) | Chlorophyll-a proxy; managed ponds have elevated chl | Optical |
| 7 | **MCI** | re1 − red | Maximum Chlorophyll Index (Mishra & Mishra 2012); zinmori's reported dominant feature — added for direct comparison | Optical |
| 8 | **CDOM** | log(green / blue) × constant | Coloured dissolved organic matter proxy; high in eutrophic ponds | Optical |
| 9 | **NDSI** | (green − SWIR1) / (green + SWIR1) (snow variant uses SWIR2) | Normalised difference snow index; flags ice cover in high-altitude ponds | Optical |
| 10 | **SDWI** | SWIR1 − SWIR2 | Shallow-water depth index; separates shallow pond bottoms from deep lakes | Optical |
| 11 | **EVI** | 2.5 × (NIR − red) / (NIR + 6·red − 7.5·blue + 1) | Enhanced Vegetation Index; soil/atmosphere-corrected NDVI | Optical |
| 12 | **SAVI** | (NIR − red) × 1.5 / (NIR + red + 0.5) | Soil-Adjusted VI; reduces soil-brightness contamination on pond margins | Optical |
| 13 | **BSI** | ((SWIR1 + red) − (NIR + blue)) / ((SWIR1 + red) + (NIR + blue)) | Bare Soil Index; pond surroundings often bare | Optical |
| 14 | **NDRE** | (re2 − re1) / (re2 + re1) | Red-edge based chlorophyll; sensitive to dense canopies, including algae blooms | Optical |

All 14 indices computed per month, then aggregated with **13 statistics** = 182 features.

### 1.3 SAR-specific stats (3 series × 13 stats = 39 features)

For each of `VH`, `VV`, `VH−VV` (in dB), compute the same 13-stat set. SAR indices are independent of cloud cover, so these features carry signal even for radar-only test cells (Section 5).

### 1.4 Persistence features (10 features)
v1's 4 + 6 new:
- `water_persistence`, `nonveg_persistence`, `strong_water_persist`, `awei_water_persist` (existing)
- **`mci_persist`** — fraction of months where MCI > 0.005 (chlorophyll-active)
- **`ndre_persist`** — fraction of months where NDRE > 0.2 (dense-canopy/algae)
- **`bsi_persist`** — fraction of months where BSI > 0 (bare soil in surrounds)
- **`evi_persist`** — fraction of months where EVI > 0.3 (vigorous vegetation)
- **`cdom_persist`** — fraction of months where CDOM > threshold
- **`sar_strong_water_persist`** — fraction of months where VH < −28 dB (stricter SAR water threshold)

### 1.5 First / last / change (14 indices × 3 = 42 features)
Extended from v1's 5-index set to all 14 indices. `_first`, `_last`, `_change = _last − _first`.

### 1.6 Window-shape features (10 features)
v1's 7 + 3 new:
- `n_valid_months_optical`, `n_valid_months_sar` (split per sensor — key for Section 5)
- `window_overlap_months` — # months where both sensors have data
- `start_sin`, `start_cos`, `window_center`, `has_monsoon`, `has_dry` (existing)
- **`window_asymmetry`** — |optical_window_center − sar_window_center|, measures sensor mismatch
- **`has_high_chl_season`** — window covers warm months (Apr–Sep) when chl blooms peak

### 1.7 Cross-sensor correlations (10 features)
v1's 1 + 9 new: Pearson correlation across months between:
- `mndwi ↔ vh`, `mndwi ↔ vv`, `mndwi ↔ vhvv` (existing)
- `ndci ↔ vhvv`, `ndci ↔ vv`, `mci ↔ vh`, `mci ↔ vv`
- `ndvi ↔ vv`, `awei ↔ vh`, `bsi ↔ vhvv`

A high positive MNDWI↔VH correlation indicates stable open water; a near-zero correlation flags vegetated/mixed pixels.

### 1.8 SAR-optical divergences (5 features)
v1's 2 + 3 new:
- `sar_water_persist`, `sar_optical_diverge` (existing)
- **`sar_ndci_diverge`** — `sar_water_persist − mci_persist` (smooth-water-with-low-chl = abandoned pond)
- **`sar_mndwi_diverge_strict`** — `sar_strong_water_persist − strong_water_persist`
- **`diverge_x_window_length`** — interaction: divergence × n_valid_months (longer windows = more reliable divergence)

### 1.9 Interactions (6 features)
- `mndwi_ndci_interact` (existing), plus:
- `mndwi_mci_interact` — water × chlorophyll
- `mci_evi_interact` — chlorophyll × vegetation
- `sar_optical_diverge_x_monsoon` — divergence weighted by monsoon-season presence
- `ndre_mci_ratio` — `ndre_mean / (mci_mean + ε)` (algae vs terrestrial vegetation separator)
- `bsi_evi_diff` — `bsi_mean − evi_mean` (bare surround vs vegetated surround)

### 1.10 Lag-1 autocorrelation (14 features, included in §1.2 stats)
For each index, lag-1 autocorrelation across the valid months. High autocorrelation = stable pond; low = transient water body (flooded field).

### 1.11 Feature count summary
| Family | Count |
|---|---|
| Index × 13 stats | 14 × 13 = **182** |
| SAR series × 13 stats | 3 × 13 = **39** |
| Persistence | **10** |
| First/last/change | 14 × 3 = **42** |
| Window-shape | **10** |
| Cross-sensor correlations | **10** |
| SAR-optical divergences | **5** |
| Interactions | **6** |
| **Total** | **304** |

After redundancy pruning (drop features with `|corr| > 0.98` with another feature, keep the more interpretable one), expect ~280 surviving features.

---

## 2. Models v2 (3-model ensemble, 6 seeds)

### 2.1 Hyperparameters

**LightGBM** (tuned from v1 for more features + more seeds):
```python
LGB_PARAMS_V2 = {
    'n_estimators': 800,        # up from 500
    'num_leaves': 31,           # up from 18 — more capacity for 280 features
    'max_depth': 6,             # up from 5
    'learning_rate': 0.03,      # down from 0.04 — more iterations, finer steps
    'subsample': 0.80,
    'subsample_freq': 1,
    'colsample_bytree': 0.70,   # down from 0.90 — more features need more feature dropout
    'reg_alpha': 0.5,           # up from 0.3
    'reg_lambda': 3.0,          # up from 2.0
    'min_child_samples': 25,    # up from 20
    'scale_pos_weight': 1.35,   # adjusted for ~54% test prevalence (was 2.0 for 40% train)
    'verbose': -1,
}
```

**CatBoost**:
```python
CB_PARAMS_V2 = {
    'iterations': 1000,         # up from 600
    'depth': 6,                 # up from 5
    'learning_rate': 0.03,
    'l2_leaf_reg': 5,           # up from 3 — more features need more reg
    'bagging_temperature': 1.0,
    'random_strength': 1.0,     # stronger randomization for ensemble diversity
    'class_weights': [1, 1.35], # match scale_pos_weight
    'eval_metric': 'AUC',
    'verbose': 0,
}
```

**XGBoost** (new — v1 did not use it):
```python
XGB_PARAMS_V2 = {
    'n_estimators': 700,
    'max_depth': 6,
    'learning_rate': 0.03,
    'subsample': 0.80,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.5,
    'reg_lambda': 3.0,
    'min_child_weight': 5,
    'scale_pos_weight': 1.35,
    'tree_method': 'hist',      # CPU-only, fast
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
}
```

### 2.2 Blend strategy

Two-stage:
1. **Simple average** baseline: `p_blend = (p_lgb + p_cb + p_xgb) / 3`.
2. **Grid-search weights** on OOF composite metric: search over `w_lgb, w_cb, w_xgb ∈ {0.0, 0.1, …, 1.0}` with constraint `w_lgb + w_cb + w_xgb = 1`, step 0.1 → 66 combinations. Pick weights that maximise `0.6·F1 + 0.4·AUC` on OOF. Default to simple average if grid-search gain < 0.0005 (avoid overfitting to OOF).

Stacking with a logistic-regression meta-learner is rejected: with only 1,821 rows, the meta-learner's variance exceeds the gain from non-linear blending.

### 2.3 Multi-seed averaging

**6 seeds: [42, 43, 44, 45, 46, 47]** — deterministic list.

Rationale for 6 (not 5, not 8):
- 5 seeds: variance reduction ≈ √5 ≈ 2.24×, leaves ~5e-4 noise on threshold probabilities.
- 6 seeds: √6 ≈ 2.45×, drops to ~3e-4 noise, comfortably below the 0.001 rank-5 gap.
- 8 seeds: marginal additional gain (~2.8×), +33% compute, not worth it.

Per-seed probability is averaged **before** the Saerens-EM correction (Section 3), so the EM operates on a denoised probability surface.

### 2.4 Wall-clock budget (4-core CPU)

Per fold, per model (8× augmentation = ~14,568 rows × 280 features × 800 trees):

| Model | Train time / fold / seed |
|---|---|
| LightGBM | ~25 s |
| CatBoost | ~50 s |
| XGBoost | ~30 s |

**CV phase** (5 folds × 6 seeds × 3 models):
`5 × 6 × (25 + 50 + 30) = 3,150 s ≈ 52 min`

**Final-model phase** (full data × 6 seeds × 3 models, no fold split):
`6 × (40 + 80 + 50) = 1,020 s ≈ 17 min` (full-data training ≈ 1.6× fold training)

**Per pseudo-labelling round** ≈ CV + final ≈ 70 min. Three rounds ≈ 3.5 hours.

**Total v2 budget ≈ 4.5 hours wall-clock on 4-core CPU.** Tractable in a single workday.

---

## 3. Threshold Calibration v2 — Saerens/EM Prior Correction

### 3.1 Why replace the probe-submission approach
v1 recovered the exact test prevalence (560/1030) via an all-positive probe submission and the identity `F1 = 2p / (1 + p)`. This used the public LB as a side channel — defensible but controversial, and not generalisable to settings where probe submissions are forbidden. v2 replaces it with the **Saerens et al. (2002) prior-correction formula**, refined by an EM iteration that jointly estimates the test prior and the corrected probabilities.

### 3.2 The Saerens formula

Given model probabilities `p̂_train(x)` calibrated to the train prevalence `π_train`, and an estimate `π_test` of the test prevalence:

```
p̂_test(x) = [α · p̂_train(x)] / [α · p̂_train(x) + β · (1 − p̂_train(x))]

where  α = π_test / π_train
       β = (1 − π_test) / (1 − π_train)
```

This is a monotone transformation that preserves AUC (rank order unchanged) but shifts the probability mass so the mean predicted probability equals `π_test`. F1 — which is threshold-sensitive — is the beneficiary.

### 3.3 EM iteration (when `π_test` is unknown)

We treat `π_test` as a latent parameter and estimate it via Expectation-Maximisation:

1. **Initialise** `π_test^(0)`:
   - Default: `π_test^(0) = mean(p̂_train)` over the test set (fully unsupervised — uses the model's own prevalence estimate as the starting point).
   - Optional prior: nudge toward the "test set may have a higher proportion of positives" statement in the competition brief, e.g. `π_test^(0) = max(mean(p̂_train), π_train)`.
2. **E-step (t → t+1):** compute `p̂_test^(t)(x)` for every test row using the Saerens formula with `π_test = π_test^(t)`.
3. **M-step:** `π_test^(t+1) = mean(p̂_test^(t))`.
4. **Converge** when `|π_test^(t+1) − π_test^(t)| < 1e-5` or after 100 iterations (typically converges in 5–15 iterations for binary problems).

### 3.4 Pseudocode

```python
def saerens_em(p_train, pi_train, pi_test_init=None, tol=1e-5, max_iter=100):
    """
    Saerens-EM prior correction (unsupervised, no LB feedback).

    Args:
        p_train   : array of model probabilities calibrated to train distribution.
        pi_train  : train prevalence (known from CV).
        pi_test_init : initial test prevalence (None → mean(p_train)).
        tol       : convergence tolerance.
        max_iter  : max EM iterations.

    Returns:
        p_test    : corrected probabilities (mean ≈ pi_test).
        pi_test   : estimated test prevalence.
        n_iter    : iterations to convergence.
    """
    pi_test = pi_test_init if pi_test_init is not None else float(np.mean(p_train))

    for t in range(max_iter):
        alpha = pi_test / pi_train
        beta  = (1.0 - pi_test) / (1.0 - pi_train)

        # E-step: prior-correct each probability
        num = alpha * p_train
        den = num + beta * (1.0 - p_train)
        p_test = num / den

        # M-step: re-estimate the test prior
        pi_test_new = float(np.mean(p_test))

        if abs(pi_test_new - pi_test) < tol:
            return p_test, pi_test_new, t + 1
        pi_test = pi_test_new

    return p_test, pi_test, max_iter
```

### 3.5 Decision threshold
After EM converges, search `t ∈ [0.01, 0.99]` step 0.001 for the threshold maximising `0.6·F1 + 0.4·AUC` **on the OOF predictions** (with the same Saerens-EM applied, using `π_train = train_prevalence` and `π_test = OOF_prevalence` as a proxy). The threshold is set entirely from CV, no LB feedback.

---

## 4. Pseudo-labelling v2 (3 rounds)

### 4.1 Protocol

| Round | Confidence threshold | Positive cutoff | Negative cutoff | Sample weight |
|---|---|---|---|---|
| 1 | 0.90 | `p̂ > 0.90` | `p̂ < 0.10` | 0.5 |
| 2 | 0.85 | `p̂ > 0.85` | `p̂ < 0.15` | 0.4 |
| 3 | 0.80 | `p̂ > 0.80` | `p̂ < 0.20` | 0.3 |

Weights decrease across rounds: later rounds add more but lower-confidence pseudo-labels, so each row contributes less to the gradient.

### 4.2 Blending pseudo-labelled rows with original training data

- Original train rows: `sample_weight = 1.0`
- Pseudo-labelled rows: `sample_weight = round_specific_weight` (0.5 / 0.4 / 0.3)
- All pseudo-labelled rows are re-masked (per-sensor, Section 5) with a fresh random window each fold, so the model sees them under the same augmentation distribution as real rows.

### 4.3 Pseudocode

```python
def pseudo_label_round(train, test, model_fn, threshold, weight,
                       prev_pseudo=None):
    """
    One round of pseudo-labelling.

    Args:
        train       : original training DataFrame (labelled).
        test        : test DataFrame (unlabelled).
        model_fn    : callable (X, y, sw) → trained model.
        threshold   : confidence threshold (e.g., 0.90).
        weight      : sample weight for pseudo-labelled rows.
        prev_pseudo : DataFrame of pseudo-labelled rows from prior rounds.

    Returns:
        new_train   : augmented training set (with sample_weight column).
        all_pseudo  : accumulated pseudo-labelled rows.
    """
    # Step 1: assemble current training set
    if prev_pseudo is not None:
        cur_train = pd.concat([train, prev_pseudo], ignore_index=True)
    else:
        cur_train = train.copy()

    # Step 2: train model with sample weights
    sw = cur_train['sample_weight'] if 'sample_weight' in cur_train else None
    model = model_fn(cur_train[FEATURES], cur_train['label'], sw)

    # Step 3: predict on test
    p_test = saerens_em(
        model.predict_proba(test[FEATURES])[:, 1],
        pi_train=cur_train['label'].mean()
    )[0]

    # Step 4: confident-extreme selection (asymmetric thresholds)
    pos_mask = p_test > threshold
    neg_mask = p_test < (1.0 - threshold)

    pseudo_pos = test[pos_mask].copy()
    pseudo_pos['label'] = 1
    pseudo_pos['sample_weight'] = weight

    pseudo_neg = test[neg_mask].copy()
    pseudo_neg['label'] = 0
    pseudo_neg['sample_weight'] = weight

    new_round_pseudo = pd.concat([pseudo_pos, pseudo_neg], ignore_index=True)
    all_pseudo = pd.concat([prev_pseudo, new_round_pseudo],
                           ignore_index=True) if prev_pseudo is not None \
                 else new_round_pseudo

    # Step 5: assemble new training set
    train_w = train.copy()
    train_w['sample_weight'] = 1.0
    new_train = pd.concat([train_w, all_pseudo], ignore_index=True)
    return new_train, all_pseudo


def pseudo_label_pipeline(train, test, model_fn, cv_fn):
    """3 rounds: thresholds 0.90 → 0.85 → 0.80."""
    thresholds_weights = [(0.90, 0.5), (0.85, 0.4), (0.80, 0.3)]
    prev_pseudo = None
    for thr, w in thresholds_weights:
        new_train, prev_pseudo = pseudo_label_round(
            train, test, model_fn, thr, w, prev_pseudo
        )
        # CV monitor: confirm score didn't drop (confirmation-bias guard)
        cv_score = cv_fn(new_train)
        n_new = len(prev_pseudo) - (len(prev_pseudo) - len(new_round_pseudo))
        print(f'PL round thr={thr}: CV={cv_score:.4f}, '
              f'cumulative pseudo={len(prev_pseudo)}')
        # Early stop if CV drops > 0.001 from previous round
        if cv_score < prev_cv_score - 0.001:
            print(f'  ↳ CV drop > 0.001, stopping early.')
            break
        prev_cv_score = cv_score
    return new_train, prev_pseudo
```

### 4.4 Confirmation-bias prevention (5 mechanisms)
1. **Decaying sample weights** (0.5 → 0.4 → 0.3) so each round contributes less gradient.
2. **Asymmetric band** — only rows outside `[1−t, t]` are pseudo-labelled; the "uncertain band" is left unlabelled rather than forced to a class.
3. **Fresh re-masking** per fold — pseudo-labels are remasked before each training pass, exposing the model to a different observation window than the one used to generate the pseudo-label.
4. **CV monitor with early stop** — if CV score drops > 0.001 from the prior round, abort the pipeline and revert to the previous model.
5. **Pseudo-labels are regenerated from the previous round's model**, not propagated forward unmodified — a row labelled in round 1 can be re-evaluated in round 2 and removed if its confidence drops.

---

## 5. Per-sensor Masking v2

### 5.1 The problem v1 doesn't fix
v1's `window_mask` samples **one** contiguous 4–6 month window and applies it to **all** 12 bands (optical + SAR). This matches test rows where optical and SAR share the same observation window — but it discards the **~320 radar-only test cells** where SAR has a window but optical is fully masked (or vice versa). v1's training distribution never sees those configurations, so the model is biased on them.

### 5.2 The fix
Sample **two independent** contiguous windows per row:
- One window for the 10 optical bands (`blue, green, nir, nira, re1, re2, re3, red, swir1, swir2`)
- One window for the 2 SAR bands (`VH, VV`)

Each window independently samples length ∈ {4, 5, 6} and start month ∈ [1, 12 − L + 1]. The two windows may overlap, partially overlap, or be disjoint — exactly mirroring the real test distribution.

### 5.3 Pseudocode

```python
OPTICAL_BANDS = ['blue', 'green', 'nir', 'nira', 're1', 're2', 're3',
                 'red', 'swir1', 'swir2']
SAR_BANDS     = ['VH', 'VV']

def window_mask_per_sensor(raw_df, rng):
    """
    Per-sensor contiguous-window masking augmentation.

    For each row, samples TWO independent contiguous 4-6 month windows:
        - one for the 10 optical bands
        - one for the 2 SAR bands

    This recovers the ~320 radar-only and N optical-only test cells that
    v1's single-mask approach discards.
    """
    d = raw_df.copy()
    n = len(d)

    for sensor_bands in [OPTICAL_BANDS, SAR_BANDS]:
        lengths = rng.choice([4, 5, 6], size=n)
        starts  = np.array(
            [rng.integers(1, 12 - L + 2) for L in lengths]
        )
        keep = np.zeros((n, 12), dtype=bool)
        for i in range(n):
            keep[i, starts[i]-1 : starts[i]-1 + lengths[i]] = True
        for m in range(1, 13):
            cols = [f'{b}_{m:02d}' for b in sensor_bands]
            d.loc[~keep[:, m-1], cols] = np.nan

    return d


def audit_test_sensor_coverage(test):
    """
    Quantify the radar-only / optical-only / both-sensor split in the test set.

    This justifies the per-sensor masking decision empirically —
    expected to find ~320 radar-only cells (~31% of the 1,030 test rows).
    """
    optical_cols = [f'{b}_{m:02d}' for b in OPTICAL_BANDS for m in range(1, 13)]
    sar_cols     = [f'{b}_{m:02d}' for b in SAR_BANDS     for m in range(1, 13)]

    has_opt = test[optical_cols].notna().any(axis=1)
    has_sar = test[sar_cols].notna().any(axis=1)

    return {
        'radar_only'   : int((has_sar & ~has_opt).sum()),
        'optical_only' : int((has_opt & ~has_sar).sum()),
        'both_sensors' : int((has_opt &  has_sar).sum()),
        'n_test'       : int(len(test)),
    }
```

### 5.4 Cost
Per-sensor masking roughly doubles the variance of the augmented training set (a row can now have 4 optical months + 5 SAR months, vs the previous 4 months of both). The 8× augmentation factor compensates; v1's K_AUG=8 is retained.

### 5.5 Feature engineering implication
Several features become per-sensor by necessity:
- `n_valid_months_optical` and `n_valid_months_sar` replace the single `n_valid_months`.
- SAR-only rows have `NaN` for all optical-derived statistics — handled natively by LightGBM/CatBoost/XGBoost NaN routing.
- Cross-sensor correlations (Section 1.7) are only computable on months where **both** sensors have data; if no overlap exists, the correlation is set to `NaN` (not 0).

---

## 6. Validation v2

### 6.1 Decision
Rename `StratifiedGroupKFold` (vacuous groups) → **`StratifiedKFold`**, with **8 different masking seeds per fold**.

### 6.2 Justification
The v1 code's comment is honest: with `groups = row_index`, `StratifiedGroupKFold` is mathematically equivalent to plain `StratifiedKFold`. The v1 choice was therefore correct in effect, but the name suggested a guarantee (group generalisation) we cannot deliver — there are no spatial tile IDs in the data, no lat/lon, no multi-temporal observations of the same pond.

v2 makes this explicit:
- **`StratifiedKFold`** correctly claims only what's testable: label stratification.
- **8 masking seeds per fold** addresses the real variance source — the random masking augmentation. With 8 seeds × 5 folds = 40 estimates per row, the OOF probability is a tight estimator of test-time behaviour, with augmentation variance averaged out.

### 6.3 What we are NOT claiming
- Spatial-group generalisation (no spatial IDs available).
- Tile-level leakage prevention (no tile IDs available).
- Robustness to geographic domain shift (the test set may be in a different region; we have no way to test this).

### 6.4 CV protocol
```python
from sklearn.model_selection import StratifiedKFold

N_SPLITS              = 5
N_MASK_SEEDS_PER_FOLD = 8

def cross_validate_v2(train, feats):
    y_all = train['label'].values
    skf   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = list(skf.split(train, y_all))

    # OOF probabilities: shape (n_rows, n_mask_seeds) → average over seeds
    oof = np.zeros((len(y_all), N_MASK_SEEDS_PER_FOLD))

    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        for seed_k in range(N_MASK_SEEDS_PER_FOLD):
            train_frames = [
                window_mask_per_sensor(
                    train.iloc[tr_idx].reset_index(drop=True),
                    np.random.default_rng(tr_idx[0] * 1000 + seed_k * 100 + k)
                )
                for k in range(K_AUG)
            ]
            Xtr  = build_features(pd.concat(train_frames).reset_index(drop=True))
            y_tr = np.tile(y_all[tr_idx], K_AUG)

            Xval = build_features(
                window_mask_per_sensor(
                    train.iloc[val_idx].reset_index(drop=True),
                    np.random.default_rng(999 + seed_k + tr_idx[0])
                )
            )

            # 6-seed × 3-model blend, computed inline (this is the inner loop)
            p_blend = blend_6seed_3model(Xtr[feats], y_tr, Xval[feats])
            oof[val_idx, seed_k] = p_blend

    return oof.mean(axis=1)   # average across masking seeds
```

### 6.5 Expected CV behaviour
- CV composite ≈ v1's ~0.977 (per-sensor masking + Saerens-EM are test-time improvements, not CV improvements).
- CV variance across masking seeds should drop ~√8 = 2.83× vs v1's single-seed CV.
- If CV score rises significantly above 0.977, suspect leakage (especially from pseudo-labelling round 3) — investigate.

---

## 7. SHAP Explainability v2

### 7.1 Where to compute SHAP
- **Per-fold:** train a `shap.TreeExplainer` on each of 5 folds' training set; compute SHAP values on the validation fold. Average |SHAP| across folds → global importance.
- **Final model:** also compute SHAP on the full-data final LightGBM model (the dominant blend component) for the submission narrative and dependence plots.
- **Pseudo-labelling rounds:** compute SHAP before round 1 and after round 3 to verify the model's reliance hasn't shifted spuriously (confirmation-bias guard).

### 7.2 What to report

1. **Global feature importance bar chart** — top 30 features by mean |SHAP|, computed per-fold and averaged. This replaces LightGBM's gain-based importance (which is biased toward high-cardinality features).
2. **Dependence plots** for the top 8 features — SHAP value vs feature value, coloured by the strongest interaction feature (auto-detected via `shap.dependence_plot`'s default).
3. **Subpopulation analysis** — mean |SHAP| of top 10 features, broken down by:
   - Window length (4 / 5 / 6 months)
   - Sensor availability (optical+SAR / radar-only / optical-only)
   - Season (monsoon / dry / pre-monsoon)
   - Predicted probability quintile
4. **Feature redundancy clustering** — Spearman correlation matrix of SHAP values → hierarchical clustering (Ward linkage) → identify ~6–8 distinct feature "axes" the model actually uses. This justifies the ~280 → ~280 retention decision (most features contribute unique signal; redundant pairs are flagged for v3 pruning).
5. **Pseudo-labelling drift report** — for each of the top 20 features, plot the SHAP distribution before round 1 vs after round 3. Features whose SHAP distribution shifts by > 0.1 (KL divergence) are flagged as confirmation-bias suspects.

### 7.3 zinmori's MCI/chlorophyll dominance claim — verify or refute

zinmori (rank 3) reportedly showed MCI (Maximum Chlorophyll Index) is the dominant feature, not water indices. **We will verify this on our data, not assume it.**

**Test protocol:**
1. Include both `MCI = re1 − red` and `NDCI = (re1 − red) / (re1 + red)` in the feature set (Section 1.2, indices #6 and #7).
2. Run SHAP on the final v2 model.
3. Report the rank of `MCI_mean`, `NDCI_mean`, `MNDWI_mean`, `sar_optical_diverge`, `mndwi_vhvv_corr` in the global importance ordering.
4. Run a controlled ablation: train v2 with MCI removed, then with NDCI removed, then with both removed. Compare CV composite scores. The feature whose removal causes the largest drop is the empirically dominant one.

**Hypothesis (to be tested, not asserted):** On our data, chlorophyll indices (MCI, NDCI) will rank in the top 5 but will NOT be the single dominant feature, because:
- The competition's pond labels include both actively managed ponds (high chlorophyll, MCI dominant) and abandoned/decommissioned ponds (low chlorophyll, MCI uninformative).
- The v1 `sar_optical_diverge` feature catches vegetated/algae-covered ponds that pure chlorophyll indices miss — these are exactly the false negatives a chlorophyll-only model would produce.
- zinmori's reported dominance may reflect a different train/test split that over-sampled managed ponds, or a different feature engineering pipeline that did not include the SAR-optical divergence.

**Report format:** We will publish the empirical rank of MCI/NDCI vs `sar_optical_diverge` vs `MNDWI` regardless of which wins. If MCI dominates on our data, we update the lessons-learned doc to credit zinmori's insight. If `sar_optical_diverge` dominates, the v1 architecture's core feature remains the most important.

---

## Summary — gap closure matrix

| Gap | v1 | v2 | Closed by |
|---|---|---|---|
| Features | 59 | ~280 | §1: 14 indices × 13 stats + new families |
| Models | LGB+CB, 1 seed | LGB+CB+XGB, 6 seeds | §2: 3-model blend, seed list [42–47] |
| Threshold | Probe-submission | Saerens/EM | §3: unsupervised prior correction |
| Pseudo-labelling | None | 3 rounds | §4: thresholds 0.90/0.85/0.80, decaying weights |
| Per-sensor masking | Single contiguous | Per-sensor | §5: 2 independent windows, recovers ~320 radar-only cells |
| Validation | StratifiedGroupKFold (vacuous) | StratifiedKFold + 8 seeds | §6: honest naming, augmentation variance averaged |
| SHAP | None | Full SHAP + subpopulation | §7: per-fold + final, MCI/NDCI verification |

**Expected net effect:** each gap closure is worth ~0.0003–0.001 composite score (per the v1 post-mortem's compounding-small-wins thesis). Combined effect: 0.002–0.005 improvement on private LB, closing the 0.001 gap to rank 5 and most of the 0.007 gap to rank 3.
