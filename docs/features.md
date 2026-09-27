# Feature Engineering — 59 Features

The design is deliberately **physics-first**: rather than letting the model re-learn spectral signatures from 1,821 rows, we hand it established remote sensing indices whose mathematical form encodes decades of domain knowledge.

## 1. Physics-based indices (24 features)

Each index is computed per month, then aggregated as **mean / median / std / linear-trend** across the available months.

| Index | Formula | Why it matters |
|---|---|---|
| NDVI  | (NIR − red) / (NIR + red) | Vegetation vigour; pond surroundings are often vegetated |
| MNDWI | (green − SWIR1) / (green + SWIR1) | Open water (high positive) vs built-up/vegetation (negative) |
| NDWI  | (green − NIRa) / (green + NIRa) | Water body extent. Uses S2 B8A (narrow NIR, 865 nm) instead of B8 (broad NIR, 842 nm) — B8A sits closer to a water-vapour absorption feature and shows a sharper water/land contrast in wetland studies. McFeeters (1996) originally used broad NIR; this is a deliberate substitution. |
| VH−VV | VH − VV (dB) | SAR polarimetric difference; smooth water has very negative VH |
| AWEI  | blue + 2.5·green − 1.5·(SWIR1 + SWIR2) − 0.25·NIR | Shadow-discriminating water index (AWEI_sh, Feyisa et al. 2014); fixes MNDWI's failure on shaded ponds |
| NDCI  | (re1 − red) / (re1 + red) | Chlorophyll-a proxy; managed aquaculture ponds have elevated chlorophyll |

The trend is fit by `np.polyfit` on the available months only; rows with < 2 valid months get `NaN` for the trend and the GBDT handles it natively.

## 2. Domain-specific engineered features

### Persistence features (4)
- **water_persistence** — fraction of observed months where MNDWI > 0. Robust to a single cloudy month.
- **nonveg_persistence** — fraction of months where NDVI < 0.1 (bare soil or water rather than vegetation).
- **strong_water_persist** — fraction of months where MNDWI > 0.1 (stricter threshold reduces false positives from wet vegetation).
- **awei_water_persist** — fraction of months where AWEI > 0.

### SAR features (5)
- `vh_mean`, `vv_mean` — mean backscatter across observed months
- `vh_min`, `vv_min` — minimum (smooth water has very low VH)
- `vh_std` — variability

### Correlation (1)
- **mndwi_vhvv_corr** — Pearson correlation between MNDWI and VH-VV across observed months; high correlation indicates a stable, consistent water body.

### First / last / change (15)
For each of MNDWI, NDVI, VHVV, AWEI, NDCI:
- `_first` — value at the first observed month
- `_last` — value at the last observed month
- `_change` — `_last` − `_first`

## 3. Seasonality & window-shape features (7)
- `n_valid_months` — how many months were observed (4, 5, or 6)
- `window_start` — the first observed month
- `window_center` — midpoint of the observed window
- `start_sin`, `start_cos` — cyclical encoding of window start month
- `has_monsoon` — does the window include June-September?
- `has_dry` — does the window include November-March?

## 4. The key feature: SAR-optical divergence (3)
- **sar_water_persist** — fraction of months where VH < −26 dB (empirically calibrated water threshold for Sentinel-1).
- **sar_optical_diverge** — `sar_water_persist − water_persistence`. **The single most important feature by LightGBM gain.** It flags the vegetated/algae-covered ponds where optical indices say "not water" but SAR backscatter says "smooth surface". This feature was derived from false-negative error analysis on the public-LB submission.
- **mndwi_ndci_interact** — `MNDWI_mean × NDCI_mean`. Captures the joint condition "water surface + elevated chlorophyll" that uniquely characterises aquaculture vs natural lakes.

## Feature selection & normalisation

- **Selection:** No explicit selection step. All 59 features are retained. LightGBM's `reg_alpha = 0.3` (L1) and CatBoost's `l2_leaf_reg = 3` act as implicit regularisers; per-feature SHAP analysis on a held-out fold confirmed no feature was actively harmful.
- **Normalisation:** None. Both LightGBM and CatBoost are tree-based and scale-invariant; normalisation would add no benefit and a small risk of numerical artefacts on near-zero denominators in the index formulas.
