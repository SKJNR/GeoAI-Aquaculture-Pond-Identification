# v2 Feature Engineering — Design Document

**Author:** Senior Satellite Remote Sensing Specialist
**Scope:** Replace v1's 59-feature, 6-index set with a v2 set engineered against the rank-3 (zinmori) solution, whose 316-feature pipeline proved via SHAP that **MCI (chlorophyll), not water indices, is the dominant aquaculture discriminant** (26.5% of total influence; subpopulation AUC 0.946 on water-only rows where every water index is useless by construction). v2 closes the gap by adopting zinmori's MCI/CDOM/red-edge family, fixing two bugs found in v1 (AWEI coefficient swap, EVI L-constant scaling), adding the Justclemax test-validated vegetation/soil indices (SAVI, BSI, NDRE, CIG), and porting zinmori's per-sensor masking to recover the 320 radar-only test cells v1 discards.

This is a **design-only** document — no implementation. Pseudocode illustrates the tricky parts (per-sensor masking, NaN-aware moments, cross-sensor correlation).

---

## 1. Index Verification Table

Sentinel-2 bands (used throughout): B2=blue (490 nm), B3=green (560 nm), B4=red (665 nm), B5=re1 (705 nm), B6=re2 (740 nm), B7=re3 (783 nm), B8=nir (842 nm), B8A=nira (865 nm), B11=swir1 (1610 nm), B12=swir2 (2190 nm). Sentinel-1: VH, VV (dB). Surface reflectance is stored ×10000.

Verdict key: **KEEP** = already in v1, retain. **ADD** = new in v2. **SKIP** = rejected, with reason. **CONDITIONAL** = add with caveats. **FIX** = already in v1 but has a bug; correct.

| # | Index | Formula (verified) | S2 bands | Citation | Verdict | Rationale |
|---|---|---|---|---|---|---|
| 1 | NDVI | (NIR − Red) / (NIR + Red) | B8, B4 | Rouse et al. 1973 | KEEP | Vegetation vigour around pond berms; the 26.6% "other" SHAP bucket includes NDVI. |
| 2 | MNDWI | (Green − SWIR1) / (Green + SWIR1) | B3, B11 | Xu 2006 | KEEP | Open-water detector; zinmori SHAP = 10.0%. |
| 3 | NDWI (McFeeters) | (Green − NIR) / (Green + NIR) | B3, B8 | McFeeters 1996 | ADD | Canonical NDWI. v1 used B8A (non-standard); v2 adds the B8 form and keeps B8A as a separate `ndwi_b8a` variant (B8A is more sensitive to turbid water). |
| 4 | NDWI_b8a (v1 "NDWI") | (Green − NIRa) / (Green + NIRa) | B3, B8A | McFeeters 1996 (variant) | KEEP | Keep v1's B8A variant — it captures the water-vapour absorption shoulder at 865 nm and is empirically better for turbid aquaculture water. |
| 5 | VH − VV (dB) | VH − VV | S1 VH, VV | RMS SAR convention | KEEP | Polarimetric difference; equals 10·log₁₀(VH_lin / VV_lin). Smooth-water VH plummets. |
| 6 | AWEI_sh (v1 "AWEI") | Blue + 2.5·Green − 1.5·(NIR + SWIR1) − 0.25·SWIR2 | B2, B3, B8, B11, B12 | Feyisa et al. 2014 | FIX | **v1 has the NIR/SWIR2 coefficients swapped** (1.5 on SWIR2, 0.25 on NIR). Documented in AUDIT-3. v2 corrects to the canonical Feyisa 2014 form. |
| 7 | NDCI | (RE1 − Red) / (RE1 + Red) | B5, B4 | Mishra & Mishra 2012 | KEEP | Red-edge chlorophyll-a proxy; zinmori's MCI supersedes it but they are complementary (NDCI normalised, MCI baseline-subtracted). |
| 8 | MCI | RE1 − Red − 0.5333·(RE2 − Red) | B5, B4, B6 | Gower et al. 2005 (OLCI); S2 adaptation Toming et al. 2016 | ADD | **THE critical v2 addition.** 3-point baseline MCI adapted from Sentinel-3 OLCI to S2 red-edge. The 0.5333 coefficient = (705−665)/(740−665). zinmori's SHAP showed `mci_p90`, `mci_p75`, `mci_max`, `mci_p95` are the top-4 features — eutrophic pond chlorophyll discriminates aquaculture from natural lakes at AUC 0.946 even when water indices are uninformative. |
| 9 | CDOM | Green / Red | B3, B4 | Kutser 2012; Toming et al. 2016 | ADD | Coloured dissolved organic matter proxy. Aquaculture ponds fed with fertiliser leach CDOM; the simple green/red ratio is the standard S2-CDOM proxy. zinmori's `cdom_p25` is the 5th-most-important SHAP feature. |
| 10 | NDSI (snow, Hall 1995) | (Green − SWIR1) / (Green + SWIR1) | B3, B11 | Hall et al. 1995 | SKIP | **Mathematically identical to MNDWI (Xu 2006).** Both are (Green − SWIR1)/(Green + SWIR1). Hall's snow index and Xu's modified water index were independently derived and produce the same pixel value. Adding it would be a redundant copy of #2 — adds a feature but zero information. |
| 11 | NDSI_sw2 (zinmori's "NDSI") | (Green − SWIR2) / (Green + SWIR2) | B3, B12 | Non-canonical (zinmori variant) | CONDITIONAL | Not the Hall 1995 snow index. Uses B12 (2190 nm) instead of B11 (1610 nm). SWIR2 is more absorbed by water than SWIR1, so this acts as a "second-derivative" water discriminant orthogonal to MNDWI. zinmori SHAP "AWEI/NDSI water" bucket = 8.0%. **Add under a distinct name `ndsi_sw2`** to avoid the snow-index ambiguity; **skip if SHAP shows collinearity with MNDWI in v2's first run.** |
| 12 | SDWI (SAR dual-pol water index) | log₁₀(10·VV_lin·VH_lin) − 8 | S1 VH, VV | Markert et al. 2018 (S1WI lineage); zinmori empirical form | ADD | A SAR-based water detector complementary to optical indices. Smooth water has very low backscatter in both pols, driving the log product below the −8 threshold. zinmori's `sdwi_min` is the 6th-most-important SHAP feature; `water_freq_sdwi` is also in the top tier. **The −8 baseline is empirical** (Markert 2018 S1WI uses σ_VV/(σ_VV+σ_VH), so SDWI here is a zinmori-adapted form). Document this in code. |
| 13 | EVI | 2.5·(NIR − Red) / (NIR + 6·Red − 7.5·Blue + L) | B8, B4, B2 | Huete et al. 2002 | ADD (with FIX) | Enhanced Vegetation Index; atmospheric-resistance via blue-band correction. **zinmori uses L=1 which is wrong for ×10000 reflectance data** (denominator collapses to ≈NIR+6·Red, huge EVI values). Justclemax uses L=10000 (correct). v2 uses **L=10000** and clips EVI to [−5, 5] to guard against blue≈0 water pixels. |
| 14 | SAVI | (1+L)·(NIR − Red) / (NIR + Red + L), L=0.5 | B8, B4 | Huete 1988 | ADD | Soil-Adjusted VI; reduces soil-background contamination on pond berms. For ×10000 reflectance the denominator offset is **L=5000**. Justclemax correctly uses 5000; v2 copies. Distinct from NDVI on sparse-berm pixels. |
| 15 | BSI | ((SWIR1 + Red) − (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue)) | B11, B4, B8, B2 | Lambin & Ehrlich 1996; Rikimaru 2002 | ADD | Bare Soil Index. High on drained-pond berms and exposed sediments — directly targets zinmori's recall deficit on drained ponds (the "no water, no chlorophyll" failure mode). |
| 16 | NDRE | (NIR − RE1) / (NIR + RE1) | B8, B5 | Barnes et al. 2000; Gitelson & Merzlyak 1994 | ADD | Normalized Difference Red Edge. Chlorophyll proxy saturating at higher LAI than NDVI. **Use the (B8 − B5)/(B8 + B5) form, NOT (B5 − B6)/(B5 + B6)** which is "NDRE2" (Sentinel Hub nomenclature) — the B8/B5 form is the Barnes 2000 original and is what Justclemax's tests verify. |
| 17 | CIG | (NIR / Green) − 1 | B8, B3 | Gitelson et al. 2003 | ADD | Chlorophyll Index Green. Sensitive to high chlorophyll concentrations that saturate NDVI/NDCI. Complements MCI as a second chlorophyll angle (MCI uses red-edge, CIG uses green-NIR contrast). |
| 18 | NDBI | (SWIR1 − NIR) / (SWIR1 + NIR) | B11, B8 | Zha et al. 2003 | ADD | Built-up detector; pond infrastructure (bunds, intake pipes, aerator housings) returns high NDBI. Useful to discriminate aquaculture from natural wetlands. |
| 19 | SAR_RVI | 4·VH_lin / (VV_lin + VH_lin) | S1 VH, VV | Periasamy 2018 | ADD | Radar Vegetation Index. RVI ∈ [0, 2]; smooth water drives it toward 0 (both pols equally absorbed), vegetation toward 1, double-bounce toward 2. **Must be computed in LINEAR space** (dB→linear via 10^(dB/10)). zinmori form: `4·vh_lin/(vv_lin+vh_lin)`. |
| 20 | SAR_backscatter_sum | VH_lin + VV_lin | S1 VH, VV | zinmori composite | ADD | Total backscatter magnitude; zinmori's "sar" series. Low for water (both pols absorbed), high for built-up (double bounce). Carries SAR SHAP = 23.5% (across `sar`, `sdwi`, `rvi`, `vh_vv`). |
| 21 | TwobDA | RE1 / Red | B5, B4 | Brienza et al. (2023) aquaculture context | ADD | Two-band Brienza Algorithm. Simple red-edge/red ratio; zinmori's `twobda` — a coarse chlorophyll proxy when MCI's 3-point baseline is over-fit by noise. **Citation imprecise**: the form originates in 2-band red-edge chlorophyll literature (Gitelson 2003, Mishra 2012) but the specific name "TwobDA" appears in Brienza's 2023 pond work. Document this ambiguity in code comments. |
| 22 | AWEI_nsh | 4·(Green − SWIR1) − (0.25·NIR + 2.75·SWIR2) | B3, B11, B8, B12 | Feyisa et al. 2014 | ADD | No-shadow variant of AWEI. **Correct form has `−2.75·SWIR2`, not `+2.75·SWIR2`** — Justclemax has a sign error in their AWEI_nsh (`+2.75*S2`). v2 uses the canonical form. Use **both AWEI_sh and AWEI_nsh**: sh is more robust to shadow, nsh is more robust to cloud-top reflectance. Their difference is itself a shadow/cloud diagnostic. |
| 23 | water_product (composite) | MNDWI × (1 − NDVI) | B3, B11, B8, B4 | zinmori-defined | ADD | Composite: high when MNDWI high (water) AND NDVI low (non-vegetated). Amplifies the open-water signal; suppresses vegetated wetlands that v1 misclassifies. |
| 24 | awei_mndwi_diff (composite) | AWEI_nsh − MNDWI | (derived) | zinmori-defined | ADD | When both indices agree on water, difference ≈ 0; divergence flags shadow/cloud/vegetation edge cases. zinmori uses `awei − mndwi` with the nsh form; v2 inherits. |

**Total: 23 indices** (6 KEEP/FIX from v1, 16 ADD, 1 SKIP, plus 1 CONDITIONAL → 22 implemented + ndsi_sw2 if first SHAP confirms decorrelation from MNDWI).

---

## 2. Recommended v2 Index Set

Implement these **22 indices** (23 if `ndsi_sw2` survives the SHAP-collinearity check after the first fold):

**Optical family (14 series, all masked by `opt_mask`):**
`ndvi, mndwi, ndwi, ndwi_b8a, awei_sh, awei_nsh, ndci, mci, cdom, evi, savi, bsi, ndre, cig, ndbi, twobda, water_product, awei_mndwi_diff, ndsi_sw2`

(That's 19 optical series; the four composite/diff series are derived from the others but are precomputed once and aggregated independently because their moment statistics are not linear in the inputs.)

**SAR family (5 series, all masked by `sar_mask`):**
`sar_sum (vh_lin+vv_lin), sdwi, vh_vv, rvi, vh_vv_ratio`

**Why this set:**
1. **MCI is the headline addition.** zinmori's SHAP attributes 26.5% of influence to MCI alone, and 4 of the top-6 single features are `mci_p{90,75,95}` + `mci_max`. v1's NDCI is a weak proxy; v2's MCI is the correct OLCI-derived form adapted to S2 red-edge. **Without MCI, v2 cannot reach rank-3-level performance.**
2. **The CDOM family (CDOM, CIG, TwobDA, NDRE) is the second chlorophyll angle.** Each measures chlorophyll through a different band combination; SHAP may concentrate on MCI but the ensemble's noise robustness comes from having 4 correlated-but-not-identical chlorophyll proxies.
3. **The BSI / NDBI pair targets the drained-pond failure mode** that zinmori explicitly documents ("recall deficit on drained ponds"): a drained pond has neither water nor chlorophyll, but it does have exposed bare soil (BSI high) and pond infrastructure (NDBI high). v2 bets that these two indices will move the needle on the hardest subpopulation.
4. **Both AWEI variants** (sh + nsh) are kept because their **difference** is itself a shadow/cloud diagnostic; their union covers the cloud-shadow discrimination that v1's single-variant design lacks.
5. **The 5 SAR series** include three "shapes" of the S1 dual-pol signal: total backscatter (`sar_sum`), water-thresholded index (`sdwi`), vegetation-sensitive ratio (`rvi`), and the two dB-space differences (`vh_vv`, `vh_vv_ratio` = `vh_lin/vv_lin`). They are not collinear: `vh_vv` is a dB difference, `vh_vv_ratio` is a linear ratio, and the model can split on either depending on which threshold is cleaner.

---

## 3. Statistical Aggregations v2

v1 uses 4 stats per index (mean / median / std / trend). zinmori uses **16 stats** per index. v2 adopts zinmori's 16 plus a mean = **17 stats per index**, plus selective gradient / autocorrelation on high-value series.

**Per-index stat list (17):**

| # | Stat | Justification for aquaculture |
|---|---|---|
| 1 | `mean` | Centre of mass; complementary to median for skewed chlorophyll distributions. (zinmori omits mean; we keep it because trees split on raw magnitudes and `mean = p50` only when symmetric — many chlorophyll series are right-skewed.) |
| 2 | `p5` | Lower tail — catches the "drained month" within a mostly-water window. |
| 3 | `p10` | zinmori uses this as `water_floor` input (`mndwi_p10 − sar_p10`). |
| 4 | `p25` | Lower quartile. `cdom_p25` is zinmori's 5th-most-important SHAP. |
| 5 | `p50` (median) | Centre; robust to outliers. |
| 6 | `p75` | Upper quartile. `mci_p75` is zinmori's #2 SHAP. |
| 7 | `p90` | Upper tail. `mci_p90` is zinmori's #1 SHAP. |
| 8 | `p95` | Extreme upper tail. `mci_p95` is zinmori's #4 SHAP. |
| 9 | `min` | Extremes — drained-month signal. |
| 10 | `max` | Extremes — bloom-peak signal. `mci_max` is zinmori's #3 SHAP. |
| 11 | `range` (max − min) | Variability range — managed ponds cycle (drain/fill) wider than natural lakes. |
| 12 | `iqr` (p75 − p25) | Robust variability — less sensitive to single-month outliers than range. |
| 13 | `std` | Dispersion; captures pond management cycle amplitude. |
| 14 | `cv` (std / |mean|) | **Coefficient of variation.** Permanent water has CV ≈ 0; managed ponds have CV > 0.3. v1's `mndwi_std` alone misses this because it's not scale-normalised. |
| 15 | `skew` | Distributional asymmetry. Eutrophic ponds with intermittent blooms have positive skew on MCI/CDOM. Requires n ≥ 3 valid months. |
| 16 | `kurtosis` | Tail weight. Distinguishes "occasional extreme bloom" (high kurtosis) from "consistent bloom" (low kurtosis). Requires n ≥ 4. |
| 17 | `frac_pos` | Fraction of months where index > 0. Direct persistence measure; more robust than threshold-based `water_persistence` because no magic threshold. |

**Gradient features (3 stats × 6 series = 18):** mean, std, and max-abs-successive-difference of monthly differences, computed on `mndwi, ndvi, sar_sum, ndsi_sw2, rvi, water_product`. These capture pond management transition dynamics — the "drained→reflooded" cycle that distinguishes managed ponds from stable natural water.

**Autocorrelation features (2 stats × 4 series = 8):** lag-1 autocorrelation + mean absolute successive difference (MASD) on `mndwi, ndvi, sar_sum, sdwi`. Low autocorrelation = transition month in window; high MASD = high-frequency management signal.

**Total stats-derived features:** 19 optical × 17 + 5 SAR × 17 + 18 gradient + 8 autocorr = **454** base temporal features.

---

## 4. Per-Sensor Masking Design

### Why this matters

The Test.csv contains **4,827 dual-sensor month-cells** and **320 cells where radar is present but optics are missing** (cloud). v1's single contiguous mask discards all 320 — i.e., v1 throws away 6.4% of the SAR information in the test set. zinmori's per-sensor masking recovers them, contributing the radar-only features for those months while leaving optical features as NaN (which GBDTs handle natively).

### How to detect which months have optical-only vs SAR-only

```python
def build_per_sensor_masks(df, n_months=12):
    """Return (opt_mask, sar_mask), each (N, n_months) boolean."""
    opt_bands = ['blue', 'green', 'red', 'nir', 'swir1']  # minimum set v2 needs
    sar_bands = ['VH', 'VV']
    opt_mask = np.ones((len(df), n_months), dtype=bool)
    sar_mask = np.ones((len(df), n_months), dtype=bool)
    for m in range(1, n_months + 1):
        ms = f'{m:02d}'
        for b in opt_bands:
            opt_mask[:, m-1] &= df[f'{b}_{ms}'].notna().values & (df[f'{b}_{ms}'].values != -9999)
        for b in sar_bands:
            sar_mask[:, m-1] &= df[f'{b}_{ms}'].notna().values & (df[f'{b}_{ms}'].values != -9999)
    return opt_mask, sar_mask

def mask_train_like_test(train_df, test_df, seed):
    """Assign each train row a real test pattern; apply per-sensor independently."""
    rng = np.random.default_rng(seed)
    opt_t, sar_t = build_per_sensor_masks(test_df)
    N = len(train_df)
    idx = rng.integers(0, len(test_df), size=N)  # one real test pattern per train row
    o, s = opt_t[idx], sar_t[idx]
    out = train_df.replace(-9999, np.nan).copy()
    for m in range(1, 13):
        ms = f'{m:02d}'
        opt_rows = np.where(~o[:, m-1])[0]
        sar_rows = np.where(~s[:, m-1])[0]
        if len(opt_rows):
            out.loc[out.index[opt_rows], [f'{b}_{ms}' for b in OPT_BANDS]] = np.nan
        if len(sar_rows):
            out.loc[out.index[sar_rows], [f'{b}_{ms}' for b in SAR_BANDS]] = np.nan
    return out
```

### Feature extraction pseudocode (per-sensor aware)

```python
def extract_features(df, opt_mask, sar_mask):
    """Each index series carries its own sensor's validity mask."""
    # ... compute per-month index matrices (N, 12) per index ...
    
    # Optical series (MNDWI, NDVI, MCI, CDOM, AWEI, ...) → opt_mask
    # SAR series (sar_sum, sdwi, vh_vv, rvi, vh_vv_ratio) → sar_mask
    
    for name, arr in optical_series.items():
        arr_masked = np.where(opt_mask, arr, np.nan)
        feats.update(temporal_stats(arr_masked, name))  # 17 stats
    
    for name, arr in sar_series.items():
        arr_masked = np.where(sar_mask, arr, np.nan)
        feats.update(temporal_stats(arr_masked, name))
    
    # Cross-sensor correlations use BOTH masks (intersection)
    both = opt_mask & sar_mask
    feats['mndwi_sar_corr'] = row_wise_corr(mndwi, sar_sum, both)
    # ... etc
```

### Reproducing zinmori's masking verification

After masking, mean observed months should be ≈5.0 (matching test's 4.997), mean optical months ≈4.70 (test 4.69), mean radar-only months ≈0.31 (test 0.311). v2's masking module must include an assertion that these match within 0.05; deviations indicate a bug.

### Decoupling features

Two extra features are needed to let the model distinguish "month missing" from "month present but constant":
- `n_optical_months` — count of months with optical data
- `n_saronly_months` — count of months with SAR but no optical

Without these, a 6-month window with 2 SAR-only months looks identical (in the optical features) to a 4-month window with 0 SAR-only months. The model needs the count to interpret the NaN structure.

---

## 5. Cross-Sensor Features

zinmori uses 4 cross-sensor correlations. v2 expands to **8** based on physical reasoning about what *should* co-vary in aquaculture ponds:

| # | Pair | Mask | Physical rationale |
|---|---|---|---|
| 1 | `mndwi_sar_corr` | both | Pearson(MNDWI, sar_sum) across months where both observed. Permanent water: high positive (both consistently low). Drained pond: low (SAR still sees mud, MNDWI collapses). |
| 2 | `mndwi_ndvi_corr` | opt | Anti-correlation expected (water vs vegetation). Deviations flag algae-scum ponds where NDVI is paradoxically high over water. |
| 3 | `ndsi_sw2_mndwi_corr` | opt | If `ndsi_sw2` is decorrelated from MNDWI, this correlation is informative; if collinear, the correlation is ≈1 and adds no info (will be skipped after first-fold SHAP). |
| 4 | `rvi_mndwi_corr` | both | RVI (vegetation SAR) vs MNDWI (water optical). Aquaculture: negative (water + no vegetation). Wetland: positive (mixed water + vegetation). |
| 5 | `mci_sar_corr` | both | MCI (chlorophyll) vs SAR sum. Aquaculture pond: chlorophyll high AND SAR low (smooth water surface). Strong negative correlation is the joint signature zinmori's SHAP revealed. |
| 6 | `mci_ndvi_corr` | opt | MCI (red-edge chlorophyll) vs NDVI (NIR-red vegetation). Both measure vegetation but through different band pairs; their correlation reveals whether the chlorophyll is from aquatic algae (MCI high, NDVI low) or emergent plants (both high). |
| 7 | `awei_mci_corr` | opt | AWEI_nsh (water/shadow) vs MCI (chlorophyll). Aquaculture: high-high (eutrophic water). Lake: high-low (clear water). |
| 8 | `bsi_sar_corr` | both | BSI (bare soil) vs SAR sum. Drained pond: high-high (exposed soil, double-bounce). Flooded pond: low-low. Targets the drained-pond failure mode directly. |

Plus **5 water-composite features** from zinmori (carried over to v2):
- `water_freq_strict` = frac{months both observed AND MNDWI > 0.5 AND sar_sum < 0.1}
- `water_freq_sdwi` = frac{SAR months AND SDWI > −1.5}
- `water_score` = `mndwi_p25 × (1 − min(sar_p25, 1))`
- `water_floor` = `mndwi_p10 − sar_p10`
- `water_consist` = `mndwi_frac_pos × (1 − sar_frac_pos)`

Plus **1 persistence feature** adapted from v1:
- `water_perm` = frac{opt months AND MNDWI > 0 AND NDVI < 0.3}

---

## 6. Total Feature Count Estimate

| Block | Count |
|---|---|
| Optical indices (19 series × 17 stats) | 323 |
| SAR indices (5 series × 17 stats) | 85 |
| Gradient features (6 series × 3 stats) | 18 |
| Autocorrelation (4 series × 2 stats) | 8 |
| Cross-sensor correlations (8) | 8 |
| Water composite features (5 + 1) | 6 |
| Persistence features (4 from v1) | 4 |
| Observation structure (n_months, first, last, block_len, mid, density, n_opt, n_saronly) | 8 |
| Window-shape (start, center, sin, cos, has_monsoon, has_dry) | 6 |
| **TOTAL** | **458** |

This is ~45% larger than zinmori's 316 and ~8× larger than v1's 59. The inflation is deliberate: v2's 19 optical indices vs zinmori's 12 reflects the addition of SAVI, BSI, NDRE, CIG, NDBI (Justclemax-inspired vegetation/soil/built-up family) that zinmori doesn't have, plus v1's NDCI, NDWI_b8a, and the conditional ndsi_sw2.

**First-fold SHAP pruning plan:** If any of the 19 optical series shows <0.5% total SHAP share AND is correlated >0.95 with another series, drop it. Expected to remove 4-6 redundant features (likely candidates: `twobda` if it collinear with `mci`, `ndsi_sw2` if it collinear with `mndwi`, `awei_mndwi_diff` if its variance is dominated by AWEI). Target post-prune: **~450 features**.

---

## 7. Implementation Notes

### 7.1 Bug fixes carried from v1 and competitor analysis

1. **AWEI_sh coefficient swap (v1 bug, AUDIT-3)**: v1 implements `blue + 2.5·green − 1.5·(swir1 + swir2) − 0.25·nir`, which puts the 1.5 coefficient on SWIR2 and the 0.25 on NIR. Canonical Feyisa 2014 is `blue + 2.5·green − 1.5·(nir + swir1) − 0.25·swir2` — i.e., 1.5 on NIR and 0.25 on SWIR2. v2 fixes this. **This is a breaking change**: v2 features will not match v1 features, and v2's model cannot be initialised from v1's checkpoint.
2. **EVI L-constant scaling (zinmori bug)**: zinmori uses L=1 in EVI's denominator, but the data is ×10000 reflectance. With L=1 the denominator is dominated by 6·Red + NIR (both ~1000s), so EVI values inflate by ~10⁴. Justclemax correctly uses L=10000. v2 uses **L=10000**. Trees are scale-invariant so zinmori's bug doesn't cost accuracy, but our documentation should be correct.
3. **AWEI_nsh sign error (Justclemax bug)**: Justclemax implements `4·(G − S1) − 0.25·N + 2.75·S2` with +2.75 on SWIR2. Canonical is `−2.75·S2`. v2 uses the canonical form. **Do not copy Justclemax's AWEI_nsh implementation.**
4. **NDWI canonical vs B8A variant**: v1's "NDWI" used B8A (nira). The canonical McFeeters 1996 uses B8 (nir). v2 keeps v1's B8A variant under the name `ndwi_b8a` and adds the canonical `ndwi` with B8. Both are useful: B8A is more sensitive to turbid water (closer to water-vapour absorption), B8 is the published standard.

### 7.2 Numerical stability

- All ratios carry `+1e-8` denominator guard (zinmori convention; Justclemax uses 1e-6 — both negligible for ×10000 data).
- `np.errstate(all='ignore')` wraps every index computation.
- Final pipeline: `df.replace([np.inf, -np.inf], np.nan).fillna(0)`. NaN→0 is appropriate for GBDTs (they can split on "is zero" as a missingness signal).
- SAR linear conversion: `vh_lin = 10**(vh_dB / 10)`. Clip dB to `[−40, 0]` before conversion (Justclemax convention) to avoid `10**4` blow-ups from corrupted −9999 sentinels that slip through.
- MCI coefficient 0.5333 = 40/75 = (705−665)/(740−665). Document this derivation; future authors may "fix" the round-off and not understand it's a wavelength ratio.

### 7.3 NaN handling per sensor

- Each index series carries its own sensor's mask, not the global mask. A month with optical-only data contributes to optical-index stats but is NaN for SAR-index stats (and vice versa).
- All moments use each series' **own** valid-month count, not a global one. zinmori's code computes `n_s = np.sum(mask, axis=1)` per series and uses it in the moment denominator. v2 must do the same.
- Min-count guards: skewness requires n ≥ 3, kurtosis n ≥ 4, gradient n ≥ 2, autocorrelation n ≥ 4. Below the threshold, return 0.0 (not NaN) so the feature column has no NaN in the final matrix (GBDT splits on 0 cleanly; NaN→0 is fine here).

### 7.4 Composite feature ordering

`water_product = mndwi × (1 − ndvi)` and `awei_mndwi_diff = awei_nsh − mndwi` must be computed **before** temporal aggregation, not after. The moment of a product ≠ product of moments (Jensen's inequality). zinmori computes them per-month then aggregates, which is correct.

### 7.5 Reproducibility

- Seed list `[42, 123, 456, 789, 1337, 2024, 7, 99]` (zinmori's 8 seeds) for the masking-augmentation ensemble.
- Each seed produces one masked view of the training set, used across all 5 folds and all 3 models in that phase.
- Determinism: `random.Random(seed)` for masking assignment, `random_state=seed+fold` for the GBDT models. v2 inherits this exactly.

### 7.6 What NOT to add (deliberate exclusions)

- **NDSI (snow, Hall 1995)**: identical to MNDWI. SKIP.
- **Raw band temporal stats** (Justclemax adds 6 stats × 12 raw bands = 72 features): v1 omits these and zinmori omits these. Trees can recover raw-band information from any index that includes those bands, but the redundancy is noise. v2 SKIPS raw-band stats to keep the feature count manageable.
- **Illumination-invariant shapes** (zinmori tested an 80-feature illumination-invariant block, it scored worse): v2 does not add these. zinmori's lesson is explicit.
- **Lat/lon features** (muts-prog uses these, violates competition rules): v2 does not add these. Competition rules forbid geolocation; zinmori and Justclemax confirm.

### 7.7 Validation strategy

 zinmori explicitly documents that **no internal CV protocol ranks pipelines correctly on this dataset** (their nine protocols all invert the leaderboard ordering). v2 inherits this constraint: internal CV is a smoke test and calibration fitter only, NOT a model selector. Every design decision in this doc is justified by zinmori's SHAP/leaderboard evidence, not by internal CV.

---

**End of v2 feature engineering design.** Implementation handoff: each section above maps to a function in `src/features_v2.py`. Total estimated implementation effort: 1 working day for the index library, 1 day for the per-sensor masking + stats pipeline, 0.5 day for tests verifying each formula against the citation table. First-fold SHAP run on a single seed is the validation gate before expanding to the full 8-seed × 5-fold × 3-model ensemble.
