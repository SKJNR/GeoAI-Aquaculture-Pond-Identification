# Limitations

This document honestly scopes what the rank-6 solution does and does not do. It exists to set correct expectations for anyone (recruiter, hiring manager, potential client, fellow researcher) evaluating this repo for a real-world use case.

## What this repo IS

- A **portfolio piece** documenting a top-0.5% finish (rank 6 of 1,318) in the Zindi GeoAI Aquaculture Pond Identification Challenge (FAO & ITU, closed August 16, 2026).
- A **clean, modular, tested, CI-backed Python codebase** that reproduces the rank-6 approach on the official Zindi competition data.
- An **honest post-mortem** (`docs/lessons_learned.md`) of what cost the 0.001 gap to the prize cutoff.
- A **design proposal for v2** (`docs/v2_design.md`, `docs/v2_feature_engineering.md`, `docs/v2_mlops_design.md`) that closes the technical gaps vs the rank-3 solution — but **v2 is not implemented or validated**.

## What this repo is NOT

### 1. NOT a production-ready commercial product

The pipeline produces a CSV file for a competition scoring system. It does not:
- Expose a REST API for real-time inference
- Have a dashboard for non-technical users
- Persist model artifacts for production deployment (the v2 MLOps design doc proposes this, but it is not implemented)
- Have drift monitoring, alerting, or rollback
- Have any SLA, uptime, or support model

If you want to deploy this for paying customers, expect 3-6 months of additional engineering.

### 2. NOT validated for real-world deployment

The model was trained and evaluated on the Zindi competition dataset:
- 1,821 training rows from **2 pilot regions** (specific locations not disclosed by Zindi)
- Test set of 1,030 rows from the **same 2 pilot regions**

The model has **not** been validated on:
- Aquaculture ponds in Bangladesh, Vietnam, Thailand, Ecuador, Nigeria, Egypt, or any other country
- Different spectral signatures (different atmospheric conditions, different pond turbidity, different algae species)
- Different pond types (shrimp vs tilapia vs catfish vs seaweed vs integrated rice-fish systems)
- Different sensor configurations (Sentinel-2 L1C vs L2A, Sentinel-1 IW vs EW)

Real-world deployment requires a retraining + validation cycle on locally-labeled data. Expect 1-3 months of data collection + labeling before any deployment.

### 3. NOT licensed for commercial use of the trained model

**Important:** The Zindi competition data may have a non-commercial use clause. The pipeline code in this repo is MIT-licensed and free to use commercially. **But a model trained on Zindi's `Train.csv` may not be deployable in a commercial product without retraining on independently-labeled data.**

Before any commercial deployment:
1. Read the Zindi Terms of Use for this specific competition
2. If non-commercial: retrain on publicly available Sentinel-1/2 imagery with independently-labeled pond boundaries (e.g., via GeoWiki, Collect Earth, or manual digitization)
3. Validate that the retrained model achieves similar performance to the rank-6 result

This repo's MIT license covers the **code**, not the **trained model weights**. The weights are not committed to the repo (only the training code is), but if you run `python -m src.run_pipeline` and save the resulting model, the Zindi data license may restrict how you use that trained model.

### 4. NOT technically the strongest in the landscape

See `docs/competitor_audit.md` for the full comparison. The rank-3 solution (zinmori, public GitHub) is technically superior on every ML axis:

| Dimension | Ours (rank 6) | zinmori (rank 3) |
|---|---|---|
| Features | 59 | 316 |
| Model fits | 10 | 480 |
| Threshold calibration | Probe-submission (LB side-channel) | Saerens/EM (principled, unsupervised) |
| Pseudo-labelling | None | 3 rounds (+0.026) |
| SHAP explainability | None | Full |
| Runtime | ~10-12 min | 3h 15min |

Our genuine differentiators are:
- **Reproducibility** — 19× faster runtime than zinmori
- **Professional packaging** — only repo in the landscape with MIT LICENSE + 8 tests + CI on 3 Python versions + community files
- **Honest documentation** — candid post-mortem + this limitations doc

We do NOT claim to be the most technically sophisticated. We claim to be the most reproducible, the most professionally packaged, and the most honestly documented.

### 5. NOT a medical/veterinary device

If used for shrimp disease early warning (the most commercially compelling use case), this pipeline is a **decision-support tool**, not a diagnostic device. It cannot replace:
- PCR testing (e.g., Genics Shrimp MultiPath2.0)
- Veterinary inspection
- On-site water quality sensors

Any commercial deployment for disease warning must include clear disclaimers: "This tool flags chlorophyll anomalies detectable from satellite imagery. It does not diagnose disease. Confirm anomalies with on-site testing."

## What you CAN use this repo for

| Use case | OK? | Notes |
|---|---|---|
| Portfolio piece for job applications | ✅ | Strongest use case — verified rank 6, modular code, tests, CI |
| Reference for learning satellite ML techniques | ✅ | The 6 physics indices, SAR-optical divergence, and contiguous-window masking are all standard RS techniques |
| Starting point for academic research | ✅ | Cite the repo + the Zindi competition |
| Reproducing the rank-6 submission | ✅ | Run `python -m src.run_pipeline` on the Zindi data |
| Internal R&D at a geospatial ML company | ✅ | Adapt the feature engineering to your own data |
| Direct commercial deployment | ⚠️ | Requires retraining on non-Zindi data + validation on target geography + building an API + UI |
| Selling the trained model as-is | ❌ | Zindi data license may forbid this; check before any commercial use |

## Known technical limitations

### v1 (current)

1. **StratifiedGroupKFold with vacuous groups** — the `groups=np.arange(len(train))` argument collapses the group constraint, making it equivalent to plain `StratifiedKFold`. Documented in `src/train.py`. v2 proposes either deriving real spatial groups (if lat/lon were available — they're not) or honestly renaming to `StratifiedKFold`.

2. **Threshold calibration via probe-submission** — uses the public leaderboard as a side-channel to recover the test-set marginal positive count (560/1030). Defensible (private > public is the opposite of LB overfitting) but borderline. v2 proposes replacing with Saerens/EM prior correction (unsupervised, no LB feedback).

3. **Single random seed** — no ensemble variance reduction. The 0.001 gap to rank 5 is within model noise; a 5-seed ensemble would likely have closed it.

4. **No SHAP explainability** — feature importance is asserted by LightGBM gain, not validated via SHAP. v2 proposes per-fold SHAP on a held-out sample.

5. **AWEI formula was incorrect in initial release** — NIR and SWIR2 coefficients were swapped vs Feyisa et al. 2014. Fixed in v1.1.0 (see `CHANGELOG.md`).

### v2 (design only, NOT implemented)

The v2 design docs (`docs/v2_*.md`) propose closing all 5 v1 limitations + the gaps vs zinmori. **v2 is not implemented and not validated.** Implementing v2 would require:
- ~2 weeks of engineering (per the v2 MLOps design doc estimate)
- ~4-6 hours of compute for the 8-seed × 5-fold × 3-model ensemble (per the v2 ML design doc estimate)
- Validation against the user's original rank-6 submission CSV to confirm no regression

## Reference

- Competition page: https://zindi.world/competitions/geoai-aquaculture-pond-identification-challenge
- Zindi Terms of Use: https://zindi.africa/terms
- Full competitor audit: `docs/competitor_audit.md`
- v1 post-mortem: `docs/lessons_learned.md`
- v2 design proposals: `docs/v2_design.md`, `docs/v2_feature_engineering.md`, `docs/v2_mlops_design.md`

---

*This document is part of the honest disclosure practice documented in the v1 post-mortem. If you find a limitation that is not listed here, please open a GitHub issue — the goal is for this doc to be the single source of truth about what this repo can and cannot do.*
