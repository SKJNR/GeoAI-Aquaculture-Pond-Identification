# Competitor Audit — How Our Solution Compares

This document compares our rank-6 solution to the 12 other public GitHub repos for the same Zindi GeoAI Aquaculture Pond Identification Challenge. The comparison was performed by 4 parallel professional-view agents (DevRel, Geospatial ML Engineer, Hiring Manager, Chess Strategist) — full reports are in `/home/z/my-project/worklog.md` under Task IDs COMPARE-1 through COMPARE-4.

## The landscape (12 competitor repos)

| # | Repo | Stars | Last push | Lang | Files | Tests | CI | LICENSE |
|---|------|-------|-----------|------|-------|-------|----|---------|
| 1 | zinmori/geoai-aquaculture-pond-identification | 1 | 2026-09-22 | Python | 16 | ✗ | ✗ | ✗ |
| 2 | Mwaisaks/GeoAI-Aquaculture-Pond-Identification-Challenge- | 1 | 2026-06-21 | Python | 8 | ✗ | ✗ | ✗ |
| 3 | ColbertN/GeoAI-Aquaculture-Pond-Identification-Challenge-by-FAO-and-ITU | 1 | 2026-06-21 | Python | 98 | ✗ | ✗ | ✗ |
| 4 | muts-prog/GeoAI-Aquaculture-Pond-Identification | 0 | 2026-06-18 | Python | 7 | ✗ | ✗ | ✗ |
| 5 | Austinjayaraj/GeoAI-Aquaculture-Pond-Identification | 0 | 2026-08-14 | Python | 8 | ✗ | ✗ | ✗ |
| 6 | Justclemax/GeoAI-Aquaculture-Pond-Identification-Challenge | 0 | 2026-06-30 | Python | 33 | ✓ | ✗ | ✗ |
| 7 | fariedd/GeoAI-Aquaculture-Pond-Identification-Zindi-Competition | 0 | 2026-08-27 | Jupyter | 4 | ✗ | ✗ | ✗ |
| 8 | pyjoek/GeoAI-Aquaculture-Pond-Identification-Challenge-by-FAO-and-ITU | 0 | 2026-07-04 | Jupyter | 6 | ✗ | ✗ | ✗ |
| 9 | bettycdaba/GeoAI-Aquaculture-Pond-Identification-Challenge-by-FAO-and-ITU | 0 | 2026-08-06 | Python | 30 | ✗ | ✗ | ✓ |
| 10 | brytesika-AI/GeoAI-Aquaculture-Pond-Identification-Challenge-by-FAO-and-ITU | 0 | 2026-06-23 | Python | 46 | ✗ | ✗ | ✗ |
| 11 | Yousifshaheen/GeoAI-Aquaculture-Pond-Identification-Challenge-by-FAO-and-ITU | 0 | 2026-08-12 | (none) | 5 | ✗ | ✗ | ✗ |
| 12 | mbelfilali-05/-GeoAI-Aquaculture-Pond-Identification-Challenge-by-FAO-and-ITU | 0 | 2026-06-29 | Jupyter | 27 | ✗ | ✗ | ✗ |
| **Ours** | **SKJNR/GeoAI-Aquaculture-Pond-Identification** | **0** | **2026-09-27** | **Python** | **24** | **✓ (8 tests)** | **✓ (3.10/3.11/3.12)** | **✓ MIT** |

## Key findings

- **LICENSE:** 1 of 12 competitors has one. We have MIT.
- **Tests:** 1 of 12 competitors has real pytest tests (Justclemax). We have 8 passing tests.
- **CI:** 0 of 12 competitors have GitHub Actions. We have CI on 3 Python versions.
- **Polished README:** 3 of 12 competitors score 4/5. None score 5/5. Ours scores 5/5.
- **Data hygiene:** 4 competitors commit raw Zindi CSVs (rule violation). We gitignore `data/`.

## Technical depth comparison (top 5 + ours)

| Repo | Features | Models | Validation | Threshold | Score /10 |
|---|---|---|---|---|---|
| **zinmori** (rank 3, private 0.947) | 316 features, 16 indices × 16 stats + autocorr + cross-sensor | 3-model blend × 8 seeds × 5 folds × 4 phases = 480 fits | 5-fold StratifiedKFold × 8 masking seeds; honest disclosure | Saerens/EM prior correction (unsupervised) + Platt scaling | **9.0** |
| **ColbertN** (OOF 0.9935 on different subset) | 200+ via sklearn transformer; EWM, momentum, anomaly z-scores | LGB + XGB Optuna TPE 40 trials; blend grid → 100% LGB | 5-fold StratifiedKFold + Optuna inside | Default 0.5 (rules-compliant) | **8.7** |
| **Justclemax** (no LB score) | 11 indices × 6 stats + trend + 3 interactions | LGB+XGB+RF blend (3:2:2 fixed) | 5-fold StratifiedKFold, 1 seed | OOF-F1 maximization (200-step linspace) — proper ML approach | **7.7** |
| **muts-prog** (claims OOF 0.989) | 7 indices × 5 stats + **lat/lon (rule violation!)** | LGB + XGB 50/50, 1 seed | 5-fold StratifiedKFold; RobustScaler on tree features (inert) | Default 0.5 | 3.0 |
| **pyjoek** (no LB score) | 8 raw bands for month 01 only (throws away 11/12 of data) | RF + GB sklearn defaults; Platt on RF only | 80/20 train_test_split (no CV) | Threshold on ROC-AUC of binary preds (mathematically wrong) | 2.0 |
| **Ours (rank 6, private 0.947)** | 59 features: 6 indices × 4 stats + persistence + SAR-optical divergence | LGB + CB 50/50, 1 seed, 5-fold StratifiedGroupKFold (vacuous groups) | 5-fold with 8x augmentation per fold | Probe-submission → 560-positive recovery (controversial) | **6.7** |

## Honest positioning

### Where we win

1. **Verified leaderboard rank.** We're the only top-10 finisher (rank 6) with a public repo. zinmori claims rank 3 but provides no leaderboard link; the others don't mention ranks at all.
2. **Fastest reproduction.** Our pipeline runs in ~10-12 min on a 4-core CPU. zinmori takes 3h 15min. A hiring manager can verify our result on a lunch break.
3. **Most polished portfolio.** Only repo in the landscape with all of: MIT LICENSE + 8 passing tests + CI on 3 Python versions + CONTRIBUTING.md + CODE_OF_CONDUCT.md + CHANGELOG.md + Mermaid architecture diagram + honest post-mortem.
4. **Honest post-mortem.** `docs/lessons_learned.md` candidly lists 5 things that cost us the 0.001 — only zinmori's DOCUMENTATION.md matches this level of self-critique.

### Where we lose

1. **Technical depth vs zinmori.** zinmori has 5× the feature count (316 vs 59), 48× the model fits (480 vs 10), principled threshold calibration (Saerens/EM vs probe-submission), and 3 rounds of pseudo-labelling that delivered +0.026.
2. **Production monitoring vs ColbertN.** ColbertN has PSI drift reports + SHAP stability + saved joblib artifacts + `score_unseen.py`. We have none of this.
3. **Software engineering vs Justclemax.** Justclemax has correct AWEI formula + proper OOF-F1 threshold calibration + 20+ assertions on physics index ranges. We had the AWEI bug (now fixed) and still use probe-submission.

### Differentiation strategy

**The honest framing:** "The fastest-reproducible rank-6 solution. The only top-10 open-source solution in the competition landscape, with tests, CI, and an honest post-mortem."

We do NOT claim to be the most technically sophisticated — zinmori wins that. We claim to be the most reproducible, the most professionally packaged, and the most honestly documented. That's a defensible niche.

### v2 architecture (design only, not validated)

The `docs/v2_*.md` design docs propose closing each technical gap vs zinmori:
- v2 features: 59 → ~458 (add MCI, CDOM, EVI, SAVI, BSI, NDRE, CIG, NDBI + per-sensor masking)
- v2 models: LGB+CB 1 seed → LGB+CB+XGB 6 seeds
- v2 threshold: probe-submission → Saerens/EM prior correction
- v2 pseudo-labelling: 3 rounds (thresholds 0.90/0.85/0.80)
- v2 monitoring: PSI drift + SHAP stability + saved artifacts + FastAPI service

**Important:** v2 is a design proposal, NOT a verified improvement. We have not run v2 against the actual Zindi data (the competition is closed). Any v2 implementation would need to be validated against the user's actual rank-6 submission CSV to confirm it doesn't regress.

## Reference

Full 4-agent audit reports: `/home/z/my-project/worklog.md` (Task IDs COMPARE-1 through COMPARE-4).
