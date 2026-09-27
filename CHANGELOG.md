# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-19

### Added
- Initial public release of the rank-6 solution to the Zindi GeoAI Aquaculture Pond Identification Challenge.
- `notebook/Solution_Notebook_v2.ipynb` — 23-cell documented solution covering all 9 Zindi-required sections + Appendix A (multi-seed reproducibility) + Appendix B (transparency on threshold calibration).
- `src/` modular Python package: `config.py`, `data.py`, `features.py`, `models.py`, `train.py`, `predict.py`, `run_pipeline.py`.
- `docs/architecture.md`, `docs/features.md`, `docs/lessons_learned.md`.
- `configs/default.yaml` hyperparameter mirror.
- `requirements.txt` with pinned versions (lightgbm 4.3.0, catboost 1.2.5, scikit-learn 1.5.0, pandas 2.2.0, numpy 1.26.0).
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `tests/test_features.py`, `.github/workflows/ci.yml`.

### Leaderboard result
- **Rank:** 6 of 1,318 teams (top 0.5%)
- **Public score:** 0.922151053
- **Private score:** 0.946981266
- **Private AUC:** 0.970627769
- **Private F1:** 0.931216931
- **Submissions used:** 30

### Known limitations
- `StratifiedGroupKFold` is used with vacuous row-index groups (collapses to plain `StratifiedKFold`) because the competition data has no natural group structure. See `src/train.py` for the explanatory comment.
- The 5-fold CV score (~0.977) is asserted as "expected" — running the notebook will produce the actual measured value.
- The probe-submission → prevalence-recovery → threshold-calibration loop is documented transparently in the notebook's Appendix B but is a borderline tactic; a v2 should re-derive the threshold from validation F1-max instead.
