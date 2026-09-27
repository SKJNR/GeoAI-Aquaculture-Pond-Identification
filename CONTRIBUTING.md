# Contributing to GeoAI-Aquaculture-Pond-Identification

Thanks for your interest in this project. This is a portfolio artifact for the rank-6 submission to the Zindi GeoAI Aquaculture Pond Identification Challenge, so the contribution model is a little unusual.

## What kind of contributions are welcome

### ✅ Welcome
- **Bug reports** — if the notebook doesn't reproduce the rank-6 submission on your machine, please open an issue with: OS, Python version, library versions, the exact error, and the CV score you got.
- **Reproducibility fixes** — if you find a missing `mkdir`, a version-incompatible API call, or a documentation typo, please open a PR.
- **Feature requests for v2** — ideas for what a v2 of this pipeline would do differently (multi-seed ensemble, spatial-grouped CV, additional physics indices, GPU port). Open an issue with the `v2-idea` label.
- **Documentation improvements** — clearer explanations, additional citations to remote-sensing literature, diagrams, translations.

### ⚠️ Restricted
- **Pull requests that change `src/features.py`, `src/models.py`, `src/train.py`, or `src/predict.py`** will not be merged into `main`. The `main` branch is frozen at the rank-6 submission; any code change there will diverge the reproduced submission file. If you have a meaningful improvement, please target a new `v2` branch instead and explain in the PR description what score delta it produces.
- **Do not** submit the reproduced CSV to Zindi yourself — that would be a violation of Zindi's competition rules.

## Development setup

```bash
# Clone and install
git clone https://github.com/SKJNR/GeoAI-Aquaculture-Pond-Identification.git
cd GeoAI-Aquaculture-Pond-Identification
pip install -r requirements.txt
pip install pytest  # for running tests

# Download Zindi data into data/
# https://zindi.world/competitions/geoai-aquaculture-pond-identification-challenge/data
mkdir -p data submissions

# Run the test suite
pytest tests/

# Run the pipeline end-to-end
python -m src.run_pipeline
```

## How to report a bug

Open a GitHub issue with:

1. **Environment** — OS, Python version, output of `pip freeze | grep -E "lightgbm|catboost|sklearn|pandas|numpy"`
2. **What you did** — exact commands run
3. **What you expected** — e.g., "CV score ~0.977, submission file with 560 positives"
4. **What happened** — full error traceback or unexpected output
5. **Reproducibility** — did you set `OMP_NUM_THREADS=1`? Same random state?

## How to propose a v2 feature

Open an issue with the `v2-idea` label:

1. **The problem** — what limitation of v1 are you addressing?
2. **The proposed change** — concrete: which file, which function, what new behaviour?
3. **Expected score delta** — back-of-envelope estimate (e.g., "+0.001 from 5-seed ensemble")
4. **Validation plan** — how would you verify the improvement without leaking LB info?

## Code style

- Python 3.10+, type hints where they add clarity, no strict formatter enforced.
- Docstrings on every public function (Google style preferred).
- Pinned library versions in `requirements.txt`; do not introduce a new dependency without justification.

## Testing convention (when adding tests)

- Place new tests in `tests/` named `test_<module>.py`.
- Test on synthetic toy inputs where possible — do not commit real Zindi data.
- One assertion per test where reasonable; group related assertions with clear names.

## License

By contributing, you agree that your contributions will be licensed under the MIT license that covers this repository.
