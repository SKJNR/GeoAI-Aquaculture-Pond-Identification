"""
Sanity tests for src/features.py and src/data.py.

Run with:  pytest tests/

These tests verify the feature engineering on toy synthetic inputs.
They do NOT require the actual Zindi data files.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BANDS, EXACT_TEST_POSITIVES
from src.features import build_features
from src.data import window_mask
from src.predict import calibrate_threshold


# ---------------------------------------------------------------------------
# Toy data fixtures
# ---------------------------------------------------------------------------
def _make_toy_row(n_rows: int = 2, n_months: int = 12, seed: int = 0):
    """Build a tiny synthetic dataframe with the expected column schema."""
    rng = np.random.default_rng(seed)
    cols = {}
    for b in BANDS:
        for m in range(1, n_months + 1):
            cols[f'{b}_{m:02d}'] = rng.uniform(-50, 100, size=n_rows)
    df = pd.DataFrame(cols)
    df['ID'] = [f'row_{i}' for i in range(n_rows)]
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_build_features_returns_expected_column_count():
    """build_features should return exactly 59 columns (the documented count)."""
    df = _make_toy_row(n_rows=3)
    feats = build_features(df)
    assert feats.shape[1] == 59, f'Expected 59 features, got {feats.shape[1]}'


def test_build_features_preserves_row_count():
    """build_features should not change the number of rows."""
    df = _make_toy_row(n_rows=5)
    feats = build_features(df)
    assert feats.shape[0] == 5, f'Expected 5 rows, got {feats.shape[0]}'


def test_build_features_handles_all_nan_row():
    """An all-NaN row should produce all-NaN features (not crash)."""
    df = _make_toy_row(n_rows=1)
    spectral_cols = [c for c in df.columns if c != 'ID']
    df[spectral_cols] = np.nan
    feats = build_features(df)
    assert feats.shape == (1, 59)


def test_build_features_no_inf():
    """No feature should be ±inf after the final replace."""
    df = _make_toy_row(n_rows=10, seed=42)
    feats = build_features(df)
    inf_count = np.isinf(feats.select_dtypes(include=np.number).values).sum()
    assert inf_count == 0, f'Found {inf_count} inf values in features'


def test_ndvi_formula_on_known_input():
    """Verify NDVI = (NIR - red) / (NIR + red) on a hand-computed input."""
    df = pd.DataFrame({
        'nir_01': [10.0], 'red_01': [4.0],
        **{f'{b}_{m:02d}': [np.nan] for b in BANDS for m in range(1, 13)
           if not (b == 'nir' and m == 1) and not (b == 'red' and m == 1)}
    })
    feats = build_features(df)
    expected = (10.0 - 4.0) / (10.0 + 4.0 + 1e-10)
    assert abs(feats['NDVI_mean'].iloc[0] - expected) < 1e-6, \
        f'NDVI expected {expected}, got {feats["NDVI_mean"].iloc[0]}'


def test_window_mask_produces_contiguous_window():
    """window_mask should produce exactly one contiguous non-NaN window per row."""
    df = _make_toy_row(n_rows=20, seed=99)
    rng = np.random.default_rng(7)
    masked = window_mask(df, rng)
    for i in range(20):
        valid_months = []
        for m in range(1, 13):
            cols = [f'{b}_{m:02d}' for b in BANDS]
            if not masked.loc[i, cols].isna().all():
                valid_months.append(m)
        assert 4 <= len(valid_months) <= 6, \
            f'Row {i}: window length {len(valid_months)} not in [4, 5, 6]'
        if len(valid_months) > 1:
            diffs = np.diff(valid_months)
            assert (diffs == 1).all(), \
                f'Row {i}: window not contiguous, months {valid_months}'


def test_calibrate_threshold_returns_exact_target():
    """calibrate_threshold should return a threshold yielding exactly N positives."""
    rng = np.random.default_rng(123)
    probs = rng.uniform(0, 1, size=1030)
    threshold = calibrate_threshold(probs, target_positives=560)
    n_pos = int((probs >= threshold).sum())
    assert n_pos == 560, f'Expected 560 positives, got {n_pos}'


def test_calibrate_threshold_within_tolerance():
    """For a range of target counts, calibrate_threshold should hit the target within ±1."""
    rng = np.random.default_rng(456)
    probs = rng.uniform(0, 1, size=1030)
    for target in [100, 200, 560, 800, 1000]:
        threshold = calibrate_threshold(probs, target_positives=target)
        n_pos = int((probs >= threshold).sum())
        assert abs(n_pos - target) <= 1, f'Target {target}: got {n_pos} positives'
