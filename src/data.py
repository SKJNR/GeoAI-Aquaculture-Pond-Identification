"""
ETL: extract raw CSVs, strip sentinel values, apply contiguous-window masking.

This module handles the "Extract" and "Transform" stages of the pipeline.
The "Load" stage is implicit — all frames live in memory as pd.DataFrames
because the competition corpus (< 3 MB) fits comfortably in RAM.
"""
from pathlib import Path
import os
import pandas as pd
import numpy as np

from .config import (
    DATA_DIR, BANDS, SENTINELS, SAMPLE_SUBMISSION_CANDIDATES,
)


def load_data(data_dir: Path = None):
    """
    Load Train.csv, Test.csv, and SampleSubmission.csv from data_dir.

    Replaces sentinel values (-9999, -999, -32768, 9999) with NaN.

    Raises FileNotFoundError if no sample submission CSV is found.
    """
    data_dir = Path(data_dir or DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(
            f'Data directory {data_dir!r} does not exist. '
            f'Create it and download Train.csv, Test.csv, SampleSubmission.csv '
            f'from https://zindi.world/competitions/geoai-aquaculture-pond-identification-challenge/data'
        )
    train = pd.read_csv(data_dir / 'Train.csv')
    test  = pd.read_csv(data_dir / 'Test.csv')

    # Handle multiple possible sample-submission filenames seen in the wild.
    ss = None
    for name in SAMPLE_SUBMISSION_CANDIDATES:
        path = data_dir / name
        if path.exists():
            ss = pd.read_csv(path)
            print(f'Loaded sample submission: {name}')
            break
    if ss is None:
        raise FileNotFoundError(
            f'No sample submission CSV found in {data_dir!r}. '
            f'Expected one of: {SAMPLE_SUBMISSION_CANDIDATES}'
        )

    spectral_cols = [c for c in train.columns if c not in ['ID', 'label']]
    for df in [train, test]:
        df[spectral_cols] = df[spectral_cols].replace(SENTINELS, np.nan)
    return train, test, ss


def window_mask(raw_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Apply contiguous-window masking augmentation.

    Replicates the test distribution: a single contiguous 4-6 month window
    per row. This is the core innovation — training data must mirror the
    test structure, not use random per-month dropout.
    """
    d = raw_df.copy()
    n = len(d)
    lengths = rng.choice([4, 5, 6], size=n)
    starts  = np.array([rng.integers(1, 12 - L + 2) for L in lengths])
    keep    = np.zeros((n, 12), dtype=bool)
    for i in range(n):
        keep[i, starts[i]-1 : starts[i]-1+lengths[i]] = True
    for m in range(1, 13):
        cols = [f'{b}_{m:02d}' for b in BANDS]
        d.loc[~keep[:, m-1], cols] = np.nan
    return d
