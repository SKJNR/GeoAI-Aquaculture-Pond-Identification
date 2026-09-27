"""
Configuration for the GeoAI Aquaculture Pond Identification solution.

All hyperparameters are validated — do not change them or the reproduced
submission will diverge from the rank-6 submission file.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / 'data'
OUTPUT_DIR    = PROJECT_ROOT / 'submissions'

# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------
BANDS     = ['VH', 'VV', 'blue', 'green', 'nir', 'nira', 're1', 're2', 're3',
             'red', 'swir1', 'swir2']
SENTINELS = [-9999, -999, -32768, 9999]  # invalid-value sentinels to strip

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
SCALE_POS_WEIGHT      = 2.0      # train ~40% positive; test ~54%
K_AUG                 = 8        # augmented copies per fold
N_SPLITS              = 5        # 5-fold StratifiedGroupKFold
EXACT_TEST_POSITIVES  = 560      # measured via all-positive probe submission
RANDOM_STATE          = 42

LGB_PARAMS = {
    'max_depth': 5,
    'num_leaves': 18,
    'subsample': 0.80,
    'colsample_bytree': 0.90,
    'reg_alpha': 0.3,
    'reg_lambda': 2.0,
    'min_child_samples': 20,
    'learning_rate': 0.04,
    'n_estimators': 500,
    'scale_pos_weight': SCALE_POS_WEIGHT,
    'random_state': RANDOM_STATE,
    'verbose': -1,
}

CB_PARAMS = {
    'iterations': 600,
    'depth': 5,
    'learning_rate': 0.04,
    'l2_leaf_reg': 3,
    'class_weights': [1, SCALE_POS_WEIGHT],
    'verbose': 0,
    'eval_metric': 'AUC',
    'random_state': RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# Sample submission filename candidates (handle multiple Zindi releases)
# ---------------------------------------------------------------------------
SAMPLE_SUBMISSION_CANDIDATES = [
    'SampleSubmission__3_.csv',
    'SampleSubmission.csv',
    'sample_submission.csv',
]
