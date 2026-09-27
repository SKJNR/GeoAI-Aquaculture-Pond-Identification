"""
Training pipeline: 5-fold StratifiedGroupKFold CV + final model training.

CV verifies the model reproduces the expected ~0.977 composite score
before the final submission is built.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from .config import N_SPLITS, K_AUG, RANDOM_STATE
from .data   import window_mask
from .features import build_features
from .models  import train_lgb, train_cb, zindi_metric


def cross_validate(train: pd.DataFrame, feats: list):
    """
    Run 5-fold StratifiedGroupKFold CV.

    Returns oof (out-of-fold probabilities) and per-fold metrics.
    """
    y_all = train['label'].values
    # NOTE: StratifiedGroupKFold is used with groups = row indices here because
    # the tabular competition data has no natural group structure (no spatial
    # tile IDs, no multi-temporal observations of the same pond). This effectively
    # collapses to plain StratifiedKFold. If spatial cluster IDs were available,
    # grouping by tile would prevent spatial-autocorrelation leakage across folds.
    sgkf  = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=RANDOM_STATE)
    folds = list(sgkf.split(train, y_all, groups=np.arange(len(train))))
    oof_lgb = np.zeros(len(y_all))
    oof_cb  = np.zeros(len(y_all))

    t_start = time.time()
    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        t_fold = time.time()
        # 8x augmented training data
        frames = [
            window_mask(
                train.iloc[tr_idx].reset_index(drop=True),
                np.random.default_rng(tr_idx[0] * 1000 + k)
            )
            for k in range(K_AUG)
        ]
        Xtr  = build_features(pd.concat(frames, axis=0).reset_index(drop=True))
        y_tr = np.tile(y_all[tr_idx], K_AUG)

        # Single-masked validation (matches test distribution)
        rng  = np.random.default_rng(999 + tr_idx[0])
        Xval = build_features(
            window_mask(train.iloc[val_idx].reset_index(drop=True), rng)
        )

        ml = train_lgb(Xtr[feats], y_tr)
        mc = train_cb(Xtr[feats], y_tr)

        oof_lgb[val_idx] = ml.predict_proba(Xval[feats])[:, 1]
        oof_cb[val_idx]  = mc.predict_proba(Xval[feats])[:, 1]
        blend = 0.5 * oof_lgb[val_idx] + 0.5 * oof_cb[val_idx]
        elapsed = time.time() - t_fold
        print(f'  Fold {fold}: CV={zindi_metric(y_all[val_idx], blend):.4f}  '
              f'AUC={roc_auc_score(y_all[val_idx], blend):.4f}  ({elapsed:.1f}s)')

    oof = 0.5 * oof_lgb + 0.5 * oof_cb
    total = time.time() - t_start
    print(f'\nCV: {zindi_metric(y_all, oof):.4f}  '
          f'AUC={roc_auc_score(y_all, oof):.4f}  ({total:.1f}s)')
    return oof


def train_final(train: pd.DataFrame, feats: list):
    """
    Train final LightGBM + CatBoost on the full dataset with 8x augmentation.
    Returns (lgb_model, cb_model).
    """
    y_all = train['label'].values
    frames_f = [
        window_mask(
            train.reset_index(drop=True),
            np.random.default_rng(2026 + k)
        )
        for k in range(K_AUG)
    ]
    Xfull  = build_features(pd.concat(frames_f, axis=0).reset_index(drop=True))
    y_full = np.tile(y_all, K_AUG)

    ml_f = train_lgb(Xfull[feats], y_full)
    mc_f = train_cb(Xfull[feats], y_full)
    return ml_f, mc_f
