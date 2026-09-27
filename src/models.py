"""
Models: LightGBM + CatBoost 50/50 probability blend.
"""
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier

from .config import LGB_PARAMS, CB_PARAMS


def train_lgb(X, y):
    """Train a LightGBM classifier."""
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X, y)
    return model


def train_cb(X, y):
    """Train a CatBoost classifier."""
    model = CatBoostClassifier(**CB_PARAMS)
    model.fit(X, y)
    return model


def predict_blend(models_lgb, models_cb, X):
    """
    Average predicted probabilities across a list of LightGBM and CatBoost
    models. Models must be provided as parallel lists of equal length.
    """
    p_lgb = np.mean([m.predict_proba(X)[:, 1] for m in models_lgb], axis=0)
    p_cb  = np.mean([m.predict_proba(X)[:, 1] for m in models_cb],  axis=0)
    return 0.5 * p_lgb + 0.5 * p_cb


def zindi_metric(y_true, y_prob, t=0.5):
    """Competition scoring metric: 0.6 * F1 + 0.4 * AUC."""
    from sklearn.metrics import f1_score, roc_auc_score
    return (0.6 * f1_score(y_true, (y_prob >= t).astype(int))
            + 0.4 * roc_auc_score(y_true, y_prob))
