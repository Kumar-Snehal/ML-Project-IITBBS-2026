"""Correlation-aware feature selection, shared by the analysis step and the models.

Why: several features measure the same thing (team strength), e.g. elo_diff vs ppg_diff
have r = 0.96 on the training set. Naive Bayes multiplies one likelihood per feature, so
k near-copies of the same signal are counted k times and the predicted probabilities become
far too extreme. The selection below keeps the most informative feature of each correlated
group and drops the rest.

Everything here must be fitted on TRAINING data only.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.config import RANDOM_STATE
from src.features.feature_engineering import FEATURE_COLUMNS, FEATURE_INFO


def mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Mutual information of each feature with the result, sorted high -> low."""
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def check_no_linear_dependence(X: pd.DataFrame) -> None:
    """Fail loudly if any feature is an exact linear combination of the others."""
    centred = (X - X.mean()).to_numpy(float)
    rank = np.linalg.matrix_rank(centred)
    if rank < X.shape[1]:
        raise ValueError(f"feature matrix has rank {rank} < {X.shape[1]} columns: "
                         "some features are exact linear combinations of others")


def select_features(X: pd.DataFrame, y: pd.Series, corr_threshold: float,
                    allow_derived: bool, candidates=None):
    """Greedy selection.

    1. rank candidate features by mutual information with the result;
    2. walk down the ranking, adding a feature unless
       - its |Pearson r| with an already-selected feature is above corr_threshold, or
       - allow_derived is False and it is a transform of an already-selected feature
         (FEATURE_INFO 'derived_from', e.g. abs_elo_diff from elo_diff);
    3. features with zero mutual information are never selected.

    Returns (selected feature list, table explaining every decision).
    """
    candidates = list(candidates or FEATURE_COLUMNS)
    mi = mutual_information(X[candidates], y)
    corr = X[candidates].corr().abs()

    selected, log = [], []
    for feature, score in mi.items():
        derived_from = FEATURE_INFO[feature][2]
        if score <= 0:
            decision, reason = "dropped", "no mutual information with the result"
        elif not allow_derived and derived_from in selected:
            decision, reason = "dropped", f"derived from selected feature {derived_from}"
        else:
            clash = [(s, corr.loc[feature, s]) for s in selected
                     if corr.loc[feature, s] > corr_threshold]
            if clash:
                other, r = max(clash, key=lambda t: t[1])
                decision, reason = "dropped", f"|r| = {r:.2f} with selected feature {other}"
            else:
                decision, reason = "selected", ""
                selected.append(feature)
        log.append(dict(feature=feature, mutual_information=round(score, 4),
                        decision=decision, reason=reason))
    return selected, pd.DataFrame(log)
