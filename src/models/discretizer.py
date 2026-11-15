"""Quantile binning so that continuous features can be used by a categorical Naive Bayes.

Laplace (add-one) smoothing is defined for counts: P(bin b | class c) =
(count(b, c) + alpha) / (count(c) + alpha * n_bins). It needs discrete feature values,
so every feature is first mapped to a small number of bins.

The bin edges are learned in fit() -- i.e. on the training data only -- and then reused
unchanged for validation and test data, so no information leaks from those splits.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileBinner(BaseEstimator, TransformerMixin):
    """Maps each column to integer bins 0..k-1 using training-set quantiles.

    - Interior edges are the 1/n, 2/n, ... quantiles of the training column.
    - Duplicate edges are removed, so low-cardinality columns (0/1 flags, h2h_n) simply
      get fewer bins instead of empty ones.
    - Values outside the training range fall into the first or last bin.
    """

    def __init__(self, n_bins=5):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.quantile(X[:, j], quantiles)) for j in range(X.shape[1])]
        self.n_categories_ = np.array([len(e) + 1 for e in self.edges_])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        # right=True: bin i holds edges[i-1] < x <= edges[i]; keeps 0/1 flags in separate bins
        return np.column_stack([np.digitize(X[:, j], self.edges_[j], right=True)
                                for j in range(X.shape[1])])
