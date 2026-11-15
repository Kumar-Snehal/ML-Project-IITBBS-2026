"""STEP 5 - Naive Bayes with Laplace smoothing.

Run from the project root:  python -m src.models.naive_bayes

Model : QuantileBinner (edges from training data)  ->  CategoricalNB(alpha=1.0)
        alpha = 1 is exactly Laplace (add-one) smoothing of the per-bin counts.

Features: Naive Bayes assumes the features are independent given the result. Feeding it
several copies of "team strength" (elo_diff, ppg_diff, gd_pg_diff, venue_ppg_diff all have
|r| ~ 0.9) multiplies the same evidence several times and gives wildly over-confident
probabilities. So only a decorrelated subset is used (see src/features/selection.py):
|r| <= NB_CORR_THRESHOLD between any two chosen features and no derived features.

Procedure (no test data is touched here):
  1. select features on the training split;
  2. tune the number of bins on the validation split (by log loss);
  3. refit on train + validation with the chosen settings and save the model.

Output: models/naive_bayes.joblib, reports/naive_bayes_validation.csv
"""
import joblib
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline

from src.config import (LAPLACE_ALPHA, MODELS_DIR, NB_BIN_GRID, NB_CORR_THRESHOLD, NB_MODEL_FILE,
                        REPORTS_DIR)
from src.features.feature_engineering import FEATURE_COLUMNS
from src.features.selection import select_features
from src.models.common import load_splits, score
from src.models.discretizer import QuantileBinner


def make_model(n_bins: int, n_features: int) -> Pipeline:
    # min_categories guarantees every possible bin index has a (smoothed) probability,
    # even a bin that happens to be empty for some class in the training data
    return Pipeline([
        ("bins", QuantileBinner(n_bins=n_bins)),
        ("nb", CategoricalNB(alpha=LAPLACE_ALPHA, min_categories=[n_bins] * n_features)),
    ])


def main():
    train, val, _ = load_splits()

    features, _ = select_features(train[FEATURE_COLUMNS], train["y"],
                                  corr_threshold=NB_CORR_THRESHOLD, allow_derived=False)
    print(f"Naive Bayes features ({len(features)}): {', '.join(features)}\n")

    rows = []
    for feature_set, cols in [("selected", features), ("all (for comparison)", FEATURE_COLUMNS)]:
        for n_bins in NB_BIN_GRID:
            model = make_model(n_bins, len(cols)).fit(train[cols], train["y"])
            rows.append(dict(features=feature_set, n_bins=n_bins,
                             **score(val["y"], model.predict_proba(val[cols]))))
    results = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    results.round(4).to_csv(REPORTS_DIR / "naive_bayes_validation.csv", index=False)
    print("Validation results:")
    print(results.round(4).to_string(index=False))

    chosen = results[results.features == "selected"].sort_values("log_loss").iloc[0]
    best_bins = int(chosen.n_bins)
    print(f"\nChosen: {best_bins} bins (validation log loss {chosen.log_loss:.4f})")

    train_val = pd.concat([train, val], ignore_index=True)
    model = make_model(best_bins, len(features)).fit(train_val[features], train_val["y"])
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": features, "n_bins": best_bins,
                 "alpha": LAPLACE_ALPHA}, NB_MODEL_FILE)
    print(f"Wrote {NB_MODEL_FILE}  (refit on train + validation)")


if __name__ == "__main__":
    main()
