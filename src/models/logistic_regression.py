"""STEP 6 - Multinomial logistic regression.

Run from the project root:  python -m src.models.logistic_regression

Model : StandardScaler  ->  LogisticRegression (multinomial, L2 penalty)

Features: logistic regression copes with correlated inputs much better than Naive Bayes,
but near-duplicates (|r| > LR_CORR_THRESHOLD) still make the coefficients unstable and
impossible to interpret, so they are removed. Derived features such as abs_elo_diff are
kept: a linear model cannot compute |elo_diff| itself, and it is the signal for draws.

Procedure (no test data is touched here):
  1. select features on the training split;
  2. tune the regularisation strength C on the validation split (by log loss);
  3. refit on train + validation with the chosen C and save the model.

Output: models/logistic_regression.joblib, reports/logistic_regression_validation.csv,
        reports/logistic_regression_coefficients.csv
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (CLASS_LABELS, LR_C_GRID, LR_CORR_THRESHOLD, LR_MODEL_FILE, MODELS_DIR,
                        REPORTS_DIR)
from src.features.feature_engineering import FEATURE_COLUMNS
from src.features.selection import select_features
from src.models.common import load_splits, score


def make_model(C: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=5000)),
    ])


def main():
    train, val, _ = load_splits()

    features, _ = select_features(train[FEATURE_COLUMNS], train["y"],
                                  corr_threshold=LR_CORR_THRESHOLD, allow_derived=True)
    print(f"Logistic regression features ({len(features)}): {', '.join(features)}\n")

    rows = []
    for C in LR_C_GRID:
        model = make_model(C).fit(train[features], train["y"])
        rows.append(dict(C=C, **score(val["y"], model.predict_proba(val[features]))))
    results = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    results.round(4).to_csv(REPORTS_DIR / "logistic_regression_validation.csv", index=False)
    print("Validation results:")
    print(results.round(4).to_string(index=False))

    best_C = float(results.sort_values("log_loss").iloc[0].C)
    print(f"\nChosen: C = {best_C}")

    train_val = pd.concat([train, val], ignore_index=True)
    model = make_model(best_C).fit(train_val[features], train_val["y"])

    coefs = pd.DataFrame(model.named_steps["lr"].coef_.T, index=features,
                         columns=[f"coef_{c}" for c in CLASS_LABELS])
    coefs.round(4).to_csv(REPORTS_DIR / "logistic_regression_coefficients.csv")
    print("\nStandardised coefficients (per class):")
    print(coefs.round(3).to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": features, "C": best_C}, LR_MODEL_FILE)
    print(f"\nWrote {LR_MODEL_FILE}  (refit on train + validation)")


if __name__ == "__main__":
    main()
