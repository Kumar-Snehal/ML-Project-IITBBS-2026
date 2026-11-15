"""STEP 7 - Final evaluation on the TEST split (2015-16 .. 2017-18). Run this once, at the end.

Run from the project root:  python -m src.models.evaluate

Compares the two saved models against simple baselines:
  always_home        gives every match the train+validation class frequencies as its
                     probabilities, so it always predicts the most common result, "H"
  bookmaker_bet365   Bet365 closing odds turned into probabilities (margin removed);
                     a realistic upper bench-mark, never used as a feature
and runs one ablation that shows why Naive Bayes needs decorrelated features:
  naive_bayes_all_features   same Laplace NB, same bins, but all features

Output: reports/test_metrics.csv, reports/test_confusion_matrices.csv
"""
import joblib
import numpy as np
import pandas as pd

from src.config import (CLASS_LABELS, CLEAN_MATCHES_FILE, LR_MODEL_FILE, NB_MODEL_FILE,
                        REPORTS_DIR)
from src.features.feature_engineering import FEATURE_COLUMNS
from src.models.common import confusion_table, load_splits, score
from src.models.naive_bayes import make_model as make_nb


def bookmaker_probabilities(test: pd.DataFrame) -> np.ndarray:
    odds = pd.read_csv(CLEAN_MATCHES_FILE)[["Season", "Date", "HomeTeam", "AwayTeam",
                                            "B365A", "B365D", "B365H"]]
    merged = test[["Season", "Date", "HomeTeam", "AwayTeam"]].merge(
        odds, on=["Season", "Date", "HomeTeam", "AwayTeam"], how="left", validate="one_to_one")
    implied = 1 / merged[["B365A", "B365D", "B365H"]].to_numpy(float)   # column order = A, D, H
    assert not np.isnan(implied).any(), "missing Bet365 odds in the test seasons"
    return implied / implied.sum(axis=1, keepdims=True)                # remove bookmaker margin


def main():
    train, val, test = load_splits()
    train_val = pd.concat([train, val], ignore_index=True)
    y = test["y"]

    nb = joblib.load(NB_MODEL_FILE)
    lr = joblib.load(LR_MODEL_FILE)

    prior = train_val["y"].value_counts(normalize=True).sort_index().to_numpy()
    nb_all = make_nb(nb["n_bins"], len(FEATURE_COLUMNS)).fit(train_val[FEATURE_COLUMNS],
                                                             train_val["y"])
    probabilities = {
        "always_home": np.tile(prior, (len(test), 1)),
        "naive_bayes": nb["model"].predict_proba(test[nb["features"]]),
        "naive_bayes_all_features": nb_all.predict_proba(test[FEATURE_COLUMNS]),
        "logistic_regression": lr["model"].predict_proba(test[lr["features"]]),
        "bookmaker_bet365": bookmaker_probabilities(test),
    }
    # the class-frequency baseline already predicts "H" (most frequent class) everywhere
    assert CLASS_LABELS[prior.argmax()] == "H"

    metrics = pd.DataFrame({name: score(y, p) for name, p in probabilities.items()}).T
    metrics.index.name = "model"
    REPORTS_DIR.mkdir(exist_ok=True)
    metrics.round(4).to_csv(REPORTS_DIR / "test_metrics.csv")

    confusion = pd.concat({name: confusion_table(y, p) for name, p in probabilities.items()})
    confusion.to_csv(REPORTS_DIR / "test_confusion_matrices.csv")

    print(f"TEST set: {len(test)} matches, {test.Season.min()} .. {test.Season.max()}")
    print(f"Naive Bayes: {len(nb['features'])} features, {nb['n_bins']} bins, alpha={nb['alpha']}")
    print(f"Logistic regression: {len(lr['features'])} features, C={lr['C']}\n")
    print("accuracy / macro_f1: higher is better;  log_loss / brier / rps: lower is better")
    print(metrics.round(4).to_string())
    print("\nConfusion matrices (rows = true result, columns = prediction):")
    for name in ("naive_bayes", "logistic_regression"):
        print(f"\n{name}\n{confusion.loc[name].to_string()}")


if __name__ == "__main__":
    main()
