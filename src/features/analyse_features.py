"""STEP 4 - Explore the features (TRAIN SET ONLY, so the test set is never peeked at).

Run from the project root:  python -m src.features.analyse_features

Output: reports/feature_signal.csv              mutual information of each feature with the result
        reports/feature_class_means.csv         average feature value for Away / Draw / Home results
        reports/feature_correlation.png         correlation heatmap
        reports/high_correlation_pairs.csv      pairs with |r| above the report threshold
        reports/feature_selection_naive_bayes.csv          which features NB uses, and why
        reports/feature_selection_logistic_regression.csv  which features LR uses, and why
        reports/feature_signal.png              mutual-information bar chart
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (CLASS_LABELS, HIGH_CORR_REPORT_THRESHOLD, LR_CORR_THRESHOLD,
                        NB_CORR_THRESHOLD, REPORTS_DIR, TRAIN_FILE)
from src.features.feature_engineering import FEATURE_COLUMNS, FEATURE_INFO
from src.features.selection import mutual_information, select_features


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(TRAIN_FILE)
    X, y = train[FEATURE_COLUMNS], train["y"]

    # 1. how much does each feature tell us about the result?
    mi = mutual_information(X, y)
    signal = pd.DataFrame({"feature": mi.index,
                           "group": [FEATURE_INFO[f][0] for f in mi.index],
                           "mutual_information": mi.values})
    signal.to_csv(REPORTS_DIR / "feature_signal.csv", index=False)
    print("Mutual information with the match result (train set):")
    print(signal.round(4).to_string(index=False))

    # 2. average value per outcome
    means = X.groupby(train.FTR).mean().T[CLASS_LABELS]
    means.columns = ["Away win", "Draw", "Home win"]
    means.round(3).to_csv(REPORTS_DIR / "feature_class_means.csv")

    # 3. correlation matrix + highly correlated pairs
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLUMNS))); ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=90); ax.set_yticklabels(FEATURE_COLUMNS)
    fig.colorbar(im, fraction=0.04); ax.set_title("Feature correlation (train set)")
    plt.tight_layout(); plt.savefig(REPORTS_DIR / "feature_correlation.png", dpi=150); plt.close()

    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATURE_COLUMNS)
             for b in FEATURE_COLUMNS[i + 1:] if abs(corr.loc[a, b]) > HIGH_CORR_REPORT_THRESHOLD]
    pairs_df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "correlation"]) \
                 .sort_values("correlation", key=abs, ascending=False)
    pairs_df.to_csv(REPORTS_DIR / "high_correlation_pairs.csv", index=False)
    print(f"\n{len(pairs_df)} feature pairs with |r| > {HIGH_CORR_REPORT_THRESHOLD}:")
    print(pairs_df.round(3).to_string(index=False))

    # 4. feature selection per model (the models recompute exactly the same thing)
    for name, threshold, allow_derived in [("naive_bayes", NB_CORR_THRESHOLD, False),
                                           ("logistic_regression", LR_CORR_THRESHOLD, True)]:
        selected, log = select_features(X, y, threshold, allow_derived)
        log.to_csv(REPORTS_DIR / f"feature_selection_{name}.csv", index=False)
        print(f"\n{name}: {len(selected)} features selected (|r| threshold {threshold})")
        print("  " + ", ".join(selected))

    # 5. signal chart
    fig, ax = plt.subplots(figsize=(7, 6))
    signal.iloc[::-1].plot.barh(x="feature", y="mutual_information", ax=ax, legend=False, color="C0")
    ax.set_xlabel("mutual information with result"); ax.set_title("Which features carry signal?")
    plt.tight_layout(); plt.savefig(REPORTS_DIR / "feature_signal.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
