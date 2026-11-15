"""STEP 2 - Build the leak-free pre-match features.

Run from the project root:  python -m src.features.build_features

Input : processed/matches_clean.csv
Output: features/features_all.csv        one row per match: keys + target + features
        features/feature_dictionary.csv  name, group, meaning and origin of each feature
"""
import pandas as pd

from src.config import (CLEAN_MATCHES_FILE, FEATURE_DICT_FILE, FEATURES_DIR, FEATURES_FILE,
                        FEATURE_SEASONS_FIRST, FEATURE_SEASONS_LAST)
from src.features.feature_engineering import FEATURE_COLUMNS, build_features, feature_dictionary
from src.features.selection import check_no_linear_dependence


def load_feature_matches() -> pd.DataFrame:
    """Cleaned matches restricted to the complete, consecutive seasons used for features."""
    matches = pd.read_csv(CLEAN_MATCHES_FILE, parse_dates=["Date"])
    matches = matches[(matches.Season >= FEATURE_SEASONS_FIRST) &
                      (matches.Season <= FEATURE_SEASONS_LAST)]
    assert matches.SeasonComplete.all(), "an incomplete season slipped into the feature set"
    return matches.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)


def main():
    matches = load_feature_matches()
    features = build_features(matches)

    nan_counts = features[FEATURE_COLUMNS].isna().sum()
    assert nan_counts.sum() == 0, f"unexpected missing values:\n{nan_counts[nan_counts > 0]}"
    assert len(features) == len(matches)
    check_no_linear_dependence(features[FEATURE_COLUMNS])

    FEATURES_DIR.mkdir(exist_ok=True)
    out = features.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(FEATURES_FILE, index=False)
    feature_dictionary().to_csv(FEATURE_DICT_FILE, index=False)

    print(f"Wrote {FEATURES_FILE}  ({len(features)} rows, {len(FEATURE_COLUMNS)} features)")
    print(f"Wrote {FEATURE_DICT_FILE}")
    print("No feature is a linear combination of the others (full-rank check passed).")
    print("\nFeatures by group:")
    print(feature_dictionary().groupby("group").feature.apply(list).to_string())


if __name__ == "__main__":
    main()
