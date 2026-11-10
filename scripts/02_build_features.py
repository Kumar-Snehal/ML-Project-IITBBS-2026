"""STEP 2 - Build the leak-free pre-match features.

Input : processed/matches_clean.csv
Output: features/features_all.csv        one row per match: keys + target + features
        features/feature_dictionary.csv  name, group and meaning of each feature
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from config import (CLEAN_MATCHES_FILE, FEATURE_DICT_FILE, FEATURES_DIR, FEATURES_FILE,
                    FEATURE_SEASONS_FIRST, FEATURE_SEASONS_LAST)
from feature_engineering import FEATURE_COLUMNS, build_features, feature_dictionary

matches = pd.read_csv(CLEAN_MATCHES_FILE, parse_dates=["Date"])
matches = matches[(matches.Season >= FEATURE_SEASONS_FIRST) & (matches.Season <= FEATURE_SEASONS_LAST)]
assert matches.SeasonComplete.all(), "an incomplete season slipped into the feature set"

features = build_features(matches)

nan_counts = features[FEATURE_COLUMNS].isna().sum()
assert nan_counts.sum() == 0, f"unexpected missing values:\n{nan_counts[nan_counts > 0]}"
assert len(features) == len(matches)

FEATURES_DIR.mkdir(exist_ok=True)
out = features.copy()
out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
out.to_csv(FEATURES_FILE, index=False)
feature_dictionary().to_csv(FEATURE_DICT_FILE, index=False)

print(f"Wrote {FEATURES_FILE}  ({len(features)} rows, {len(FEATURE_COLUMNS)} features)")
print(f"Wrote {FEATURE_DICT_FILE}")
print("\nFeatures by group:")
print(feature_dictionary().groupby("group").feature.apply(list).to_string())
