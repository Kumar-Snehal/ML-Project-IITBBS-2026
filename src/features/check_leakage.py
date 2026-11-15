"""SAFETY CHECK - proves the features do not use future information.

Run from the project root:  python -m src.features.check_leakage

Test: take a match in the middle of the data and drastically change its result.
Every feature row at or before that match must be IDENTICAL (they were computed
before the result was known), while later rows should change.
"""
import sys

import numpy as np

from src.features.build_features import load_feature_matches
from src.features.feature_engineering import FEATURE_COLUMNS, build_features


def main() -> bool:
    matches = load_feature_matches()
    base = build_features(matches)
    rng = np.random.default_rng(0)
    all_ok = True
    for idx in rng.choice(np.arange(500, len(matches) - 500), size=5, replace=False):
        tampered = matches.copy()
        tampered.loc[idx, ["FTHG", "FTAG", "HST", "AST"]] = [9, 0, 20, 0]
        tampered.loc[idx, "FTR"] = "H"
        new = build_features(tampered)

        unchanged = np.allclose(base.loc[:idx, FEATURE_COLUMNS].to_numpy(float),
                                new.loc[:idx, FEATURE_COLUMNS].to_numpy(float))
        later_changed = not np.allclose(base.loc[idx + 1:, FEATURE_COLUMNS].to_numpy(float),
                                        new.loc[idx + 1:, FEATURE_COLUMNS].to_numpy(float))
        ok = unchanged and later_changed
        all_ok &= ok
        print(f"match #{idx:5d}  rows <= #{idx} unchanged: {unchanged}   "
              f"later rows react: {later_changed}  -> {'PASS' if ok else 'FAIL'}")

    print("\nLEAKAGE CHECK:", "PASSED - no future information in features" if all_ok else "FAILED")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
