"""SAFETY CHECK - proves the features do not use future information.

Test: take a match in the middle of the data and drastically change its result.
Every feature row at or before that match must be IDENTICAL (they were computed
before the result was known), while later rows should change.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from config import CLEAN_MATCHES_FILE, FEATURE_SEASONS_FIRST, FEATURE_SEASONS_LAST
from feature_engineering import FEATURE_COLUMNS, build_features

matches = pd.read_csv(CLEAN_MATCHES_FILE, parse_dates=["Date"])
matches = matches[(matches.Season >= FEATURE_SEASONS_FIRST) & (matches.Season <= FEATURE_SEASONS_LAST)]
matches = matches.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)

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
    print(f"match #{idx:5d}  rows <= #{idx} unchanged: {unchanged}   later rows react: {later_changed}  -> "
          f"{'PASS' if ok else 'FAIL'}")

print("\nLEAKAGE CHECK:", "PASSED - no future information in features" if all_ok else "FAILED")
sys.exit(0 if all_ok else 1)
