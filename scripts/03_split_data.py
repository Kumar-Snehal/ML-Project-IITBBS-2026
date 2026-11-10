"""STEP 3 - Chronological train / validation / test split (never random for time-ordered data).

Input : features/features_all.csv
Output: features/train.csv, features/validation.csv, features/test.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from config import (CLASS_LABELS, FEATURES_FILE, TEST_FILE, TRAIN_FILE, TRAIN_LAST_SEASON,
                    VAL_FILE, VAL_LAST_SEASON)

df = pd.read_csv(FEATURES_FILE)

splits = {
    TRAIN_FILE: df[df.Season <= TRAIN_LAST_SEASON],
    VAL_FILE:   df[(df.Season > TRAIN_LAST_SEASON) & (df.Season <= VAL_LAST_SEASON)],
    TEST_FILE:  df[df.Season > VAL_LAST_SEASON],
}
assert sum(len(s) for s in splits.values()) == len(df)

for path, part in splits.items():
    part.to_csv(path, index=False)
    balance = part.FTR.value_counts(normalize=True).reindex(CLASS_LABELS).round(3).to_dict()
    print(f"{path.name:15s} {len(part):5d} matches  {part.Season.min()} .. {part.Season.max()}  "
          f"class balance {balance}")
