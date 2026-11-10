"""Reads the original Kaggle season files and returns one tidy table (no feature engineering)."""
import re

import numpy as np
import pandas as pd

from config import DATASET_DIR, FULL_SEASON_MATCHES, TEAM_NAME_FIXES

KEEP_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A",
]
INT_COLUMNS = ["FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST",
               "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]


def season_from_filename(name: str) -> str:
    return re.match(r"EPL_(\d{4}-\d{2})", name).group(1)


def parse_dates(s: pd.Series) -> pd.Series:
    """The raw files mix dd/mm/yy and dd/mm/yyyy."""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = out.isna()
    out[short] = pd.to_datetime(s[short], format="%d/%m/%y", errors="coerce")
    return out


def load_all_seasons() -> pd.DataFrame:
    frames = []
    for path in sorted(DATASET_DIR.glob("EPL_*.csv")):
        season = season_from_filename(path.name)
        df = pd.read_csv(path, encoding="latin1")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"]).copy()
        for col in KEEP_COLUMNS:                      # seasons differ in which columns exist
            if col not in df.columns:
                df[col] = np.nan
        df = df[KEEP_COLUMNS].copy()
        df["Date"] = parse_dates(df["Date"])
        for col in ("HomeTeam", "AwayTeam"):
            df[col] = df[col].str.strip().replace(TEAM_NAME_FIXES)
        for col in INT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.insert(0, "Season", season)
        df["SeasonComplete"] = len(df) == FULL_SEASON_MATCHES
        frames.append(df)

    matches = pd.concat(frames, ignore_index=True)
    assert matches["Date"].notna().all(), "unparseable dates found"
    return matches.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
