"""STEP 1 - Read the original Kaggle season files and write ONE tidy table.
Cleaning only (consistent columns, ISO dates, team-name fix). No features here.

Input : dataset/seasons/EPL_*.csv      (never modified)
Output: processed/matches_clean.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import CLEAN_MATCHES_FILE, PROCESSED_DIR
from data_loading import load_all_seasons

matches = load_all_seasons()

PROCESSED_DIR.mkdir(exist_ok=True)
out = matches.copy()
out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
out.to_csv(CLEAN_MATCHES_FILE, index=False)

print(f"Wrote {CLEAN_MATCHES_FILE}")
print(f"  {len(matches)} matches, {matches.Season.nunique()} seasons, "
      f"{matches.HomeTeam.nunique()} teams")
partial = matches.loc[~matches.SeasonComplete, "Season"].unique().tolist()
print(f"  incomplete seasons (excluded later from features): {partial}")
