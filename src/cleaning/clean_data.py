"""STEP 1 - Read the original Kaggle season files and write ONE tidy table.
Cleaning only (consistent columns, ISO dates, team-name fix). No features here.

Run from the project root:  python -m src.cleaning.clean_data

Input : dataset/seasons/EPL_*.csv      (never modified)
Output: processed/matches_clean.csv
"""
from src.cleaning.data_loading import load_all_seasons
from src.config import CLEAN_MATCHES_FILE, PROCESSED_DIR


def main():
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


if __name__ == "__main__":
    main()
