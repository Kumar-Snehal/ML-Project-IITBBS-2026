# English Premier League Dataset (organised)

Source: Kaggle "English Premier League" (saife245), which repackages football-data.co.uk season files.
Coverage: seasons 2000-01 to 2021-22. See `MANIFEST.csv` for every file with row/column counts.

## Folder layout

| Folder | Contents | Use it for |
|---|---|---|
| `01_raw_seasons/` | One file per season, byte-for-byte original (renamed `EPL_<season>_raw.csv`) | Full detail: shots, cards, referee, many bookmakers' odds |
| `02_clean/` | `epl_matches_clean_all_seasons.csv` - all seasons in one tidy file | Most analysis and modelling |
| `03_features/` | Engineered / preprocessed feature tables | Ready-to-train ML inputs |
| `04_reference/` | League positions, team list, column dictionary | Lookups and documentation |

## Important notes

1. **Two seasons are incomplete** in the source data: `2018-19` (160 of 380 matches) and `2019-20` (260 of 380).
   They are kept, flagged `_PARTIAL_` in the raw folder and `SeasonComplete = False` in the clean file.
   Do not use them for season-to-date features.
2. **Middlesboro** (old files) was renamed **Middlesbrough** in cleaned files.
3. **Dates** in `02_clean` and `03_features` are ISO (`YYYY-MM-DD`). Raw files mix `dd/mm/yy` and `dd/mm/yyyy`.
4. **Bookmaker odds** (`B365H/D/A`) are blank for 2000-01 to 2008-09 in the clean file (not in those source files).
   `Attendance` is blank for some seasons.
5. `kaggle_preprocessed_test_split_...csv` is a **subset** of `kaggle_preprocessed_final_dataset_...csv`
   (the last three seasons), not independent data. Don't train on one and test on the other.
6. The Kaggle preprocessed files merge draws into the target (`FTR` = `H` or `NH`). Use `02_clean` for 3-class H/D/A.
7. `epl_features_engineered_...csv` target `y`: 0 = Away win, 1 = Draw, 2 = Home win. Features use only matches before each game.

## Clean file columns (`02_clean/epl_matches_clean_all_seasons.csv`)

`Season, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR, Referee, Attendance,
HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR, B365H, B365D, B365A, SeasonComplete`

Meanings (FT = full time, HT = half time, H/A = home/away): goals, result (H/D/A), shots (S), shots on target (ST),
fouls (F), corners (C), yellow (Y) and red (R) cards, Bet365 odds for home/draw/away.
Full key: `04_reference/football_data_column_dictionary.txt`.

## Which file should I use?

- Predicting match results with leak-free features: `03_features/epl_features_engineered_*.csv`
- Building your own features: `02_clean/epl_matches_clean_all_seasons.csv`
- Looking at bookmakers' odds from many firms: the relevant file in `01_raw_seasons/`
