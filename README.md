# EPL Match Outcome Prediction

Predict Home win / Draw / Away win with Naive Bayes (Laplace smoothing) and Logistic Regression.

## Project layout
```
dataset/     original Kaggle data (read-only, never modified)
processed/   cleaned match table (cleaning only, no features)
features/    engineered features, feature dictionary, train/validation/test splits
reports/     feature analysis tables and figures
models/      trained models (next step)
src/         shared code: config.py, data_loading.py, feature_engineering.py
scripts/     one runnable script per pipeline step
```

## Pipeline (run from the project root)
| Step | Script | Output |
|---|---|---|
| 1 | `scripts/01_clean_data.py` | `processed/matches_clean.csv` |
| 2 | `scripts/02_build_features.py` | `features/features_all.csv`, `features/feature_dictionary.csv` |
| - | `scripts/check_leakage.py` | proves no future information is used |
| 3 | `scripts/03_split_data.py` | `features/train.csv`, `validation.csv`, `test.csv` |
| 4 | `scripts/04_analyse_features.py` | `reports/` (signal, correlations, class means) |

Run everything: `pip install -r requirements.txt && sh run_all.sh`

## Key decisions
- Features use only seasons 2000-01 to 2017-18. 2018-19 and 2019-20 are incomplete in the source, which also
  leaves a gap before 2020-21, so season-to-date features would be wrong.
- Every feature is computed from matches BEFORE the one being predicted (verified by `check_leakage.py`).
- Chronological split: train 2000-01..2012-13, validation 2013-14..2014-15, test 2015-16..2017-18.
- Bookmaker odds stay in `processed/matches_clean.csv` for use as an evaluation benchmark only, never as features.
- All settings (paths, seasons, split points) live in `src/config.py`.

## Next steps
5. `scripts/05_train_naive_bayes.py`   6. `scripts/06_train_logistic_regression.py`   7. `scripts/07_evaluate.py`
