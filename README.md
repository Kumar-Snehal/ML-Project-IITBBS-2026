# EPL Match Outcome Prediction

Predict Home win / Draw / Away win with Naive Bayes (Laplace smoothing) and Logistic Regression.

## Project layout
```
dataset/        original Kaggle data (read-only, never modified)
processed/      cleaned match table (cleaning only, no features)
features/       engineered features, feature dictionary, train/validation/test splits
reports/        feature analysis, validation results, test metrics, figures
models/         trained models (.joblib)
src/            ALL code; every step is a module run with  python -m
  config.py               paths, seasons, split points, model settings
  cleaning/
    data_loading.py       reads and tidies the raw season files
    clean_data.py         STEP 1
  features/
    feature_engineering.py  leak-free pre-match features + feature dictionary
    selection.py          correlation-aware feature selection, full-rank check
    build_features.py     STEP 2
    check_leakage.py      safety check: no future information in features
    split_data.py         STEP 3
    analyse_features.py   STEP 4
  models/
    common.py             data loading and metrics (accuracy, log loss, Brier, RPS, macro-F1)
    discretizer.py        quantile binning for the categorical Naive Bayes
    naive_bayes.py        STEP 5
    logistic_regression.py  STEP 6
    evaluate.py           STEP 7 (test set, run once)
```

## Pipeline (run from the project root)
| Step | Command | Output |
|---|---|---|
| 1 | `python -m src.cleaning.clean_data` | `processed/matches_clean.csv` |
| 2 | `python -m src.features.build_features` | `features/features_all.csv`, `features/feature_dictionary.csv` |
| - | `python -m src.features.check_leakage` | proves no future information is used |
| 3 | `python -m src.features.split_data` | `features/train.csv`, `validation.csv`, `test.csv` |
| 4 | `python -m src.features.analyse_features` | `reports/` (signal, correlations, class means, feature selection) |
| 5 | `python -m src.models.naive_bayes` | `models/naive_bayes.joblib`, `reports/naive_bayes_validation.csv` |
| 6 | `python -m src.models.logistic_regression` | `models/logistic_regression.joblib`, `reports/logistic_regression_*.csv` |
| 7 | `python -m src.models.evaluate` | `reports/test_metrics.csv`, `reports/test_confusion_matrices.csv` |

Run everything: `pip install -r requirements.txt && sh run_all.sh`

## Key decisions
- Features use only seasons 2000-01 to 2017-18. 2018-19 and 2019-20 are incomplete in the source, which also
  leaves a gap before 2020-21, so season-to-date features would be wrong.
- Every feature is computed from matches BEFORE the one being predicted (verified by `check_leakage`).
- Chronological split: train 2000-01..2012-13, validation 2013-14..2014-15, test 2015-16..2017-18.
  Hyper-parameters are tuned on validation; final models are refit on train + validation and scored
  on the test set once.
- Bookmaker odds stay in `processed/matches_clean.csv` for use as an evaluation benchmark only, never as features.
- All settings (paths, seasons, split points, model grids) live in `src/config.py`.

## Redundant features
- No feature is an exact linear combination of others. `elo_home`/`elo_away` and `form5_home`/`form5_away`
  were removed because `elo_diff` and `form5_diff` are their differences; `build_features` now fails if the
  feature matrix is not full rank.
- Strength features are still strongly correlated (`elo_diff` vs `ppg_diff` r = 0.96). Each model therefore
  uses a subset chosen on the training set (`src/features/selection.py`): features are ranked by mutual
  information and a feature is skipped if |r| with an already chosen one exceeds the model's threshold.
  - Naive Bayes: |r| <= 0.50, and no features derived from a chosen one (`abs_elo_diff`).
  - Logistic regression: |r| <= 0.85; `abs_elo_diff` kept (a linear model cannot compute |x| itself).
  - The decision for every feature is written to `reports/feature_selection_<model>.csv`.

## Naive Bayes and Laplace smoothing
Laplace smoothing applies to counts, so continuous features are first discretised into quantile bins (edges
learned on training data only, `src/models/discretizer.py`) and a `CategoricalNB(alpha=1)` is fitted, which is
exactly add-one smoothing of the per-bin counts. The number of bins is tuned on validation.
`reports/test_metrics.csv` includes `naive_bayes_all_features` to show the effect of feeding correlated
features to Naive Bayes (test log loss 1.55 vs 0.98 with the selected features).
