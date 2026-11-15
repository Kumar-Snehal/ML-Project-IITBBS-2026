"""Central configuration: folder paths, season splits, model settings.
Every module imports from here so nothing is hard-coded elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- folders -------------------------------------------------------------------------
DATASET_DIR = ROOT / "dataset" / "seasons"     # original Kaggle data (never modified)
PROCESSED_DIR = ROOT / "processed"             # cleaned data (not features)
FEATURES_DIR = ROOT / "features"               # engineered features + splits
REPORTS_DIR = ROOT / "reports"                 # figures and analysis tables
MODELS_DIR = ROOT / "models"                   # trained models

# ---- files ---------------------------------------------------------------------------
CLEAN_MATCHES_FILE = PROCESSED_DIR / "matches_clean.csv"
FEATURES_FILE = FEATURES_DIR / "features_all.csv"
FEATURE_DICT_FILE = FEATURES_DIR / "feature_dictionary.csv"
TRAIN_FILE = FEATURES_DIR / "train.csv"
VAL_FILE = FEATURES_DIR / "validation.csv"
TEST_FILE = FEATURES_DIR / "test.csv"
NB_MODEL_FILE = MODELS_DIR / "naive_bayes.joblib"
LR_MODEL_FILE = MODELS_DIR / "logistic_regression.joblib"

# ---- target --------------------------------------------------------------------------
CLASS_LABELS = ["A", "D", "H"]                 # class index 0, 1, 2 (Away, Draw, Home)
LABEL_TO_INT = {label: i for i, label in enumerate(CLASS_LABELS)}

# ---- seasons -------------------------------------------------------------------------
# Only complete, consecutive seasons are used for features. 2018-19 and 2019-20 are
# incomplete in the source data, which also leaves a gap before 2020-21.
FEATURE_SEASONS_FIRST = "2000-01"
FEATURE_SEASONS_LAST = "2017-18"
FULL_SEASON_MATCHES = 380

# Chronological split (season labels compare correctly as strings)
TRAIN_LAST_SEASON = "2012-13"        # train      : 2000-01 .. 2012-13
VAL_LAST_SEASON = "2014-15"          # validation : 2013-14 .. 2014-15
                                     # test       : 2015-16 .. 2017-18

# ---- cleaning ------------------------------------------------------------------------
TEAM_NAME_FIXES = {"Middlesboro": "Middlesbrough"}

# ---- feature selection (computed on the TRAIN split only) -----------------------------
# Features are ranked by mutual information with the result and added greedily; a feature
# is skipped if its |correlation| with an already-selected feature exceeds the threshold.
# Naive Bayes assumes independent features, so it gets a much stricter threshold and also
# skips features that are deterministic functions of a selected one (e.g. abs_elo_diff).
NB_CORR_THRESHOLD = 0.50
LR_CORR_THRESHOLD = 0.85
HIGH_CORR_REPORT_THRESHOLD = 0.85

# ---- Naive Bayes ---------------------------------------------------------------------
# Continuous features are discretised into quantile bins (edges learned on training data
# only) so that a categorical Naive Bayes with Laplace (add-one) smoothing can be used.
LAPLACE_ALPHA = 1.0
NB_BIN_GRID = [3, 4, 5, 6, 8, 10]    # number of bins, tuned on the validation split

# ---- Logistic regression -------------------------------------------------------------
LR_C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]   # inverse regularisation strength

RANDOM_STATE = 0
