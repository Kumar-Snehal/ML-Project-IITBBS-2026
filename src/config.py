"""Central configuration: folder paths, season splits, constants.
Every script imports from here so nothing is hard-coded elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- folders -------------------------------------------------------------------------
DATASET_DIR = ROOT / "dataset" / "seasons"     # original Kaggle data (never modified)
PROCESSED_DIR = ROOT / "processed"             # cleaned data (not features)
FEATURES_DIR = ROOT / "features"               # engineered features + splits
REPORTS_DIR = ROOT / "reports"                 # figures and analysis tables
MODELS_DIR = ROOT / "models"                   # trained models (later steps)

# ---- files ---------------------------------------------------------------------------
CLEAN_MATCHES_FILE = PROCESSED_DIR / "matches_clean.csv"
FEATURES_FILE = FEATURES_DIR / "features_all.csv"
FEATURE_DICT_FILE = FEATURES_DIR / "feature_dictionary.csv"
TRAIN_FILE = FEATURES_DIR / "train.csv"
VAL_FILE = FEATURES_DIR / "validation.csv"
TEST_FILE = FEATURES_DIR / "test.csv"

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
