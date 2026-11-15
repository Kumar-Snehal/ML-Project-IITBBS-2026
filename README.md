# EPL Match Outcome Prediction

**ML Project, IIT Bhubaneswar 2026**

Predict the result of an English Premier League match — **Home win / Draw / Away win** — using only
information that was available *before* kick-off. Two models are built and compared:

- **Naive Bayes** with **Laplace (add-one) smoothing**
- **Multinomial Logistic Regression**

Both are benchmarked against a naive baseline and against the **Bet365 bookmaker's odds**.

> **Headline result (test set, 1,140 matches from 2015-16 to 2017-18):**
> Logistic Regression reaches **54.1 % accuracy**, the same as the bookmaker (54.1 %), and a log loss of
> 0.970 against the bookmaker's 0.962. Naive Bayes reaches 53.0 % accuracy and 0.980 log loss. Always
> predicting "Home win" gives 45.4 %.

---

## Table of contents

1. [What this project does (in plain words)](#1-what-this-project-does-in-plain-words)
2. [Results at a glance](#2-results-at-a-glance)
3. [Repository structure](#3-repository-structure)
4. [Getting started](#4-getting-started)
5. [The pipeline, step by step](#5-the-pipeline-step-by-step)
6. [The data](#6-the-data)
7. [Feature engineering](#7-feature-engineering)
8. [Feature selection](#8-feature-selection)
9. [The models](#9-the-models)
10. [Evaluation](#10-evaluation)
11. [Output files reference](#11-output-files-reference)
12. [Configuration](#12-configuration)
13. [Reproducibility](#13-reproducibility)
14. [Troubleshooting](#14-troubleshooting)
15. [Limitations and ideas for extension](#15-limitations-and-ideas-for-extension)
16. [Changes in this version of the repository](#16-changes-in-this-version-of-the-repository)

---

## 1. What this project does (in plain words)

**The question.** Given two teams about to play, can we estimate the probabilities of a home win, a draw and
an away win — using only what is known before the match?

**The approach.**

1. **Data.** 22 seasons of Premier League results from Kaggle (originally from football-data.co.uk).
2. **Cleaning.** Merge the season files into one consistent table.
3. **Features.** For every match, compute numbers describing both teams *as they stood just before the
   match*: an Elo rating, points per game, recent form, head-to-head record, rest days, and so on. Each
   feature is a difference between the home and away team where that makes sense.
4. **Split by time.** Train on the oldest seasons, tune on the next two, test on the most recent three.
   A random split would be wrong here, because it would let the model learn from matches played *after*
   the ones it is tested on.
5. **Models.** Fit Naive Bayes and Logistic Regression, choosing their settings on the validation seasons.
6. **Evaluation.** Score both once on the untouched test seasons and compare against baselines and the
   bookmaker.

**Three ideas run through the whole project:**

| Idea | Why it matters | How it is enforced |
|---|---|---|
| **No future information** | A model that peeks at the future looks brilliant and is useless | Features are computed in date order, from past matches only. `check_leakage` tests this. |
| **Chronological evaluation** | Predicting later matches from earlier ones is the real use case | Train, validation and test are consecutive blocks of seasons |
| **Test set used once** | Tuning on the test set gives over-optimistic results | All tuning uses validation. `evaluate` is run once at the end. |

---

## 2. Results at a glance

Test set: 1,140 matches (2015-16, 2016-17, 2017-18). Models were refit on train + validation before testing.

| Model | Accuracy ↑ | Log loss ↓ | Brier ↓ | RPS ↓ | Macro-F1 ↑ | Draws predicted |
|---|---|---|---|---|---|---|
| Always predict Home (baseline) | 0.4535 | 1.0669 | 0.6447 | 0.2275 | 0.2080 | 0 |
| **Naive Bayes** (4 features, 5 bins, α = 1) | 0.5298 | 0.9797 | 0.5834 | 0.1983 | 0.3899 | 0 |
| **Logistic Regression** (11 features, C = 0.01) | **0.5412** | **0.9699** | **0.5756** | **0.1946** | 0.3964 | 0 |
| Bookmaker (Bet365, margin removed) | 0.5412 | 0.9624 | 0.5709 | 0.1923 | 0.4014 | 0 |
| *Ablation:* Naive Bayes, all 19 features | 0.5088 | 1.5502 | 0.7177 | 0.2534 | 0.4630 | 191 |

↑ higher is better, ↓ lower is better. See [Evaluation](#10-evaluation) for what each metric means.

**What to take from this:**

- Both models clearly beat the baseline, and Logistic Regression is essentially level with the bookmaker on
  accuracy and close on probability quality.
- **Neither model ever predicts a draw.** Draws are about 25 % of matches but are rarely the single most
  likely outcome, so an "argmax" prediction almost never picks them. This is normal for football and is
  why probability-based metrics (log loss, Brier, RPS) are reported alongside accuracy.
- **The ablation row shows why feature selection matters for Naive Bayes.** Giving it all 19 features
  (many near-copies of "team strength") makes its probabilities badly over-confident: log loss rises from
  0.98 to 1.55. Its macro-F1 looks better only because it guesses draws more often, not because the
  probabilities are better.

---

## 3. Repository structure

```
ML-Project-IITBBS-2026/
├── README.md
├── requirements.txt          Python dependencies
├── run_all.sh                runs every pipeline step in order
├── .gitignore
│
├── dataset/                  ORIGINAL Kaggle data, never modified
│   ├── README.md             notes on the raw data
│   ├── MANIFEST.csv          every file with row / column counts
│   ├── seasons/              22 season files, EPL_2000-01.csv ... EPL_2021-22.csv
│   └── reference/            final league positions, column dictionary, Kaggle description
│
├── processed/                CLEANED data (cleaning only, no features)
│   └── matches_clean.csv     one table, 8,020 matches, all 22 seasons
│
├── features/                 ENGINEERED features and the splits used for modelling
│   ├── features_all.csv      6,840 matches x 19 features (complete seasons only)
│   ├── feature_dictionary.csv  name, group, description and origin of each feature
│   ├── train.csv             2000-01 .. 2012-13   (4,940 matches)
│   ├── validation.csv        2013-14 .. 2014-15   (  760 matches)
│   └── test.csv              2015-16 .. 2017-18   (1,140 matches)
│
├── models/                   trained models (.joblib)
│   ├── naive_bayes.joblib
│   └── logistic_regression.joblib
│
├── reports/                  analysis tables, validation results, test metrics, figures
│
└── src/                      ALL code. Every step is a module run with `python -m`
    ├── config.py             paths, seasons, split points, model settings (single source of truth)
    ├── cleaning/
    │   ├── data_loading.py     reads and tidies the raw season files
    │   └── clean_data.py       STEP 1
    ├── features/
    │   ├── feature_engineering.py  leak-free pre-match features + feature dictionary
    │   ├── selection.py        correlation-aware feature selection + full-rank check
    │   ├── build_features.py   STEP 2
    │   ├── check_leakage.py    safety check: no future information in the features
    │   ├── split_data.py       STEP 3
    │   └── analyse_features.py STEP 4
    └── models/
        ├── common.py           data loading and metrics
        ├── discretizer.py      quantile binning for the categorical Naive Bayes
        ├── naive_bayes.py      STEP 5
        ├── logistic_regression.py  STEP 6
        └── evaluate.py         STEP 7 (test set, run once)
```

**Data flow:**

```
dataset/seasons/*.csv
        │  Step 1  clean_data
        ▼
processed/matches_clean.csv
        │  Step 2  build_features   (+ check_leakage)
        ▼
features/features_all.csv
        │  Step 3  split_data
        ▼
features/train.csv   validation.csv   test.csv
        │                  │              │
        │  Step 4  analyse_features (train only) ──► reports/
        │  Step 5  naive_bayes         (train → tune on validation → refit train+val) ──► models/
        │  Step 6  logistic_regression (train → tune on validation → refit train+val) ──► models/
        └──────────────────────────────── Step 7  evaluate (test, once) ──► reports/
```

`processed/`, `features/`, `models/` and `reports/` are **generated** folders. They are included in the
repository for convenience, but you can delete them and recreate them with `run_all.sh`.

---

## 4. Getting started

### 4.1 Prerequisites

- **Python 3.12** (this is what the project was run on; other recent Python 3 versions are likely fine)
- `pip`
- A terminal opened in the **project root** (the folder that contains `src/` and `run_all.sh`)

### 4.2 Set up an environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`.
The results in this README were produced with pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1,
matplotlib 3.11.2 and joblib 1.6.0.

### 4.3 Run the whole pipeline

```bash
sh run_all.sh
```

To use a specific interpreter (for example the virtual environment's):

```bash
PYTHON=.venv/bin/python sh run_all.sh
```

> **Windows:** `sh` is not available in plain PowerShell. Use **Git Bash** or **WSL**, or simply run the
> eight commands in the next section one after another.

### 4.4 Run the steps one by one

Always run these from the **project root**, using `python -m` (not `python src/...`).

```bash
python -m src.cleaning.clean_data            # 1. clean raw data
python -m src.features.build_features        # 2. build features
python -m src.features.check_leakage         #    verify no future information (optional but recommended)
python -m src.features.split_data            # 3. train / validation / test split
python -m src.features.analyse_features      # 4. feature analysis and selection report
python -m src.models.naive_bayes             # 5. train Naive Bayes
python -m src.models.logistic_regression     # 6. train Logistic Regression
python -m src.models.evaluate                # 7. final test-set evaluation
```

The steps depend on each other, so run them **in this order** the first time. Afterwards you can re-run
any step on its own (for example, change `NB_BIN_GRID` in `config.py` and re-run only step 5, then step 7).

### 4.5 What a successful run looks like

- Step 1 prints `8020 matches, 22 seasons, 44 teams` and lists the two incomplete seasons.
- Step 2 prints `6840 rows, 19 features` and `full-rank check passed`.
- `check_leakage` prints five `PASS` lines and `LEAKAGE CHECK: PASSED`.
- Step 3 prints the three split sizes: 4,940, 760 and 1,140.
- Step 7 prints the results table shown in [section 2](#2-results-at-a-glance).

Re-running the whole pipeline reproduces every committed CSV in `features/` and `reports/` byte for byte.

---

## 5. The pipeline, step by step

### Step 1 — Clean the raw data
`python -m src.cleaning.clean_data`

| | |
|---|---|
| **Reads** | `dataset/seasons/EPL_*.csv` |
| **Writes** | `processed/matches_clean.csv` |

The 22 raw files are inconsistent with each other. This step makes them one table:

- keeps the same 25 columns from every season, adding empty columns where a season lacks one (older
  seasons have far fewer columns, and Bet365 odds only exist from 2009-10);
- converts dates to `YYYY-MM-DD` (the raw files mix `dd/mm/yy` and `dd/mm/yyyy`);
- reads files as Latin-1, which is how the raw files are encoded;
- renames `Middlesboro` to `Middlesbrough` so the club is not counted twice;
- adds a `Season` column and a `SeasonComplete` flag (true if the season has all 380 matches).

No features are created here. 8,020 matches, 22 seasons and 44 teams come out.

### Step 2 — Build the features
`python -m src.features.build_features`

| | |
|---|---|
| **Reads** | `processed/matches_clean.csv`, restricted to seasons 2000-01 to 2017-18 |
| **Writes** | `features/features_all.csv`, `features/feature_dictionary.csv` |

Builds the 19 pre-match features described in [section 7](#7-feature-engineering). Before writing, it
checks that:

- the output has one row per match and **no missing values**;
- **no feature is an exact linear combination of the others** (a full-rank check), because duplicated
  information makes Naive Bayes over-confident and makes regression coefficients meaningless;
- only complete seasons were used.

### Safety check — Leakage test
`python -m src.features.check_leakage`

This is the project's guarantee that the features are honest. The test picks five random matches, then for
each one **rewrites its result** (a 9–0 home win with 20 shots on target) and rebuilds all the features.
Two things must hold:

1. every feature row at or **before** that match is unchanged, because those features were computed before
   the result existed;
2. rows **after** it do change, which proves the test can actually detect a change.

If any feature used future information, check 1 would fail.

### Step 3 — Split by time
`python -m src.features.split_data`

| | Seasons | Matches | Role |
|---|---|---|---|
| `train.csv` | 2000-01 to 2012-13 | 4,940 | fit models |
| `validation.csv` | 2013-14 to 2014-15 | 760 | choose settings |
| `test.csv` | 2015-16 to 2017-18 | 1,140 | final, one-time scoring |

The split is **chronological**. Class balance (Away / Draw / Home) is 27 / 26 / 47 % in train,
31 / 22.5 / 46 % in validation and 29 / 25 / 45 % in test.

### Step 4 — Analyse the features
`python -m src.features.analyse_features`

Uses the **training set only**, so the test set is never looked at. It produces:

- the **mutual information** of every feature with the result (how much it tells us about who wins);
- the **average value of each feature** for away wins, draws and home wins;
- a **correlation heatmap** and a list of feature pairs with |r| > 0.85;
- the **feature selection decisions** for each model, with a reason for every dropped feature
  (see [section 8](#8-feature-selection)).

### Step 5 — Train Naive Bayes
`python -m src.models.naive_bayes`

1. Select features on the training set (strict decorrelation, see section 8).
2. For each number of bins in `[3, 4, 5, 6, 8, 10]`, fit on train and score on validation.
3. Pick the number of bins with the lowest **validation log loss** (5 bins).
4. Refit on train + validation and save to `models/naive_bayes.joblib`.

The validation table also scores a version that uses all features, for comparison. The same comparison is repeated on the test set in step 7 (the ablation).

### Step 6 — Train Logistic Regression
`python -m src.models.logistic_regression`

1. Select features on the training set (looser decorrelation, see section 8).
2. For each regularisation strength `C` in `[0.001 … 3.0]`, fit on train and score on validation.
3. Pick the `C` with the lowest **validation log loss** (C = 0.01).
4. Refit on train + validation, save to `models/logistic_regression.joblib`, and write the
   standardised coefficients to `reports/logistic_regression_coefficients.csv`.

### Step 7 — Final evaluation
`python -m src.models.evaluate`

> **Run this once, at the end.** Re-running it after changing models in response to the test scores would
> turn the test set into a second validation set and make the results over-optimistic.

Loads both saved models, scores them on the test set together with the baselines and the ablation, and
writes `reports/test_metrics.csv` and `reports/test_confusion_matrices.csv`.

---

## 6. The data

**Source.** Kaggle dataset *English Premier League* (author: saife245), whose files originate from
[football-data.co.uk](https://www.football-data.co.uk). The original files are kept untouched in
`dataset/seasons/`; only their names were standardised.

**Contents.** 22 season files (2000-01 to 2021-22). Each row is one match with the result, half-time
score, shots, shots on target, fouls, corners, cards and (in later seasons) betting odds. The meaning of
every column is in `dataset/reference/column_dictionary.txt`.

**Things to know about this data:**

| Issue | What we do |
|---|---|
| **2018-19 and 2019-20 are incomplete** (160 and 260 of 380 matches) | Excluded from features. Season-to-date features would be wrong on a partial season. They are flagged `_PARTIAL_` in the filenames. |
| **Gap before 2020-21** | 2020-21 and 2021-22 are complete, but there is a break of two seasons before them, so they are also excluded from features. |
| Columns differ between seasons | Only a common set of 25 columns is kept (see Step 1). |
| Mixed date formats and Latin-1 encoding | Handled in `data_loading.py`. |
| `Middlesboro` vs `Middlesbrough` | Unified. |
| Bet365 odds only from 2009-10 | Used **only** as an evaluation benchmark on the test seasons (all of which have odds), never as a feature. |

Result: **18 complete, consecutive seasons, 6,840 matches**, are used for modelling.

---

## 7. Feature engineering

Code: `src/features/feature_engineering.py`. The definition of every feature is also saved to
`features/feature_dictionary.csv`.

### 7.1 How leakage is prevented

Matches are processed **in date order**. For each match the code does two things, strictly in this order:

1. **Record the features** from each team's current state (everything known *before* the match).
2. **Then update** each team's state with the result (Elo, points, form, head-to-head).

A feature can therefore never contain the match it describes or anything after it.

### 7.2 The 19 features

Most are **differences** (home minus away), so a positive value favours the home team.

| Group | Feature | Meaning |
|---|---|---|
| **strength** | `elo_diff` | Elo rating of the home team minus the away team, before the match |
| | `abs_elo_diff` | \|`elo_diff`\|: how evenly matched the teams are (closer matches are more often drawn) |
| **season** | `ppg_diff` | Points per game this season, home minus away, blended with last season's value |
| | `gd_pg_diff` | Goal difference per game this season, home minus away, shrunk toward 0 early in the season |
| | `venue_ppg_diff` | Home team's points per game *at home* minus the away team's points per game *away* |
| **form** | `form5_diff` | Points from the last 5 matches, home minus away |
| | `gf5_diff` | Average goals scored in the last 5, home minus away |
| | `ga5_diff` | Average goals conceded in the last 5, home minus away |
| | `sot5_diff` | Average shots on target in the last 5, home minus away |
| **draw** | `draw_rate_home` | Share of the home team's last 10 matches that ended in a draw (smoothed) |
| | `draw_rate_away` | Same for the away team |
| | `total_goals10` | Average total goals per match in both teams' last 10 (low-scoring teams draw more) |
| **h2h** | `h2h_pts` | Home team's average points in the last 5 meetings of these two teams (1.0 if none) |
| | `h2h_n` | Number of previous meetings found (0 to 5) |
| **context** | `rest_home`, `rest_away` | Days since each team's last match (capped at 14) |
| | `matchweek` | Matches the home team has already played this season, plus 1 |
| | `home_promoted`, `away_promoted` | 1 if the team was not in the league the previous season |

### 7.3 Key definitions

**Elo rating.** A team starts at 1,450 the first time it appears, and ratings are updated after each match:
`K = 20`, multiplied by `1 + ln(1 + goal difference)` so that bigger wins move ratings more, with a
home-advantage bonus of 60 points when computing the expected result. At the start of each season, every
rating is pulled 20 % of the way back toward 1,500, because squads change over the summer.

**Blending with last season.** Early in a season, points per game is based on very few matches. So it is
blended with last season's value, as if the team had already played 5 "prior" matches at that rate:
`ppg = (points + 5 × last_season_ppg) / (games + 5)`. Promoted teams use 1.0. The same idea (a prior of
5 matches) smooths goal difference, home/away PPG and the draw rates.

**Form.** The sum of the points won in the last 5 matches. If fewer than 5 exist (start of a team's
history), the gap is padded with last season's PPG.

**Promoted teams.** A team is "promoted" if it did not appear in the league in the previous season. Its
recent-match history is reset, since it was playing in a different league.

### 7.4 Why `elo_home`, `elo_away`, `form5_home`, `form5_away` are *not* features

An earlier version stored both teams' values *and* their difference. The difference is an exact linear
combination of the other two, so the same information was counted twice. That makes Naive Bayes
over-confident and makes regression coefficients impossible to interpret. Only the differences are kept.
The `build_features` step now **fails** if the feature matrix is ever not full rank, so this cannot happen
by accident.

### 7.5 What the features tell us

From `reports/feature_signal.csv` (mutual information with the result, training set), the strongest
features are `elo_diff` (0.086), `ppg_diff` (0.061), `gd_pg_diff` (0.058), `form5_diff` (0.053),
`venue_ppg_diff` (0.049) and `sot5_diff` (0.048). Features about rest days, total goals and the number of
head-to-head meetings carry essentially no signal.

`reports/feature_class_means.csv` shows the pattern directly. For example, the average `elo_diff` is
−86 before away wins, −14 before draws and +55 before home wins. `abs_elo_diff` is lowest before draws
(109, against 133 for away wins and 124 for home wins), which supports the idea that evenly matched teams
draw more.

---

## 8. Feature selection

Code: `src/features/selection.py`. Run on the **training set only**.

**The problem.** Several features measure the same thing, team strength. For example `elo_diff` and
`ppg_diff` have a correlation of 0.96. Four pairs exceed |r| = 0.85
(`reports/high_correlation_pairs.csv`).

- **Naive Bayes** multiplies one probability per feature, assuming they are independent. With several
  near-copies of the same signal, the same evidence is counted several times, and the predicted
  probabilities become far too extreme.
- **Logistic Regression** copes better, but near-duplicates still make its coefficients unstable.

**The method (greedy selection):**

1. Rank features by mutual information with the result, highest first.
2. Walk down the list and **keep** a feature unless:
   - its |correlation| with an already-kept feature is above the model's threshold, or
   - (Naive Bayes only) it is a transform of an already-kept feature, such as `abs_elo_diff` from
     `elo_diff`;
3. Features with zero mutual information are never kept.

| | Threshold | Derived features | Features kept |
|---|---|---|---|
| **Naive Bayes** | \|r\| ≤ 0.50 | not allowed | 4: `elo_diff`, `matchweek`, `away_promoted`, `draw_rate_home` |
| **Logistic Regression** | \|r\| ≤ 0.85 | allowed | 11: `elo_diff`, `form5_diff`, `venue_ppg_diff`, `sot5_diff`, `ga5_diff`, `gf5_diff`, `h2h_pts`, `abs_elo_diff`, `matchweek`, `away_promoted`, `draw_rate_home` |

`abs_elo_diff` is kept for Logistic Regression because a linear model cannot compute an absolute value on
its own, and it is the signal for evenly matched (drawish) games. Naive Bayes does not need it, because
it learns a separate distribution of `elo_diff` for each outcome.

The decision and reason for **every** feature is in `reports/feature_selection_naive_bayes.csv` and
`reports/feature_selection_logistic_regression.csv`. For example: *"ppg_diff — dropped — |r| = 0.96 with
selected feature elo_diff"*.

---

## 9. The models

### 9.1 Naive Bayes with Laplace smoothing

Code: `src/models/discretizer.py`, `src/models/naive_bayes.py`.

**Pipeline:** `QuantileBinner` → `CategoricalNB(alpha = 1.0)`.

Naive Bayes classifies by combining a prior for each outcome with a probability for each feature value
given the outcome:

```
P(outcome | features)  ∝  P(outcome) × Π  P(feature_i takes this value | outcome)
```

**Why the binning step.** Laplace smoothing is defined for *counts*, so the features must be discrete.
`QuantileBinner` cuts every feature into `n_bins` groups of roughly equal size. The cut points are learned
from the **training data only** and then reused unchanged for validation and test data, so nothing leaks
from them. Binary features end up with fewer bins automatically.

**Why Laplace smoothing.** Without it, a bin that never occurred for some outcome in the training data
would get probability **0**, and because probabilities are multiplied together, a single zero erases the
entire prediction. Add-one smoothing prevents this:

```
P(bin b | outcome c) = ( count(b, c) + α ) / ( count(c) + α × n_bins )        with α = 1
```

Every bin therefore keeps a small non-zero probability. The code also sets `min_categories` so that every
possible bin index has a smoothed probability, even a bin that is empty in the training data.

**Settings chosen on validation:** 5 bins, α = 1 (log loss 0.9766 on validation).

### 9.2 Logistic Regression

Code: `src/models/logistic_regression.py`.

**Pipeline:** `StandardScaler` → multinomial `LogisticRegression` with an L2 penalty.

Features are standardised (mean 0, spread 1) so the penalty treats them equally and the coefficients are
comparable. The regularisation strength `C` (smaller = stronger penalty) is chosen on validation:
**C = 0.01** (log loss 0.9677 on validation).

**Reading the coefficients** (`reports/logistic_regression_coefficients.csv`, standardised, one column per
outcome). `elo_diff` dominates: −0.378 for away win and +0.411 for home win, so a larger Elo advantage
makes a home win more likely and an away win less likely. `sot5_diff` is the next strongest (−0.139 for
away, +0.115 for home). `abs_elo_diff` has a negative coefficient for draws (−0.069): the more mismatched
the teams, the fewer draws.

### 9.3 How the final models are produced

For both models: **fit on train → choose settings on validation → refit on train + validation with the
chosen settings → save.** Using the validation seasons for the final fit gives the model 15 seasons of
data instead of 13, while the test seasons stay untouched.

The saved `.joblib` file holds the fitted pipeline and also the feature list and chosen settings, so
`evaluate` can reload everything without recomputing the selection.

---

## 10. Evaluation

### 10.1 Metrics

All are in `src/models/common.py`. The models output a probability for each of the three outcomes.

| Metric | What it measures | Better |
|---|---|---|
| **Accuracy** | Share of matches where the most probable outcome was correct | higher |
| **Log loss** | Penalises confident wrong predictions heavily. The **main metric**, used to choose settings. | lower |
| **Brier score** | Mean squared error of the probability vector | lower |
| **RPS** (ranked probability score) | Like Brier, but treats outcomes as ordered (Away < Draw < Home), so predicting a draw gets partial credit when the result was a narrow win | lower |
| **Macro-F1** | F1 averaged over the three outcomes, treating draws as important as wins | higher |
| **Draws predicted** | How many times "Draw" was the top prediction | n/a |

**Why not just accuracy?** Draws make up a quarter of matches and are rarely the single most likely
outcome. A model can reach a respectable accuracy without ever predicting one. Probability-quality
metrics show whether the model's probabilities are sensible even when its top pick is never a draw.

### 10.2 Baselines and ablation

| Row | What it is | Purpose |
|---|---|---|
| `always_home` | Gives every match the historical outcome frequencies, so it always picks "Home" | The minimum any model must beat |
| `bookmaker_bet365` | Bet365's odds converted to probabilities, with the bookmaker's margin removed (`1/odds`, then normalised to sum to 1) | A strong external benchmark. **Never used as a feature.** |
| `naive_bayes_all_features` | The same Naive Bayes (same bins, same α) but fed all 19 features | Shows what goes wrong without feature selection |

### 10.3 Confusion matrices (test set)

Rows are the true result, columns are the prediction.

**Naive Bayes**

| | pred A | pred D | pred H |
|---|---|---|---|
| **true A** | 178 | 0 | 155 |
| **true D** | 95 | 0 | 195 |
| **true H** | 91 | 0 | 426 |

**Logistic Regression**

| | pred A | pred D | pred H |
|---|---|---|---|
| **true A** | 167 | 0 | 166 |
| **true D** | 71 | 0 | 219 |
| **true H** | 67 | 0 | 450 |

The empty "pred D" column is the draw problem described in section 2. Draws are classified as home or
away wins, mostly home wins.

---

## 11. Output files reference

### `reports/`

| File | Produced by | Contents |
|---|---|---|
| `feature_signal.csv` / `.png` | Step 4 | Mutual information of each feature with the result |
| `feature_class_means.csv` | Step 4 | Average feature value for away wins, draws, home wins |
| `feature_correlation.png` | Step 4 | Correlation heatmap of all features |
| `high_correlation_pairs.csv` | Step 4 | Feature pairs with \|r\| > 0.85 |
| `feature_selection_naive_bayes.csv` | Step 4 | Keep/drop decision and reason for each feature (NB) |
| `feature_selection_logistic_regression.csv` | Step 4 | Same, for Logistic Regression |
| `naive_bayes_validation.csv` | Step 5 | Validation metrics for each bin count, selected vs all features |
| `logistic_regression_validation.csv` | Step 6 | Validation metrics for each value of C |
| `logistic_regression_coefficients.csv` | Step 6 | Standardised coefficients per outcome |
| `test_metrics.csv` | Step 7 | Final test metrics for all models and baselines |
| `test_confusion_matrices.csv` | Step 7 | Confusion matrix for every model |

### Columns of `features/features_all.csv` (and the splits)

`Season, Date, HomeTeam, AwayTeam` (identify the match) · `FTR` (result: `H`, `D` or `A`) · `y` (the same
result as a number: 0 = Away, 1 = Draw, 2 = Home) · then the 19 feature columns.

Goals, shots and cards of the match itself are deliberately **not** in this file, so they cannot be used
by accident.

---

## 12. Configuration

Everything adjustable is in **`src/config.py`**. No other file hard-codes paths, seasons or settings.

| Setting | Default | Controls |
|---|---|---|
| `FEATURE_SEASONS_FIRST` / `_LAST` | `2000-01` / `2017-18` | Seasons used for features |
| `TRAIN_LAST_SEASON` | `2012-13` | End of the training period |
| `VAL_LAST_SEASON` | `2014-15` | End of the validation period (the rest is test) |
| `NB_CORR_THRESHOLD` | `0.50` | How strictly Naive Bayes features must be decorrelated |
| `LR_CORR_THRESHOLD` | `0.85` | Same, for Logistic Regression |
| `HIGH_CORR_REPORT_THRESHOLD` | `0.85` | |r| above which pairs are listed in the report |
| `LAPLACE_ALPHA` | `1.0` | Laplace smoothing strength (1 = add-one) |
| `NB_BIN_GRID` | `[3, 4, 5, 6, 8, 10]` | Bin counts tried for Naive Bayes |
| `LR_C_GRID` | `[0.001 … 3.0]` | Regularisation strengths tried for Logistic Regression |
| `RANDOM_STATE` | `0` | Seed for the mutual-information estimate |

Settings for the feature definitions themselves (Elo constants, window lengths, smoothing priors) are
constants at the top of `src/features/feature_engineering.py`.

**To add a feature:** add an entry to `FEATURE_INFO` and a matching value in the row dictionary inside
`build_features()`, both in `feature_engineering.py`. Then re-run steps 2 to 7.

**To add a model:** create a module in `src/models/` following `naive_bayes.py`, save it with `joblib`,
and add it to `probabilities` in `evaluate.py`.

---

## 13. Reproducibility

- **Deterministic.** The only random component (the mutual-information estimate) uses `RANDOM_STATE`.
  Re-running the entire pipeline regenerates every CSV in `features/` and `reports/` identically.
- **Environment.** Produced with Python 3.12.3, pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1,
  matplotlib 3.11.2 and joblib 1.6.0. `requirements.txt` does not pin versions.
- **Saved models.** `.joblib` files depend on the scikit-learn version that created them. If you load
  them with a different version, re-run steps 5 and 6 to regenerate them.

---

## 14. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | You ran a file directly or from the wrong folder. Run from the **project root** with `python -m src.<package>.<module>`. |
| `FileNotFoundError: ... features/train.csv` (or similar) | An earlier step has not been run. Run the steps in order, or use `sh run_all.sh`. |
| `sh: command not found` (Windows) | Use Git Bash or WSL, or run the eight commands in section 4.4 individually. |
| `python: command not found` | Use `python3`, or set `PYTHON=python3`. |
| Warning or error when loading a `.joblib` model | scikit-learn version mismatch. Re-run steps 5 and 6. |
| `AssertionError: an incomplete season slipped into the feature set` | `FEATURE_SEASONS_FIRST/LAST` in `config.py` now include 2018-19 or 2019-20. Restore the range. |
| `AssertionError: missing Bet365 odds in the test seasons` | The test split was changed to include seasons without odds (before 2009-10). |
| `ValueError: feature matrix has rank ...` | A newly added feature is an exact linear combination of others. Remove or replace it. |

---

## 15. Limitations and ideas for extension

**Observations about the current results:**

- **Draws are never predicted.** Both models assign draws a probability that is rarely the highest. A
  threshold or cost-sensitive rule for draws, or a dedicated "evenly matched" model, could be explored.
- **Naive Bayes is close to an Elo-only model.** After decorrelation it keeps four features, but only
  `elo_diff` has substantial signal. `matchweek`, `away_promoted` and `draw_rate_home` have very low
  mutual information (0.005 or less) and add little. The strict decorrelation threshold (0.50) is what
  leaves so few useful candidates. Trying other thresholds is a cheap experiment.
- **Logistic Regression and the bookmaker are close but not equal.** The gap in log loss is about 0.008.
  The bookmaker's odds reflect information we do not have (injuries, line-ups, transfers).
- **Mutual information is a rough guide.** Values this small are noisy, so the selection should be read as
  a sensible heuristic rather than a precise ranking.

**Data limits:** only 18 complete consecutive seasons are used. The 2020-21 and 2021-22 seasons are
complete but cannot be used without handling the two-season gap, so they are left out.

**Ideas for extension:**

- Add the 2020-21 / 2021-22 seasons as a second, later test set, with a rule for the gap in team history.
- Calibrate the probabilities (for example with isotonic or Platt calibration) and plot reliability curves.
- Compare with other models (gradient boosting, ordinal logistic regression exploiting Away < Draw < Home).
- Add squad-level information (player ratings, injuries) or market values.
- Use time-series cross-validation (expanding window) instead of a single validation block.

---

## 16. Changes in this version of the repository

Compared with the first version (feature engineering only):

| Area | Before | Now |
|---|---|---|
| **Code layout** | Separate files in `scripts/` plus helpers in `src/` | Everything in `src/`, grouped into `cleaning/`, `features/`, `models/`. Each step is a module run with `python -m`. |
| **Features** | 23, including `elo_home`, `elo_away`, `form5_home`, `form5_away` | **19.** The four redundant columns were removed, and a full-rank check now guards against redundancy. |
| **Feature selection** | None | New `selection.py`: correlation-aware, mutual-information-ranked, with a written reason for every decision. |
| **Modelling** | Not present | Naive Bayes with Laplace smoothing and Logistic Regression, tuned on validation, saved to `models/`. |
| **Naive Bayes support** | n/a | New `discretizer.py` (quantile binning learned on training data only). |
| **Evaluation** | n/a | New `evaluate.py` with six metrics, baselines, the bookmaker benchmark and an ablation. |
| **Metrics** | n/a | Accuracy, log loss, Brier, RPS, macro-F1, number of draws predicted. |
| **Config** | Paths and seasons | Also holds selection thresholds, Laplace α, and the tuning grids. |
| **`run_all.sh`** | Ran scripts | Runs the eight module commands, with an optional `PYTHON` variable. |
| **Dependencies** | pandas, numpy, scikit-learn, matplotlib | Adds `joblib`. |
| **`.gitignore`** | None | Ignores `.vscode`, `__pycache__`, `.venv`, `.env`. |