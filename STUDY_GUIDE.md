# Study Guide: EPL Match Outcome Prediction

A walkthrough of the project's data, code and results, written so you can explain every decision in a
report or a viva. It follows the order the pipeline runs in:

1. [The data: what is in it and what the columns mean](#1-the-data)
2. [Cleaning](#2-cleaning)
3. [Leakage: what it is and how this project prevents it](#3-leakage)
4. [Feature engineering: how each feature is computed](#4-feature-engineering)
5. [Feature selection: how features are chosen](#5-feature-selection)
6. [Which features were used and which were dropped](#6-features-used-and-dropped)
7. [The models: theory and formulas](#7-the-models)
8. [Evaluation metrics: formulas](#8-evaluation-metrics)
9. [Results: why we got them, and how they compare with benchmarks](#9-results-and-why)
10. [Quick revision: likely questions](#10-quick-revision-likely-questions)

All numbers below come from the files the pipeline writes to `reports/` and `features/`. Code references
point into `src/`.

---

## 0. The big picture in one paragraph

We want to predict the result of a Premier League match: **Home win (H), Draw (D) or Away win (A)**, using
only information available **before kick-off**. Raw match records (22 seasons) are cleaned into one table.
For every match we compute 19 numbers that describe both teams *as they stood before the match*: an Elo
rating, points per game, recent form, head-to-head record and so on. We split the matches **by time**
(old seasons for training, recent seasons for testing), choose a decorrelated subset of features for each
model, train **Naive Bayes** and **Logistic Regression**, and compare them with a trivial baseline and with
the **Bet365 bookmaker's odds**.

```
dataset/seasons/*.csv  ──clean──►  processed/matches_clean.csv  ──features──►  features/features_all.csv
                                                                                    │ split by season
                                                       train (2000-13) / validation (2013-15) / test (2015-18)
                                                                                    │
                                     feature selection (train only) ──► NB and LR ──► evaluate on test once
```

---

## 1. The data

### 1.1 Source and files

- **Source:** the Kaggle dataset *English Premier League* (author saife245). The data comes from
  [football-data.co.uk](https://www.football-data.co.uk), a standard source for football research.
- **Files:** `dataset/seasons/EPL_2000-01.csv` to `EPL_2021-22.csv`: **22 season files**, one row per match.
  A full Premier League season has 20 teams, each playing every other team home and away:
  $20 \times 19 = 380$ matches.
- **Reference files** (`dataset/reference/`): the official column key (`column_dictionary.txt`), the Kaggle
  description, and final league positions 2000-2016. The final league positions file is **not** used by
  the pipeline.
- `dataset/MANIFEST.csv` lists each file with its row and column count.

The raw files are not consistent with each other:

| Seasons | Columns | Why |
| --- | --- | --- |
| 2000-01 to 2008-09 | 23–28 | results and match statistics, a few odds |
| 2009-10 to 2018-19 | 62–71 | many bookmakers' odds added (Bet365 from 2009-10) |
| 2019-20 to 2021-22 | 106 | even more betting markets |

### 1.2 The raw columns that are kept

The cleaning step keeps 25 columns that are useful and available (or can be filled in) for every season.
"Avg" is the average over all 8,020 cleaned matches.

| Column | Meaning | Avg | Known before kick-off? |
| --- | --- | --- | --- |
| `Date` | Match date | | yes |
| `HomeTeam`, `AwayTeam` | Team names | | yes |
| `FTHG`, `FTAG` | **F**ull-**T**ime **H**ome / **A**way **G**oals | 1.52 / 1.16 | **no** |
| `FTR` | **F**ull-**T**ime **R**esult: `H`, `D`, `A`. **This is the target.** | | **no** |
| `HTHG`, `HTAG`, `HTR` | Half-time goals and result | | **no** |
| `Referee` | Referee name | | yes (not used) |
| `HS`, `AS` | Home / Away **S**hots | 13.5 / 10.6 | **no** |
| `HST`, `AST` | Home / Away **S**hots on **T**arget | 6.2 / 4.8 | **no** |
| `HF`, `AF` | **F**ouls committed | | **no** |
| `HC`, `AC` | **C**orners | 6.1 / 4.8 | **no** |
| `HY`, `AY` | **Y**ellow cards | 1.4 / 1.75 | **no** |
| `HR`, `AR` | **R**ed cards | 0.06 / 0.09 | **no** |
| `B365H`, `B365D`, `B365A` | Bet365 decimal odds for H / D / A (only from 2009-10) | | yes, but used only as a benchmark |

Two derived columns are added during cleaning: `Season` (e.g. `2012-13`) and `SeasonComplete`
(True if the season has all 380 matches).

The last column matters most. **Almost every raw column describes what happened *during* the match.**
None of them can be fed directly to a model that predicts the match before it is played. That is why the
project builds its own pre-match features ([Section 4](#4-feature-engineering)). It uses past values of
these columns, never the current match's.

Columns dropped at the cleaning stage include `Div` (always "E0"), attendance (missing in most seasons),
odds from other bookmakers, over/under 2.5 goals markets and Asian handicap markets.

### 1.3 The target variable

`FTR` is mapped to an integer `y` (see `src/config.py`):

| Label | `y` | Share of all matches | Train | Validation | Test |
| --- | --- | --- | --- | --- | --- |
| Away win `A` | 0 | 29.1 % | 27 % | 31 % | 29.2 % |
| Draw `D` | 1 | 25.2 % | 26 % | 22.5 % | 25.4 % |
| Home win `H` | 2 | 45.7 % | 47 % | 46 % | 45.4 % |

The order A < D < H is deliberate. From the home team's point of view the outcomes are **ordered**
(lose < draw < win), and the RPS metric ([Section 8](#8-evaluation-metrics)) uses that order.

The **home advantage** is visible: home teams win about 46 % of matches and away teams about 29 %. The
home-win share also drifts slightly downward over time (47 % in train against 45.4 % in test).

### 1.4 Known problems in the data and what we do about them

| Problem | Consequence if ignored | What the project does |
| --- | --- | --- |
| 2018-19 has 160/380 matches and 2019-20 has 260/380 | Season-to-date features (points per game etc.) would be wrong | Excluded from features |
| Two-season gap before 2020-21 | Team histories (Elo, form) would jump across a gap | 2020-21 and 2021-22 excluded too |
| Different columns per season | Can't concatenate | Keep a common set of 25, fill missing ones with NaN |
| Dates in `dd/mm/yy` and `dd/mm/yyyy` | Wrong dates, wrong ordering, leakage | Parse both formats (`parse_dates`) |
| Latin-1 encoding | Read errors | `encoding="latin1"` |
| `Middlesboro` vs `Middlesbrough` | One club counted as two teams, histories split | Renamed (`TEAM_NAME_FIXES`) |
| Bet365 odds only from 2009-10 | Can't use as a feature everywhere | Used only as a benchmark on the test seasons |

**Final modelling data:** 18 complete, consecutive seasons (2000-01 to 2017-18) = **6,840 matches**.

---

## 2. Cleaning

Code: `src/cleaning/data_loading.py`, `src/cleaning/clean_data.py`. Output: `processed/matches_clean.csv`.

For each season file, `load_all_seasons()`:

1. reads it as Latin-1;
2. drops rows with no `HomeTeam`, `AwayTeam` or `FTR` (some raw files have empty trailing rows);
3. adds any of the 25 kept columns that the season lacks, filled with NaN;
4. parses dates (tries `dd/mm/yyyy` first, then `dd/mm/yy`);
5. strips whitespace from team names and applies the name fix;
6. converts the numeric columns to numbers;
7. adds `Season` and `SeasonComplete`.

The seasons are then concatenated and **sorted by date** (ties broken by home team name). It asserts that
every date parsed. Result: 8,020 matches, 22 seasons, 44 distinct teams.

**Cleaning creates no features.** Separating cleaning from feature engineering keeps each step testable
and means the cleaned file is still a faithful copy of the raw data.

---

## 3. Leakage

### 3.1 What leakage is

**Data leakage** happens when information that would not be available at prediction time ends up in the
training data or the features. The model looks excellent during evaluation and fails in real use.

In this project there are four ways it could happen:

| Type of leakage | Example | How this project prevents it |
| --- | --- | --- |
| **Same-match leakage** | Using `HST` (shots on target) of the match being predicted. Shots on target are strongly linked to the result. | Match statistics are only used from **previous** matches, as rolling averages. `features_all.csv` does not even contain the current match's goals or shots. |
| **Future leakage in features** | A season-average PPG computed over the *whole* season includes matches played after the one being predicted | Features are computed in a single pass in date order: **record features first, then update** (Section 3.2) |
| **Temporal leakage in the split** | A random train/test split lets the model train on 2017 matches and test on 2005 matches | **Chronological split** by season |
| **Preprocessing / selection leakage** | Fitting the scaler, bin edges, or feature selection on all data including test | Feature selection, `StandardScaler` and `QuantileBinner` are fit on **training data only**. The test set is touched exactly once, in `evaluate.py` |

Also, **bookmaker odds are never a feature.** They are known before kick-off, so using them would not be
leakage. But the project's goal is to see how close we can get to the bookmaker *without* using it, so the
odds are kept as a benchmark only.

### 3.2 The "record, then update" rule

`build_features()` in `src/features/feature_engineering.py` keeps a `TeamState` object per team (Elo,
season points, last 10 matches, last match date). It walks through matches **in date order**:

```
for each match (in date order):
    1. RECORD FEATURES  ← read home and away TeamState (contains only earlier matches)
    2. UPDATE STATE     ← apply this match's result to Elo, points, form, head-to-head
```

Because step 1 always runs before step 2, a feature row cannot contain its own match or any later one.

### 3.3 The leakage test

`src/features/check_leakage.py` checks this rule automatically instead of trusting it:

1. Pick 5 random matches (seeded, away from the edges of the data).
2. For each one, **change its result** to a 9-0 home win with 20 shots on target, and rebuild all features.
3. Check:
   - all feature rows **up to and including** that match are **identical** (the match's own features must
     not depend on its result);
   - rows **after** it **do change** (proves the test can detect a change).

If any feature looked into the future, check (a) would fail. All five checks pass.

### 3.4 The chronological split

`src/features/split_data.py`:

| Split | Seasons | Matches | Used for |
| --- | --- | --- | --- |
| Train | 2000-01 … 2012-13 | 4,940 | fitting models, feature selection, feature analysis |
| Validation | 2013-14 … 2014-15 | 760 | choosing hyperparameters (bins, C) |
| Test | 2015-16 … 2017-18 | 1,140 | final, one-time evaluation |

After tuning, each model is **refit on train + validation** (15 seasons) and then scored on test. The test
seasons are never used to make any decision.

---

## 4. Feature engineering

Code: `src/features/feature_engineering.py`. Output: `features/features_all.csv` (6,840 rows × 19
features) and `features/feature_dictionary.csv`.

**Design principle:** most features are a **difference, home minus away**. A positive value favours the
home team. The result depends on how strong the two teams are *relative* to each other, so the difference
is what matters. It also cancels slow drifts that affect both teams equally, such as league-wide Elo
inflation.

### 4.1 Elo rating (`elo_diff`, `abs_elo_diff`)

Elo is a rating system originally used in chess. Each team has a number $R$. Before a match, the
**expected score** of the home team is

$$
E_H = \frac{1}{1 + 10^{-(R_H + h - R_A)/400}}, \qquad h = 60 \text{ (home advantage in Elo points)}
$$

This is a logistic curve: a 400-point advantage means 10:1 odds. After the match, with actual score
$S_H = 1$ (win), $0.5$ (draw) or $0$ (loss):

$$
\Delta = K \cdot \big(1 + \ln(1 + |G_H - G_A|)\big) \cdot (S_H - E_H), \qquad K = 20
$$

$$
R_H \leftarrow R_H + \Delta, \qquad R_A \leftarrow R_A - \Delta
$$

- The **margin multiplier** $1 + \ln(1 + |\text{goal difference}|)$ makes a 4-0 win move ratings more than a
  1-0 win. The logarithm keeps the effect from growing too fast.
- **Zero-sum:** what one team gains the other loses.
- **New teams** start at 1,450 (a little below the 1,500 average, because newly appearing teams are
  usually promoted and weaker).
- **Season regression:** at the start of each season every rating is pulled 20 % toward 1,500:
  $R \leftarrow 1500 + 0.8\,(R - 1500)$. Squads change over the summer, so last season's rating is only
  partly trusted.

Features:

$$
\texttt{elo\_diff} = R_H - R_A, \qquad \texttt{abs\_elo\_diff} = |R_H - R_A|
$$

`elo_diff` ranges from about −650 to +560 (std ≈ 162). `abs_elo_diff` measures how evenly matched the
teams are. Evenly matched teams draw more often. A linear model can't compute $|x|$ from $x$, so this is
given to it explicitly.

### 4.2 Season-to-date features (`ppg_diff`, `gd_pg_diff`, `venue_ppg_diff`)

Early in a season a team may have played only 1 or 2 matches, so a raw average is very noisy. The project
uses a **Bayesian-style shrinkage**. It pretends the team has already played $m = 5$ "prior" matches at
last season's rate:

$$
\text{PPG} = \frac{\text{points} + m \cdot \text{PPG}_{\text{last season}}}{\text{games} + m}, \qquad m = 5
$$

- After 0 games, PPG = last season's PPG. After 38 games, the prior is worth only 5/43 ≈ 12 %.
- Promoted teams (and the very first season) use $\text{PPG}_{\text{last season}} = 1.0$.
- This is the same idea as Laplace smoothing in Naive Bayes: add pseudo-counts to avoid extreme
  estimates from small samples.

Then:

| Feature | Formula |
| --- | --- |
| `ppg_diff` | $\text{PPG}_H - \text{PPG}_A$ |
| `gd_pg_diff` | $\dfrac{GD_H}{n_H + 5} - \dfrac{GD_A}{n_A + 5}$ (goal difference shrunk toward 0) |
| `venue_ppg_diff` | $\text{HomePPG}_H - \text{AwayPPG}_A$, where $\text{HomePPG} = \dfrac{\text{home pts} + 5\cdot\text{PPG}}{\text{home games} + 5}$ (and likewise away) |

`venue_ppg_diff` asks how good the home team is *at home* compared with how good the away team is *away*.
Its mean is +0.28, positive because of home advantage.

### 4.3 Recent form (`form5_diff`, `gf5_diff`, `ga5_diff`, `sot5_diff`)

Each team keeps a queue of its last 10 matches. Over the **last 5**:

| Feature | Formula |
| --- | --- |
| `form5_diff` | (points in last 5)$_H$ − (points in last 5)$_A$; range −15 to +15 |
| `gf5_diff` | mean goals scored in last 5, H − A |
| `ga5_diff` | mean goals conceded in last 5, H − A (positive = home team concedes more = bad for home) |
| `sot5_diff` | mean shots on target in last 5, H − A |

If a team has fewer than 5 past matches, `form_points` pads the missing ones with last season's PPG, and
the averages fall back to defaults (1.35 goals, 4.5 shots on target).

This is where the in-match statistics (`HST`, `AST`, goals) are used legally: only from **past** matches.
`sot5_diff` captures how many chances a team creates, which is less noisy than goals.

### 4.4 Draw tendency (`draw_rate_home`, `draw_rate_away`, `total_goals10`)

Draws are the hardest class, so three features target them:

$$
\texttt{draw\_rate} = \frac{\#\text{draws in last 10} + 0.26 \cdot 5}{\#\text{matches in window} + 5}
$$

(the smoothing prior is the league draw rate of about 0.26, worth 5 matches), and

$$
\texttt{total\_goals10} = \tfrac{1}{2}\big(\text{TG}_H + \text{TG}_A\big), \quad
\text{TG} = \frac{\sum_{\text{last 10}} (gf + ga) + 2.7 \cdot 3}{n + 3}
$$

The idea: low-scoring teams and draw-prone teams draw more. As the results show, these features turned out
to carry almost no signal.

### 4.5 Head-to-head (`h2h_pts`, `h2h_n`)

For each unordered pair of teams, the project stores past meetings. `h2h_pts` is the **current home
team's** average points in the last 5 meetings, counting meetings at either ground and converting points
to the current home team's perspective (3↔0, 1↔1). If they never met, it is 1.0 (neutral). `h2h_n` is the
number of meetings found (0–5).

### 4.6 Context (`rest_home`, `rest_away`, `matchweek`, `home_promoted`, `away_promoted`)

| Feature | Meaning |
| --- | --- |
| `rest_home`, `rest_away` | Days since the team's last league match, capped at 14 (also 14 if none) |
| `matchweek` | Home team's games played this season + 1 (1 to 38) |
| `home_promoted`, `away_promoted` | 1 if the team was not in the league last season. Its recent history is reset, because it was playing in a weaker league |

### 4.7 Redundancy guard: the full-rank check

An earlier version stored `elo_home`, `elo_away` **and** `elo_diff`. But
$\texttt{elo\_diff} = \texttt{elo\_home} - \texttt{elo\_away}$ exactly, so the feature matrix had a column that
was a **linear combination** of others. That causes two problems:

- In **Naive Bayes** the same information gets multiplied into the posterior more than once
  (Section 7.3).
- In **Logistic Regression** the coefficients are not identifiable: infinitely many weight vectors
  give the same predictions.

So only differences are kept, and `check_no_linear_dependence()` (in `selection.py`) computes the rank of
the centred feature matrix:

$$
\operatorname{rank}(X - \bar{X}) = p \quad (p = 19 \text{ columns})
$$

If the rank is less than 19, the build fails. Centring matters: it also catches a column that is a
constant plus a combination of others.

---

## 5. Feature selection

Code: `src/features/selection.py`, called from `analyse_features.py` (reporting) and from each model
script. Everything is computed on the **training split only**.

### 5.1 Why selection is needed

Many features measure the same thing, **team strength**. On the training set:

| Pair | Pearson r |
| --- | --- |
| `elo_diff` – `ppg_diff` | 0.96 |
| `ppg_diff` – `venue_ppg_diff` | 0.90 |
| `ppg_diff` – `gd_pg_diff` | 0.90 |
| `elo_diff` – `gd_pg_diff` | 0.89 |

Several other features have no relationship with the result at all (rest days, number of h2h meetings).
Strongly correlated features break Naive Bayes's independence assumption and make Logistic Regression
coefficients unstable. Useless features only add noise.

### 5.2 Tool 1: Mutual information (relevance)

**Mutual information** measures how much knowing a feature $X$ reduces uncertainty about the result $Y$:

$$
I(X; Y) = \sum_{y}\int p(x, y)\,\log\frac{p(x, y)}{p(x)\,p(y)}\,dx = H(Y) - H(Y \mid X)
$$

- $I = 0$ means $X$ and $Y$ are independent. Larger means more informative.
- Unlike correlation, it catches **non-linear** relationships. For example `abs_elo_diff` relates to draws
  in a non-monotonic way.
- scikit-learn's `mutual_info_classif` estimates it for continuous $X$ using a k-nearest-neighbour method.
  The estimate is noisy for small values and is clipped at 0. A fixed seed (`RANDOM_STATE = 0`) makes it
  reproducible.

For scale: the entropy of the result is $H(Y) \approx 1.06$ nats on the training set. The best feature, `elo_diff`, has
$I \approx 0.086$, so it removes only about 8 % of the uncertainty. **Football is noisy**, and that sets
an upper limit on any model.

### 5.3 Tool 2: Pearson correlation (redundancy)

$$
r_{jk} = \frac{\sum_i (x_{ij} - \bar{x}_j)(x_{ik} - \bar{x}_k)}{\sqrt{\sum_i (x_{ij} - \bar{x}_j)^2}\sqrt{\sum_i (x_{ik} - \bar{x}_k)^2}}
$$

$|r|$ near 1 means two features carry nearly the same linear information.

### 5.4 The greedy algorithm (`select_features`)

```
rank all features by mutual information, highest first
selected = []
for each feature f in that order:
    if MI(f) == 0:                                  → drop  ("no mutual information")
    elif (not allow_derived) and f is derived from a selected feature
                                                    → drop  ("derived from …")
    elif |r(f, s)| > threshold for some selected s  → drop  ("|r| = … with …")
    else                                            → keep
```

This is a simplified version of **mRMR** (maximum relevance, minimum redundancy). It takes the most
relevant feature first, then only adds features that are not redundant with what is already chosen.

Each model gets different settings, because the models react differently to correlated inputs:

| | Correlation threshold | Derived features (`abs_elo_diff`)? | Why |
| --- | --- | --- | --- |
| **Naive Bayes** | $\lvert r\rvert \le 0.50$ (strict) | Not allowed | NB *assumes* independence. Correlated features double-count evidence. NB already learns a separate distribution of `elo_diff` per class, so it can see that draws sit near 0 without needing $\lvert\cdot\rvert$. |
| **Logistic Regression** | $\lvert r\rvert \le 0.85$ (loose) | Allowed | LR estimates weights *jointly*, so moderate correlation is handled by the fit itself. Only near-duplicates are removed. LR is linear in its inputs, so it needs $\lvert\texttt{elo\_diff}\rvert$ handed to it to model "evenly matched → draw". |

The decision and reason for every feature is in `reports/feature_selection_naive_bayes.csv` and
`reports/feature_selection_logistic_regression.csv`.

---

## 6. Features used and dropped

### 6.1 Feature signal (training set)

| Rank | Feature | Group | Mutual information | Class means (A / D / H) |
| --- | --- | --- | --- | --- |
| 1 | `elo_diff` | strength | 0.0859 | −86 / −14 / +55 |
| 2 | `ppg_diff` | season | 0.0614 | −0.33 / −0.06 / +0.19 |
| 3 | `gd_pg_diff` | season | 0.0582 | −0.37 / −0.08 / +0.21 |
| 4 | `form5_diff` | form | 0.0534 | −1.83 / −0.48 / +0.77 |
| 5 | `venue_ppg_diff` | season | 0.0490 | −0.02 / +0.26 / +0.52 |
| 6 | `sot5_diff` | form | 0.0484 | −1.06 / −0.26 / +0.43 |
| 7 | `ga5_diff` | form | 0.0255 | +0.28 / +0.07 / −0.11 |
| 8 | `gf5_diff` | form | 0.0161 | −0.28 / −0.09 / +0.13 |
| 9 | `h2h_pts` | h2h | 0.0150 | 1.03 / 1.26 / 1.40 |
| 10 | `abs_elo_diff` | strength | 0.0052 | 133 / **109** / 124 |
| 11 | `matchweek` | context | 0.0043 | 19.4 / 19.2 / 19.7 |
| 12 | `away_promoted` | context | 0.0036 | 0.09 / 0.12 / 0.18 |
| 13 | `draw_rate_home` | draw | 0.0025 | 0.263 / 0.260 / 0.258 |
| 14–19 | `draw_rate_away`, `total_goals10`, `rest_home`, `h2h_n`, `rest_away`, `home_promoted` | | 0.0 | flat |

How to read the class means: for strength features the mean moves cleanly from A → D → H. That is a strong,
monotone signal. `abs_elo_diff` is **lowest for draws** (109), which confirms that evenly matched games
draw more. The draw-tendency features barely move across classes, so they are useless in practice.

A caveat: `home_promoted` gets MI = 0 while `away_promoted` gets 0.0036, although the class means show
both carry a small signal (a promoted home team wins less often). This is **estimator noise** for very
small MI values. The ranking below about 0.01 should be treated as rough.

### 6.2 What each model uses

| Feature | Naive Bayes | Logistic Regression | Reason when dropped |
| --- | --- | --- | --- |
| `elo_diff` | ✅ | ✅ | |
| `ppg_diff` | ❌ | ❌ | $\lvert r\rvert$ = 0.96 with `elo_diff` |
| `gd_pg_diff` | ❌ | ❌ | $\lvert r\rvert$ = 0.89 with `elo_diff` |
| `form5_diff` | ❌ | ✅ | NB: $\lvert r\rvert$ = 0.72 with `elo_diff` |
| `venue_ppg_diff` | ❌ | ✅ | NB: $\lvert r\rvert$ = 0.85 with `elo_diff` (just under LR's threshold) |
| `sot5_diff` | ❌ | ✅ | NB: $\lvert r\rvert$ = 0.53 with `elo_diff` |
| `ga5_diff` | ❌ | ✅ | NB: $\lvert r\rvert$ = 0.53 with `elo_diff` |
| `gf5_diff` | ❌ | ✅ | NB: $\lvert r\rvert$ = 0.58 with `elo_diff` |
| `h2h_pts` | ❌ | ✅ | NB: $\lvert r\rvert$ = 0.53 with `elo_diff` |
| `abs_elo_diff` | ❌ | ✅ | NB: derived from `elo_diff` |
| `matchweek` | ✅ | ✅ | |
| `away_promoted` | ✅ | ✅ | |
| `draw_rate_home` | ✅ | ✅ | |
| `draw_rate_away` | ❌ | ❌ | MI = 0 |
| `total_goals10` | ❌ | ❌ | MI = 0 |
| `rest_home`, `rest_away` | ❌ | ❌ | MI = 0 |
| `h2h_n` | ❌ | ❌ | MI = 0 |
| `home_promoted` | ❌ | ❌ | MI = 0 (estimator noise, see above) |
| **Total** | **4** | **11** | |

**Raw columns never used as features:** the current match's goals, shots, corners, fouls, cards and
half-time score (leakage), referee (not engineered), Bet365 odds (benchmark only).

**Features removed at the design stage** (before selection): `elo_home`, `elo_away`, `form5_home`,
`form5_away`. They were exact linear combinations together with the difference features (Section 4.7).

### 6.3 What the selection means

- **Naive Bayes is effectively an Elo model.** Of its 4 features, only `elo_diff` has real signal.
  The other three have MI ≤ 0.004. They were kept because they are *uncorrelated* with Elo, not because
  they are strong. The 0.50 threshold removes every other strength or form feature.
- **Logistic Regression gets Elo plus form, shots, venue and h2h.** These are correlated with Elo
  (0.5–0.85) but each adds a bit of *independent* information that LR can weigh correctly.

---

## 7. The models

### 7.1 The common protocol

Both models follow the same three steps:

1. **Select features** on train.
2. **Tune one hyperparameter** on validation, choosing the value with the lowest **validation log loss**.
   Log loss is chosen over accuracy because it rewards good probabilities, it is smooth, and accuracy
   barely changes between settings.
3. **Refit on train + validation** with the chosen setting, then save (`models/*.joblib`).

### 7.2 Baselines (the reference points)

**Always-Home (class-prior) baseline.** Every match gets the same probability vector, the class
frequencies from train + validation, roughly $(0.28, 0.25, 0.47)$ for (A, D, H). The argmax is always H.
Any useful model must beat this.

**Bookmaker (Bet365).** Decimal odds $o_k$ imply probabilities $1/o_k$. These add up to more than 1, and
the excess is the bookmaker's **margin** ("overround"). In the test seasons it averages about **4 %**:

$$
\sum_{k \in \{A,D,H\}} \frac{1}{o_k} \approx 1.04
$$

The margin is removed by normalising:

$$
\hat{p}_k = \frac{1/o_k}{\sum_j 1/o_j}
$$

Bookmakers have strong incentives and inside information (injuries, line-ups, team news, the betting
market itself). Their odds are widely regarded as the hardest benchmark in football prediction.

### 7.3 Model 1: Naive Bayes with Laplace smoothing

Code: `src/models/naive_bayes.py`, `src/models/discretizer.py`. Pipeline:
`QuantileBinner(n_bins) → CategoricalNB(alpha=1)`.

#### Bayes' theorem

For a class $c \in \{A, D, H\}$ and feature vector $\mathbf{x} = (x_1, \dots, x_p)$:

$$
P(c \mid \mathbf{x}) = \frac{P(c)\,P(\mathbf{x} \mid c)}{P(\mathbf{x})}
$$

#### The naive assumption

Estimating the joint $P(\mathbf{x} \mid c)$ is impossible with few data points. Naive Bayes assumes the
features are **conditionally independent given the class**:

$$
P(\mathbf{x} \mid c) = \prod_{i=1}^{p} P(x_i \mid c)
$$

so

$$
P(c \mid \mathbf{x}) = \frac{P(c)\prod_{i} P(x_i \mid c)}{\sum_{c'} P(c')\prod_{i} P(x_i \mid c')}
$$

In practice this is computed in log space (sums of logs) to avoid numerical underflow:

$$
\log P(c \mid \mathbf{x}) = \log P(c) + \sum_i \log P(x_i \mid c) - \log Z
$$

The prediction is $\hat{c} = \arg\max_c P(c \mid \mathbf{x})$.

#### Why the independence assumption makes feature selection necessary

Suppose the same feature were included $k$ times. The log-odds between two classes becomes

$$
\log\frac{P(H\mid\mathbf{x})}{P(A\mid\mathbf{x})} = \log\frac{P(H)}{P(A)} + k\cdot\log\frac{P(x\mid H)}{P(x\mid A)}
$$

The evidence gets multiplied by $k$, and the probabilities are pushed toward 0 and 1. Highly correlated
features (`elo_diff`, `ppg_diff`, `gd_pg_diff`, `venue_ppg_diff`) behave almost like copies, so feeding them
all to NB makes it **overconfident**. The ablation in Section 9 shows this directly.

#### Step A: discretisation (`QuantileBinner`)

`CategoricalNB` needs discrete values, so each continuous feature is cut into $n$ bins of roughly equal
population, using training-set quantiles:

$$
\text{edges}_j = \operatorname{unique}\Big\{ Q_j\!\left(\tfrac{1}{n}\right), Q_j\!\left(\tfrac{2}{n}\right), \dots, Q_j\!\left(\tfrac{n-1}{n}\right) \Big\}
$$

$$
\text{bin}(x) = b \quad\text{such that}\quad \text{edge}_{b-1} < x \le \text{edge}_b
$$

- Edges are learned in `fit()` (training data only) and reused for validation and test, so no leakage.
- Duplicate edges are removed. A 0/1 flag like `away_promoted` (86 % zeros) ends up with just 2 bins.
- Values outside the training range fall into the first or last bin.

Binning makes NB **non-parametric**: no Gaussian shape is assumed. The cost is that some information within
each bin is lost.

#### Step B: Laplace (add-one) smoothing

The maximum-likelihood estimate $P(x_i = b \mid c) = N_{c,b}/N_c$ is **zero** for any bin never seen with
class $c$ in training. One zero factor makes the whole product zero, whatever the other features say.
Laplace smoothing adds $\alpha$ pseudo-counts to every bin:

$$
P(x_i = b \mid c) = \frac{N_{c,i,b} + \alpha}{N_c + \alpha\, n_i}, \qquad \alpha = 1
$$

where $N_{c,i,b}$ is the number of training matches of class $c$ with feature $i$ in bin $b$, $N_c$ is the
number of class-$c$ matches, and $n_i$ is the number of bins for feature $i$.

- $\alpha = 1$ is "add-one" or Laplace smoothing. In Bayesian terms it is the posterior mean under a uniform
  Dirichlet prior.
- `min_categories = n_bins` guarantees every possible bin index has a smoothed probability, even if it
  never appears in training. (For a 2-bin flag this slightly enlarges the denominator. The effect is
  negligible with thousands of matches.)

The class prior is the empirical frequency: $P(c) = N_c / N$.

#### Hyperparameter tuning: number of bins

Validation log loss for the 4 selected features (`reports/naive_bayes_validation.csv`):

| bins | 3 | 4 | **5** | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| log loss | 0.9873 | 0.9811 | **0.9766** | 0.9784 | 0.9779 | 0.9785 |

This is a **bias–variance trade-off.** Too few bins are too coarse (bias). Too many bins have few matches
each, so their probabilities are noisy (variance). **5 bins** was chosen.

### 7.4 Model 2: Multinomial Logistic Regression

Code: `src/models/logistic_regression.py`. Pipeline:
`StandardScaler → LogisticRegression(C, L2 penalty, multinomial)`.

#### Standardisation

Each feature is rescaled with its training mean and standard deviation:

$$
z_j = \frac{x_j - \mu_j}{\sigma_j}
$$

This is needed so that (a) the L2 penalty treats all features equally (otherwise `elo_diff`, with
std ≈ 160, would be penalised far less per unit than `draw_rate_home`, with std ≈ 0.09), and (b)
coefficients are comparable: each one is the effect of a **1 standard deviation** change.

#### The model: softmax over linear scores

Each class $k$ has a weight vector $\mathbf{w}_k$ and bias $b_k$. The linear score (logit) is
$s_k = \mathbf{w}_k^\top \mathbf{z} + b_k$, and probabilities come from the **softmax**:

$$
P(y = k \mid \mathbf{z}) = \frac{e^{\,\mathbf{w}_k^\top\mathbf{z} + b_k}}{\sum_{j \in \{A,D,H\}} e^{\,\mathbf{w}_j^\top\mathbf{z} + b_j}}
$$

The log-odds between any two classes is **linear** in the features:

$$
\log\frac{P(H\mid\mathbf{z})}{P(A\mid\mathbf{z})} = (\mathbf{w}_H - \mathbf{w}_A)^\top\mathbf{z} + (b_H - b_A)
$$

With two classes this reduces to the familiar sigmoid $\sigma(s) = 1/(1+e^{-s})$.

#### Training: maximum likelihood with L2 regularisation

The parameters minimise the **cross-entropy (negative log-likelihood)** plus an L2 penalty. In
scikit-learn's form:

$$
\min_{W, \mathbf{b}} \; \frac{1}{2}\lVert W\rVert_2^2 \;-\; C\sum_{i=1}^{N} \log P(y_i \mid \mathbf{z}_i)
$$

- $C$ is the **inverse** regularisation strength. Small $C$ means a strong penalty and weights pulled
  toward 0 (simpler model, more bias). Large $C$ means close to plain maximum likelihood.
- The objective is **convex**, so there is a single global optimum. It is solved with L-BFGS, a
  quasi-Newton method.
- The gradient for class $k$ has a clean form:
  $\nabla_{\mathbf{w}_k} = \mathbf{w}_k - C\sum_i \big(\mathbb{1}[y_i = k] - P(k \mid \mathbf{z}_i)\big)\mathbf{z}_i$.
  Each match pushes the weights by "observed − predicted".
- **Why the coefficients across the three classes sum to zero:** softmax only depends on *differences*
  between class scores, so adding the same vector to every $\mathbf{w}_k$ changes nothing. The L2 penalty
  picks the smallest-norm solution among these, which is the one with $\sum_k \mathbf{w}_k = 0$. Check:
  for `elo_diff`, $-0.378 - 0.033 + 0.411 = 0$.

Unlike NB, LR is **discriminative**. It models $P(y \mid \mathbf{x})$ directly and fits all weights
jointly, so if two features overlap it splits the credit between them instead of counting it twice.
NB is **generative**: it models $P(\mathbf{x} \mid y)$ and $P(y)$ and applies Bayes' rule.

#### Hyperparameter tuning: C

`reports/logistic_regression_validation.csv`:

| C | 0.001 | 0.003 | **0.01** | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|---|
| log loss | 0.9746 | 0.9693 | **0.9677** | 0.9678 | 0.9683 | 0.9685 | 0.9686 | 0.9686 |

The curve is flat above C ≈ 0.01. The data has ~5,000 matches and few parameters (11 × 3 + 3), so
overfitting is mild either way. A fairly strong penalty helps slightly by shrinking the noisy weights of
weak, correlated features. **C = 0.01** was chosen.

#### Reading the coefficients

Standardised coefficients after refitting on train + validation
(`reports/logistic_regression_coefficients.csv`):

| Feature | A | D | H |
| --- | --- | --- | --- |
| `elo_diff` | −0.378 | −0.033 | **+0.411** |
| `sot5_diff` | −0.139 | +0.024 | +0.115 |
| `form5_diff` | +0.071 | +0.019 | −0.090 |
| `venue_ppg_diff` | −0.071 | +0.010 | +0.061 |
| `abs_elo_diff` | +0.033 | **−0.069** | +0.036 |
| `h2h_pts` | −0.048 | +0.052 | −0.005 |
| `gf5_diff` | +0.041 | −0.046 | +0.005 |
| `ga5_diff` | +0.024 | +0.012 | −0.035 |
| `away_promoted` | −0.019 | −0.019 | +0.038 |
| `matchweek` | +0.002 | −0.021 | +0.019 |
| `draw_rate_home` | −0.003 | −0.005 | +0.007 |

How to interpret:

- **`elo_diff` dominates.** One standard deviation more Elo advantage (≈ 160 points) raises the H-vs-A
  log-odds by $0.411 - (-0.378) = 0.789$, multiplying the odds of home win over away win by
  $e^{0.789} \approx 2.2$.
- **`sot5_diff` is second.** Creating more chances recently predicts winning, even after accounting for Elo.
- **`abs_elo_diff` has a negative draw weight.** The more mismatched the teams, the less likely a draw.
  This is exactly what the feature was added for.
- **`form5_diff` has the "wrong" sign** (positive for away). Coefficients are **partial effects**: the
  effect of form *holding Elo and shots fixed*. Elo already includes recent results (weighted by margin),
  so the extra part of form beyond Elo and shots is mostly luck that tends to even out (regression to the
  mean). Correlated predictors (r = 0.72 with Elo) can flip signs this way. The coefficient is small and
  should not be read causally.

---

## 8. Evaluation metrics

Both models output a probability vector $\mathbf{p}_i = (p_{iA}, p_{iD}, p_{iH})$ for each match $i$.
Let $\mathbf{o}_i$ be the one-hot true outcome and $N$ the number of matches. Code:
`src/models/common.py`.

| Metric | Formula | Range / reading |
| --- | --- | --- |
| **Accuracy** | $\frac{1}{N}\sum_i \mathbb{1}[\arg\max_k p_{ik} = y_i]$ | higher is better; ignores confidence |
| **Log loss** (cross-entropy) | $-\frac{1}{N}\sum_i \log p_{i,y_i}$ | lower is better; uniform guess = $\ln 3 = 1.099$; punishes confident mistakes very hard |
| **Brier score** | $\frac{1}{N}\sum_i \sum_k (p_{ik} - o_{ik})^2$ | lower is better; 0 to 2; uniform guess = 0.667 |
| **RPS** (ranked probability score) | $\frac{1}{N}\sum_i \frac{1}{K-1}\sum_{k=1}^{K-1}\Big(\sum_{j\le k} p_{ij} - \sum_{j\le k} o_{ij}\Big)^2$, $K = 3$ | lower is better; uses order A < D < H |
| **Macro-F1** | $\frac{1}{3}\sum_k F1_k$, $F1_k = \frac{2\,\text{Prec}_k\,\text{Rec}_k}{\text{Prec}_k + \text{Rec}_k}$ | higher is better; a class never predicted has F1 = 0 |
| **Draws predicted** | count of $\arg\max = D$ | diagnostic |

**Why RPS?** It compares *cumulative* distributions. If the true result is a home win, predicting "draw"
is closer than predicting "away win". RPS gives partial credit for that, while log loss and Brier treat the
three outcomes as unordered. RPS is the standard metric in football forecasting research (Constantinou &
Fenton, 2012).

**Why log loss is the tuning metric:** it is a **proper scoring rule**. It is minimised in expectation
only by reporting the true probabilities, so it rewards honest, well-calibrated forecasts. Brier and RPS
are proper too. Accuracy is not.

---

## 9. Results and why

### 9.1 The final table (test set: 1,140 matches, 2015-16 to 2017-18)

| Model | Accuracy ↑ | Log loss ↓ | Brier ↓ | RPS ↓ | Macro-F1 ↑ | Draws predicted |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform guess (1/3 each), reference | – | 1.0986 | 0.6667 | ≈ 0.235 | – | – |
| Always Home (class prior) | 0.4535 | 1.0669 | 0.6447 | 0.2275 | 0.2080 | 0 |
| **Naive Bayes** (4 features, 5 bins, α = 1) | 0.5298 | 0.9797 | 0.5834 | 0.1983 | 0.3899 | 0 |
| **Logistic Regression** (11 features, C = 0.01) | **0.5412** | **0.9699** | **0.5756** | **0.1946** | 0.3964 | 0 |
| Bookmaker (Bet365, margin removed) | 0.5412 | 0.9624 | 0.5709 | 0.1923 | 0.4014 | 0 |
| *Ablation:* NB with all 19 features | 0.5088 | 1.5502 | 0.7177 | 0.2534 | 0.4630 | 191 |

(The uniform row is computed analytically from the test class balance and is not produced by the code.)

Confusion matrices (rows = true, columns = predicted):

| | **NB** A | D | H | **LR** A | D | H | **Bet365** A | D | H |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| true A (333) | 178 | 0 | 155 | 167 | 0 | 166 | 188 | 0 | 145 |
| true D (290) | 95 | 0 | 195 | 71 | 0 | 219 | 84 | 0 | 206 |
| true H (517) | 91 | 0 | 426 | 67 | 0 | 450 | 88 | 0 | 429 |

### 9.2 Why both models beat the baseline

The baseline's log loss (1.067) equals the **entropy of the outcome distribution**: it knows how often each
result happens but nothing about the teams. Both models reduce log loss by ~0.09 nats. That is about the
size of the mutual information of `elo_diff` (0.086), so **most of the gain comes from knowing relative
team strength.** Accuracy rises from 45 % to 53–54 %: the models now correctly pick away wins when the
away team is clearly stronger (167–178 correct away predictions, against 0 for the baseline).

### 9.3 Why Logistic Regression beats Naive Bayes

The margin is small but consistent across all four probability metrics. Reasons:

1. **More information.** NB's strict decorrelation leaves it essentially Elo-only. LR also uses form,
   shots on target, venue PPG and head-to-head. These correlate with Elo but carry some independent
   signal, and LR can use them because it estimates weights jointly.
2. **No discretisation loss.** NB sees `elo_diff` in only 5 bins: an Elo gap of +60 and one of +200 may
   land in the same bin and get identical treatment. LR uses the exact value through a smooth
   logistic curve, which suits the Elo relationship (Elo is itself defined by a logistic curve).
3. **Discriminative vs generative.** With enough data, a discriminative model that directly optimises
   $P(y\mid\mathbf{x})$ usually reaches lower error than a generative one whose modelling assumptions
   (independence) are only approximately true. (Ng & Jordan, 2001, showed NB can win with *little* data
   but LR tends to win as data grows. Here we have ~5,700 training matches.)

### 9.4 Why we match the bookmaker on accuracy but not quite on log loss

- **Accuracy is identical (617/1140 correct for both).** Both essentially pick the stronger team, and on
  most matches they agree.
- **Log loss gap: 0.9699 vs 0.9624 (≈ 0.0075).** The bookmaker's *probabilities* are slightly sharper and
  better calibrated. It knows things our features can't capture: injuries, suspensions, line-ups, managerial
  changes, transfers, motivation (e.g. a team with nothing to play for), and the information carried by
  the betting market itself.
- **For context**, a model built only from historical results that gets this close to a major bookmaker is
  a good result. In the forecasting literature, beating bookmaker odds with public historical data is
  rare. Elo-style ratings are known to get close but usually fall slightly short (e.g. Hvattum & Arntzen,
  2010).

### 9.5 Why no model ever predicts a draw

This is the most important thing to understand about football prediction. For "Draw" to be the argmax we
need $p_D > p_H$ and $p_D > p_A$. Since $p_H + p_A = 1 - p_D$, the larger of the two is at least
$(1 - p_D)/2$, so

$$
p_D > \frac{1 - p_D}{2} \;\Longrightarrow\; p_D > \frac{1}{3}
$$

**A draw can only be predicted if its probability exceeds 33 %.** In real football, draw probability
stays within a narrow band (roughly 20–30 %) and is highest when teams are even. But exactly then, home
advantage gives H a slightly higher probability still. Even the bookmaker never makes "draw" its top pick
in these 1,140 matches.

Consequences:

- Draws (25 % of matches) are always "wrong" under argmax, which caps accuracy at around 75 % even for a
  perfect win/loss predictor. Realistically it keeps accuracy in the 50s.
- Most true draws are classified as **home wins** (195–219), because that is the default lean.
- Macro-F1 suffers because the draw class has F1 = 0.
- That's why the project reports **probability metrics**. The models do assign sensible draw probabilities
  (≈ 25 %, higher for even games). They just never exceed 1/3.

### 9.6 Why the "all features" Naive Bayes is so much worse (ablation)

Feeding NB all 19 features:

- **Log loss jumps from 0.98 to 1.55**, worse than the baseline (1.07) and even worse than uniform
  guessing (1.10).
- Reason: `elo_diff`, `ppg_diff`, `gd_pg_diff`, `venue_ppg_diff` and `form5_diff` (|r| 0.7–0.96) are
  nearly copies of one another. NB multiplies their likelihoods as if they were independent evidence, so
  the strength signal is counted about 5 times (Section 7.3). Probabilities become extreme (e.g. 0.95),
  and when the confident prediction is wrong, $-\log(0.05) \approx 3$ is added to the loss.
- **Macro-F1 is higher (0.46)** only because extreme probabilities sometimes push draws past 1/3 by
  accident (191 draw predictions, 60 correct), so draw F1 is no longer 0. Accuracy still falls (50.9 %).
  This shows why **macro-F1 alone can be misleading**: a metric based on hard labels rewarded a model whose
  probabilities are badly wrong.

The validation set showed the same pattern for every bin count (log loss 1.51–1.63), so this is a
reliable effect, not chance.

### 9.7 Is the model overfitting?

| Model | Validation log loss | Test log loss |
| --- | --- | --- |
| Naive Bayes | 0.9766 | 0.9797 |
| Logistic Regression | 0.9677 | 0.9699 |

Test is only slightly worse than validation (≈ 0.003), and the ranking of models is the same. There is
**no sign of overfitting** to the validation set, and the chronological protocol held up.

### 9.8 Are the differences statistically meaningful?

With $N = 1140$ and accuracy ≈ 0.54, the standard error of accuracy is

$$
SE = \sqrt{\frac{p(1-p)}{N}} = \sqrt{\frac{0.54 \times 0.46}{1140}} \approx 0.015
$$

so a 95 % interval is about ±3 percentage points. Therefore:

- **Models vs baseline (+8 to 9 pp):** clearly significant.
- **NB vs LR vs bookmaker (≤ 1.1 pp):** **not** distinguishable by accuracy alone. The log-loss differences
  are consistent across metrics, which is more convincing, but a formal claim would need a paired test on
  per-match losses (e.g. a paired t-test or bootstrap).

Say this carefully in a report: "LR performs on par with the bookmaker in accuracy and slightly below it in
probability quality."

### 9.9 Comparison with published benchmarks

General points from the football prediction literature, to give your numbers context. **Check the exact
figures in the original papers before citing them.**

- **Three-way accuracy** of match-outcome models (EPL and other top leagues) is typically reported in the
  **~50–55 %** range. Bookmakers sit near the top of that range. Our 53–54 % is in line.
- **RPS**: in the 2017 *Soccer Prediction Challenge* (Dubitzky et al., *Machine Learning*, 2019), the best
  submissions across many leagues had RPS around **0.20**. Our 0.195 on the EPL is comparable. (The EPL is
  somewhat more predictable than average because of its large strength gaps, so the comparison is
  approximate.)
- **Elo for football**: Hvattum & Arntzen (2010) found Elo ratings to be a strong, simple predictor that
  comes close to, but does not beat, bookmaker odds. Our results show the same.
- **Draws**: almost all studies report that draws are rarely or never the predicted class. This is a known
  property of the problem, not a bug in this project.

### 9.10 Summary of "why"

| Observation | Explanation |
| --- | --- |
| Models ≫ baseline | Relative team strength (mainly Elo) carries real signal |
| LR > NB (slightly) | LR uses more features without double-counting, and keeps continuous values |
| LR ≈ bookmaker | Public historical data captures most, but not all, of what the market knows |
| No draws predicted | Draw probability rarely exceeds 1/3, the threshold needed to be the argmax |
| NB-all-features fails | Correlated features violate NB's independence assumption, causing overconfidence |
| Accuracy ceiling ~55 % | Football is low-scoring and noisy. The best feature explains only ~8 % of outcome entropy |

---

## 10. Quick revision: likely questions

**Q: Why not a random train/test split?**
Matches are time-ordered. A random split would train on future seasons and test on past ones. Team
strengths drift, so this inflates scores and doesn't reflect real use.

**Q: Why are the current match's shots not features?**
They are only known after the match, so using them is leakage. Shots from *previous* matches are used
(`sot5_diff`).

**Q: How do you know there is no leakage?**
"Record, then update" in one chronological pass, plus `check_leakage.py`: rewriting a match's result
leaves all earlier and same-match feature rows unchanged.

**Q: Why discretise for Naive Bayes?**
To use a categorical NB with Laplace smoothing, which is defined for counts, and to avoid assuming a
Gaussian shape.

**Q: What does Laplace smoothing fix?**
Zero probabilities for bin/class combinations not seen in training, which would otherwise wipe out the
whole product.

**Q: Why different correlation thresholds for the two models?**
NB assumes independence, so correlated features double-count. LR fits weights jointly and only needs
near-duplicates removed.

**Q: Why keep `abs_elo_diff` for LR but not NB?**
LR is linear and can't compute |x| itself. NB learns a full per-class distribution of `elo_diff`, which
already shows that draws cluster near 0.

**Q: Why tune on log loss, not accuracy?**
Log loss is a proper scoring rule that rewards good probabilities. Accuracy is coarse and ignores
confidence.

**Q: Why does `form5_diff` have a negative coefficient for home wins?**
It is a partial effect given Elo and shots (correlated predictors). It should not be read causally.

**Q: Why does the always-home baseline have log loss ≈ 1.067?**
It predicts the class frequencies, so its log loss equals the entropy of the outcome distribution.

**Q: What would you try next?**
A decision rule or cost-sensitive threshold for draws; probability calibration (Platt or isotonic);
ordinal logistic regression (exploits A < D < H); gradient boosting; expanding-window time-series
cross-validation; player-level or market-value features; using 2020-22 as a second test set.

---

### References (for further reading)

- Constantinou, A. C., & Fenton, N. E. (2012). Solving the problem of inadequate scoring rules for
  assessing probabilistic football forecast models. *Journal of Quantitative Analysis in Sports*.
- Hvattum, L. M., & Arntzen, H. (2010). Using ELO ratings for match result prediction in association
  football. *International Journal of Forecasting*.
- Dubitzky, W., Lopes, P., Davis, J., & Berrar, D. (2019). The Open International Soccer Database for
  machine learning. *Machine Learning*. (Soccer Prediction Challenge 2017.)
- Ng, A. Y., & Jordan, M. I. (2001). On discriminative vs. generative classifiers: A comparison of
  logistic regression and naive Bayes. *NeurIPS*.
- Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information (mRMR). *IEEE
  TPAMI*.
