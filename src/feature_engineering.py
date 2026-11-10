"""Leak-free pre-match feature engineering.

Golden rule: the features of a match are computed from matches played BEFORE it.
Implementation: matches are processed in date order; for each match we first record
the features from the current team states, and only then update the states with the
match result.

Feature groups
--------------
strength    Elo ratings
season      season-to-date points / goal difference (blended with last season)
form        last-5-match results, goals and shots on target
draw        draw tendency and low-scoring tendency (helps with the hard "Draw" class)
h2h         head-to-head history
context     rest days, matchweek, promoted-team flags
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from config import CLASS_LABELS, LABEL_TO_INT

# ---- hyper-parameters of the feature definitions ----------------------------------------
K_ELO = 20.0
ELO_HOME_ADVANTAGE = 60.0
ELO_SEASON_REGRESSION = 0.20      # pull ratings 20% toward 1500 at the start of each season
NEW_TEAM_ELO = 1450.0
PRIOR_STRENGTH = 5                # pseudo-matches of last season's PPG blended into this season's
PROMOTED_PPG = 1.0                # assumed PPG for a team with no recent top-flight history
SHORT_WINDOW = 5
LONG_WINDOW = 10
H2H_WINDOW = 5
REST_CAP_DAYS = 14
DEFAULT_GOALS_FOR = 1.35          # fallbacks when a team has no history yet
DEFAULT_SHOTS_ON_TARGET = 4.5
DEFAULT_DRAW_RATE = 0.26
DEFAULT_TOTAL_GOALS = 2.7

KEY_COLUMNS = ["Season", "Date", "HomeTeam", "AwayTeam"]
TARGET_COLUMNS = ["FTR", "y"]

# name -> (group, description).  Order here = column order in the output file.
FEATURE_INFO = {
    # strength
    "elo_home":        ("strength", "Elo rating of the home team before the match"),
    "elo_away":        ("strength", "Elo rating of the away team before the match"),
    "elo_diff":        ("strength", "elo_home - elo_away"),
    "abs_elo_diff":    ("strength", "|elo_diff|: how evenly matched the teams are (closer = more draws)"),
    # season-to-date
    "ppg_diff":        ("season", "Season points-per-game difference (home - away), blended with last season"),
    "gd_pg_diff":      ("season", "Season goal-difference-per-game difference (home - away)"),
    "venue_ppg_diff":  ("season", "Home team's PPG at home minus away team's PPG away"),
    # recent form
    "form5_home":      ("form", "Points won by the home team in its last 5 matches (0-15)"),
    "form5_away":      ("form", "Points won by the away team in its last 5 matches (0-15)"),
    "form5_diff":      ("form", "form5_home - form5_away"),
    "gf5_diff":        ("form", "Avg goals scored last 5 (home team) minus (away team)"),
    "ga5_diff":        ("form", "Avg goals conceded last 5 (home team) minus (away team)"),
    "sot5_diff":       ("form", "Avg shots on target last 5 (home team) minus (away team)"),
    # draw tendency
    "draw_rate_home":  ("draw", "Share of the home team's last 10 matches that were draws (smoothed)"),
    "draw_rate_away":  ("draw", "Share of the away team's last 10 matches that were draws (smoothed)"),
    "total_goals10":   ("draw", "Avg total goals per match in both teams' last 10 matches"),
    # head to head
    "h2h_pts":         ("h2h", "Home team's avg points per game in last 5 meetings (neutral 1.0 if none)"),
    "h2h_n":           ("h2h", "Number of previous meetings used (0-5)"),
    # context
    "rest_home":       ("context", "Days since the home team's last match (capped at 14)"),
    "rest_away":       ("context", "Days since the away team's last match (capped at 14)"),
    "matchweek":       ("context", "Matches already played by the home team this season + 1"),
    "home_promoted":   ("context", "1 if the home team was not in the league last season"),
    "away_promoted":   ("context", "1 if the away team was not in the league last season"),
}
FEATURE_COLUMNS = list(FEATURE_INFO)


def match_points(goals_for, goals_against):
    return 3 if goals_for > goals_against else (1 if goals_for == goals_against else 0)


class TeamState:
    """Everything we know about one team at a point in time."""

    def __init__(self, elo):
        self.elo = elo
        self.recent = deque(maxlen=LONG_WINDOW)   # dicts: pts, gf, ga, sot
        self.last_date = None
        self.prev_ppg = PROMOTED_PPG
        self.start_new_season()

    def start_new_season(self):
        self.games = self.points = self.goal_diff = 0
        self.home_games = self.home_points = 0
        self.away_games = self.away_points = 0

    # ---- season-to-date (blended with last season's PPG) ----------------------------
    @property
    def ppg(self):
        return (self.points + PRIOR_STRENGTH * self.prev_ppg) / (self.games + PRIOR_STRENGTH)

    @property
    def home_ppg(self):
        return (self.home_points + PRIOR_STRENGTH * self.ppg) / (self.home_games + PRIOR_STRENGTH)

    @property
    def away_ppg(self):
        return (self.away_points + PRIOR_STRENGTH * self.ppg) / (self.away_games + PRIOR_STRENGTH)

    @property
    def gd_per_game(self):
        return self.goal_diff / (self.games + PRIOR_STRENGTH)

    # ---- recent form ----------------------------------------------------------------
    def last(self, n):
        return list(self.recent)[-n:]

    def form_points(self, n=SHORT_WINDOW):
        window = self.last(n)
        return sum(r["pts"] for r in window) + (n - len(window)) * self.prev_ppg

    def mean_recent(self, key, n, default):
        values = [r[key] for r in self.last(n) if not np.isnan(r[key])]
        return float(np.mean(values)) if values else default

    def draw_rate(self, n=LONG_WINDOW):
        window = self.last(n)
        draws = sum(r["pts"] == 1 for r in window)
        return (draws + DEFAULT_DRAW_RATE * PRIOR_STRENGTH) / (len(window) + PRIOR_STRENGTH)

    def total_goals(self, n=LONG_WINDOW):
        window = self.last(n)
        total = sum(r["gf"] + r["ga"] for r in window)
        return (total + DEFAULT_TOTAL_GOALS * 3) / (len(window) + 3)

    def rest_days(self, date):
        if self.last_date is None:
            return REST_CAP_DAYS
        return min((date - self.last_date).days, REST_CAP_DAYS)

    def record_match(self, date, goals_for, goals_against, shots_on_target, at_home):
        pts = match_points(goals_for, goals_against)
        self.games += 1
        self.points += pts
        self.goal_diff += goals_for - goals_against
        if at_home:
            self.home_games += 1
            self.home_points += pts
        else:
            self.away_games += 1
            self.away_points += pts
        self.recent.append(dict(pts=pts, gf=goals_for, ga=goals_against, sot=shots_on_target))
        self.last_date = date


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """matches: cleaned match table (Season, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HST, AST ...)
    containing only complete, consecutive seasons. Returns one feature row per match."""
    matches = matches.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    teams = {}
    h2h = defaultdict(list)                         # frozenset({a, b}) -> [(home_team, home_pts), ...]
    rows = []
    previous_season, previous_teams = None, set()

    for season, season_df in matches.groupby("Season", sort=True):
        season_teams = set(season_df.HomeTeam) | set(season_df.AwayTeam)

        # ---- season roll-over ---------------------------------------------------------
        if previous_season is not None:
            for t in previous_teams:
                teams[t].prev_ppg = teams[t].points / max(teams[t].games, 1)
        promoted = (season_teams - previous_teams) if previous_season is not None else set()
        for t in season_teams:
            if t not in teams:
                teams[t] = TeamState(NEW_TEAM_ELO)
            else:
                teams[t].elo = 1500 + (1 - ELO_SEASON_REGRESSION) * (teams[t].elo - 1500)
            if t in promoted:
                teams[t].prev_ppg = PROMOTED_PPG
                teams[t].recent.clear()
                teams[t].last_date = None
            teams[t].start_new_season()

        for m in season_df.itertuples(index=False):
            home, away = teams[m.HomeTeam], teams[m.AwayTeam]
            pair = frozenset((m.HomeTeam, m.AwayTeam))
            meetings = h2h[pair][-H2H_WINDOW:]
            meeting_pts = [p if ht == m.HomeTeam else {3: 0, 1: 1, 0: 3}[p] for ht, p in meetings]

            # ============ 1. RECORD FEATURES (state before this match) ============
            rows.append(dict(
                Season=season, Date=m.Date, HomeTeam=m.HomeTeam, AwayTeam=m.AwayTeam,
                FTR=m.FTR, y=LABEL_TO_INT[m.FTR],
                elo_home=home.elo, elo_away=away.elo,
                elo_diff=home.elo - away.elo, abs_elo_diff=abs(home.elo - away.elo),
                ppg_diff=home.ppg - away.ppg,
                gd_pg_diff=home.gd_per_game - away.gd_per_game,
                venue_ppg_diff=home.home_ppg - away.away_ppg,
                form5_home=home.form_points(), form5_away=away.form_points(),
                form5_diff=home.form_points() - away.form_points(),
                gf5_diff=home.mean_recent("gf", SHORT_WINDOW, DEFAULT_GOALS_FOR)
                         - away.mean_recent("gf", SHORT_WINDOW, DEFAULT_GOALS_FOR),
                ga5_diff=home.mean_recent("ga", SHORT_WINDOW, DEFAULT_GOALS_FOR)
                         - away.mean_recent("ga", SHORT_WINDOW, DEFAULT_GOALS_FOR),
                sot5_diff=home.mean_recent("sot", SHORT_WINDOW, DEFAULT_SHOTS_ON_TARGET)
                          - away.mean_recent("sot", SHORT_WINDOW, DEFAULT_SHOTS_ON_TARGET),
                draw_rate_home=home.draw_rate(), draw_rate_away=away.draw_rate(),
                total_goals10=(home.total_goals() + away.total_goals()) / 2,
                h2h_pts=float(np.mean(meeting_pts)) if meeting_pts else 1.0,
                h2h_n=len(meeting_pts),
                rest_home=home.rest_days(m.Date), rest_away=away.rest_days(m.Date),
                matchweek=home.games + 1,
                home_promoted=int(m.HomeTeam in promoted), away_promoted=int(m.AwayTeam in promoted),
            ))

            # ============ 2. ONLY NOW UPDATE STATE WITH THE RESULT ============
            home_pts = match_points(m.FTHG, m.FTAG)
            expected_home = 1 / (1 + 10 ** (-(home.elo + ELO_HOME_ADVANTAGE - away.elo) / 400))
            actual_home = {3: 1.0, 1: 0.5, 0: 0.0}[home_pts]
            margin_multiplier = 1 + np.log1p(abs(m.FTHG - m.FTAG))
            delta = K_ELO * margin_multiplier * (actual_home - expected_home)
            home.elo += delta
            away.elo -= delta
            home.record_match(m.Date, m.FTHG, m.FTAG, m.HST, at_home=True)
            away.record_match(m.Date, m.FTAG, m.FTHG, m.AST, at_home=False)
            h2h[pair].append((m.HomeTeam, home_pts))

        previous_season, previous_teams = season, season_teams

    features = pd.DataFrame(rows)
    return features[KEY_COLUMNS + TARGET_COLUMNS + FEATURE_COLUMNS]


def feature_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [(name, group, desc) for name, (group, desc) in FEATURE_INFO.items()],
        columns=["feature", "group", "description"],
    )
