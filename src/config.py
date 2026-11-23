"""Paths, seasons, team names and model constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"

DATASET_PATH = PROCESSED_DIR / "matches.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.csv"

SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]

# Cups are only used to compute rest days
CUP_COMPETITIONS = ["ucl", "uel", "uecl", "coppa_italia"]

# Every source is mapped to the football-data team names
TEAM_ALIASES = {
    "AC Milan": "Milan",
    "AC Monza": "Monza",
    "AC Pisa": "Pisa",
    "ACF Fiorentina": "Fiorentina",
    "AS Roma": "Roma",
    "Atalanta BC": "Atalanta",
    "Benevento Calcio": "Benevento",
    "Bologna FC": "Bologna",
    "Cagliari Calcio": "Cagliari",
    "Como 1907": "Como",
    "Empoli FC": "Empoli",
    "FC Crotone": "Crotone",
    "FC Internazionale": "Inter",
    "Frosinone Calcio": "Frosinone",
    "Genoa CFC": "Genoa",
    "Hellas Verona": "Verona",
    "Inter Milan": "Inter",
    "Internazionale": "Inter",
    "Juventus FC": "Juventus",
    "Lazio Roma": "Lazio",
    "Parma Calcio 1913": "Parma",
    "Pisa SC": "Pisa",
    "SS Lazio": "Lazio",
    "SSC Napoli": "Napoli",
    "Sassuolo Calcio": "Sassuolo",
    "Spezia Calcio": "Spezia",
    "Torino FC": "Torino",
    "UC Sampdoria": "Sampdoria",
    "US Cremonese": "Cremonese",
    "US Lecce": "Lecce",
    "US Salernitana 1919": "Salernitana",
    "Udinese Calcio": "Udinese",
    "Venezia FC": "Venezia",
}

# Over/under 2.5 odds: market average first, then fallbacks
OVER_ODDS_COLS = ["Avg>2.5", "Max>2.5", "P>2.5", "B365>2.5"]
UNDER_ODDS_COLS = ["Avg<2.5", "Max<2.5", "P<2.5", "B365<2.5"]

# Elo: every team starts each season at 1500
ELO_START = 1500.0
ELO_K = 32
ELO_HOME_ADVANTAGE = 45

# Chosen with src/feature_analysis.py: only "total goals" features that are
# stable across seasons, no duplicates (see README)
FEATURES = [
    "combined_xg_last5",
    "combined_goals_last5",
    "shots_on_target_total_last5",
    "xg_conceded_total_last5",
    "shots_on_target_conceded_total_last5",
    "home_over25_rate_last10",
    "away_over25_rate_last10",
    "home_goals_conceded_last5",
]

# Original v6 feature set, kept for comparison
FEATURES_V6 = [
    "combined_xg_last5",
    "shots_on_target_total_last5",
    "xg_conceded_total_last5",
    "shots_on_target_conceded_total_last5",
    "defensive_efficiency",
    "offensive_efficiency",
    "home_over25_rate_last10",
    "away_over25_rate_last10",
    "home_xg_last5",
    "away_xg_last5",
    "home_goals_conceded_last5",
    "away_goals_conceded_last5",
    "points_diff",
    "home_to_relegation",
    "elo_diff_pre",
    "home_attack_vs_away_defense",
    "away_attack_vs_home_defense",
]
TARGET = "over25"

# Final model, chosen with src/tuning.py on 2021-22 ... 2024-25
MODEL_PARAMS = dict(
    n_estimators=50, max_depth=1, learning_rate=0.05, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, random_state=42,
    eval_metric="logloss",
)
DECAY_LAMBDA = 0.002   # weight of a match = exp(-DECAY_LAMBDA * days ago)
PRIOR_STRENGTH = 80    # how many matches the historical over-rate is worth

# Original v6 model, kept only to reproduce the analysis in feature_analysis.py / tuning.py
TOPUP_WINDOW = 250
DECAY_LAMBDA_GLOBAL = 0.008
DECAY_LAMBDA_TOPUP = 0.008
DECAY_LAMBDA_TEAM = 0.003
TOPUP_BOOST = 2.0
TEAM_WEIGHT_BOOST = 3.0

BASE_PARAMS = dict(
    n_estimators=500, max_depth=3, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, gamma=1.0,
    reg_lambda=2.0, min_child_weight=5, random_state=42,
    eval_metric="logloss",
)
TOPUP_PARAMS = dict(
    n_estimators=150, max_depth=3, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, gamma=1.0,
    reg_lambda=2.0, min_child_weight=3, random_state=42,
    eval_metric="logloss",
)

# Betting
BET_MARGIN = 0.04        # bet only if my probability beats 1/odd by 4 points (chosen on 2023-24 + 2024-25)
KELLY_FRAC_OVER = 0.15   # fraction of the full Kelly stake
KELLY_FRAC_UNDER = 0.10
MAX_BET_FRAC = 0.05      # never more than 5% of the bankroll on one bet
DRAWDOWN_THRESHOLD = 0.15
