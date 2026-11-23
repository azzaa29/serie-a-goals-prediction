"""Hyperparameter tuning on the validation seasons (2025-26 is never used).

    python src/tuning.py

Step 1: grid search of the base XGBoost model.
        Train on all seasons before S, log-loss on S, for S = 2021-22 ... 2024-25.
Step 2: with the best base model, test the v6 extras (per-match top-up and the
        season over-rate correction) from 10 February, where the bets are placed.
Step 3: instead of the top-up, retrain the model before every match day with the
        matches of the current season weighted more (x1, x2, x3, x5).

    python src/tuning.py --season-weight   # run only step 3
"""

import sys
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from config import (
    DATASET_PATH, DECAY_LAMBDA_TEAM, DECAY_LAMBDA_TOPUP, FEATURES, FEATURES_V6, TARGET,
    TEAM_WEIGHT_BOOST, TOPUP_BOOST, TOPUP_WINDOW,
)
from model import bayesian_season_rate, label_shift_correction, time_weights


def topup_weights(topup, home, away):
    """v6 top-up weights: recent matches, and above all those of the two teams, count more."""
    w = time_weights(topup["date"], DECAY_LAMBDA_TOPUP)
    same_teams = (topup["home"].isin([home, away]) | topup["away"].isin([home, away])).to_numpy()
    w_team = time_weights(topup["date"], DECAY_LAMBDA_TEAM)
    w[same_teams] = w_team[same_teams] * TEAM_WEIGHT_BOOST
    return w * TOPUP_BOOST

SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
VALIDATION_SEASONS = SEASONS[1:]
START_DAY = "02-10"

GRID = {
    "n_estimators": [50, 100, 200, 400],
    "max_depth": [1, 2, 3],
    "learning_rate": [0.01, 0.02, 0.05],
    "min_child_weight": [5, 20, 50],
    "decay": [0.0, 0.002, 0.008],
}
# Result of step 1 with the new features
BEST_BASE = {"n_estimators": 50, "max_depth": 1, "learning_rate": 0.05, "min_child_weight": 20, "decay": 0.002}
SEASON_BOOSTS = [1, 2, 3, 5]

FIXED = dict(subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, random_state=42, eval_metric="logloss")


def make_model(p):
    return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                         learning_rate=p["learning_rate"], min_child_weight=p["min_child_weight"], **FIXED)


def fit(model, train, features, decay):
    weights = time_weights(train["date"], decay) if decay > 0 else None
    model.fit(train[features], train[TARGET], sample_weight=weights)
    return model


def constant_logloss(df, seasons, window=False):
    scores = []
    for season in seasons:
        test = df[df["season"] == season]
        if window:
            test = after_start_day(test)
        rate = df[df["season"] < season][TARGET].mean()
        scores.append(log_loss(test[TARGET], np.full(len(test), rate)))
    return np.mean(scores)


def after_start_day(df):
    start = pd.to_datetime(df["season"].str[5:] + "-" + START_DAY)
    return df[df["date"] >= start]


def step1_grid(df, features):
    rows = []
    for values in product(*GRID.values()):
        p = dict(zip(GRID, values))
        scores = []
        for season in VALIDATION_SEASONS:
            model = fit(make_model(p), df[df["season"] < season], features, p["decay"])
            test = df[df["season"] == season]
            scores.append(log_loss(test[TARGET], model.predict_proba(test[features])[:, 1]))
        rows.append({**p, "logloss": np.mean(scores), "worst_season": max(scores)})
    return pd.DataFrame(rows).sort_values("logloss")


def step2_extras(df, features, p, topup_trees, use_label_shift):
    """Walk-forward from 10 February, optionally with top-up and label shift."""
    scores = []
    for season in VALIDATION_SEASONS:
        base = fit(make_model(p), df[df["season"] < season], features, p["decay"])
        pi_train = df[df["season"] < season][TARGET].mean()
        test = after_start_day(df[df["season"] == season])
        probs = []
        for date, day in test.groupby("date"):
            history = df[df["date"] < date]
            played = history[history["season"] == season]
            pi_new = bayesian_season_rate(played[TARGET].sum(), len(played), pi_train)
            topup = history.tail(TOPUP_WINDOW)
            for _, match in day.iterrows():
                x = match[features].to_frame().T.astype(float)
                if topup_trees:
                    m = make_model({**p, "n_estimators": topup_trees})
                    m.fit(topup[features], topup[TARGET],
                          sample_weight=topup_weights(topup, match["home"], match["away"]),
                          xgb_model=base.get_booster())
                    prob = float(m.predict_proba(x)[0, 1])
                else:
                    prob = float(base.predict_proba(x)[0, 1])
                if use_label_shift:
                    prob = label_shift_correction(prob, pi_train, pi_new)
                probs.append(prob)
        scores.append(log_loss(test[TARGET], probs))
    return np.mean(scores), np.round(scores, 3)


def step3_season_weight(df, features, p, boost, use_label_shift):
    """Retrain before every match day; current-season matches get `boost` times more weight."""
    scores = []
    for season in VALIDATION_SEASONS:
        pi_train = df[df["season"] < season][TARGET].mean()
        test = after_start_day(df[df["season"] == season])
        probs = []
        for date, day in test.groupby("date"):
            history = df[df["date"] < date]
            weights = time_weights(history["date"], p["decay"])
            weights = weights * np.where(history["season"] == season, boost, 1.0)
            model = make_model(p)
            model.fit(history[features], history[TARGET], sample_weight=weights)
            day_probs = model.predict_proba(day[features])[:, 1]

            played = history[history["season"] == season]
            pi_new = bayesian_season_rate(played[TARGET].sum(), len(played), pi_train)
            if use_label_shift:
                day_probs = [label_shift_correction(pr, pi_train, pi_new) for pr in day_probs]
            probs.extend(day_probs)
        scores.append(log_loss(test[TARGET], probs))
    return np.mean(scores), np.round(scores, 3)


def run_step3(df):
    print(f"\nSTEP 3 - season weight, from 10 February "
          f"(constant over-rate: {constant_logloss(df, VALIDATION_SEASONS, True):.4f})")
    for boost, shift in product(SEASON_BOOSTS, [False, True]):
        mean, per_season = step3_season_weight(df, FEATURES, BEST_BASE, boost, shift)
        print(f"  current season x{boost:<3} label shift={str(shift):<6} log-loss={mean:.4f}  {per_season}")


if __name__ == "__main__":
    data = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    data = data[data["season"].isin(SEASONS)]

    if "--season-weight" in sys.argv:
        run_step3(data.dropna(subset=FEATURES).sort_values("date").reset_index(drop=True))
        sys.exit()

    for name, features in [("v6 features (17)", FEATURES_V6), ("new features (8)", FEATURES)]:
        df = data.dropna(subset=features).sort_values("date").reset_index(drop=True)
        print(f"\nSTEP 1 - {name}  (constant over-rate: {constant_logloss(df, VALIDATION_SEASONS):.4f})")
        results = step1_grid(df, features)
        print(results.head(10).round(4).to_string(index=False))

    best = results.iloc[0][list(GRID)].to_dict()
    best = {k: (float(v) if k in ("learning_rate", "decay") else int(v)) for k, v in best.items()}
    print(f"\nBest base model: {best}")

    print(f"\nSTEP 2 - from 10 February (constant over-rate: {constant_logloss(df, VALIDATION_SEASONS, True):.4f})")
    for topup_trees, shift in product([0, 25, 75, 150], [False, True]):
        mean, per_season = step2_extras(df, FEATURES, best, topup_trees, shift)
        print(f"  top-up trees={topup_trees:<4} label shift={str(shift):<6} log-loss={mean:.4f}  {per_season}")
