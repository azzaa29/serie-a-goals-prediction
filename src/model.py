"""Final model: a small XGBoost trained before the season, whose probabilities
are then moved towards the current season's over 2.5 rate.

For every match of the test season the model only sees matches played on
earlier days.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import DECAY_LAMBDA, FEATURES, MODEL_PARAMS, PRIOR_STRENGTH, TARGET


def time_weights(dates, lam):
    days_ago = (dates.max() - dates).dt.days.to_numpy()
    return np.exp(-lam * days_ago)


def bayesian_season_rate(n_over, n_total, prior_rate):
    """Over 2.5 rate of the current season, starting from the historical rate."""
    alpha = prior_rate * PRIOR_STRENGTH + n_over
    beta = (1 - prior_rate) * PRIOR_STRENGTH + (n_total - n_over)
    return alpha / (alpha + beta)


def label_shift_correction(prob, pi_train, pi_new):
    """Move a probability from the training over-rate to the current season's one."""
    if abs(pi_new - pi_train) < 0.01:
        return prob
    r = (pi_new / (1 - pi_new)) / (pi_train / (1 - pi_train))
    return float(np.clip(prob * r / (prob * r + (1 - prob)), 1e-6, 1 - 1e-6))


def bookmaker_probability(odd_over, odd_under):
    # remove the bookmaker margin
    return (1 / odd_over) / (1 / odd_over + 1 / odd_under)


def predict_season(df, season):
    model_df = df.dropna(subset=FEATURES + [TARGET]).sort_values("date").reset_index(drop=True)
    train = model_df[model_df["season"] < season]
    test = model_df[model_df["season"] == season]

    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(train[FEATURES], train[TARGET], sample_weight=time_weights(train["date"], DECAY_LAMBDA))
    pi_train = train[TARGET].mean()

    rows = []
    for date, day in test.groupby("date", sort=True):
        played = test[test["date"] < date]
        pi_new = bayesian_season_rate(played[TARGET].sum(), len(played), pi_train)
        probs = model.predict_proba(day[FEATURES])[:, 1]

        for (_, match), prob in zip(day.iterrows(), probs):
            rows.append({
                "season": season,
                "date": date,
                "home": match["home"],
                "away": match["away"],
                "actual": int(match[TARGET]),
                "prob": label_shift_correction(float(prob), pi_train, pi_new),
                "p_season_rate": pi_new,
                "p_bookmaker": bookmaker_probability(match["odd_over"], match["odd_under"]),
                "odd_over": match["odd_over"],
                "odd_under": match["odd_under"],
            })
    print(f"  {season}: {len(rows)} matches predicted")
    return pd.DataFrame(rows)
