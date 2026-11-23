"""Feature analysis on the training/validation seasons only (2025-26 is never used).

    python src/feature_analysis.py

1. Multicollinearity of the current features (correlations, VIF)
2. Predictive power of each single feature, and whether it is stable across seasons
3. Drop-one-feature test with the base XGBoost model, validated walk-forward:
   train on all seasons before S, evaluate on S, for S = 2021-22 ... 2024-25
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

from config import BASE_PARAMS, DATASET_PATH, DECAY_LAMBDA_GLOBAL, FEATURES_V6 as FEATURES, TARGET
from model import time_weights

ANALYSIS_SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
VALIDATION_SEASONS = ANALYSIS_SEASONS[1:]

# columns that are not pre-match information
NOT_FEATURES = {
    "season", "date", "home", "away", "home_goals", "away_goals", "home_shots", "away_shots",
    "home_shots_on_target", "away_shots_on_target", "home_corners", "away_corners",
    "odd_over", "odd_under", "HxG", "AxG", "over25", "total_goals",
}


def vif(df):
    out = {}
    for col in df.columns:
        others = df.drop(columns=col)
        r2 = LinearRegression().fit(others, df[col]).score(others, df[col])
        out[col] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out).sort_values(ascending=False)


def multicollinearity(df):
    print("1. MULTICOLLINEARITY (current 17 features)")
    corr = df[FEATURES].corr()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:]
             if abs(corr.loc[a, b]) > 0.7]
    print("\nPairs with |correlation| > 0.7:")
    for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])):
        print(f"  {r:+.2f}  {a}  <->  {b}")
    print("\nVIF (above 10 = strongly redundant):")
    print(vif(df[FEATURES]).round(1).to_string())


def single_feature_power(df, candidates):
    print("\n2. SINGLE FEATURE vs OVER 2.5 (2020-21 to 2024-25)")
    rows = []
    for col in candidates:
        d = df[[col, TARGET, "total_goals", "season"]].dropna()
        auc = roc_auc_score(d[TARGET], d[col])
        rho = spearmanr(d[col], d["total_goals"]).statistic
        per_season = [spearmanr(g[col], g["total_goals"]).statistic for _, g in d.groupby("season")]
        same_sign = sum(np.sign(r) == np.sign(rho) for r in per_season)
        rows.append({
            "feature": col,
            "in_model": col in FEATURES,
            "auc": max(auc, 1 - auc),  # direction does not matter here
            "corr_total_goals": rho,
            "seasons_same_sign": f"{same_sign}/{len(per_season)}",
        })
    table = pd.DataFrame(rows).sort_values("auc", ascending=False)
    print(table.round(3).to_string(index=False))
    return table


def walk_forward_score(df, features):
    """Log-loss of the base model on each validation season, plus the average."""
    scores = {}
    for season in VALIDATION_SEASONS:
        train = df[df["season"] < season].dropna(subset=features)
        test = df[df["season"] == season].dropna(subset=features)
        model = XGBClassifier(**BASE_PARAMS)
        model.fit(train[features], train[TARGET], sample_weight=time_weights(train["date"], DECAY_LAMBDA_GLOBAL))
        scores[season] = log_loss(test[TARGET], model.predict_proba(test[features])[:, 1])
    scores["mean"] = np.mean(list(scores.values()))
    return scores


def drop_one_feature(df):
    print("\n3. DROP-ONE-FEATURE TEST (base XGBoost, log-loss, lower = better)")
    # same rows for every run: matches where all current features exist
    df = df.dropna(subset=FEATURES)

    baseline = {}
    for season in VALIDATION_SEASONS:
        rate = df[df["season"] < season][TARGET].mean()
        y = df[df["season"] == season][TARGET]
        baseline[season] = log_loss(y, np.full(len(y), rate))
    baseline["mean"] = np.mean(list(baseline.values()))

    full = walk_forward_score(df, FEATURES)
    print(f"  constant over-rate (no model): {baseline['mean']:.4f}")
    print(f"  all 17 features:               {full['mean']:.4f}")
    print("\n  Without feature ->  mean log-loss  (change; negative = model better without it)")
    rows = []
    for col in FEATURES:
        s = walk_forward_score(df, [f for f in FEATURES if f != col])
        rows.append((col, s["mean"], s["mean"] - full["mean"]))
    for col, score, delta in sorted(rows, key=lambda x: x[2]):
        print(f"  {col:<40}{score:.4f}  ({delta:+.4f})")


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    df = df[df["season"].isin(ANALYSIS_SEASONS)]

    multicollinearity(df.dropna(subset=FEATURES))
    candidates = [c for c in df.columns if c not in NOT_FEATURES]
    single_feature_power(df, candidates)
    drop_one_feature(df)
