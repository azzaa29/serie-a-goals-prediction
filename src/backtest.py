"""Walk-forward evaluation of the model and betting backtest.

    python src/backtest.py           # reuse saved predictions if they exist
    python src/backtest.py --refit   # recompute the predictions

Seasons:
  2020-21 ... 2022-23  training only
  2023-24, 2024-25     validation: used to choose the betting margin
  2025-26              test

Every match is predicted with a model that only saw earlier matches.
Bets are placed from 10 February, when half of the current season is known.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from betting import simulate
from config import BET_MARGIN, DATASET_PATH, FIGURES_DIR, PREDICTIONS_PATH
from model import predict_season

VALIDATION_SEASONS = ["2023-2024", "2024-2025"]
TEST_SEASON = "2025-2026"
PREDICT_SEASONS = VALIDATION_SEASONS + [TEST_SEASON]
START_DAY = "02-10"
MARGINS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]


def after_start_day(preds):
    start = pd.to_datetime(preds["season"].str[5:] + "-" + START_DAY)
    return preds[preds["date"] >= start]


def get_predictions(refit):
    if PREDICTIONS_PATH.exists() and not refit:
        preds = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
        return preds[preds["season"].isin(PREDICT_SEASONS)]
    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    preds = pd.concat([predict_season(df, s) for s in PREDICT_SEASONS], ignore_index=True)
    preds.to_csv(PREDICTIONS_PATH, index=False)
    return preds


def print_metrics(preds):
    print("PROBABILITY QUALITY (AUC: higher = better, Brier: lower = better)")
    print(f"{'season':<12}{'forecast':<20}{'AUC':>7}{'Brier':>8}")
    for season, g in preds.groupby("season"):
        for name, col in [("model", "prob"), ("season over-rate", "p_season_rate"),
                          ("bookmaker", "p_bookmaker")]:
            print(f"{season:<12}{name:<20}{roc_auc_score(g['actual'], g[col]):>7.3f}"
                  f"{brier_score_loss(g['actual'], g[col]):>8.4f}")

    print("\nCALIBRATION of the model (all seasons)")
    bins = pd.cut(preds["prob"], [0, 0.4, 0.45, 0.5, 0.55, 1])
    table = preds.groupby(bins, observed=True).agg(n=("actual", "size"), predicted=("prob", "mean"),
                                                    observed=("actual", "mean"))
    print(table.round(3).to_string())


def bets_line(label, bets):
    if len(bets) == 0:
        return f"{label:<14}{0:>6}"
    two_se = 2 * bets["profit"].std() / np.sqrt(len(bets))
    return (f"{label:<14}{len(bets):>6}{bets['won'].mean():>7.1%}{bets['profit'].mean():>+8.1%}"
            f"{two_se:>9.1%}{bets['bankroll'].iloc[-1]:>9.2f}x")


def choose_margin(preds):
    print("\nMARGIN CHOICE on the validation seasons (2023-24 + 2024-25)")
    print(f"{'margin':<14}{'bets':>6}{'win%':>7}{'ROI':>8}{'+/- 2SE':>9}")
    validation = preds[preds["season"].isin(VALIDATION_SEASONS)]
    for margin in MARGINS:
        bets = pd.concat([simulate(g, margin) for _, g in validation.groupby("season")])
        print(bets_line(f"{margin:.2f}", bets)[:44])
    print(f"-> chosen margin: {BET_MARGIN:.2f} (set in config.py)")


def run_betting(preds):
    print(f"\nBETTING with margin {BET_MARGIN:.2f}")
    print(f"{'season':<14}{'bets':>6}{'win%':>7}{'ROI':>8}{'+/- 2SE':>9}{'bankroll':>10}")
    results = {}
    for season in PREDICT_SEASONS:
        bets = simulate(preds[preds["season"] == season])
        role = "test" if season == TEST_SEASON else "validation"
        print(bets_line(f"{season} ({role[0]})", bets))
        results[season] = bets
    return results


def make_plots(preds, test_bets):
    FIGURES_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, col in [("Model", "prob"), ("Bookmaker", "p_bookmaker")]:
        bins = pd.cut(preds[col], np.arange(0.3, 0.75, 0.05))
        t = preds.groupby(bins, observed=True).agg(p=(col, "mean"), y=("actual", "mean"), n=("actual", "size"))
        t = t[t["n"] >= 15]
        ax.plot(t["p"], t["y"], marker="o", label=name)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.3, 0.7)
    ax.set_xlabel("Predicted probability of over 2.5")
    ax.set_ylabel("Observed over 2.5 rate")
    ax.set_title("Calibration (2023-24 to 2025-26)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=150)

    if len(test_bets):
        b = test_bets.sort_values("date")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(b["date"], b["bankroll"], marker=".")
        ax.axhline(1, color="grey", lw=1)
        ax.set_ylabel("Bankroll (start = 1)")
        ax.set_title(f"Kelly bankroll, {TEST_SEASON} test season")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "bankroll.png", dpi=150)


if __name__ == "__main__":
    preds = after_start_day(get_predictions(refit="--refit" in sys.argv))
    print_metrics(preds)
    choose_margin(preds)
    results = run_betting(preds)
    make_plots(preds, results[TEST_SEASON])
    print(f"\nFigures saved in {FIGURES_DIR}")
