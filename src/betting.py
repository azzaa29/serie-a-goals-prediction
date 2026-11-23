"""Which bets to place, how much to stake, and the bankroll simulation.

Rule: bet on a side when my probability is higher than the probability implied
by the odds (1 / odd, bookmaker margin included) by at least BET_MARGIN.
Stake: fractional Kelly, capped at MAX_BET_FRAC of the bankroll.
"""

import pandas as pd

from config import BET_MARGIN, DRAWDOWN_THRESHOLD, KELLY_FRAC_OVER, KELLY_FRAC_UNDER, MAX_BET_FRAC


def kelly_stake(prob, odd, fraction):
    edge = prob * odd - 1
    if edge <= 0:
        return 0.0
    return min(edge / (odd - 1) * fraction, MAX_BET_FRAC)


def decide_bet(prob, odd_over, odd_under, margin=BET_MARGIN):
    """Return "OVER", "UNDER" or None. If both sides qualify, take the bigger edge."""
    if pd.isna(odd_over) or pd.isna(odd_under):
        return None
    edge_over = prob - 1 / odd_over
    edge_under = (1 - prob) - 1 / odd_under
    if edge_over >= margin and edge_over >= edge_under:
        return "OVER"
    if edge_under >= margin:
        return "UNDER"
    return None


def simulate(preds, margin=BET_MARGIN, drawdown_guard=True):
    bankroll, peak = 1.0, 1.0
    bets = []
    for row in preds.sort_values("date").itertuples(index=False):
        side = decide_bet(row.prob, row.odd_over, row.odd_under, margin)
        if side is None:
            continue

        if side == "OVER":
            prob, odd, won, fraction = row.prob, row.odd_over, row.actual == 1, KELLY_FRAC_OVER
        else:
            prob, odd, won, fraction = 1 - row.prob, row.odd_under, row.actual == 0, KELLY_FRAC_UNDER

        # halve the stakes while the bankroll is more than 15% below its peak
        if drawdown_guard and (peak - bankroll) / peak > DRAWDOWN_THRESHOLD:
            fraction *= 0.5
        stake = kelly_stake(prob, odd, fraction)

        profit = odd - 1 if won else -1.0
        bankroll *= 1 + stake * profit
        peak = max(peak, bankroll)
        bets.append({"season": row.season, "date": row.date, "home": row.home, "away": row.away,
                     "side": side, "odd": odd, "stake": stake, "won": won, "profit": profit,
                     "bankroll": bankroll})
    return pd.DataFrame(bets)
