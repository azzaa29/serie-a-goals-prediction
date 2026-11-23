"""Pre-match features for every Serie A match.

The features of a match played on day D only use matches played before D.
"""

import numpy as np
import pandas as pd

from config import ELO_HOME_ADVANTAGE, ELO_K, ELO_START
from data_loading import load_fixtures, load_league, load_xg

ROLLING_STATS = [
    "goals", "goals_conceded", "xg", "xg_conceded",
    "shots", "shots_conceded", "shots_on_target", "shots_on_target_conceded",
    "corners", "corners_conceded",
]
ROLLING_STATS_LAST3 = ROLLING_STATS[:8]


# Elo (every team restarts at 1500 each season)

def expected_home_score(home_elo, away_elo):
    return 1 / (1 + 10 ** (-(home_elo + ELO_HOME_ADVANTAGE - away_elo) / 400))


def goal_diff_multiplier(goal_diff):
    goal_diff = abs(goal_diff)
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    if goal_diff == 3:
        return 1.8
    return 2.0


def add_elo(matches):
    teams = set(matches["home"]) | set(matches["away"])
    elo = dict.fromkeys(teams, ELO_START)
    home_pre, away_pre = [], []

    for home, away, hg, ag in matches[["home", "away", "home_goals", "away_goals"]].itertuples(index=False):
        h, a = elo[home], elo[away]
        home_pre.append(h)
        away_pre.append(a)

        actual_home = 1.0 if hg > ag else 0.0 if hg < ag else 0.5
        delta = ELO_K * goal_diff_multiplier(hg - ag) * (actual_home - expected_home_score(h, a))
        elo[home] = h + delta
        elo[away] = a - delta

    matches["elo_home_pre"] = home_pre
    matches["elo_away_pre"] = away_pre
    matches["elo_diff_pre"] = matches["elo_home_pre"] - matches["elo_away_pre"]
    return matches


# Rest days and number of games in the last 12 days (all competitions)

def add_rest_features(matches, fixtures, window_days=12):
    games = pd.concat([
        fixtures[["date", "home"]].rename(columns={"home": "team"}),
        fixtures[["date", "away"]].rename(columns={"away": "team"}),
    ]).drop_duplicates().sort_values(["team", "date"]).reset_index(drop=True)

    games["rest"] = games.groupby("team")["date"].diff().dt.days

    def games_in_window(dates):
        values = dates.to_numpy(dtype="datetime64[ns]")
        first = np.searchsorted(values, values - np.timedelta64(window_days, "D"), side="left")
        return pd.Series(np.arange(len(values)) - first, index=dates.index)

    games["matches_last12"] = games.groupby("team")["date"].transform(games_in_window)

    for side in ["home", "away"]:
        matches = matches.merge(
            games.rename(columns={"team": side, "rest": f"rest_{side}",
                                  "matches_last12": f"matches_last12_{side}"}),
            on=["date", side], how="left",
        )
    matches["rest_diff"] = matches["rest_home"] - matches["rest_away"]
    matches["matches12_diff"] = matches["matches_last12_home"] - matches["matches_last12_away"]
    return matches


# League table before each match day

def add_standings(matches):
    teams = sorted(set(matches["home"]) | set(matches["away"]))
    points = dict.fromkeys(teams, 0)
    rows = {}

    for _, day in matches.groupby("date", sort=True):
        table = sorted(points.values(), reverse=True)
        fourth, safe_line = table[3], table[17]
        for idx, m in day.iterrows():
            ph, pa = points[m["home"]], points[m["away"]]
            rows[idx] = {
                "home_points": ph,
                "away_points": pa,
                "home_position": 1 + sum(p > ph for p in table),
                "away_position": 1 + sum(p > pa for p in table),
                "home_to_champions": max(0, fourth - ph),
                "away_to_champions": max(0, fourth - pa),
                "home_to_relegation": ph - safe_line,
                "away_to_relegation": pa - safe_line,
                "points_diff": ph - pa,
            }
        # results of the day are added only after the whole day
        for m in day.itertuples():
            if m.home_goals > m.away_goals:
                points[m.home] += 3
            elif m.home_goals < m.away_goals:
                points[m.away] += 3
            else:
                points[m.home] += 1
                points[m.away] += 1

    return matches.join(pd.DataFrame.from_dict(rows, orient="index"))


# Form: averages over the last N league games of each team

def team_view(matches):
    """One row per team per match, seen from that team's side."""
    sides = []
    for side, other in [("home", "away"), ("away", "home")]:
        sides.append(pd.DataFrame({
            "season": matches["season"],
            "date": matches["date"],
            "team": matches[side],
            "goals": matches[f"{side}_goals"],
            "goals_conceded": matches[f"{other}_goals"],
            "xg": matches["HxG" if side == "home" else "AxG"],
            "xg_conceded": matches["AxG" if side == "home" else "HxG"],
            "shots": matches[f"{side}_shots"],
            "shots_conceded": matches[f"{other}_shots"],
            "shots_on_target": matches[f"{side}_shots_on_target"],
            "shots_on_target_conceded": matches[f"{other}_shots_on_target"],
            "corners": matches[f"{side}_corners"],
            "corners_conceded": matches[f"{other}_corners"],
            "over25": matches["over25"],
        }))
    return pd.concat(sides, ignore_index=True).sort_values(["team", "date"]).reset_index(drop=True)


def add_form_features(matches):
    team = team_view(matches)

    # shift(1) so the current match is never included; form restarts every season
    by_season = team.groupby(["season", "team"])
    for n, stats in [(5, ROLLING_STATS), (3, ROLLING_STATS_LAST3)]:
        for col in stats:
            team[f"{col}_last{n}"] = by_season[col].transform(
                lambda s, n=n: s.shift(1).rolling(n, min_periods=n).mean()
            )

    # over 2.5 rate keeps the history of previous seasons
    by_team = team.groupby("team")["over25"]
    team["over25_rate_last5"] = by_team.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    team["over25_rate_last10"] = by_team.transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())

    new_cols = [c for c in team.columns if c.endswith(("_last5", "_last3", "_last10"))]
    for side in ["home", "away"]:
        side_cols = team[["date", "team"] + new_cols].rename(
            columns={"team": side, **{c: f"{side}_{c}" for c in new_cols}}
        )
        matches = matches.merge(side_cols, on=["date", side], how="left")
    return matches


def add_combined_features(df):
    df["combined_xg_last5"] = df["home_xg_last5"] + df["away_xg_last5"]
    df["combined_goals_last5"] = df["home_goals_last5"] + df["away_goals_last5"]
    df["xg_conceded_total_last5"] = df["home_xg_conceded_last5"] + df["away_xg_conceded_last5"]
    df["goals_conceded_total_last5"] = df["home_goals_conceded_last5"] + df["away_goals_conceded_last5"]
    df["shots_on_target_total_last5"] = df["home_shots_on_target_last5"] + df["away_shots_on_target_last5"]
    df["shots_on_target_conceded_total_last5"] = (
        df["home_shots_on_target_conceded_last5"] + df["away_shots_on_target_conceded_last5"]
    )

    # goals compared with xG (finishing / goalkeeping)
    df["offensive_efficiency"] = (df["combined_goals_last5"] / df["combined_xg_last5"].clip(lower=0.5)).clip(0, 2.5)
    df["defensive_efficiency"] = (
        df["goals_conceded_total_last5"] / df["xg_conceded_total_last5"].clip(lower=0.5)
    ).clip(0, 2.5)

    # attack of one team vs defence of the other
    df["home_attack_vs_away_defense"] = (df["home_xg_last5"] / df["away_xg_conceded_last5"].clip(lower=0.5)).clip(0, 3)
    df["away_attack_vs_home_defense"] = (df["away_xg_last5"] / df["home_xg_conceded_last5"].clip(lower=0.5)).clip(0, 3)
    return df


def load_season_matches(season):
    league = load_league(season)
    # merge on the teams, not the date: each home/away pair plays once per season,
    # and the sources sometimes disagree by a day (time zones, postponed matches)
    xg = load_xg(season).drop(columns="date")
    if xg.duplicated(["home", "away"]).any():
        raise ValueError(f"{season}: duplicated matches in the xG file")
    matches = league.merge(xg, on=["home", "away"], how="left")
    missing = matches["HxG"].isna().sum()
    if missing:
        print(f"  {season}: {missing} matches without xG")
    matches["over25"] = (matches["home_goals"] + matches["away_goals"] > 2).astype(int)
    matches["total_goals"] = matches["home_goals"] + matches["away_goals"]
    return matches


def build_dataset(seasons):
    frames = []
    for season in seasons:
        matches = load_season_matches(season)
        matches = add_elo(matches)
        matches = add_rest_features(matches, load_fixtures(season))
        matches = add_standings(matches)
        frames.append(matches)

    df = pd.concat(frames, ignore_index=True)
    df = add_form_features(df)
    df = add_combined_features(df)
    return df.sort_values(["date", "home"]).reset_index(drop=True)
