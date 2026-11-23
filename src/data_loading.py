"""Read the raw files of one season and give them consistent column/team names."""

import pandas as pd

from config import CUP_COMPETITIONS, OVER_ODDS_COLS, RAW_DIR, TEAM_ALIASES, UNDER_ODDS_COLS

LEAGUE_COLUMNS = {
    "Date": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
}


def normalize_team(name):
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def load_league(season):
    """Results, match stats and over/under odds (football-data.co.uk)."""
    raw = pd.read_csv(RAW_DIR / "football_data" / f"{season}.csv")
    df = raw[list(LEAGUE_COLUMNS)].rename(columns=LEAGUE_COLUMNS)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed").dt.normalize()
    df["home"] = df["home"].map(normalize_team)
    df["away"] = df["away"].map(normalize_team)

    # take the first odds column that is not missing
    df["odd_over"] = raw[[c for c in OVER_ODDS_COLS if c in raw]].bfill(axis=1).iloc[:, 0]
    df["odd_under"] = raw[[c for c in UNDER_ODDS_COLS if c in raw]].bfill(axis=1).iloc[:, 0]

    df.insert(0, "season", season)
    return df.sort_values(["date", "home"]).reset_index(drop=True)


def load_xg(season):
    """Expected goals per match (Understat)."""
    df = pd.read_csv(RAW_DIR / "xg" / f"{season}.csv", parse_dates=["date"])
    df["home"] = df["home"].map(normalize_team)
    df["away"] = df["away"].map(normalize_team)
    return df[["date", "home", "away", "HxG", "AxG"]]


def load_fixtures(season):
    """All matches (Serie A + cups) of the season, used for rest days."""
    frames = [load_league(season)[["date", "home", "away"]].assign(competition="serie_a")]
    for comp in CUP_COMPETITIONS:
        path = RAW_DIR / "fixtures" / comp / f"{season}.csv"
        if not path.exists():
            continue
        cup = pd.read_csv(path, parse_dates=["date"])
        cup["home"] = cup["home"].map(normalize_team)
        cup["away"] = cup["away"].map(normalize_team)
        frames.append(cup[["date", "home", "away"]].assign(competition=comp))
    out = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out
