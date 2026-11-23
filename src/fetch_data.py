"""Download the raw data of a season into data/raw/.

    python src/fetch_data.py 2025-2026

Coppa Italia fixtures have no public CSV and are written by hand in
data/raw/fixtures/coppa_italia/.
"""

import io
import sys

import pandas as pd
import requests
from understatapi import UnderstatClient

from config import RAW_DIR, TEAM_ALIASES

CUPS = {"ucl": "champions-league", "uel": "europa-league", "uecl": "conference-league"}


def fetch_league(season):
    code = season[2:4] + season[7:9]  # "2025-2026" -> "2526"
    resp = requests.get(f"https://www.football-data.co.uk/mmz4281/{code}/I1.csv", timeout=60)
    resp.raise_for_status()
    (RAW_DIR / "football_data" / f"{season}.csv").write_bytes(resp.content)


def fetch_xg(season):
    with UnderstatClient() as understat:
        matches = understat.league(league="Serie_A").get_match_data(season=season[:4])
    rows = [{
        "date": pd.Timestamp(m["datetime"]).date(),
        "home": TEAM_ALIASES.get(m["h"]["title"], m["h"]["title"]),
        "away": TEAM_ALIASES.get(m["a"]["title"], m["a"]["title"]),
        "HxG": float(m["xG"]["h"]),
        "AxG": float(m["xG"]["a"]),
        "source": "understat",
    } for m in matches if m["isResult"]]
    pd.DataFrame(rows).sort_values("date").to_csv(RAW_DIR / "xg" / f"{season}.csv", index=False)


def fetch_cups(season):
    for folder, name in CUPS.items():
        url = f"https://fixturedownload.com/download/{name}-{season[:4]}-UTC.csv"
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200 or resp.text.lstrip().startswith("<"):
            print(f"  no {name} file for {season}")
            continue
        raw = pd.read_csv(io.StringIO(resp.text))
        out = pd.DataFrame({
            "date": pd.to_datetime(raw["Date"], dayfirst=True).dt.date,
            "home": raw["Home Team"],
            "away": raw["Away Team"],
        })
        out.to_csv(RAW_DIR / "fixtures" / folder / f"{season}.csv", index=False)


if __name__ == "__main__":
    season = sys.argv[1]
    fetch_league(season)
    fetch_xg(season)
    fetch_cups(season)
    print(f"Downloaded {season}")
