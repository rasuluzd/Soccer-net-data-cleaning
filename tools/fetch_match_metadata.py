"""Fetch a match's lineup + metadata from football-data.org and write
``Labels-caption.json`` in the same shape as SoccerNet's labels.

Free-tier limits: 10 req/min, top competitions only. For Allsvenskan etc.
we'd need API-Football fallback (separate module).

Usage:
    set FOOTBALLDATA_API_KEY=...                 # PowerShell: $env:FOOTBALLDATA_API_KEY = "..."
    python tools/fetch_match_metadata.py \\
        --competition PL --home Chelsea --away Liverpool --date 2016-09-16 \\
        --output "path/to/SoccerNet/.../2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool/Labels-caption.json"

Competition codes:
    PL  = Premier League              CL  = Champions League
    BL1 = Bundesliga                  EL  = Europa League
    PD  = La Liga (Primera División)  DED = Eredivisie
    SA  = Serie A                     PPL = Primeira Liga
    FL1 = Ligue 1                     ELC = Championship

Notes on free-tier coverage: football-data.org's free tier usually has
match metadata + score + referees for the supported competitions, but
detailed line-ups (starting XI / subs) may be limited to recent matches
or paid plans. The script reports what it got and warns if lineup is
empty so you know to either upgrade or use API-Football fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests


API_BASE = "https://api.football-data.org/v4"


def get_token(cli_token: Optional[str]) -> str:
    token = cli_token or os.environ.get("FOOTBALLDATA_API_KEY", "").strip()
    if not token:
        print("ERROR: provide --token or set FOOTBALLDATA_API_KEY env var", file=sys.stderr)
        sys.exit(2)
    return token


def _request(path: str, token: str, params: Optional[dict] = None) -> dict:
    url = f"{API_BASE}{path}"
    r = requests.get(url, headers={"X-Auth-Token": token}, params=params, timeout=15)
    # Surface rate-limit info from headers
    remaining = r.headers.get("X-RequestsAvailable", "?")
    reset = r.headers.get("X-RequestCounter-Reset", "?")
    if r.status_code == 429:
        wait = int(reset) if reset.isdigit() else 60
        print(f"  [rate-limited] sleeping {wait}s ...", file=sys.stderr)
        time.sleep(wait + 1)
        r = requests.get(url, headers={"X-Auth-Token": token}, params=params, timeout=15)
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} on {path}: {r.text[:200]}", file=sys.stderr)
        return {}
    print(f"  [ok] {path}  remaining={remaining}, reset_in={reset}s")
    return r.json()


def find_match(token: str, competition: str, home: str, away: str, date: str) -> Optional[dict]:
    """Find a match in given competition between given teams on given date.

    Searches a +/-1 day window because broadcast date sometimes differs from
    listed match date (timezone shifts).
    """
    from datetime import datetime, timedelta
    d = datetime.strptime(date, "%Y-%m-%d")
    date_from = (d - timedelta(days=1)).strftime("%Y-%m-%d")
    date_to = (d + timedelta(days=1)).strftime("%Y-%m-%d")
    data = _request(
        f"/competitions/{competition}/matches", token,
        params={"dateFrom": date_from, "dateTo": date_to},
    )
    matches = data.get("matches") or []
    for m in matches:
        h = (m.get("homeTeam", {}).get("name") or "").lower()
        a = (m.get("awayTeam", {}).get("name") or "").lower()
        if home.lower() in h and away.lower() in a:
            return m
        if home.lower() in a and away.lower() in h:
            print(f"  WARN: home/away appear swapped — found {m.get('homeTeam',{}).get('name')} vs {m.get('awayTeam',{}).get('name')}")
            return m
    print(f"  No match found for {home} vs {away} on {date} in {competition}", file=sys.stderr)
    if matches:
        print("  Available matches in window:")
        for m in matches[:10]:
            print(f"    {m.get('utcDate')} - {m.get('homeTeam',{}).get('name')} vs {m.get('awayTeam',{}).get('name')}")
    return None


def get_match_details(token: str, match_id: int) -> dict:
    """Get full match details including lineup if available."""
    return _request(f"/matches/{match_id}", token)


def get_team_squad(token: str, team_id: int) -> list[dict]:
    """Fetch full team squad as fallback when match lineup is empty
    (free tier often has squad but not per-match XI)."""
    data = _request(f"/teams/{team_id}", token)
    return data.get("squad") or []


def _player_to_lineup_entry(player: dict) -> dict:
    """Map football-data.org player object → SoccerNet Labels-caption player shape."""
    full_name = player.get("name") or player.get("shortName") or ""
    parts = full_name.split()
    short_name = parts[-1] if parts else full_name  # surname as short_name
    return {
        "name": full_name,
        "long_name": full_name,
        "short_name": short_name,
        "shirt_number": player.get("shirtNumber"),
        "position": player.get("position"),
        "country": (player.get("nationality") or ""),
    }


def build_labels(match: dict, home_squad: list[dict], away_squad: list[dict],
                 home_lineup: list[dict], away_lineup: list[dict]) -> dict:
    """Build SoccerNet Labels-caption.json shape from football-data.org match."""
    home_team = match.get("homeTeam", {}).get("name", "")
    away_team = match.get("awayTeam", {}).get("name", "")
    referee = ""
    refs = match.get("referees") or []
    for r in refs:
        if (r.get("type") or "").upper() in ("REFEREE", "MAIN_REFEREE", ""):
            referee = r.get("name") or ""
            break
    if not referee and refs:
        referee = refs[0].get("name") or ""

    venue = match.get("venue") or ""
    score = match.get("score", {}).get("fullTime") or {}

    # Prefer per-match lineup (more accurate). Fall back to full squad if empty.
    home_players = home_lineup if home_lineup else home_squad
    away_players = away_lineup if away_lineup else away_squad

    return {
        "gameDate": match.get("utcDate", "")[:10],
        "gameHomeTeam": home_team,
        "gameAwayTeam": away_team,
        "home": home_team,
        "away": away_team,
        "score": f"{score.get('home','?')}-{score.get('away','?')}",
        "teams": [home_team, away_team],
        "referee": referee,
        "venue": venue,
        "lineup": {
            "home": {
                "players": [_player_to_lineup_entry(p) for p in home_players],
                "coach": [],
            },
            "away": {
                "players": [_player_to_lineup_entry(p) for p in away_players],
                "coach": [],
            },
        },
        "annotations": [],
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--competition", required=True, help="e.g. PL, BL1, PD, SA, FL1")
    p.add_argument("--home", required=True, help="Home team substring (e.g. Chelsea)")
    p.add_argument("--away", required=True, help="Away team substring (e.g. Liverpool)")
    p.add_argument("--date", required=True, help="Match date YYYY-MM-DD")
    p.add_argument("--output", type=Path, required=True, help="Where to write Labels-caption.json")
    p.add_argument("--token", default=None, help="API key (else read from $FOOTBALLDATA_API_KEY)")
    args = p.parse_args()

    token = get_token(args.token)

    print(f"\nLooking up {args.home} vs {args.away} on {args.date} in {args.competition} ...")
    match = find_match(token, args.competition, args.home, args.away, args.date)
    if not match:
        return 1
    print(f"  Found match id={match['id']}")

    print(f"\nFetching match details ...")
    details = get_match_details(token, match["id"])

    # football-data.org puts lineup in match.homeTeam.lineup / .bench (paid plans);
    # free tier usually returns empty arrays, so we fall back to team squad.
    home_lineup = (details.get("homeTeam", {}).get("lineup") or [])
    away_lineup = (details.get("awayTeam", {}).get("lineup") or [])
    home_lineup += (details.get("homeTeam", {}).get("bench") or [])
    away_lineup += (details.get("awayTeam", {}).get("bench") or [])

    home_id = match["homeTeam"]["id"]
    away_id = match["awayTeam"]["id"]
    home_squad: list[dict] = []
    away_squad: list[dict] = []
    if not home_lineup:
        print(f"\nNo per-match lineup for home team — fetching squad as fallback ...")
        home_squad = get_team_squad(token, home_id)
    if not away_lineup:
        print(f"\nNo per-match lineup for away team — fetching squad as fallback ...")
        away_squad = get_team_squad(token, away_id)

    labels = build_labels(details, home_squad, away_squad, home_lineup, away_lineup)

    # Quick sanity check
    n_home = len(labels["lineup"]["home"]["players"])
    n_away = len(labels["lineup"]["away"]["players"])
    print(f"\nLabels built: home={n_home} players, away={n_away} players, "
          f"referee={labels['referee']!r}, venue={labels['venue']!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
