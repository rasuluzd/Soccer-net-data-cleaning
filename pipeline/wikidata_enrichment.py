"""
Wikidata SPARQL Enrichment — fetches EPL squad lists for expanded gazetteers.

Commentators often reference players not in the match lineup:
  - Players from other matches mentioned in comparison
  - Transfer rumors, former players, etc.

This module queries Wikidata to get ALL EPL players for a given season,
caches the results locally, and merges them into the gazetteer.
"""

import json
import time
from pathlib import Path
from typing import Optional

import requests

from pipeline.config import (
    WIKIDATA_ENDPOINT,
    WIKIDATA_CACHE_PATH,
)


# ─── SPARQL Query Template ──────────────────────────────────────────
# Fetches all footballers who were members of a Premier League team
# within the specified season range.
SPARQL_QUERY_TEMPLATE = """
SELECT ?player ?playerLabel ?playerAltLabel ?team ?teamLabel WHERE {{
  ?player wdt:P106 wd:Q937857 .        # occupation: association football player
  ?player p:P54 ?teamStmt .            # member of sports team (statement)
  ?teamStmt ps:P54 ?team .             # the actual team
  ?team wdt:P118 wd:Q9448 .           # team is in Premier League
  ?teamStmt pq:P580 ?start .          # start date qualifier
  FILTER(YEAR(?start) >= {year_start} && YEAR(?start) <= {year_end})
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en".
  }}
}}
LIMIT 3000
"""


def fetch_epl_players(year_start: int, year_end: int) -> list[dict]:
    """
    Query Wikidata SPARQL for all EPL players in the given year range.

    Args:
        year_start: earliest year (e.g. 2014)
        year_end: latest year (e.g. 2016)

    Returns:
        List of dicts: [{"name": "...", "team": "...", "aliases": ["..."]}, ...]
    """
    query = SPARQL_QUERY_TEMPLATE.format(
        year_start=year_start, year_end=year_end
    )

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "SoccerASR-DataCleaning/1.0 (bachelor thesis project)",
    }

    try:
        response = requests.get(
            WIKIDATA_ENDPOINT,
            params={"query": query},
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ⚠ Wikidata query failed: {e}")
        return []

    results = response.json().get("results", {}).get("bindings", [])

    # Group by player to collect aliases
    players_by_name: dict[str, dict] = {}
    for row in results:
        name = row.get("playerLabel", {}).get("value", "")
        team = row.get("teamLabel", {}).get("value", "")
        alt_labels = row.get("playerAltLabel", {}).get("value", "")

        if not name or name.startswith("Q"):  # Skip unnamed items
            continue

        if name not in players_by_name:
            players_by_name[name] = {
                "name": name,
                "team": team,
                "aliases": set(),
            }

        # Parse alternative labels (pipe-separated)
        if alt_labels and not alt_labels.startswith("Q"):
            for alias in alt_labels.split(", "):
                alias = alias.strip()
                if alias and alias != name and not alias.startswith("Q"):
                    players_by_name[name]["aliases"].add(alias)

    # Convert sets to lists for JSON serialization
    player_list = []
    for p in players_by_name.values():
        p["aliases"] = list(p["aliases"])
        player_list.append(p)

    return player_list


def _extract_surname(full_name: str) -> str:
    """Extract the surname from a full name."""
    parts = full_name.strip().split()
    if len(parts) <= 1:
        return full_name.strip()
    return " ".join(parts[1:])


def load_or_fetch_cache(
    year_start: int, year_end: int
) -> list[dict]:
    """
    Load EPL players from local cache, or fetch from Wikidata if not cached.

    Cache key is the year range string (e.g., "2014-2016").

    Args:
        year_start: earliest year
        year_end: latest year

    Returns:
        List of player dicts
    """
    cache_key = f"{year_start}-{year_end}"

    # Try loading from cache
    if WIKIDATA_CACHE_PATH.exists():
        try:
            with open(WIKIDATA_CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if cache_key in cache:
                players = cache[cache_key]
                print(f"  Wikidata cache: loaded {len(players)} players for {cache_key}")
                return players
        except (json.JSONDecodeError, KeyError):
            pass

    # Fetch from Wikidata
    print(f"  Fetching EPL players from Wikidata for {cache_key}...")
    players = fetch_epl_players(year_start, year_end)
    print(f"  Wikidata: fetched {len(players)} players")

    if players:
        # Save to cache
        WIKIDATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache = {}
        if WIKIDATA_CACHE_PATH.exists():
            try:
                with open(WIKIDATA_CACHE_PATH, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except json.JSONDecodeError:
                cache = {}

        cache[cache_key] = players
        with open(WIKIDATA_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    return players


def enrich_gazetteer(
    gazetteer: dict[str, str],
    year_start: int = 2014,
    year_end: int = 2016,
) -> dict[str, str]:
    """
    Expand a match gazetteer with Wikidata EPL player names.

    Adds players not already in the gazetteer (avoids overwriting
    match-specific data that is more authoritative).

    Args:
        gazetteer: existing gazetteer dict (variant → canonical)
        year_start: earliest year for the Wikidata query
        year_end: latest year for the Wikidata query

    Returns:
        Enriched gazetteer dict
    """
    players = load_or_fetch_cache(year_start, year_end)

    added = 0
    for player in players:
        name = player["name"]
        surname = _extract_surname(name)

        # Only add names NOT already in the gazetteer
        # (match-specific data is more authoritative)
        if name not in gazetteer:
            gazetteer[name] = name
            added += 1
        if surname and surname not in gazetteer:
            gazetteer[surname] = name
            added += 1

        # Add aliases
        for alias in player.get("aliases", []):
            if alias and alias not in gazetteer:
                gazetteer[alias] = name
                added += 1

    print(f"  Wikidata enrichment: +{added} name entries (total: {len(gazetteer)})")
    return gazetteer


if __name__ == "__main__":
    # Quick test
    players = load_or_fetch_cache(2014, 2016)
    print(f"\nTotal players: {len(players)}")
    for p in players[:10]:
        aliases = ", ".join(p["aliases"][:3]) if p["aliases"] else "none"
        print(f"  {p['name']} ({p['team']}) aliases: {aliases}")
