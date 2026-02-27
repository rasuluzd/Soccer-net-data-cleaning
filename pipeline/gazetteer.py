"""
Gazetteer Builder — creates a name lookup dictionary from Labels-caption.json.

The gazetteer is the "source of truth" for correct names. It maps every known
name variant (surname, short_name, long_name) to its canonical full form.
This is what the fuzzy matcher compares ASR entities against.
"""

import json
from pathlib import Path
from typing import Optional

from pipeline.config import LEARNED_CORRECTIONS_PATH


def extract_names_from_labels(labels: dict) -> dict[str, str]:
    """
    Extract all proper names from a Labels-caption.json and build a
    gazetteer mapping every variant to its canonical form.

    Sources of names:
        - Players (home + away, starting + subs): long_name, short_name, name
        - Coaches: long_name, short_name
        - Referees: referee_matched or referee field
        - Teams: gameHomeTeam, gameAwayTeam, team names
        - Venue: from the venue field

    Args:
        labels: parsed Labels-caption.json dict

    Returns:
        Dict mapping name variants -> canonical name.
        Example: {"De Bruyne": "Kevin De Bruyne", "Kevin De Bruyne": "Kevin De Bruyne"}
    """
    gazetteer: dict[str, str] = {}

    def add_name(canonical: str, *variants: str):
        """Register a canonical name and all its variants."""
        canonical = canonical.strip()
        if not canonical:
            return
        # The canonical name maps to itself
        gazetteer[canonical] = canonical
        for v in variants:
            v = v.strip()
            if v:
                gazetteer[v] = canonical

    def extract_surname(full_name: str) -> str:
        """Extract likely surname from a full name. Handles multi-word surnames."""
        parts = full_name.strip().split()
        if len(parts) <= 1:
            return full_name.strip()
        # For names like "Kevin De Bruyne", surname is "De Bruyne"
        # For names like "Sergio Agüero", surname is "Agüero"
        # Heuristic: if the second part is lowercase (de, van, el, etc.), include it
        first = parts[0]
        rest = " ".join(parts[1:])
        if rest:
            return rest
        return first

    # ── Players ──────────────────────────────────────────────────────
    for side in ("home", "away"):
        lineup = labels.get("lineup", {}).get(side, {})
        players = lineup.get("players", [])
        for player in players:
            long_name = player.get("long_name", "")
            short_name = player.get("short_name", "")
            name = player.get("name", "")

            if long_name:
                surname = extract_surname(long_name)
                add_name(long_name, short_name, name, surname)
            elif short_name:
                add_name(short_name, name)

        # Coaches
        coaches = lineup.get("coach", [])
        for coach in coaches:
            long_name = coach.get("long_name", "")
            short_name = coach.get("short_name", "")
            name = coach.get("name", "")
            if long_name:
                surname = extract_surname(long_name)
                add_name(long_name, short_name, name, surname)

    # ── Referees ─────────────────────────────────────────────────────
    for ref_name in labels.get("referee_matched", labels.get("referee", [])):
        if ref_name:
            surname = extract_surname(ref_name)
            add_name(ref_name, surname)

    # ── Teams ────────────────────────────────────────────────────────
    home_team = labels.get("gameHomeTeam", "")
    away_team = labels.get("gameAwayTeam", "")
    if home_team:
        add_name(home_team)
    if away_team:
        add_name(away_team)

    # Also add team name variants from the "home"/"away" objects
    for side in ("home", "away"):
        team_obj = labels.get(side, {})
        main_name = team_obj.get("name", "")
        if main_name:
            add_name(main_name)
        for alt_name in team_obj.get("names", []):
            if alt_name:
                gazetteer[alt_name] = main_name or alt_name

    # ── Venue ────────────────────────────────────────────────────────
    for venue in labels.get("venue", []):
        if venue:
            add_name(venue)

    return gazetteer


def load_learned_corrections() -> dict[str, dict]:
    """
    Load the self-learning correction dictionary.

    Returns:
        Dict mapping lowercase misspelling -> {"correct": str, "confidence": float, "seen_count": int}
    """
    if not LEARNED_CORRECTIONS_PATH.exists():
        return {}

    with open(LEARNED_CORRECTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_learned_corrections(corrections: dict[str, dict]) -> None:
    """Save the updated learned corrections dictionary."""
    LEARNED_CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEARNED_CORRECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(corrections, f, indent=2, ensure_ascii=False)


def build_gazetteer(
    labels: Optional[dict],
    include_learned: bool = True
) -> dict[str, str]:
    """
    Build the complete gazetteer for a match.

    Combines:
        1. Names extracted from Labels-caption.json (match-specific)
        2. Previously learned corrections (cross-match knowledge)

    Args:
        labels: parsed Labels-caption.json dict (or None)
        include_learned: whether to merge the learned correction dictionary

    Returns:
        Complete gazetteer dict: variant -> canonical name
    """
    gazetteer = {}

    # Step 1: Match-specific names from Labels
    if labels:
        gazetteer = extract_names_from_labels(labels)

    # Step 2: Merge learned corrections
    if include_learned:
        learned = load_learned_corrections()
        for misspelling, info in learned.items():
            correct = info.get("correct", "")
            if correct and misspelling not in gazetteer:
                gazetteer[misspelling] = correct

    return gazetteer


if __name__ == "__main__":
    # Quick test: build gazetteer for a sample match
    from pipeline.loader import discover_matches

    matches = discover_matches()
    if matches:
        m = matches[0]
        print(f"Building gazetteer for: {m.match_name}\n")
        gaz = build_gazetteer(m.labels)
        print(f"Total entries: {len(gaz)}\n")
        print("Sample entries:")
        for variant, canonical in sorted(gaz.items())[:20]:
            if variant != canonical:
                print(f"  '{variant}' → '{canonical}'")
            else:
                print(f"  '{canonical}'")
