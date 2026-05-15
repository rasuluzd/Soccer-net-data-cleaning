"""Build a name-variant -> canonical map from Labels-caption.json.
Stage E retrieves against this gazetteer for every detected entity."""

import json
from typing import Optional

from pipeline.config import LEARNED_CORRECTIONS_PATH, LEARNED_MIN_SEEN_COUNT, LEARNED_MIN_CONFIDENCE


def extract_names_from_labels(labels: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Pull every name (players/coaches/referees/teams/venue) out of Labels-caption.json.
    Returns (variant -> canonical, canonical -> entity_type)."""
    gazetteer: dict[str, str] = {}
    entity_types: dict[str, str] = {}  # canonical_name → type

    def add_name(canonical: str, *variants: str, entity_type: str = "player"):
        canonical = canonical.strip()
        if not canonical:
            return
        # The canonical name maps to itself
        gazetteer[canonical] = canonical
        entity_types[canonical] = entity_type
        for v in variants:
            v = v.strip()
            if v:
                gazetteer[v] = canonical

    def extract_surname(full_name: str) -> str:
        """Best-effort surname extraction. Keeps "De Bruyne" together."""
        parts = full_name.strip().split()
        if len(parts) <= 1:
            return full_name.strip()
        first = parts[0]
        rest = " ".join(parts[1:])
        if rest:
            return rest
        return first

    def bigram_variants(full_name: str) -> list[str]:
        """For a 3+ word name, produce 2-word subsequences and initial-abbreviated
        forms. Catches mishearings of slurred adjacent tokens.

        "Mads Döhr Thychosen" -> "Mads Döhr", "Mads Thychosen", "Döhr Thychosen",
        "M. Thychosen", "M.D. Thychosen"."""
        parts = full_name.strip().split()
        variants: list[str] = []
        if len(parts) < 3:
            return variants
        # 2-grams (surname is already added, so we keep all pairs).
        for i, a in enumerate(parts):
            for b in parts[i + 1:]:
                variants.append(f"{a} {b}")
        # Initial-letter abbreviations.
        if len(parts) >= 2:
            initial = f"{parts[0][0]}."
            variants.append(f"{initial} {parts[-1]}")
            if len(parts) >= 3:
                initials = ".".join(p[0] for p in parts[:-1]) + "."
                variants.append(f"{initials} {parts[-1]}")
        return variants

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
                add_name(long_name, short_name, name, surname, entity_type="player")
                for variant in bigram_variants(long_name):
                    if variant not in gazetteer:
                        gazetteer[variant] = long_name
            elif short_name:
                add_name(short_name, name, entity_type="player")

        # Coaches
        coaches = lineup.get("coach", [])
        for coach in coaches:
            long_name = coach.get("long_name", "")
            short_name = coach.get("short_name", "")
            name = coach.get("name", "")
            if long_name:
                surname = extract_surname(long_name)
                add_name(long_name, short_name, name, surname, entity_type="coach")

    # ── Referees ─────────────────────────────────────────────────────
    for ref_name in labels.get("referee_matched", labels.get("referee", [])):
        if ref_name:
            surname = extract_surname(ref_name)
            add_name(ref_name, surname, entity_type="referee")

    # ── Teams ────────────────────────────────────────────────────────
    home_team = labels.get("gameHomeTeam", "")
    away_team = labels.get("gameAwayTeam", "")
    if home_team:
        add_name(home_team, entity_type="team")
    if away_team:
        add_name(away_team, entity_type="team")

    # Also add team name variants from the "home"/"away" objects
    for side in ("home", "away"):
        team_obj = labels.get(side, {})
        main_name = team_obj.get("name", "")
        if main_name:
            add_name(main_name, entity_type="team")
        for alt_name in team_obj.get("names", []):
            if alt_name:
                gazetteer[alt_name] = main_name or alt_name
                entity_types[main_name or alt_name] = "team"

    # Top-level "teams" lists rivals / league-table mentions. Register them
    # so the pipeline doesn't fuzzy-correct them to player surnames.
    for extra_team in labels.get("teams", []):
        if extra_team and extra_team not in gazetteer:
            add_name(extra_team, entity_type="team")

    # ── Venue ────────────────────────────────────────────────────────
    for venue in labels.get("venue", []):
        if venue:
            # "Stamford Bridge (London)" -> "Stamford Bridge"
            import re
            venue_clean = re.sub(r"\s*\(.*?\)\s*$", "", venue).strip()
            if venue_clean:
                add_name(venue_clean, entity_type="venue")

    return gazetteer, entity_types


def load_learned_corrections() -> dict[str, dict]:
    """Load the on-disk learned-corrections JSON. Returns {} on empty/missing/bad."""
    if not LEARNED_CORRECTIONS_PATH.exists():
        return {}

    if LEARNED_CORRECTIONS_PATH.stat().st_size == 0:
        return {}

    try:
        with open(LEARNED_CORRECTIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_learned_corrections(corrections: dict[str, dict]) -> None:
    LEARNED_CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEARNED_CORRECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(corrections, f, indent=2, ensure_ascii=False)


def get_team_words(
    entity_types: dict[str, str],
    gazetteer: dict[str, str],
) -> set[str]:
    """Lowercase words appearing in any team or venue name (3+ chars).
    Used to detect 'Palace' as part of 'Crystal Palace', 'Ham' from 'West Ham', etc."""
    team_words: set[str] = set()
    for canonical, etype in entity_types.items():
        if etype == "team":
            for word in canonical.split():
                if len(word) >= 3:
                    team_words.add(word.lower())
    for canonical, etype in entity_types.items():
        if etype == "venue":
            for word in canonical.split():
                if len(word) >= 3:
                    team_words.add(word.lower())
    return team_words


def build_firstname_map(
    gazetteer: dict[str, str],
    entity_types: dict[str, str],
) -> dict[str, list[str]]:
    """Map first name (lowercase) -> list of canonical full names.
    Skips first names <4 chars (Ed/Mo collide with common words).
    Players + coaches only."""
    firstname_map: dict[str, list[str]] = {}
    seen_canonicals: set[str] = set()

    for canonical, etype in entity_types.items():
        if etype not in ("player", "coach"):
            continue
        if canonical in seen_canonicals:
            continue
        seen_canonicals.add(canonical)

        parts = canonical.split()
        if len(parts) < 2:
            continue

        first_name = parts[0].lower()
        if len(first_name) < 4:
            continue

        if first_name not in firstname_map:
            firstname_map[first_name] = []
        firstname_map[first_name].append(canonical)

    return firstname_map


def build_gazetteer(
    labels: Optional[dict],
    include_learned: bool = True
) -> tuple[dict[str, str], dict[str, str]]:
    """Build the gazetteer for a match. Merges Labels-caption.json names with
    the on-disk learned dict (high-confidence entries only)."""
    gazetteer = {}
    entity_types = {}

    if labels:
        gazetteer, entity_types = extract_names_from_labels(labels)

    if include_learned:
        learned = load_learned_corrections()
        for misspelling, info in learned.items():
            correct = info.get("correct", "")
            seen = info.get("seen_count", 0)
            conf = info.get("confidence", 0.0)
            if (correct
                    and misspelling not in gazetteer
                    and seen >= LEARNED_MIN_SEEN_COUNT
                    and conf >= LEARNED_MIN_CONFIDENCE):
                gazetteer[misspelling] = correct

    return gazetteer, entity_types


if __name__ == "__main__":
    # Quick smoke test: build the gazetteer for the first match found.
    from pipeline.loader import discover_matches

    matches = discover_matches()
    if matches:
        m = matches[0]
        print(f"Building gazetteer for: {m.match_name}\n")
        gaz, etypes = build_gazetteer(m.labels)
        print(f"Total entries: {len(gaz)}\n")
        print(f"Entity types: {len(etypes)} canonical names typed\n")
        print("Sample entries:")
        for variant, canonical in sorted(gaz.items())[:20]:
            if variant != canonical:
                print(f"  '{variant}' → '{canonical}'")
            else:
                print(f"  '{canonical}'")
