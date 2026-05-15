"""
pipeline/lineup_query_expander.py — query-time entity expansion using
lineup data + phonetic matching, instead of indexing-time cleaning.

The radical architectural shift: don't pre-clean the transcript at all.
Index the raw Whisper output as-is. Then at query time, take whatever
the user types and expand it into a disjunction of all phonetically-
similar lineup entries. The search engine sees:

  Original query:   "Sturridge goal"
  Expanded query:   "(Sturridge OR Daniel OR Daniel Sturridge OR
                     Starridge OR Sturage OR Daniel Klain) goal"

The expansion is built per-match from the lineup (Labels-caption.json):
- Phonetic codes via metaphone (jellyfish library)
- Surface variants harvested from the raw Whisper transcript itself
  (any token within fuzz.ratio≥70 of a lineup name)
- Positional aliases ("venstreback" → known left-back) — future work

This trades:
  Cleaning cost: O(per_match × ML_models)         [hours per match]
For:
  Expansion cost: O(per_query × lineup_size)      [milliseconds per query]

For a production deployment with N matches and M queries/day, the new
architecture wins as soon as M > 0 because indexing cost was the
dominant fixed cost.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class LineupEntry:
    canonical: str          # "Daniel Sturridge"
    short: str              # "Sturridge"
    side: str               # "home" / "away"
    role: str               # "player" / "coach" / "team" / "referee"
    metaphone_codes: list[str]  # one per word in canonical


def _metaphone(s: str) -> str:
    """Simple metaphone via jellyfish if available, else return upper."""
    try:
        import jellyfish
        return jellyfish.metaphone(s) or s.upper()
    except Exception:
        return s.upper()


def build_lineup_index(labels: dict) -> list[LineupEntry]:
    """Parse Labels-caption.json into a flat lineup index."""
    entries: list[LineupEntry] = []

    # Teams
    for team in labels.get("teams", []):
        if team:
            entries.append(LineupEntry(
                canonical=team, short=team, side="", role="team",
                metaphone_codes=[_metaphone(team)],
            ))

    # Players + coaches
    for side in ("home", "away"):
        lineup = labels.get("lineup", {}).get(side, {})
        for p in lineup.get("players", []):
            long_ = (p.get("long_name") or "").strip()
            short = (p.get("short_name") or "").strip()
            if not long_ and not short:
                continue
            canonical = long_ or short
            sname = short or canonical.split()[-1]
            codes = [_metaphone(w) for w in canonical.split() if len(w) >= 3]
            entries.append(LineupEntry(
                canonical=canonical, short=sname, side=side, role="player",
                metaphone_codes=codes,
            ))
        for coach in lineup.get("coach", []):
            long_ = (coach.get("long_name") or "").strip()
            if long_:
                codes = [_metaphone(w) for w in long_.split() if len(w) >= 3]
                entries.append(LineupEntry(
                    canonical=long_, short=long_.split()[-1],
                    side=side, role="coach", metaphone_codes=codes,
                ))

    # Referee, stadium
    for ref in labels.get("referee", []):
        if isinstance(ref, dict):
            name = (ref.get("long_name") or ref.get("name") or "").strip()
        else:
            name = str(ref).strip()
        if name:
            codes = [_metaphone(w) for w in name.split() if len(w) >= 3]
            entries.append(LineupEntry(
                canonical=name, short=name.split()[-1],
                side="", role="referee", metaphone_codes=codes,
            ))

    return entries


def harvest_surface_variants(raw_text: str, lineup: list[LineupEntry],
                             min_fuzz: int = 70) -> dict[str, set[str]]:
    """Scan raw Whisper text for surface variants of each lineup entry.

    For each lineup canonical, collect all capitalised tokens (and
    bigrams) in the raw text that fuzz-match it. These are the actual
    misspellings Whisper produced — the strongest signal we have for
    what to expand at query time.

    Returns: {canonical: {variant1, variant2, ...}}
    """
    from rapidfuzz import fuzz
    variants: dict[str, set[str]] = defaultdict(set)

    words = [w.strip(".,;:!?\"'()[]{}").rstrip("'s").rstrip("'")
             for w in raw_text.split()]
    cap_words = [w for w in words if w and w[0].isupper() and len(w) >= 4]
    cap_bigrams = []
    for i in range(len(words) - 1):
        if words[i] and words[i+1] and words[i][0].isupper() and len(words[i]) >= 3:
            cap_bigrams.append(f"{words[i]} {words[i+1]}")

    for entry in lineup:
        target = entry.canonical
        # Single-word match: scan cap_words
        if " " not in target:
            for w in cap_words:
                if abs(len(w) - len(target)) > 4:
                    continue
                if fuzz.ratio(w.lower(), target.lower()) >= min_fuzz:
                    variants[target].add(w)
        else:
            # Multi-word: scan bigrams
            for bg in cap_bigrams:
                if abs(len(bg) - len(target)) > 5:
                    continue
                if fuzz.ratio(bg.lower(), target.lower()) >= min_fuzz:
                    variants[target].add(bg)
            # Also match on last word (surname only)
            last = target.split()[-1]
            for w in cap_words:
                if abs(len(w) - len(last)) > 3:
                    continue
                if fuzz.ratio(w.lower(), last.lower()) >= min_fuzz:
                    variants[target].add(w)
    return dict(variants)


def expand_query(user_query: str,
                 lineup: list[LineupEntry],
                 surface_variants: dict[str, set[str]] | None = None,
                 max_expansions_per_token: int = 8) -> dict:
    """Expand a user query into a disjunction of lineup-aware variants.

    For each capitalised, multi-char token in the query:
      1. Compute its metaphone code.
      2. Find all lineup entries that share any metaphone code.
      3. Collect their surface variants (from harvest_surface_variants).
      4. Build the expansion list.

    Returns a dict suitable for an ES bool/should query body:
      {
        'original': "Sturridge goal",
        'must_terms': ["goal"],            # non-entity terms
        'should_terms': ["Sturridge", "Daniel Sturridge", "Starridge",
                         "Sturage", "Daniel Klain"],  # expansion
        'matched_entities': [{'token': 'Sturridge',
                              'canonical': 'Daniel Sturridge',
                              'expansions': [...]}],
      }
    """
    surface_variants = surface_variants or {}
    must_terms: list[str] = []
    should_terms: set[str] = set()
    matched_entities: list[dict] = []

    # Build metaphone lookup
    code_to_entries: dict[str, list[LineupEntry]] = defaultdict(list)
    for entry in lineup:
        for code in entry.metaphone_codes:
            code_to_entries[code].append(entry)

    for raw_token in user_query.split():
        token = raw_token.strip(".,;:!?\"'()[]{}")
        if not token:
            continue
        # Heuristic for entity-likeness: capitalised + ≥4 chars
        is_entity_like = (token[0].isupper() and len(token) >= 4)
        if not is_entity_like:
            must_terms.append(token)
            continue

        # Compute metaphone, find lineup entries with matching code
        token_code = _metaphone(token)
        matched = code_to_entries.get(token_code, [])
        if not matched:
            # No phonetic match — treat as regular search term
            must_terms.append(token)
            continue

        expansions = set()
        for entry in matched[:max_expansions_per_token]:
            expansions.add(entry.canonical)
            expansions.add(entry.short)
            # Add Whisper-observed surface variants for this canonical
            for v in surface_variants.get(entry.canonical, set()):
                expansions.add(v)

        matched_entities.append({
            "token": token,
            "metaphone": token_code,
            "canonical": matched[0].canonical,
            "expansions": sorted(expansions),
        })
        should_terms.update(expansions)

    return {
        "original": user_query,
        "must_terms": must_terms,
        "should_terms": sorted(should_terms),
        "matched_entities": matched_entities,
    }
