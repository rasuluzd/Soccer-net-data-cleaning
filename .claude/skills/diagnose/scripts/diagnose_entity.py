"""
Tier 2 diagnostic script — bundled with the diagnose skill.
Usage: python .claude/skills/diagnose/scripts/diagnose_entity.py <entity> <candidate>

Prints the exact multi-signal score the pipeline would compute for an entity/candidate pair,
so you can reproduce false positives without running the full pipeline.
"""
import sys
import os

# Add project root to path (4 levels up: scripts -> diagnose -> skills -> .claude -> root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import jellyfish
import rapidfuzz.fuzz as fuzz

# Weights from pipeline/config.py — keep in sync
FUZZY_WEIGHT    = 0.45
PHONETIC_WEIGHT = 0.40
CONTEXT_WEIGHT  = 0.15


def phonetic_score(a: str, b: str) -> float:
    """Metaphone-based phonetic match, 0.0–1.0."""
    pa, pb = jellyfish.metaphone(a), jellyfish.metaphone(b)
    if not pa or not pb:
        return 0.0
    return 1.0 if pa == pb else fuzz.ratio(pa, pb) / 100.0


def diagnose(entity: str, candidate: str, context_bonus: float = 0.0) -> dict:
    f = fuzz.token_sort_ratio(entity, candidate)
    p = phonetic_score(entity, candidate) * 100
    c = context_bonus * 100

    combined = (FUZZY_WEIGHT * f) + (PHONETIC_WEIGHT * p) + (CONTEXT_WEIGHT * c)

    return {
        "entity":       entity,
        "candidate":    candidate,
        "fuzzy":        round(f, 1),
        "phonetic":     round(p, 1),
        "context":      round(c, 1),
        "combined":     round(combined, 1),
        "metaphone_entity":    jellyfish.metaphone(entity),
        "metaphone_candidate": jellyfish.metaphone(candidate),
        "verdict": (
            "ACCEPT (>=75)"           if combined >= 75   else
            "UNCERTAIN -> Tier 3"     if combined >= 55   else
            "UNCERTAIN -> Tier 3 (short)" if combined >= 48 else
            "REJECT"
        ),
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diagnose_entity.py <entity> <candidate> [context_bonus 0.0-1.0]")
        sys.exit(1)

    entity    = sys.argv[1]
    candidate = sys.argv[2]
    context   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    result = diagnose(entity, candidate, context)
    print("\n=== Tier 2 Diagnostic ===")
    for k, v in result.items():
        print(f"  {k:<25} {v}")
    print()
