"""
Stage E diagnostic script — bundled with the diagnose skill.

Usage:
    python .claude/skills/diagnose/scripts/diagnose_entity.py <entity> <candidate> [language]

Shows the per-pair signals Stage E (`pipeline/entity_corrector.py`) would use
to gate a specific entity → candidate substitution: fuzz signals, reduced-word
fuzz, MCQ pre-gates, and the conservative validation gates.

NOTE: This does NOT replicate full Stage E routing. The real pipeline retrieves
candidates via TF-IDF char-bigram cosine over the entire match gazetteer (5283
canonicals across leagues), then routes based on cosine AND fuzz together.
For an end-to-end decision, run `/test-match` and inspect Stage E telemetry.
"""
import os
import sys

# Add project root to path (4 levels up: scripts -> diagnose -> skills -> .claude -> root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from rapidfuzz import fuzz

from pipeline.config import (
    CONSERVATIVE_C1_FUZZY_FLOOR,
    CONSERVATIVE_C2_LEN_TOLERANCE,
    DICTIONARY_VETO_ENABLED,
    DICTIONARY_VETO_MIN_LEN,
    MCQ_MIN_FUZZ_TO_INVOKE,
    MCQ_MIN_TOKEN_LEN,
    MCQ_SHORT_TOKEN_MIN_FUZZ,
)
from pipeline.entity_corrector import SHORTCUT_ACCEPT_TFIDF, SHORTCUT_REJECT_TFIDF
from pipeline.fuzzy_corrector import passes_conservative_gates


def _reduced_word_fuzz(entity: str, canonical: str) -> tuple[str, float]:
    """If canonical is multi-word and entity is single token, fuzz against
    the closest canonical word (replicates entity_corrector._reduce_to_best_word).
    """
    if not entity or not canonical:
        return canonical, 0.0
    if " " in entity or " " not in canonical:
        return canonical, fuzz.ratio(entity.lower(), canonical.lower())
    best_word, best_score = canonical, -1.0
    for w in canonical.split():
        s = fuzz.ratio(entity.lower(), w.lower())
        if s > best_score:
            best_score = s
            best_word = w
    return best_word, best_score


def diagnose(entity: str, candidate: str, language: str = "en") -> dict:
    """Compute Stage E per-pair signals."""
    full_fuzz = fuzz.ratio(entity.lower(), candidate.lower())
    reduced_word, reduced_fuzz = _reduced_word_fuzz(entity, candidate)
    ent_len = len(entity.strip())

    short_block = (
        ent_len < MCQ_MIN_TOKEN_LEN and reduced_fuzz < MCQ_SHORT_TOKEN_MIN_FUZZ
    )
    fuzz_block = reduced_fuzz < MCQ_MIN_FUZZ_TO_INVOKE
    c1_passes = full_fuzz >= CONSERVATIVE_C1_FUZZY_FLOOR
    len_delta = abs(len(candidate) - len(entity))
    c2_limit = max(2, CONSERVATIVE_C2_LEN_TOLERANCE * len(entity))
    c2_passes = len_delta <= c2_limit
    gates_pass = passes_conservative_gates(entity, candidate, language=language)

    if short_block:
        verdict = f"PRE-MCQ REJECT (len {ent_len}<{MCQ_MIN_TOKEN_LEN}, reduced fuzz {reduced_fuzz:.0f}<{MCQ_SHORT_TOKEN_MIN_FUZZ})"
    elif fuzz_block:
        verdict = f"PRE-MCQ REJECT (reduced fuzz {reduced_fuzz:.0f}<{MCQ_MIN_FUZZ_TO_INVOKE})"
    elif not gates_pass:
        verdict = "VALIDATION REJECT (dictionary veto or C1/C2 gate)"
    else:
        verdict = (
            f"MCQ-ELIGIBLE — would go to Qwen MCQ judge if TF-IDF cosine "
            f"in [{SHORTCUT_REJECT_TFIDF:.2f}, {SHORTCUT_ACCEPT_TFIDF:.2f})"
        )

    return {
        "entity": entity,
        "candidate": candidate,
        "language": language,
        "full_fuzz": round(full_fuzz, 1),
        "reduced_canonical_word": reduced_word,
        "reduced_word_fuzz": round(reduced_fuzz, 1),
        "token_length": ent_len,
        "short_token_block": short_block,
        "fuzz_below_invoke_floor": fuzz_block,
        "c1_passes": c1_passes,
        "c1_threshold": CONSERVATIVE_C1_FUZZY_FLOOR,
        "c2_passes": c2_passes,
        "c2_len_delta": len_delta,
        "c2_limit": round(c2_limit, 1),
        "validation_gates_pass": gates_pass,
        "dictionary_veto_enabled": DICTIONARY_VETO_ENABLED,
        "dictionary_veto_min_len": DICTIONARY_VETO_MIN_LEN,
        "verdict": verdict,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diagnose_entity.py <entity> <candidate> [language=en]")
        sys.exit(1)

    entity = sys.argv[1]
    candidate = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) > 3 else "en"

    result = diagnose(entity, candidate, language)
    print("\n=== Stage E Diagnostic ===")
    for k, v in result.items():
        print(f"  {k:<28} {v}")
    print(
        "\nNote: real Stage E routing also depends on TF-IDF cosine over the "
        "match gazetteer.\n      Run `/test-match` on the affected match for "
        "an end-to-end decision.\n"
    )
