"""Automatically harvest Whisper misspellings from raw V3 → grow
validated_corrections.json without manual review.

For every capitalised token in the raw transcript that is:
  (a) NOT already a canonical (i.e. needs correction)
  (b) NOT a real English word (per pyspellchecker dict)
  (c) Has fuzz.ratio ≥ 80 to a single gazetteer canonical
  (d) Occurs at least 2 times (signal of consistent Whisper error)

… add it to data/validated_corrections.json so the next oldschool
cleaning pass picks it up via the lookup table.

The point: this demonstrates the LEARNED_CORRECTIONS WIN — every match
we run grows the cache. After 50 matches, the cache contains every
high-frequency Whisper misspelling and oldschool cleaning becomes a
~zero-cost lookup operation that catches everything LLM-based pipelines
catch, but in milliseconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--match", default="Chelsea 1 - 2 Liverpool")
    p.add_argument("--variant", default="_v3_nbest")
    p.add_argument("--raw-root", type=Path,
                   default=Path("path/to/SoccerNet/caption-2023"))
    p.add_argument("--cache", type=Path,
                   default=REPO_ROOT / "data" / "validated_corrections.json")
    p.add_argument("--fuzz", type=int, default=80,
                   help="Minimum fuzz.ratio to harvest")
    p.add_argument("--min-occ", type=int, default=2,
                   help="Token must appear at least N times")
    p.add_argument("--apply", action="store_true",
                   help="Write to validated_corrections.json (default: dry run)")
    args = p.parse_args()

    # Find match
    match_dir = None
    for league in args.raw_root.iterdir():
        if league.is_dir():
            for s in league.iterdir():
                if s.is_dir():
                    for m in s.iterdir():
                        if m.is_dir() and args.match.lower() in m.name.lower():
                            match_dir = m
    if not match_dir:
        print("match not found", file=sys.stderr); return 1
    print(f"Harvesting from: {match_dir.name}")

    from rapidfuzz import fuzz
    from pipeline.gazetteer import build_gazetteer

    labels = json.load(open(match_dir / "Labels-caption.json", encoding="utf-8"))
    gaz, _ = build_gazetteer(labels)
    canonicals = sorted({v for v in gaz.values() if v and len(v) >= 4})
    canon_lower = {c.lower() for c in canonicals}
    canon_words_lower = set()
    for c in canonicals:
        for w in c.split():
            if len(w) >= 4:
                canon_words_lower.add(w.lower())

    # Try pyspellchecker for English-word veto
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker(language="en")
        def is_english_word(w):
            return w.lower() in spell
    except Exception:
        def is_english_word(w):
            return False

    # Tokenise raw text
    tok_freq: Counter = Counter()
    for half in (1, 2):
        rp = match_dir / "commentary_data" / f"{half}_asr{args.variant}.json"
        if not rp.exists():
            continue
        d = json.load(open(rp, encoding="utf-8"))
        for s in d.get("segments", {}).values():
            text = s.get("text") if isinstance(s, dict) else s[2]
            for w in text.split():
                clean = w.strip(".,;:!?\"'()[]{}").rstrip("'s").rstrip("'")
                if len(clean) >= 4 and clean[0].isupper():
                    tok_freq[clean] += 1
    print(f"  {len(tok_freq)} unique capitalised tokens (≥4 chars)")

    # Score each token: best canonical fuzz match
    proposals = []
    for tok, cnt in tok_freq.items():
        if cnt < args.min_occ:
            continue
        if tok.lower() in canon_words_lower:
            continue   # already a canonical word
        if is_english_word(tok):
            continue   # real English word, don't replace

        best_canon = None
        best_score = 0
        for c in canonicals:
            # Single-word canonicals only for simple substitution
            if " " in c:
                continue
            if abs(len(c) - len(tok)) > 3:
                continue
            s = fuzz.ratio(tok.lower(), c.lower())
            if s > best_score:
                best_score = s
                best_canon = c
        if best_canon and best_score >= args.fuzz:
            proposals.append({
                "token": tok, "key": tok.lower(),
                "canonical": best_canon,
                "fuzz": best_score, "occ": cnt,
                "impact": best_score * cnt,
            })

    proposals.sort(key=lambda x: -x["impact"])
    print(f"  {len(proposals)} new candidate mappings (fuzz≥{args.fuzz}, occ≥{args.min_occ}):")
    print()
    print(f"  {'TOKEN':<20s} → {'CANONICAL':<22s} fuzz  occ  impact")
    for p in proposals[:30]:
        print(f"  {p['token']:<20s} → {p['canonical']:<22s} "
              f"{p['fuzz']:>4d} {p['occ']:>4d}  {p['impact']:>5d}")

    if not args.apply:
        print()
        print(f"DRY RUN. Re-run with --apply to merge into {args.cache}.")
        return 0

    # Apply
    cache = json.load(open(args.cache, encoding="utf-8")) if args.cache.exists() else {}
    added = 0
    for p in proposals:
        if p["key"] in cache:
            continue
        cache[p["key"]] = {
            "correct": p["canonical"],
            "fuzzy_avg": float(p["fuzz"]),
            "last_updated": "2026-05-15",
            "matches_seen": [f"auto_harvest:{match_dir.name}"],
        }
        added += 1
    json.dump(cache, open(args.cache, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\n  Added {added} new mappings → {args.cache}")
    print(f"  Total cache size: {len(cache)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
