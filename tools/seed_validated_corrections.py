"""Scan raw V3 for high-confidence Whisper misspellings of lineup names
and seed validated_corrections.json with cure-rated mappings.

Logic per token in raw transcription:
  1. Skip if token IS already a gazetteer canonical (lowercase compare).
  2. Skip if token is a real common English word (pyenchant dict).
  3. Skip if token appears < MIN_OCCURRENCES times (rare → maybe genuine).
  4. Compute fuzz.ratio against every gazetteer canonical.
  5. If best fuzz ≥ FUZZ_THRESHOLD AND len(token) ≥ MIN_LEN:
       propose mapping (token.lower → best_canonical).

Output:
  - Print proposed mappings sorted by fuzz × occurrence_count.
  - With --apply: merge into data/validated_corrections.json.

Used to bootstrap entity F1 on a new match before iterative pipeline runs
have a chance to discover and cache the same mappings via MCQ judge.
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
    p.add_argument("--match", required=True)
    p.add_argument("--variant", default="_v3_nbest")
    p.add_argument("--raw-root", type=Path,
                   default=Path("path/to/SoccerNet/caption-2023"))
    p.add_argument("--cache", type=Path,
                   default=Path("data/validated_corrections.json"))
    p.add_argument("--fuzz", type=int, default=80,
                   help="Minimum fuzz.ratio to propose (default 80)")
    p.add_argument("--min-len", type=int, default=5,
                   help="Minimum token length")
    p.add_argument("--min-occ", type=int, default=2,
                   help="Token must appear at least this many times")
    p.add_argument("--apply", action="store_true",
                   help="Merge into validated_corrections.json")
    args = p.parse_args()

    # Find match
    raw_dir = None
    for league in args.raw_root.iterdir():
        if not league.is_dir():
            continue
        for season in league.iterdir():
            if not season.is_dir():
                continue
            for m in season.iterdir():
                if m.is_dir() and args.match.lower() in m.name.lower():
                    raw_dir = m
                    break
    if not raw_dir:
        print(f"match dirs not found", file=sys.stderr)
        return 1
    print(f"[seed] match: {raw_dir.name}")

    from rapidfuzz import fuzz
    from pipeline.gazetteer import build_gazetteer

    labels = json.load(open(raw_dir / "Labels-caption.json", encoding="utf-8"))
    gaz, etypes = build_gazetteer(labels)
    canonicals = sorted({v for v in gaz.values() if v and len(v) >= 3})
    canonical_lower = {c.lower() for c in canonicals}
    print(f"[seed] {len(canonicals)} canonical names in gazetteer")

    # Tokenise raw
    tok_freq: Counter = Counter()
    for half in (1, 2):
        rp = raw_dir / "commentary_data" / f"{half}_asr{args.variant}.json"
        if not rp.exists():
            continue
        d = json.load(open(rp, encoding="utf-8"))
        for s in d["segments"].values():
            text = s.get("text") if isinstance(s, dict) else s[2]
            for w in text.split():
                clean = w.strip(".,;:!?\"'()[]{}").rstrip("'s").rstrip("'")
                if len(clean) >= args.min_len and clean[0].isupper():
                    tok_freq[clean] += 1

    print(f"[seed] {len(tok_freq)} candidate tokens (capitalised, ≥{args.min_len} chars)")

    # Dictionary for veto
    try:
        import enchant
        en_dict = enchant.Dict("en_US")
    except Exception:
        en_dict = None

    proposals = []
    for tok, cnt in tok_freq.items():
        if cnt < args.min_occ:
            continue
        if tok.lower() in canonical_lower:
            continue
        # Skip real English common words (we'd be re-mapping them to a name)
        if en_dict and en_dict.check(tok):
            continue

        # Best fuzz match
        best = None
        best_score = 0
        for c in canonicals:
            s = fuzz.ratio(tok, c)
            if s > best_score:
                best_score = s
                best = c

        if best and best_score >= args.fuzz:
            proposals.append({
                "token": tok,
                "key": tok.lower(),
                "canonical": best,
                "fuzz": best_score,
                "occ": cnt,
                "score": best_score * cnt,  # rank by total impact
            })

    proposals.sort(key=lambda x: -x["score"])
    print(f"\n[seed] {len(proposals)} candidate mappings (fuzz≥{args.fuzz}, occ≥{args.min_occ}):\n")
    print(f"  {'TOKEN':<20s} {'CANONICAL':<25s} {'FUZZ':>4s} {'OCC':>4s}")
    for p in proposals[:60]:
        print(f"  {p['token']:<20s} {p['canonical']:<25s} {p['fuzz']:>4.1f} {p['occ']:>4d}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to merge into "
              f"{args.cache}.")
        return 0

    # Apply: merge into validated_corrections.json with matches_seen=['seed']
    cache = json.load(open(args.cache, encoding="utf-8"))
    added = 0
    for p in proposals:
        key = p["key"]
        if key in cache:
            continue
        cache[key] = {
            "correct": p["canonical"],
            "fuzzy_avg": float(p["fuzz"]),
            "last_updated": "2026-05-14",
            "matches_seen": [f"seed:{raw_dir.name}"],
        }
        added += 1
    json.dump(cache, open(args.cache, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\n[seed] Added {added} new mappings → {args.cache} "
          f"(total entries: {len(cache)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
