"""'Old-school' cleaning: pure fuzzy + lookup, no LLM, no transformer.

The hypothesis: if we go back to the cleaning style that worked BEFORE
LLM models were introduced (validated_corrections lookup + rapidfuzz),
we get most of the entity-F1 gain in seconds, not 37 minutes.

Algorithm per segment:
1. Lowercase normalisation pass (regex strip extra whitespace).
2. For every word: check validated_corrections.json — if exact-lower
   match, replace with canonical.
3. For every capitalised word ≥4 chars not in dict: rapidfuzz against
   gazetteer canonicals; if best ≥85, replace.
4. Apply CanonicalCaseFixer to restore canonical casing on all
   gazetteer entries.

That's it. No Qwen, no xlm-roberta, no oliverguhr punctuation.
Total per-match time: <30 seconds.

Output: cleaned_data/oldschool/{1,2}_asr_v3_nbest_cleaned.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
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
    p.add_argument("--output-dir", type=Path,
                   default=Path("cleaned_data/oldschool/caption-2023"))
    p.add_argument("--fuzzy-floor", type=int, default=85,
                   help="Min fuzz.ratio to apply replacement (default 85)")
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
    print(f"Match: {match_dir.name}")

    # Mirror cleaned_data structure: cleaned_data/oldschool/<league>/<season>/<match>/commentary_data
    rel = match_dir.relative_to(args.raw_root)
    out_match_dir = args.output_dir / rel / "commentary_data"
    out_match_dir.mkdir(parents=True, exist_ok=True)

    from rapidfuzz import fuzz
    from pipeline.gazetteer import build_gazetteer
    from pipeline.canonical_case_fixer import CanonicalCaseFixer

    labels = json.load(open(match_dir / "Labels-caption.json", encoding="utf-8"))
    gazetteer, entity_types = build_gazetteer(labels)
    canonicals = sorted({v for v in gazetteer.values() if v and len(v) >= 4})
    fixer = CanonicalCaseFixer(gazetteer=gazetteer)

    # Load validated_corrections cache
    val_path = REPO_ROOT / "data" / "validated_corrections.json"
    validated = {}
    if val_path.exists():
        raw = json.load(open(val_path, encoding="utf-8"))
        for k, v in raw.items():
            if isinstance(v, dict) and v.get("correct"):
                validated[k.lower()] = v["correct"]
    print(f"Loaded {len(validated)} validated mappings + {len(canonicals)} canonicals")

    total_validated_hits = 0
    total_fuzzy_hits = 0
    total_case_fixes = 0
    total_time = 0.0

    for half in (1, 2):
        rp = match_dir / "commentary_data" / f"{half}_asr{args.variant}.json"
        if not rp.exists():
            continue

        t_start = time.perf_counter()
        d = json.load(open(rp, encoding="utf-8"))
        out_segs = {}
        corrections = []

        for sid, seg in d.get("segments", {}).items():
            text = seg.get("text") if isinstance(seg, dict) else seg[2]
            start = seg.get("start") if isinstance(seg, dict) else seg[0]
            end = seg.get("end") if isinstance(seg, dict) else seg[1]
            original = text

            # 1. Validated lookup pass — word-by-word
            words = text.split()
            new_words = []
            for w in words:
                stripped = w.strip(".,;:!?\"'()[]{}").rstrip("'s").rstrip("'")
                key = stripped.lower()
                if key in validated and len(key) >= 4:
                    canonical = validated[key]
                    new_w = w.replace(stripped, canonical)
                    new_words.append(new_w)
                    if new_w != w:
                        total_validated_hits += 1
                        corrections.append({
                            "segment_id": sid, "half": half,
                            "original": stripped, "corrected": canonical,
                            "method": "validated_cache", "stage": "3",
                            "score": 100,
                        })
                    continue

                # 2. Fuzzy match against canonicals (single-word only, ≥4 chars)
                # Aggressive mode: also try lowercase tokens (Whisper sometimes
                # outputs lowercase player names mid-sentence)
                if (len(stripped) >= 4
                        and stripped.lower() not in {c.lower() for c in canonicals}):
                    best_canon = None
                    best_score = 0
                    for c in canonicals:
                        if " " in c:
                            continue  # only single-word canonicals here
                        if abs(len(c) - len(stripped)) > 3:
                            continue
                        s = fuzz.ratio(stripped.lower(), c.lower())
                        if s > best_score:
                            best_score = s
                            best_canon = c
                    if best_canon and best_score >= args.fuzzy_floor:
                        new_w = w.replace(stripped, best_canon)
                        new_words.append(new_w)
                        total_fuzzy_hits += 1
                        corrections.append({
                            "segment_id": sid, "half": half,
                            "original": stripped, "corrected": best_canon,
                            "method": "fuzzy_match", "stage": "3",
                            "score": best_score,
                        })
                        continue

                new_words.append(w)
            text = " ".join(new_words)

            # 3. Canonical case fix on the whole segment
            new_text = fixer.restore(text)
            if new_text != text:
                total_case_fixes += 1
            text = new_text

            out_segs[sid] = {
                "start_time": float(start), "end_time": float(end),
                "text": text,
                "start": float(start), "end": float(end),
            }

        out_path = out_match_dir / f"{half}_asr{args.variant}_cleaned.json"
        json.dump({
            "schema_version": 2,
            "language": d.get("language", "en"),
            "language_probability": d.get("language_probability", 1.0),
            "segments": out_segs,
            "cleaning_metadata": {
                "stage_timings": {"oldschool_total": round(time.perf_counter() - t_start, 3)},
                "corrections": corrections,
            },
        }, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

        elapsed = time.perf_counter() - t_start
        total_time += elapsed
        print(f"  Half {half}: {elapsed:.2f}s, {len(out_segs)} segs, "
              f"{len(corrections)} corrections")

    print()
    print(f"=== OLD-SCHOOL CLEANING SUMMARY ===")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Validated lookup hits: {total_validated_hits}")
    print(f"  Fuzzy match hits: {total_fuzzy_hits}")
    print(f"  Case fixes (segments touched): {total_case_fixes}")
    print()
    print(f"Output: {out_match_dir}")
    print(f"  Compare to V8 (full pipeline): ~37 min, F1 0.603")
    return 0


if __name__ == "__main__":
    sys.exit(main())
