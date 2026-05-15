"""Test 'lightweight cleaning' — Stage E + canonical case fix only,
without Step L (Qwen GER) or Step P (oliverguhr punctuation).

Hypothesis: most of V8's Entity-F1 gain comes from
  (a) Step E entity correction (which is fast — ~1 min)
  (b) Step P's casing restoration on the canonical names (case-sensitive F1)

If (b) can be replaced by a regex-based canonical-case-fixer, we get
the same F1 gain in 5-7 minutes instead of 37 minutes per match.

This script:
1. Loads raw V3 transcript
2. Runs Stage E + canonical case fix
3. Writes the result as a 'cleaned' output
4. Computes WER + Entity-F1 vs GOAL GT
5. Compares to V8 (full pipeline) and V5 (Stage E only)
"""

from __future__ import annotations

import argparse
import json
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
                   default=Path("cleaned_data/lightweight"))
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from pipeline.loader import load_asr_json, Segment
    from pipeline.gazetteer import build_gazetteer
    from pipeline.hallucination_filter import filter_segments, detect_commentary_language
    from pipeline.deduplicator import deduplicate_segments
    from pipeline.domain_normalizer import DomainNormalizer
    from pipeline.ner_extractor import extract_entities_batch
    from pipeline.entity_corrector import correct_match as entity_correct_match
    from pipeline.canonical_case_fixer import CanonicalCaseFixer
    from pipeline.temporal_chunker import generate_match_id

    labels = json.load(open(match_dir / "Labels-caption.json", encoding="utf-8"))
    gazetteer, entity_types = build_gazetteer(labels)

    timings = {}
    total_corrections = 0
    case_corrections = 0

    for half in (1, 2):
        rp = match_dir / "commentary_data" / f"{half}_asr{args.variant}.json"
        if not rp.exists():
            continue

        t_start = time.perf_counter()

        # === Step 1-3: Filter + dedup + normalize ===
        segments = load_asr_json(rp, half=half)
        lang = detect_commentary_language(segments)
        valid, _ = filter_segments(segments, expected_lang=lang)
        deduped, _ = deduplicate_segments(valid)
        normalizer = DomainNormalizer(lang)
        normalized, _ = normalizer.normalize_batch(deduped)
        t_step123 = time.perf_counter() - t_start

        # === Step E: entity correction ===
        t = time.perf_counter()
        ent_map = extract_entities_batch(normalized, language=lang, gazetteer=gazetteer)
        match_id = generate_match_id("test", "lightweight", match_dir.name)
        corrected, ent_corrections = entity_correct_match(
            segments=normalized, gazetteer=gazetteer, entity_types=entity_types,
            segment_entities_map=ent_map, match_id=match_id,
            match_name=match_dir.name, language=lang,
        )
        t_stepE = time.perf_counter() - t
        total_corrections += len(ent_corrections)

        # === NEW Step P-lite: canonical case fixer ===
        t = time.perf_counter()
        fixer = CanonicalCaseFixer(gazetteer=gazetteer)
        case_fixed, case_corr = fixer.restore_batch(corrected)
        t_caseFix = time.perf_counter() - t
        case_corrections += len(case_corr)

        # Save as schema-2 cleaned output for evaluate_wer.py to read
        out_path = args.output_dir / f"{half}_asr{args.variant}_cleaned.json"
        out_data = {
            "schema_version": 2,
            "language": lang,
            "language_probability": 1.0,
            "segments": {s.segment_id: {
                "start": s.start_time, "end": s.end_time, "text": s.text,
                "start_time": s.start_time, "end_time": s.end_time,
            } for s in case_fixed},
            "cleaning_metadata": {
                "stage_timings": {
                    "step123_pre": round(t_step123, 2),
                    "stepE_entity": round(t_stepE, 2),
                    "stepP_lite_case": round(t_caseFix, 2),
                    "total_pipeline": round(t_step123 + t_stepE + t_caseFix, 2),
                },
                "corrections": [
                    {"segment_id": c["segment_id"], "original": c["original"],
                     "corrected": c["corrected"], "method": c.get("method"),
                     "stage": "3", "score": c.get("score", 0)}
                    for c in ent_corrections
                ],
                "case_fixes": case_corr,
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)

        timings[f"half{half}_total"] = round(t_step123 + t_stepE + t_caseFix, 2)
        timings[f"half{half}_stepE"] = round(t_stepE, 2)
        timings[f"half{half}_caseFix"] = round(t_caseFix, 2)

    print()
    print("=== Lightweight cleaning timings ===")
    for k, v in timings.items():
        print(f"  {k}: {v}s")
    print(f"  Total entity corrections: {total_corrections}")
    print(f"  Total case fixes: {case_corrections}")
    print()
    print(f"Output dir: {args.output_dir}")
    print(f"  Compare to V8 (full pipeline): ~37 min, F1 0.603")
    print(f"  Now run: python tools/evaluate_wer.py --match \"...\" "
          f"with cleaned-root={args.output_dir.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
