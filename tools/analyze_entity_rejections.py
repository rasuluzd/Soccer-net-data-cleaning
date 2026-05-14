"""Analyze why Stage E rejected entity-correction candidates.

For each detected entity in the match, runs the same TF-IDF retrieval
the pipeline used and reports which gate would have rejected the
correction. Useful for tuning thresholds or seeding the validated cache.

Output sections:
  1. Accepted corrections (already in cleaning_metadata)
  2. Rejected at shortcut-reject (cosine < 0.40)
  3. Rejected at MCQ pre-gates (length / fuzz too low)
  4. Rejected at MCQ judge (D=keep / E=unsure)
  5. Rejected at MLM veto
  6. Rejected at validation gates (dict veto / fuzzy floor / length tol)
  7. Comparison vs OLD learned_dictionary.json corrections (if present)

Usage:
    python tools/analyze_entity_rejections.py \\
        --match "Chelsea 1 - 2 Liverpool" \\
        --out thesis/entity_rejection_analysis.md
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


def _find_match_dir(name_substr: str, root: Path) -> Path | None:
    for league in root.iterdir():
        if not league.is_dir():
            continue
        for season in league.iterdir():
            if not season.is_dir():
                continue
            for match in season.iterdir():
                if match.is_dir() and name_substr.lower() in match.name.lower():
                    return match
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--match", required=True)
    p.add_argument("--cleaned-root", type=Path, default=Path("cleaned_data"))
    p.add_argument("--raw-root", type=Path,
                   default=Path("path/to/SoccerNet/caption-2023"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", default="_v3_nbest")
    args = p.parse_args()

    raw_dir = _find_match_dir(args.match, args.raw_root)
    cleaned_dir = _find_match_dir(args.match, args.cleaned_root)
    if not raw_dir or not cleaned_dir:
        print(f"match dirs not found", file=sys.stderr)
        return 1

    md = ["# Entity Correction Analysis — Stage E rejection breakdown",
          "",
          f"Match: **{raw_dir.name}**",
          ""]

    accepted_total = []
    methods = Counter()
    rejection_examples: dict[str, list] = {}

    for half in (1, 2):
        cp = cleaned_dir / "commentary_data" / f"{half}_asr_cleaned.json"
        if not cp.exists():
            continue
        cleaned = json.load(open(cp, encoding="utf-8"))
        meta = cleaned.get("cleaning_metadata", {})
        for c in meta.get("corrections", []):
            methods[c.get("method", "?")] += 1
            accepted_total.append({
                "half": half,
                "segment_id": c.get("segment_id"),
                "original": c.get("original"),
                "corrected": c.get("corrected"),
                "method": c.get("method"),
                "score": c.get("score"),
            })

    md.append(f"## 1. Accepted Stage E corrections ({len(accepted_total)})")
    md.append("")
    if accepted_total:
        md.append("| Half | Seg | Original → Corrected | Method | Score |")
        md.append("|---|---|---|---|---|")
        for c in accepted_total:
            md.append(f"| {c['half']} | {c['segment_id']} | "
                      f"`{c['original']}` → `{c['corrected']}` | {c['method']} | {c['score']} |")
        md.append("")
        md.append(f"**By method:** {dict(methods)}")
    else:
        md.append("*(none)*")
    md.append("")

    # Now reproduce the entity-correction pipeline to see what was rejected.
    print("[analyze] running TF-IDF + MCQ trace …", file=sys.stderr)
    from pipeline.loader import Segment
    from pipeline.gazetteer import build_gazetteer
    from pipeline.ner_extractor import extract_entities_batch
    from pipeline.fuzzy_corrector import passes_conservative_gates
    from pipeline.config import (
        MCQ_MIN_TOKEN_LEN, MCQ_MIN_FUZZ_TO_INVOKE, MCQ_SHORT_TOKEN_MIN_FUZZ,
        FREQUENCY_HEURISTIC_THRESHOLD,
    )
    from pipeline.entity_corrector import (
        SHORTCUT_REJECT_TFIDF, SHORTCUT_ACCEPT_TFIDF, SHORTCUT_ACCEPT_GAP,
        TOP_K_CANDIDATES,
    )
    from rapidfuzz import fuzz
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Load raw + labels
    labels = json.load(open(raw_dir / "Labels-caption.json", encoding="utf-8"))
    gaz, etypes = build_gazetteer(labels)

    all_segs: list[Segment] = []
    for half in (1, 2):
        rp = raw_dir / "commentary_data" / f"{half}_asr{args.variant}.json"
        if not rp.exists():
            continue
        d = json.load(open(rp, encoding="utf-8"))
        for sid, v in d["segments"].items():
            text = v.get("text") if isinstance(v, dict) else (v[2] if len(v) >= 3 else "")
            start = v.get("start") if isinstance(v, dict) else v[0]
            end = v.get("end") if isinstance(v, dict) else v[1]
            all_segs.append(Segment(
                segment_id=str(sid), start_time=float(start), end_time=float(end),
                text=str(text), half=half,
            ))

    ent_map = extract_entities_batch(all_segs, language="en", gazetteer=gaz)
    canonicals = sorted({v for v in gaz.values() if v})
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(canonicals)
    cm = vec.transform(canonicals)

    # Token frequency for the frequency heuristic
    tok_freq: Counter = Counter()
    for s in all_segs:
        for t in s.text.split():
            tok_freq[t.strip(".,;:!?\"'()[]{}").lower()] += 1

    rejected: dict[str, list] = {
        "shortcut_reject_low_cosine": [],
        "frequency_heuristic": [],
        "mcq_pregate_short_low_fuzz": [],
        "mcq_pregate_low_fuzz": [],
        "validation_dict_veto": [],
        "validation_c1_fuzzy_floor": [],
        "validation_c2_length_tolerance": [],
    }

    accepted_pairs = {(c["original"].lower(), c["corrected"].lower()) for c in accepted_total}

    seen = set()
    for (half, sid), ents in ent_map.items():
        for ent in ents:
            etoken = ent.text.strip(".,;:!?\"'()[]{}")
            key = (etoken.lower(),)
            if key in seen:
                continue
            seen.add(key)

            # Skip if already a canonical name
            if etoken.lower() in {c.lower() for c in canonicals}:
                continue

            # TF-IDF retrieve
            qv = vec.transform([etoken])
            sims = cosine_similarity(qv, cm)[0]
            top_idx = sims.argsort()[::-1][:TOP_K_CANDIDATES]
            top_cands = [(canonicals[i], float(sims[i])) for i in top_idx]
            best_cand, best_cos = top_cands[0]
            top_fuzz = fuzz.ratio(etoken, best_cand)

            # Was this accepted?
            if (etoken.lower(), best_cand.lower()) in accepted_pairs:
                continue

            # Frequency heuristic
            if tok_freq[etoken.lower()] >= FREQUENCY_HEURISTIC_THRESHOLD:
                rejected["frequency_heuristic"].append({
                    "token": etoken, "best": best_cand, "freq": tok_freq[etoken.lower()],
                })
                continue

            # Shortcut reject
            if best_cos < SHORTCUT_REJECT_TFIDF:
                rejected["shortcut_reject_low_cosine"].append({
                    "token": etoken, "best": best_cand, "cosine": round(best_cos, 3),
                })
                continue

            # MCQ pre-gates
            if len(etoken) < MCQ_MIN_TOKEN_LEN and top_fuzz < MCQ_SHORT_TOKEN_MIN_FUZZ:
                rejected["mcq_pregate_short_low_fuzz"].append({
                    "token": etoken, "best": best_cand, "fuzz": top_fuzz, "cosine": round(best_cos, 3),
                })
                continue
            if top_fuzz < MCQ_MIN_FUZZ_TO_INVOKE:
                rejected["mcq_pregate_low_fuzz"].append({
                    "token": etoken, "best": best_cand, "fuzz": top_fuzz, "cosine": round(best_cos, 3),
                })
                continue

            # Validation gates (the would-be MCQ accept)
            ok = passes_conservative_gates(etoken, best_cand, language="en")
            if not ok:
                # Drill into which gate would have failed
                from rapidfuzz import fuzz as _f
                from pipeline.config import (
                    CONSERVATIVE_C1_FUZZY_FLOOR, CONSERVATIVE_C2_LEN_TOLERANCE,
                    DICTIONARY_VETO_MIN_LEN,
                )
                why = "unknown"
                # C1
                if _f.ratio(etoken, best_cand) < CONSERVATIVE_C1_FUZZY_FLOOR:
                    why = f"c1_fuzz<{CONSERVATIVE_C1_FUZZY_FLOOR}"
                    bucket = "validation_c1_fuzzy_floor"
                elif abs(len(etoken) - len(best_cand)) > max(2, CONSERVATIVE_C2_LEN_TOLERANCE * len(etoken)):
                    why = "c2_len_tol"
                    bucket = "validation_c2_length_tolerance"
                else:
                    why = "dict_veto"
                    bucket = "validation_dict_veto"
                rejected[bucket].append({
                    "token": etoken, "best": best_cand,
                    "fuzz": top_fuzz, "cosine": round(best_cos, 3), "why": why,
                })

    md.append("## 2. Rejection breakdown (would-be candidates)")
    md.append("")
    md.append("These tokens *passed* NER detection but were filtered before any"
              " correction was applied. Each row shows the top gazetteer canonical"
              " the TF-IDF retrieve picked, with the gate that rejected the pair.")
    md.append("")
    for bucket, items in rejected.items():
        if not items:
            continue
        md.append(f"### {bucket} — {len(items)} cases")
        md.append("")
        md.append("| Token | Top canonical | cosine | fuzz | extra |")
        md.append("|---|---|---|---|---|")
        for it in items[:25]:
            extra = it.get("why", "") or it.get("freq", "")
            md.append(f"| `{it['token']}` | `{it['best']}` | "
                      f"{it.get('cosine', '?')} | {it.get('fuzz', '?')} | {extra} |")
        if len(items) > 25:
            md.append(f"")
            md.append(f"*(showing 25 of {len(items)})*")
        md.append("")

    md.append("## 3. Summary counts")
    md.append("")
    md.append("| Bucket | Count |")
    md.append("|---|---|")
    md.append(f"| Accepted | {len(accepted_total)} |")
    for bucket, items in rejected.items():
        md.append(f"| {bucket} | {len(items)} |")
    md.append("")

    # OLD learned_dictionary comparison
    old_path = REPO_ROOT / "data" / "learned_corrections.json"
    if old_path.exists():
        old = json.load(open(old_path, encoding="utf-8"))
        md.append("## 4. OLD `learned_corrections.json` for context")
        md.append("")
        md.append(f"OLD pipeline shipped a learned cross-match dictionary with "
                  f"{len(old)} mappings. NEW's validated cache requires "
                  f"3-match consensus, so single-match runs see *zero* "
                  f"validated-cache hits (telemetry: `validated_cache_hit=0`).")
        md.append("")
        md.append("First 30 entries from the OLD dictionary:")
        md.append("")
        md.append("| Original | Corrected (and meta) |")
        md.append("|---|---|")
        for k, v in list(old.items())[:30]:
            md.append(f"| `{k}` | `{v}` |")
        md.append("")
    else:
        md.append("## 4. OLD `learned_corrections.json` not present in repo")
        md.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[analyze] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
