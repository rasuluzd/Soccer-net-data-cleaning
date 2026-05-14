"""Dump every entity NER+heuristics detected in the match for human verification.

Brukeren vil verifisere at vi ikke overteller. Dette skriptet kjører
samme NER-pipeline som orchestrator (extract_entities_batch), men
dumper hver detected entity til en CSV med:
  - segment_id, half, time, entity text, source (spaCy/heuristic/gazetteer-fuzz)
  - duplikatflagg (samme tekst sett tidligere i samme kamp)
  - kontekst (5 ord rundt)

Også gir den oversikt:
  - Total entities
  - Unique entities (case-folded)
  - Top-50 most frequent entity tokens
  - Per-source breakdown
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
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
    p.add_argument("--out", type=Path, default=Path("thesis/detected_entities.csv"))
    p.add_argument("--summary", type=Path, default=Path("thesis/detected_entities_summary.md"))
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

    from pipeline.loader import Segment
    from pipeline.gazetteer import build_gazetteer
    from pipeline.ner_extractor import extract_entities_batch

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
    print(f"[dump] {len(all_segs)} segments loaded", file=sys.stderr)

    ent_map = extract_entities_batch(all_segs, language="en", gazetteer=gaz)

    # Build segment lookup
    seg_lookup = {(s.half, s.segment_id): s for s in all_segs}

    rows = []
    seen_per_match = set()
    src_count: Counter = Counter()
    for (half, sid), ents in ent_map.items():
        seg = seg_lookup.get((half, sid))
        if not seg:
            continue
        for ent in ents:
            etoken = ent.text.strip(".,;:!?\"'()[]{}")
            key = etoken.lower()
            is_dup_in_match = key in seen_per_match
            seen_per_match.add(key)
            src = getattr(ent, "source", "spacy")  # spacy / heuristic / gazetteer_fuzz
            src_count[src] += 1
            rows.append({
                "half": half,
                "segment_id": sid,
                "start_time": seg.start_time,
                "entity": etoken,
                "source": src,
                "is_dup_in_match": is_dup_in_match,
                "context": seg.text[:120],
            })

    print(f"[dump] {len(rows)} total entity occurrences detected", file=sys.stderr)

    # CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["half", "segment_id", "start_time",
                                          "entity", "source", "is_dup_in_match",
                                          "context"])
        w.writeheader()
        w.writerows(rows)

    # Summary markdown
    unique_tokens = set(r["entity"].lower() for r in rows)
    canonical_set = {v.lower() for v in gaz.values() if v}
    in_gaz = sum(1 for r in rows if r["entity"].lower() in canonical_set)
    out_of_gaz = len(rows) - in_gaz

    tok_freq = Counter(r["entity"] for r in rows)
    md = []
    md.append("# Detected Entities — Verification Report")
    md.append("")
    md.append(f"Match: **{raw_dir.name}**")
    md.append(f"Source variant: `{args.variant}`")
    md.append(f"Gazetteer canonicals: **{len(canonical_set)}**")
    md.append("")
    md.append("## Counts")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| Total entity occurrences | **{len(rows)}** |")
    md.append(f"| Unique entity tokens (case-folded) | **{len(unique_tokens)}** |")
    md.append(f"| Already-canonical occurrences | {in_gaz} ({in_gaz/max(1,len(rows))*100:.0f}%) |")
    md.append(f"| Non-canonical (potential corrections) | {out_of_gaz} ({out_of_gaz/max(1,len(rows))*100:.0f}%) |")
    md.append("")
    md.append("**Note**: \"occurrences\" counts every detection — a player "
              "name said 20 times produces 20 rows. This is by design: "
              "Stage E's per-match cache short-circuits repeats so the "
              "MCQ judge only runs once per unique (entity, top-3-cands) tuple.")
    md.append("")
    md.append("## By detection source")
    md.append("")
    md.append("| Source | Count |")
    md.append("|---|---|")
    for src, n in src_count.most_common():
        md.append(f"| {src} | {n} |")
    md.append("")
    md.append("## Top 50 most-frequent detected entity tokens")
    md.append("")
    md.append("| Token | Count | In gazetteer? |")
    md.append("|---|---|---|")
    for tok, n in tok_freq.most_common(50):
        in_g = "✓" if tok.lower() in canonical_set else "✗"
        md.append(f"| `{tok}` | {n} | {in_g} |")
    md.append("")
    md.append(f"## Full data")
    md.append("")
    md.append(f"All {len(rows)} occurrences are in `{args.out}` "
              f"(open in Excel / VS Code). Filter `is_dup_in_match=True` "
              f"to see only first occurrence of each unique entity.")

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[dump] wrote {args.out} ({len(rows)} rows)", file=sys.stderr)
    print(f"[dump] wrote {args.summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
