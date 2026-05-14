"""Count surface-form variants per canonical entity in RAW vs CLEANED.

This is the test that DOES expose cleaning value. Modern fuzzy
retrieval and LLM RAG paper over surface variants at *query time*.
But knowledge-graph aggregation, event-extraction systems, and
analytics dashboards aggregate by *exact entity ID* — and there a
single player must collapse into a single surface form, or the same
player gets 4-7 separate buckets in your "top scorers" report.

For each canonical name in the gazetteer, we:
  1. Find all surface forms in the RAW transcript that fuzz-match it ≥70.
  2. Find all surface forms in the CLEANED transcript that fuzz-match it ≥70.
  3. Report the variant count and the actual surface forms seen.

Output: thesis/entity_variant_counts.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_text(path: Path) -> str:
    """Concatenate all segment text from a schema-2 ASR JSON."""
    if not path.exists():
        return ""
    d = json.load(open(path, encoding="utf-8"))
    parts = []
    for sid, seg in d.get("segments", {}).items():
        if isinstance(seg, dict):
            parts.append(seg.get("text", ""))
        elif isinstance(seg, list) and len(seg) >= 3:
            parts.append(seg[2])
    return " ".join(parts)


def _strip_punct(s: str) -> str:
    """Strip ALL punctuation, keep only letters + spaces. Lowercased."""
    out = []
    for ch in s:
        if ch.isalpha() or ch == " ":
            out.append(ch)
    return "".join(out).strip().lower()


def find_variants(text: str, canonical: str, min_fuzz: int = 70) -> Counter:
    """Find capitalised tokens / bigrams in text that fuzz-match canonical.

    Variants are normalised case + punctuation-stripped, so that
    'Sturridge', 'Sturridge.', 'Sturridge,' all count as the same form.
    """
    from rapidfuzz import fuzz
    variants = Counter()
    words = text.split()
    canon_norm = _strip_punct(canonical)

    n_canon = len(canonical.split())
    if n_canon == 1:
        for w in words:
            norm = _strip_punct(w)
            if not norm or not w[0].isupper():
                continue
            if abs(len(norm) - len(canon_norm)) > 4:
                continue
            score = fuzz.ratio(norm, canon_norm)
            if score >= min_fuzz:
                variants[norm] += 1
    elif n_canon == 2:
        # Multi-word canonical: scan bigrams
        for i in range(len(words) - 1):
            bg = " ".join(words[i:i+2])
            norm = _strip_punct(bg)
            if not norm or not words[i][0].isupper():
                continue
            if abs(len(norm) - len(canon_norm)) > 5:
                continue
            score = fuzz.ratio(norm, canon_norm)
            if score >= min_fuzz:
                variants[norm] += 1
    return variants


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--match", required=True)
    p.add_argument("--variant", default="_v3_nbest")
    p.add_argument("--cleaned-root", type=Path,
                   default=Path("cleaned_data/caption-2023"))
    p.add_argument("--raw-root", type=Path,
                   default=Path("path/to/SoccerNet/caption-2023"))
    p.add_argument("--out", type=Path, default=Path("thesis/entity_variant_counts.md"))
    args = p.parse_args()

    # Find dirs
    raw_dir = None
    cleaned_dir = None
    for league in args.raw_root.iterdir():
        if league.is_dir():
            for s in league.iterdir():
                if s.is_dir():
                    for m in s.iterdir():
                        if m.is_dir() and args.match.lower() in m.name.lower():
                            raw_dir = m
    for league in args.cleaned_root.iterdir():
        if league.is_dir():
            for s in league.iterdir():
                if s.is_dir():
                    for m in s.iterdir():
                        if m.is_dir() and args.match.lower() in m.name.lower():
                            cleaned_dir = m
    if not raw_dir or not cleaned_dir:
        print("dirs not found", file=sys.stderr); return 1

    from pipeline.gazetteer import build_gazetteer
    labels = json.load(open(raw_dir / "Labels-caption.json", encoding="utf-8"))
    gaz, etypes = build_gazetteer(labels)

    # Aggregate text from both halves
    raw_text = " ".join(
        load_text(raw_dir / "commentary_data" / f"{h}_asr{args.variant}.json")
        for h in (1, 2)
    )
    cleaned_text = " ".join(
        load_text(cleaned_dir / "commentary_data" / f"{h}_asr{args.variant}_cleaned.json")
        for h in (1, 2)
    )

    canonicals = sorted({v for v in gaz.values() if v and len(v) >= 4})

    md = ["# Entity Surface-Form Variant Counts: RAW vs CLEANED",
          "",
          f"Match: **{raw_dir.name}**",
          "",
          "For each canonical lineup name, we count how many distinct *surface",
          "forms* of that name appear in the transcript (fuzz.ratio ≥ 70 to",
          "the canonical). A downstream event-aggregation or knowledge-graph",
          "system buckets events by exact surface form — so 5 surface variants",
          "of the same player means 5 broken event-buckets per match.",
          "",
          "**This is the cleaning-pipeline value that retrieval and LLM-RAG",
          "benchmarks miss.** ES fuzzy AUTO bridges variants at query time;",
          "an analytics pipeline aggregating by exact match cannot.",
          "",
          "| Canonical | RAW variants | RAW total mentions | CLEANED variants | CLEANED total | Δ variants |",
          "|---|---|---|---|---|---|"]

    summary_raw_variants = 0
    summary_clean_variants = 0
    rows_with_change = 0
    full_rows = []
    for c in canonicals:
        raw_v = find_variants(raw_text, c)
        clean_v = find_variants(cleaned_text, c)
        n_raw = len(raw_v)
        n_clean = len(clean_v)
        tot_raw = sum(raw_v.values())
        tot_clean = sum(clean_v.values())
        if tot_raw == 0 and tot_clean == 0:
            continue
        delta = n_clean - n_raw
        marker = ""
        if delta < 0:
            marker = "✓ collapsed"
            rows_with_change += 1
        elif delta > 0:
            marker = "⚠ expanded"
        summary_raw_variants += n_raw
        summary_clean_variants += n_clean
        md.append(f"| `{c}` | **{n_raw}** | {tot_raw} | **{n_clean}** | {tot_clean} | {delta:+d} {marker} |")
        full_rows.append((c, raw_v, clean_v))

    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(f"- Total surface variants across all canonicals: **RAW {summary_raw_variants}** vs **CLEANED {summary_clean_variants}**")
    md.append(f"  (Δ = **{summary_clean_variants - summary_raw_variants:+d}** variants)")
    md.append(f"- Canonicals where cleaning *collapsed* variants: **{rows_with_change}**")
    md.append("")

    md.append("## Detailed variant breakdown (top 15 most-affected canonicals)")
    md.append("")
    full_rows.sort(key=lambda r: -(len(r[1]) - len(r[2])))
    for c, raw_v, clean_v in full_rows[:15]:
        only_raw = set(raw_v) - set(clean_v)
        only_clean = set(clean_v) - set(raw_v)
        if not only_raw and not only_clean:
            continue
        md.append(f"### `{c}` (raw {len(raw_v)} variants → cleaned {len(clean_v)} variants)")
        md.append("")
        md.append("| RAW only | CLEANED only | Both |")
        md.append("|---|---|---|")
        rows_only = sorted(only_raw, key=lambda x: -raw_v.get(x, 0))[:8]
        clean_only = sorted(only_clean, key=lambda x: -clean_v.get(x, 0))[:8]
        both = set(raw_v) & set(clean_v)
        rows_both = sorted(both, key=lambda x: -raw_v.get(x, 0))[:5]
        max_rows = max(len(rows_only), len(clean_only), len(rows_both), 1)
        for i in range(max_rows):
            ro = f"`{rows_only[i]}` ({raw_v[rows_only[i]]})" if i < len(rows_only) else ""
            co = f"`{clean_only[i]}` ({clean_v[clean_only[i]]})" if i < len(clean_only) else ""
            bo = f"`{rows_both[i]}` ({raw_v[rows_both[i]]}→{clean_v[rows_both[i]]})" if i < len(rows_both) else ""
            md.append(f"| {ro} | {co} | {bo} |")
        md.append("")

    md.append("## Why this matters for event aggregation / knowledge-graph systems")
    md.append("")
    md.append("Take a use-case: \"Show me the top scorers in this match.\" The")
    md.append("downstream event-extraction system will:")
    md.append("")
    md.append("1. NER over each segment → list of PERSON mentions")
    md.append("2. Group mentions by exact surface string (or by canonical ID")
    md.append("   if it has one)")
    md.append("3. Count goal-events per group")
    md.append("")
    md.append("With **N surface variants per player**, the same player splits")
    md.append("into N rows in the top-scorers list — `Daniel Sturridge: 1 goal`,")
    md.append("`Sturridge: 1 goal`, `Starridge: 0 goals`, `Daniel Klain: 0 goals`")
    md.append("— and the analytics is broken.")
    md.append("")
    md.append("Cleaning collapses variants → **1 row per player → correct counts.**")
    md.append("ES fuzzy doesn't help because the aggregation step happens AFTER")
    md.append("retrieval, on the raw text. LLM RAG doesn't help because the LLM")
    md.append("answers single questions, it doesn't run the analytics pipeline.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {args.out}")
    print()
    print(f"Total surface variants RAW {summary_raw_variants} -> CLEANED {summary_clean_variants}")
    print(f"({summary_clean_variants - summary_raw_variants:+d} variants — "
          f"{rows_with_change} canonicals had variants collapsed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
