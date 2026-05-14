"""Compare search quality on RAW vs CLEANED Whisper output indexed in ES.

Runs a fixed query set against two Elasticsearch match indexes and
reports, per query:
  - Top-3 hit text (truncated)
  - BM25 score (joint hybrid score)
  - Whether the hit contains the canonical entity name we expected
  - Side-by-side diff so we can SEE what changed

The query set is hand-picked to cover four failure modes:
  A) Canonical-spelling queries (user types it right, both should match)
  B) Misspelling queries (user types Whisper-style, only RAW matches)
  C) Cross-event queries (multi-entity, semantic match needed)
  D) Generic English queries that touched wrong-FP territory before

Output: thesis/search_quality_comparison.md (markdown with diff tables).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ES = "http://localhost:9200/forzasearch-windows/_search"
RAW_ID = "chelsea-liverpool-2016-RAW"
CLEAN_ID = "chelsea-liverpool-2016"


# Hand-picked queries covering 4 failure modes.
QUERIES = [
    # ----- A: canonical spelling (user types right name) -----
    ("A1", "Sturridge goal",         ["Sturridge"]),
    ("A2", "Diego Costa shot",       ["Diego Costa", "Costa"]),
    ("A3", "Hazard cross",           ["Hazard"]),
    ("A4", "Mignolet save",          ["Mignolet"]),
    ("A5", "Klopp tactics",          ["Klopp", "Jürgen"]),

    # ----- B: misspelling / Whisper-style query -----
    # If the user heard the commentator and types phonetically, they get
    # what Whisper got. Cleaned should still match because the canonical
    # was substituted in.
    ("B1", "Aspilicueta header",     ["Azpilicueta", "Aspilicueta"]),
    ("B2", "Davi Luiz pass",         ["David Luiz", "Davi Luiz"]),
    ("B3", "Diogo Costa Chelsea",    ["Diego Costa", "Diogo Costa"]),
    ("B4", "Havanovic defending",    ["Ivanovic", "Havanovic"]),
    ("B5", "Marcus Alonso run",      ["Marcos Alonso", "Marcus Alonso"]),

    # ----- C: multi-entity / semantic queries -----
    ("C1", "Coutinho Lallana goal",  ["Coutinho", "Lallana"]),
    ("C2", "Henderson midfield",     ["Henderson"]),
    ("C3", "first goal Liverpool",   ["goal", "Liverpool"]),
    ("C4", "free kick wall",         ["free kick", "wall"]),

    # ----- D: 'tricky' queries the cleaning should NOT have broken -----
    ("D1", "Conte signing",          ["Conte"]),    # Antonio Conte → Conte
    ("D2", "Willian winger",         ["Willian"]),  # William → Willian
    ("D3", "Origi striker",          ["Origi"]),    # rigi/Rigi → Origi
]


def query_es(match_id: str, q: str, k: int = 5) -> list[dict]:
    """Hybrid BM25 multi_match — same body as frontend (without k-NN)."""
    body = {
        "size": k,
        "query": {
            "bool": {
                "filter": [{"term": {"match_id": match_id}}],
                "should": [
                    {"multi_match": {"query": q, "fields": ["text^2", "text.general"], "fuzziness": "AUTO"}},
                    {"match_phrase": {"text": {"query": q, "boost": 5}}},
                    {"match_phrase": {"text.general": {"query": q, "boost": 5}}},
                ],
                "minimum_should_match": 1,
            }
        },
        "_source": ["start_sec", "end_sec", "match_minute", "text"],
    }
    req = urllib.request.Request(
        ES, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.load(r)
    return [
        {
            "score": hit["_score"],
            "start_sec": hit["_source"].get("start_sec"),
            "match_minute": hit["_source"].get("match_minute"),
            "text": hit["_source"].get("text", ""),
        }
        for hit in data.get("hits", {}).get("hits", [])
    ]


def has_target(text: str, targets: list[str]) -> str:
    """Return the first target found in text, else ''."""
    low = text.lower()
    for t in targets:
        if t.lower() in low:
            return t
    return ""


def render(qid: str, q: str, targets: list[str],
           raw_hits: list[dict], clean_hits: list[dict]) -> list[str]:
    md = []
    md.append(f"### {qid}: `\"{q}\"`  (target entities: {targets})")
    md.append("")
    md.append("| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |")
    md.append("|---|---|---|---|---|---|---|---|")
    for i in range(3):
        r = raw_hits[i] if i < len(raw_hits) else {}
        c = clean_hits[i] if i < len(clean_hits) else {}
        r_text = (r.get("text", "") or "")[:90].replace("|", "/")
        c_text = (c.get("text", "") or "")[:90].replace("|", "/")
        r_hit = "✓ " + has_target(r.get("text", ""), targets) if has_target(r.get("text", ""), targets) else "✗"
        c_hit = "✓ " + has_target(c.get("text", ""), targets) if has_target(c.get("text", ""), targets) else "✗"
        md.append(
            f"| {i+1} | "
            f"{r.get('score', 0):.2f} @{r.get('match_minute','-')}min | `{r_text}` | {r_hit} || "
            f"{c.get('score', 0):.2f} @{c.get('match_minute','-')}min | `{c_text}` | {c_hit} |"
        )
    md.append("")
    return md


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("thesis/search_quality_comparison.md"))
    args = p.parse_args()

    md = ["# Search Quality: RAW Whisper vs CLEANED pipeline output",
          "",
          "Both indexed in the same Elasticsearch instance with identical",
          "BM25 + fuzzy multi_match + phrase-match (boost 5) hybrid query.",
          "Tests whether the cleaning pipeline adds practical value when",
          "the search backend already does fuzzy matching.",
          "",
          f"- RAW index: `match_id = {RAW_ID}` (1528 segments → 510 windows)",
          f"- CLEANED index: `match_id = {CLEAN_ID}` (1524 segments → 509 windows)",
          "- Same audio, same Whisper model, only difference is the ASR-cleaning pipeline.",
          "",
          "Each query asks: does the top-3 search result contain a canonical",
          "entity name we expected (`hit?` column)? Score is the joint BM25",
          "+ phrase boost — higher means stronger lexical match.",
          "",
          "---",
          ""]

    summary = {"A": [0, 0], "B": [0, 0], "C": [0, 0], "D": [0, 0]}  # bucket: [raw_top1_hit, clean_top1_hit]
    score_summary = {"raw": [], "clean": []}

    for qid, q, targets in QUERIES:
        try:
            raw_hits = query_es(RAW_ID, q)
            clean_hits = query_es(CLEAN_ID, q)
        except Exception as e:
            md.append(f"### {qid}: `\"{q}\"`  ERROR: {e}")
            md.append("")
            continue

        md.extend(render(qid, q, targets, raw_hits, clean_hits))

        bucket = qid[0]
        if raw_hits and has_target(raw_hits[0].get("text", ""), targets):
            summary[bucket][0] += 1
        if clean_hits and has_target(clean_hits[0].get("text", ""), targets):
            summary[bucket][1] += 1
        if raw_hits:
            score_summary["raw"].append(raw_hits[0]["score"])
        if clean_hits:
            score_summary["clean"].append(clean_hits[0]["score"])

    # Aggregate summary
    md.append("---")
    md.append("")
    md.append("## Top-1 hit-rate per category")
    md.append("")
    md.append("| Category | Queries | RAW top-1 hits | CLEANED top-1 hits |")
    md.append("|---|---|---|---|")
    bucket_names = {"A": "A — canonical spelling",
                    "B": "B — Whisper-style misspelling",
                    "C": "C — multi-entity / semantic",
                    "D": "D — tricky (validation cases)"}
    n_per = {"A": 5, "B": 5, "C": 4, "D": 3}
    total_raw, total_clean, total_n = 0, 0, 0
    for k, name in bucket_names.items():
        r, c = summary[k]
        n = n_per[k]
        md.append(f"| {name} | {n} | {r}/{n} | {c}/{n} |")
        total_raw += r; total_clean += c; total_n += n
    md.append(f"| **Total** | **{total_n}** | **{total_raw}/{total_n} ({total_raw/total_n*100:.0f}%)** | **{total_clean}/{total_n} ({total_clean/total_n*100:.0f}%)** |")
    md.append("")

    if score_summary["raw"] and score_summary["clean"]:
        avg_r = sum(score_summary["raw"]) / len(score_summary["raw"])
        avg_c = sum(score_summary["clean"]) / len(score_summary["clean"])
        md.append(f"## Average top-1 BM25 score")
        md.append("")
        md.append(f"- RAW: **{avg_r:.2f}**")
        md.append(f"- CLEANED: **{avg_c:.2f}**")
        md.append(f"- Δ: **{avg_c - avg_r:+.2f}** ({(avg_c - avg_r) / avg_r * 100:+.1f}% rel)")
        md.append("")

    md.append("## Interpretation guide")
    md.append("")
    md.append("- **Category A (canonical)**: Both RAW and CLEANED *should* find the entity, because the user types the right name. If RAW does well here, the cleaning didn't help on this axis — but it shouldn't hurt either.")
    md.append("- **Category B (misspelling)**: This is where cleaning is supposed to win. The user types `Aspilicueta`, RAW only contains canonical `Aspilicueta` (because that's what Whisper said). CLEANED contains `Azpilicueta`. A user who types the canonical form should NOT find any matches in the misspelled-only RAW corpus. ES fuzzy `AUTO` may compensate within edit distance ≤2; cleaning extends the reach beyond that.")
    md.append("- **Category C (multi-entity/semantic)**: BM25 alone may struggle when only 1 of 2 entities is exact. k-NN re-ranking via embeddings should compensate; cleaning's contribution is marginal here.")
    md.append("- **Category D (tricky)**: Cases the cleaning pipeline introduced corrections for. If the cleaned corpus *removed* good matches, this is where it shows.")
    md.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {args.out} ({len(md)} lines)")

    # Console summary
    print()
    print(f"Top-1 hit rate: RAW {total_raw}/{total_n} ({total_raw/total_n*100:.0f}%) "
          f"vs CLEANED {total_clean}/{total_n} ({total_clean/total_n*100:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
