"""Test lineup-aware query expansion against ES — RAW Whisper indeks vs
CLEANED indeks, both with standard query and with expanded query.

Four test conditions per query:
  A1) RAW indeks + standard ES query (status quo for raw)
  A2) RAW indeks + lineup-expanded query (NEW — proves cleaning is unnecessary)
  B1) CLEANED indeks + standard ES query (status quo for cleaned)
  B2) CLEANED indeks + lineup-expanded query (combination)

If A2 ≥ B1, the new architecture wins: query-time expansion on raw
beats indexing-time cleaning, and the entire L+P pipeline (32 minutes
per match) can be deleted in favour of an O(milliseconds-per-query)
expansion.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ES = "http://localhost:9200/forzasearch-windows/_search"
RAW_ID = "chelsea-liverpool-2016-RAW"
CLEAN_ID = "chelsea-liverpool-2016"


# Same query suite as compare_search_quality.py for direct comparison.
QUERIES = [
    ("A1", "Sturridge goal",         ["Sturridge"]),
    ("A2", "Diego Costa shot",       ["Diego Costa", "Costa"]),
    ("A3", "Hazard cross",           ["Hazard"]),
    ("A4", "Mignolet save",          ["Mignolet"]),
    ("A5", "Klopp tactics",          ["Klopp", "Jürgen"]),
    ("B1", "Aspilicueta header",     ["Azpilicueta", "Aspilicueta"]),
    ("B2", "Davi Luiz pass",         ["David Luiz", "Davi Luiz"]),
    ("B3", "Diogo Costa Chelsea",    ["Diego Costa", "Diogo Costa"]),
    ("B4", "Havanovic defending",    ["Ivanovic", "Havanovic"]),
    ("B5", "Marcus Alonso run",      ["Marcos Alonso", "Marcus Alonso"]),
    ("C1", "Coutinho Lallana goal",  ["Coutinho", "Lallana"]),
    ("C2", "Henderson midfield",     ["Henderson"]),
    ("C3", "first goal Liverpool",   ["goal", "Liverpool"]),
    ("C4", "free kick wall",         ["free kick", "wall"]),
    ("D1", "Conte signing",          ["Conte"]),
    ("D2", "Willian winger",         ["Willian"]),
    ("D3", "Origi striker",          ["Origi"]),
]


def es_query_standard(match_id: str, q: str, k: int = 3) -> list[dict]:
    """Standard hybrid BM25 + AUTO fuzzy + phrase boost (frontend default)."""
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
    return _exec_es(body)


def es_query_expanded(match_id: str, expansion: dict,
                      original: str, k: int = 3) -> list[dict]:
    """Lineup-expanded query — append expansions to original as extra
    fuzzy-match terms, with phrase-boost on multi-word canonicals.

    Critical design choice: keep the ORIGINAL query as the primary
    match (with AUTO fuzzy + phrase boost — matches the standard query
    body so we don't lose any baseline matches). Add expansions as
    additional `should` clauses with smaller boost so they only
    influence ranking when they fire on top of the baseline.
    """
    # Filter expansions: drop bigrams that look like 'X Y' where X and Y
    # are both potentially player surnames in the SAME query — these are
    # adjacent-in-transcript spurious matches like "Lallana Henderson".
    clean_expansions = []
    for term in expansion["should_terms"]:
        if " " in term:
            # Keep only multi-word terms that look like FULL canonical names
            # (e.g. "Daniel Sturridge"). Drop "Mane Sturridge" type noise
            # by checking against canonicals from matched_entities.
            canonicals = {me["canonical"] for me in expansion["matched_entities"]}
            if term in canonicals:
                clean_expansions.append(term)
        else:
            clean_expansions.append(term)

    should = [
        # Baseline match (same as standard query)
        {"multi_match": {"query": original,
                         "fields": ["text^2", "text.general"],
                         "fuzziness": "AUTO"}},
        {"match_phrase": {"text": {"query": original, "boost": 5}}},
        {"match_phrase": {"text.general": {"query": original, "boost": 5}}},
    ]
    # Add expansion terms with mild boost
    for term in clean_expansions:
        if " " in term:
            should.append({"match_phrase": {"text.general": {"query": term, "boost": 3}}})
        else:
            should.append({"multi_match": {"query": term,
                                            "fields": ["text^2", "text.general"],
                                            "boost": 1.2}})

    body = {
        "size": k,
        "query": {
            "bool": {
                "filter": [{"term": {"match_id": match_id}}],
                "should": should,
                "minimum_should_match": 1,
            }
        },
        "_source": ["start_sec", "end_sec", "match_minute", "text"],
    }
    return _exec_es(body)


def _exec_es(body: dict) -> list[dict]:
    req = urllib.request.Request(
        ES, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.load(r)
    return [
        {"score": hit["_score"], "match_minute": hit["_source"].get("match_minute"),
         "text": hit["_source"].get("text", "")}
        for hit in data.get("hits", {}).get("hits", [])
    ]


def has_target(text: str, targets: list[str]) -> str:
    low = text.lower()
    for t in targets:
        if t.lower() in low:
            return t
    return ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--match", default="Chelsea 1 - 2 Liverpool")
    p.add_argument("--raw-root", type=Path,
                   default=Path("path/to/SoccerNet/caption-2023"))
    p.add_argument("--out", type=Path,
                   default=Path("thesis/lineup_query_expansion_test.md"))
    args = p.parse_args()

    # Find labels for the match
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

    from pipeline.lineup_query_expander import (
        build_lineup_index, harvest_surface_variants, expand_query,
    )

    labels = json.load(open(match_dir / "Labels-caption.json", encoding="utf-8"))
    lineup = build_lineup_index(labels)
    print(f"Lineup: {len(lineup)} entries")

    # Harvest surface variants from raw transcript
    raw_text = ""
    for half in (1, 2):
        rp = match_dir / "commentary_data" / f"{half}_asr_v3_nbest.json"
        if rp.exists():
            d = json.load(open(rp, encoding="utf-8"))
            for s in d["segments"].values():
                raw_text += " " + (s.get("text", "") if isinstance(s, dict) else s[2])

    surface_variants = harvest_surface_variants(raw_text, lineup)
    print(f"Surface variants harvested: {sum(len(v) for v in surface_variants.values())} variants across {len(surface_variants)} canonicals")
    print()

    # Run all four conditions per query
    md = ["# Lineup-Aware Query Expansion Test", "",
          f"Match: **{match_dir.name}**", "",
          f"Lineup index: {len(lineup)} entries (players, coaches, teams, referees)",
          f"Surface variants harvested from raw Whisper text: "
          f"{sum(len(v) for v in surface_variants.values())} total",
          "",
          "## Per-query top-1 hit?",
          "",
          "| ID | Query | RAW+std | RAW+expand | CLEANED+std | CLEANED+expand |",
          "|---|---|---|---|---|---|"]

    counts = {"raw_std": 0, "raw_exp": 0, "clean_std": 0, "clean_exp": 0}
    expansion_examples = []

    for qid, q, targets in QUERIES:
        try:
            exp = expand_query(q, lineup, surface_variants)
            raw_std = es_query_standard(RAW_ID, q)
            raw_exp = es_query_expanded(RAW_ID, exp, q)
            clean_std = es_query_standard(CLEAN_ID, q)
            clean_exp = es_query_expanded(CLEAN_ID, exp, q)
        except Exception as e:
            md.append(f"| {qid} | {q} | ERROR | ERROR | ERROR | ERROR |"); continue

        def hit(hits):
            return "✓" if hits and has_target(hits[0]["text"], targets) else "✗"

        h_rs = hit(raw_std)
        h_re = hit(raw_exp)
        h_cs = hit(clean_std)
        h_ce = hit(clean_exp)
        if h_rs == "✓": counts["raw_std"] += 1
        if h_re == "✓": counts["raw_exp"] += 1
        if h_cs == "✓": counts["clean_std"] += 1
        if h_ce == "✓": counts["clean_exp"] += 1

        md.append(f"| {qid} | `{q}` | {h_rs} | {h_re} | {h_cs} | {h_ce} |")

        if exp["matched_entities"] and len(expansion_examples) < 6:
            expansion_examples.append({"query": q, "expansion": exp})

    md.append("")
    md.append(f"## Top-1 hit-rate (out of {len(QUERIES)})")
    md.append("")
    md.append("| Configuration | Hit rate |")
    md.append("|---|---|")
    md.append(f"| RAW + standard query | **{counts['raw_std']}/{len(QUERIES)} ({counts['raw_std']/len(QUERIES)*100:.0f} %)** |")
    md.append(f"| **RAW + lineup-expanded query** | **{counts['raw_exp']}/{len(QUERIES)} ({counts['raw_exp']/len(QUERIES)*100:.0f} %)** |")
    md.append(f"| CLEANED + standard query | **{counts['clean_std']}/{len(QUERIES)} ({counts['clean_std']/len(QUERIES)*100:.0f} %)** |")
    md.append(f"| CLEANED + lineup-expanded query | **{counts['clean_exp']}/{len(QUERIES)} ({counts['clean_exp']/len(QUERIES)*100:.0f} %)** |")
    md.append("")

    # Sample expansions
    md.append("## Example query expansions")
    md.append("")
    for ex in expansion_examples:
        md.append(f"**Original:** `{ex['query']}`")
        md.append("")
        md.append("Matched entities:")
        for me in ex["expansion"]["matched_entities"]:
            exp_str = ", ".join(f"`{e}`" for e in me["expansions"])
            md.append(f"- `{me['token']}` (metaphone: `{me['metaphone']}`) → "
                      f"canonical `{me['canonical']}` → expansions: {exp_str}")
        md.append("")

    md.append("## Interpretation")
    md.append("")
    md.append("**If RAW+expand ≥ CLEANED+std:** The 32-minute Step L + Step P")
    md.append("cleaning pipeline can be replaced by lineup-aware query expansion")
    md.append("at search time. Cleaning's only remaining justification is for")
    md.append("non-search consumers (LLM RAG answer fluency, NER-based event")
    md.append("aggregation).")
    md.append("")
    md.append("**If RAW+expand << CLEANED+std:** Cleaning still earns its place")
    md.append("at indexing time. (We don't expect this — fuzzy AUTO is already")
    md.append("known to bridge edit-distance ≤ 2 errors, and our expansion")
    md.append("explicitly targets the larger errors fuzzy can't reach.)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print()
    print(f"Top-1 hit-rate summary:")
    print(f"  RAW + std       : {counts['raw_std']}/{len(QUERIES)}")
    print(f"  RAW + EXPAND    : {counts['raw_exp']}/{len(QUERIES)}")
    print(f"  CLEANED + std   : {counts['clean_std']}/{len(QUERIES)}")
    print(f"  CLEANED + EXPAND: {counts['clean_exp']}/{len(QUERIES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
