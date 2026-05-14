"""Compare LLM RAG answer quality on RAW vs CLEANED indexes.

This is the experiment that justifies the cleaning pipeline. ES retrieval
alone is fuzzy enough that RAW and CLEANED give the same hit-rate
(see search_quality_comparison.md). But the LLM that READS those hits
and synthesises a natural-language answer for the user is the
downstream consumer that actually benefits from canonical entity names.

For each query:
  1. Retrieve top-3 windows from RAW index → feed to Mistral 7B
  2. Retrieve top-3 windows from CLEANED index → feed to Mistral 7B
  3. Render both answers + the prompt for human comparison

Output: thesis/llm_answer_comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ES = "http://localhost:9200/forzasearch-windows/_search"
OLLAMA = "http://localhost:11434/api/generate"
MODEL = "mistral:7b"

QUERIES = [
    ("Q1", "Who scored the second goal for Liverpool?"),
    ("Q2", "What did Diego Costa do in the first half?"),
    ("Q3", "Who is the Chelsea right-back?"),
    ("Q4", "Tell me about Sturridge's involvement in the match."),
    ("Q5", "Did Aspilicueta have a header?"),
    ("Q6", "Who was the Liverpool goalkeeper?"),
    ("Q7", "What happened with David Luiz?"),
]


def es_query(match_id: str, q: str, k: int = 3) -> list[str]:
    body = {
        "size": k,
        "query": {
            "bool": {
                "filter": [{"term": {"match_id": match_id}}],
                "should": [
                    {"multi_match": {"query": q, "fields": ["text^2", "text.general"], "fuzziness": "AUTO"}},
                    {"match_phrase": {"text": {"query": q, "boost": 5}}},
                ],
                "minimum_should_match": 1,
            }
        },
        "_source": ["start_sec", "match_minute", "text"],
    }
    req = urllib.request.Request(
        ES, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.load(r)
    return [
        f"[{hit['_source'].get('match_minute','-')}min] {hit['_source'].get('text','')}"
        for hit in data.get("hits", {}).get("hits", [])
    ]


def ollama_answer(question: str, contexts: list[str]) -> str:
    """Same prompt the frontend uses (B6 hardened version)."""
    ctx_lines = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(contexts))
    prompt = (
        "You are a search assistant for a football match: "
        "Chelsea 1-2 Liverpool, 16 September 2016, Premier League.\n"
        "Answer the user's question in 2-3 sentences using ONLY the "
        "commentary segments below. If the segments do not contain "
        "the answer, reply NO_MATCH.\n\n"
        f"Commentary segments:\n{ctx_lines}\n\n"
        f"Question: {question}\n"
        "Answer (or NO_MATCH if segments don't contain the info):"
    )
    body = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 200},
    }
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return data.get("response", "").strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("thesis/llm_answer_comparison.md"))
    args = p.parse_args()

    md = ["# LLM Answer Quality: RAW vs CLEANED ES indexes",
          "",
          "Same retrieval (ES BM25 + AUTO fuzzy + phrase boost) feeding the",
          "same Mistral 7B (Ollama) with identical prompt. The only variable",
          "is whether the indexed text was the raw Whisper output or the",
          "ASR-cleaning pipeline's output.",
          "",
          "This is the experiment that DOES distinguish raw from cleaned —",
          "the LLM is the downstream consumer that needs canonical entity",
          "names to give a correct answer.",
          ""]

    for qid, q in QUERIES:
        md.append(f"## {qid}: {q}")
        md.append("")
        try:
            raw_ctx = es_query("chelsea-liverpool-2016-RAW", q)
            clean_ctx = es_query("chelsea-liverpool-2016", q)

            print(f"  [{qid}] {q!r}")
            print(f"    RAW context: {len(raw_ctx)} segments")
            for c in raw_ctx:
                print(f"      {c[:100]}")
            raw_answer = ollama_answer(q, raw_ctx)
            print(f"    RAW answer: {raw_answer[:200]}")

            print(f"    CLEANED context: {len(clean_ctx)} segments")
            for c in clean_ctx:
                print(f"      {c[:100]}")
            clean_answer = ollama_answer(q, clean_ctx)
            print(f"    CLEANED answer: {clean_answer[:200]}")
            print()

        except Exception as e:
            md.append(f"ERROR: {e}")
            md.append("")
            continue

        md.append(f"### Question: `{q}`")
        md.append("")
        md.append("**RAW Whisper context fed to Mistral:**")
        md.append("")
        md.append("```")
        for i, c in enumerate(raw_ctx):
            md.append(f"{i+1}. {c}")
        md.append("```")
        md.append("")
        md.append(f"**Mistral answer (RAW):**")
        md.append("")
        md.append("> " + raw_answer.replace("\n", "\n> "))
        md.append("")
        md.append("**CLEANED context fed to Mistral:**")
        md.append("")
        md.append("```")
        for i, c in enumerate(clean_ctx):
            md.append(f"{i+1}. {c}")
        md.append("```")
        md.append("")
        md.append(f"**Mistral answer (CLEANED):**")
        md.append("")
        md.append("> " + clean_answer.replace("\n", "\n> "))
        md.append("")
        md.append("---")
        md.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
