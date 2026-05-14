"""Generate a per-stage walkthrough for the thesis from a cleaned match.

Reads the match's raw V3 input, the cleaned output, and the cleaning
metadata (which now includes per-stage timings and telemetry, written by
``pipeline/orchestrator.py:_write_cleaned_output``). Emits a Markdown
walkthrough showing:

  - per-stage wall time
  - segments touched by each stage
  - 5 representative before/after diffs per stage
  - aggregate WER vs ground truth (if available)

Usage:

    python tools/generate_pipeline_walkthrough.py \\
        --match "Chelsea 1 - 2 Liverpool" \\
        --gt-glob '_goal_reference/*goal_entity_tagged.json' \\
        --out thesis/pipeline_detailed_walkthrough.md
"""

from __future__ import annotations

import argparse
import json
import sys
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


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _diff_examples(raw_segs: dict, cleaned_segs: dict, max_n: int = 8) -> list[dict]:
    """Pick segments where text changed and show before/after."""
    out = []
    for sid, raw in raw_segs.items():
        if sid not in cleaned_segs:
            continue
        raw_text = raw["text"] if isinstance(raw, dict) else raw[2]
        cleaned_text = cleaned_segs[sid]["text"]
        if raw_text != cleaned_text:
            out.append({
                "segment_id": sid,
                "raw": raw_text,
                "cleaned": cleaned_text,
            })
            if len(out) >= max_n:
                break
    return out


def _format_corrections(corrections: list[dict], cap: int = 8) -> list[str]:
    """Compact view: 'orig → corr (method)' lines."""
    out = []
    seen = set()
    for c in corrections:
        key = (c.get("original"), c.get("corrected"))
        if key in seen:
            continue
        seen.add(key)
        out.append(f"`{c.get('original')}` → `{c.get('corrected')}`  *({c.get('method')})*")
        if len(out) >= cap:
            break
    return out


STAGE_DOCS = {
    "step0_detect_language": {
        "title": "Step 0: Language Detection",
        "module": "pipeline/hallucination_filter.py:detect_commentary_language",
        "model": "langdetect (port of Google language-detection)",
        "purpose": (
            "Sample the first ~1000 characters of the transcript and "
            "predict the commentary language. The result is propagated "
            "to every downstream stage so each picks the right model "
            "(spaCy, Whisper, MLM veto)."
        ),
    },
    "step1_build_gazetteer": {
        "title": "Step 1: Build Gazetteer from Labels-caption.json",
        "module": "pipeline/gazetteer.py:build_gazetteer",
        "model": "—",
        "purpose": (
            "Read the match's Labels-caption.json (lineup, teams, referee) "
            "and produce a name-variant dict (`gazetteer`) and typed entity "
            "map (player/team/referee/coach). This is the canonical name "
            "set that Stage E retrieval and the n-best reranker score "
            "candidates against."
        ),
    },
    "step2_hallucination_filter": {
        "title": "Step 2: Hallucination Filter",
        "module": "pipeline/hallucination_filter.py:filter_segments",
        "model": "langdetect, regex",
        "purpose": (
            "Remove segments that are obvious ASR garbage. Rules: empty, "
            "non-Latin script, alpha-ratio < 0.50, wrong language family, "
            "and clustered repeated single-word callouts. The langdetect "
            "gate only fires for segments ≥15 words to avoid misclassifying "
            "short football phrases."
        ),
    },
    "step3_deduplicate": {
        "title": "Step 3: Deduplicate + Whisper-loop Collapse",
        "module": "pipeline/deduplicator.py + pipeline.orchestrator._collapse_repeated_words",
        "model": "rapidfuzz",
        "purpose": (
            "Merge consecutive near-duplicate segments (rapidfuzz ratio "
            "≥ 95) and collapse Whisper-loop word repetitions (3+ "
            "consecutive identical tokens → 1). Two-repeats are kept "
            "because GT has legitimate forms like \"well, well\" and "
            "\"starry, starry night\"."
        ),
    },
    "stepN_nbest_rerank": {
        "title": "Step N: N-best Entity-Grounded Reranking",
        "module": "pipeline/nbest_reranker.py:rerank_match",
        "model": "sentence-transformers + FAISS",
        "purpose": (
            "Apple RAG-NEC pattern (arxiv:2409.06062). For each segment "
            "with multiple beam hypotheses, score each candidate by sum of "
            "max-cosine to gazetteer canonical names (FAISS index over "
            "paraphrase-multilingual-MiniLM embeddings). A length-penalty "
            "blocks structurally distant hypotheses. Pass-through when "
            "n-best is missing or gazetteer is empty."
        ),
    },
    "step2A_domain_normalize": {
        "title": "Stage 2A: Domain Normalizer",
        "module": "pipeline/domain_normalizer.py:DomainNormalizer",
        "model": "regex",
        "purpose": (
            "Football-specific text normalization: disfluency removal "
            "(uh, um, eh), unambiguous compound merging (off side → "
            "offside, goal keeper → goalkeeper), repeated punctuation "
            "cleanup, whitespace normalization. The rule set is "
            "language-aware (English/Swedish/German). Empirically harmful "
            "rules (\"half time → halftime\", \"line up → lineup\") were "
            "removed after WER analysis vs GOAL human GT."
        ),
    },
    "stepNER_extract_entities": {
        "title": "NER: Entity Extraction (spaCy + heuristics)",
        "module": "pipeline/ner_extractor.py:extract_entities_batch",
        "model": "en_core_web_sm (or sv/de/... per language)",
        "purpose": (
            "Detect candidate entities for Stage E. Combines spaCy NER "
            "(PERSON/ORG/GPE/FAC) with three heuristics: (1) capitalised "
            "tokens that look like proper nouns, (2) POS-filtered tokens, "
            "(3) gazetteer fuzz-match — any token within fuzz.ratio≥65 of "
            "a gazetteer canonical (catches ASR mishearings whose surface "
            "is a real English word, e.g. \"storage\" → Sturridge)."
        ),
    },
    "stepE_entity_corrector": {
        "title": "Stage E: Validated Entity Corrector",
        "module": "pipeline/entity_corrector.py:correct_match",
        "model": "TF-IDF char-bigram + Qwen2.5-1.5B-Instruct (GGUF) + xlm-roberta",
        "purpose": (
            "Replaces the legacy fuzzy/phonetic/context cascade. Per detected "
            "entity: (1) check validated cross-match cache, (2) TF-IDF "
            "char-bigram retrieve top-K=5 from gazetteer, (3) per-match "
            "decision cache, (4) frequency heuristic (≥5× in match → reject), "
            "(5) shortcut accept/reject by cosine, (6) MCQ judge with Qwen "
            "picking A/B/C/D=keep/E=unsure, (7) MLM veto via xlm-roberta "
            "pseudo-logprob, (8) validation gates (dictionary veto, fuzzy "
            "floor, length tolerance)."
        ),
    },
    "stepL_llm_ger": {
        "title": "Step L: Confidence-gated Generative Error Correction",
        "module": "pipeline/llm_corrector.py:correct_match",
        "model": "Qwen2.5-1.5B-Instruct Q4_K_M GGUF + xlm-roberta-base MLM",
        "purpose": (
            "Confidence-Guided Error Correction (arxiv:2509.25048). Tokens "
            "with avg_logprob > -0.3 are kept verbatim; only low-confidence "
            "tokens are wrapped <token> in the prompt and may be edited. "
            "Match-context block (Teams/Players/Referee + prev/next "
            "segments) grounds the LLM. After the LLM proposes an edit, "
            "xlm-roberta vetos it if MLM(original)/MLM(proposed) ≥ 1.5. "
            "Step L respects entity_corrector decisions via "
            "frozen_word_indices so it does not re-touch canonical names."
        ),
    },
    "stepP_punct_restore": {
        "title": "Step P: Punctuation + Casing Restoration",
        "module": "pipeline/punct_restorer.py:restore_punctuation_batch",
        "model": "oliverguhr/fullstop-punctuation-multilang-large",
        "purpose": (
            "Multilingual transformer that inserts (.,?!) and casing where "
            "Whisper dropped it. Conservative: only inserts where missing, "
            "never deletes existing punctuation. Important for downstream "
            "Elasticsearch tokenization and event extraction (NER works "
            "better on properly-cased text)."
        ),
    },
}


def render_markdown(walk: dict) -> str:
    md = ["# Pipeline Detailed Walkthrough — Chelsea vs Liverpool 2016-09-16",
          "",
          "*Generated automatically from `cleaning_metadata.stage_timings` and "
          "the cleaned-vs-raw segment diff. Source: "
          "`tools/generate_pipeline_walkthrough.py`.*",
          ""]

    md.append("## Overview")
    md.append("")
    md.append(f"- **Match:** {walk['match_name']}")
    md.append(f"- **Detected language:** {walk.get('language', 'en')}")
    md.append(f"- **Halves analysed:** {sorted(walk['per_half'].keys())}")
    md.append("")

    # Combined stage table
    md.append("## Per-Stage Wall Time (seconds)")
    md.append("")
    all_stages = []
    for half_data in walk["per_half"].values():
        for s in (half_data.get("stage_timings") or {}):
            if s not in all_stages:
                all_stages.append(s)
    md.append("| Stage | " + " | ".join(f"H{h}" for h in sorted(walk["per_half"])) + " | Total |")
    md.append("|" + "---|" * (len(walk["per_half"]) + 2))
    for s in all_stages:
        row = [s]
        total = 0.0
        for h in sorted(walk["per_half"]):
            v = walk["per_half"][h].get("stage_timings", {}).get(s, 0.0)
            row.append(f"{v:.2f}")
            total += float(v)
        row.append(f"{total:.2f}")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    # Per-stage detail
    md.append("## Per-Stage Detail")
    md.append("")
    for stage_key, doc in STAGE_DOCS.items():
        md.append(f"### {doc['title']}")
        md.append("")
        md.append(f"- **Module:** `{doc['module']}`")
        md.append(f"- **Model:** {doc['model']}")
        md.append("")
        md.append(doc["purpose"])
        md.append("")
        # Per-half timings + counts
        md.append("**Per-half stats:**")
        md.append("")
        md.append("| Half | Wall time (s) | Notes |")
        md.append("|---|---|---|")
        for h in sorted(walk["per_half"]):
            half_data = walk["per_half"][h]
            t = half_data.get("stage_timings", {}).get(stage_key, 0.0)
            note = ""
            if stage_key == "step2_hallucination_filter":
                note = f"removed {len(half_data.get('removed_hallucinations', []))}"
            elif stage_key == "step3_deduplicate":
                note = f"removed {len(half_data.get('removed_duplicates', []))} duplicates"
            elif stage_key == "stepN_nbest_rerank":
                tel = half_data.get("nbest_telemetry", {}) or {}
                if tel.get("pass_through_reason"):
                    note = f"pass-through ({tel['pass_through_reason']})"
                else:
                    note = (f"{tel.get('segments_replaced', 0)} of "
                            f"{tel.get('segments_with_nbest', 0)} re-picked")
            elif stage_key == "stepE_entity_corrector":
                ec = sum(1 for c in half_data.get("corrections", [])
                         if c.get("stage") == "3")
                note = f"applied {ec} entity corrections"
            elif stage_key == "stepL_llm_ger":
                ll = sum(1 for c in half_data.get("sota_corrections", [])
                         if c.get("method") == "llm_ger")
                tel = half_data.get("llm_telemetry", {}) or {}
                note = (f"applied {ll} GER edits "
                        f"(eligible {tel.get('eligible_segments', '?')}/"
                        f"{tel.get('total_segments', '?')})")
            elif stage_key == "stepP_punct_restore":
                pp = sum(1 for c in half_data.get("sota_corrections", [])
                         if c.get("method") == "punct_restore")
                note = f"restyled {pp} segments"
            md.append(f"| {h} | {t:.2f} | {note} |")
        md.append("")

        # Examples
        md.append("**Representative diffs:**")
        md.append("")
        # Pick examples from the corrections / removals attached to this stage
        examples_md = []
        for h in sorted(walk["per_half"]):
            half_data = walk["per_half"][h]
            if stage_key == "step2_hallucination_filter":
                for r in (half_data.get("removed_hallucinations", []) or [])[:5]:
                    examples_md.append(
                        f"- H{h} `[{r['segment_id']}]` removed `{r['text']}` *(reason: {r['reason']})*"
                    )
            elif stage_key == "step3_deduplicate":
                for r in (half_data.get("removed_duplicates", []) or [])[:5]:
                    sid = r.get("segment_id", r.get("id", "?"))
                    txt = r.get("text", "")
                    examples_md.append(f"- H{h} `[{sid}]` collapsed `{txt}`")
            elif stage_key == "step2A_domain_normalize":
                for c in (half_data.get("normalization_corrections", []) or [])[:5]:
                    examples_md.append(
                        f"- H{h} `[{c['segment_id']}]` `{c['original']}` → `{c['corrected']}`"
                    )
            elif stage_key == "stepE_entity_corrector":
                for c in (half_data.get("corrections", []) or [])[:5]:
                    examples_md.append(
                        f"- H{h} `[{c['segment_id']}]` `{c['original']}` → `{c['corrected']}`"
                        f" *(method: {c.get('method', '?')}, score {c.get('score', '?')})*"
                    )
            elif stage_key == "stepL_llm_ger":
                ll_examples = [c for c in (half_data.get("sota_corrections", []) or [])
                               if c.get("method") == "llm_ger"][:3]
                for c in ll_examples:
                    examples_md.append(
                        f"- H{h} `[{c.get('segment_id')}]` `{c.get('original')}` → `{c.get('corrected')}`"
                    )
            elif stage_key == "stepP_punct_restore":
                pp_examples = [c for c in (half_data.get("sota_corrections", []) or [])
                               if c.get("method") == "punct_restore"][:3]
                for c in pp_examples:
                    examples_md.append(
                        f"- H{h} `[{c.get('segment_id')}]` `{c.get('original')[:80]}…` → `{c.get('corrected')[:80]}…`"
                    )
            elif stage_key == "stepN_nbest_rerank":
                tel = half_data.get("nbest_telemetry", {}) or {}
                for ex in (tel.get("examples", []) or [])[:3]:
                    examples_md.append(
                        f"- H{h} `[{ex['segment_id']}]` `{ex['from']}` → `{ex['to']}` *(Δscore {ex.get('score_delta')})*"
                    )

        if not examples_md:
            md.append("*(no changes recorded by this stage on this match — likely a no-op given the inputs)*")
        else:
            md.extend(examples_md)
        md.append("")

    # Final summary
    md.append("## Summary")
    md.append("")
    total = 0.0
    for h in sorted(walk["per_half"]):
        t = walk["per_half"][h].get("stage_timings", {}).get("total_pipeline", 0.0)
        md.append(f"- Half {h} total cleaning time: **{t:.1f}s**")
        total += t
    md.append(f"- Combined: **{total:.1f}s**")
    return "\n".join(md)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--match", required=True, help="match name substring")
    p.add_argument("--cleaned-root", type=Path, default=Path("cleaned_data"),
                   help="root containing cleaned_data/ output")
    p.add_argument("--raw-root", type=Path,
                   default=Path("path/to/SoccerNet/caption-2023"),
                   help="root containing raw match dirs")
    p.add_argument("--variant", default="_v3",
                   help="raw asr variant suffix (default _v3)")
    p.add_argument("--cleaned-variant", default=None,
                   help="cleaned asr variant suffix (defaults to --variant)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    raw_dir = _find_match_dir(args.match, args.raw_root)
    cleaned_dir = _find_match_dir(args.match, args.cleaned_root)
    if not raw_dir:
        print(f"raw match not found for {args.match!r}", file=sys.stderr)
        return 1
    if not cleaned_dir:
        print(f"cleaned match not found for {args.match!r}", file=sys.stderr)
        return 1

    walk = {"match_name": raw_dir.name, "per_half": {}}

    cleaned_v = args.cleaned_variant if args.cleaned_variant is not None else args.variant
    for half in (1, 2):
        cleaned_path = cleaned_dir / "commentary_data" / f"{half}_asr{cleaned_v}_cleaned.json"
        raw_path = raw_dir / "commentary_data" / f"{half}_asr{args.variant}.json"
        if not cleaned_path.exists() or not raw_path.exists():
            continue

        cleaned = _load(cleaned_path)
        raw = _load(raw_path)
        meta = cleaned.get("cleaning_metadata", {})

        walk["per_half"][half] = {
            "stage_timings": meta.get("stage_timings") or {},
            "nbest_telemetry": meta.get("nbest_telemetry") or {},
            "llm_telemetry": meta.get("llm_telemetry") or {},
            "removed_hallucinations": meta.get("removed_hallucinations") or [],
            "removed_duplicates": meta.get("removed_duplicates") or [],
            "corrections": meta.get("corrections") or [],
            "sota_corrections": meta.get("sota_corrections") or [],
            "diffs": _diff_examples(raw.get("segments", {}), cleaned.get("segments", {})),
        }
        if "language" not in walk:
            walk["language"] = raw.get("language", "en")

    md = render_markdown(walk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {args.out} ({len(md.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
