# SoccerNet ASR Cleaning Pipeline

Multi-stage NLP pipeline that cleans Whisper ASR football commentary into
search-friendly transcripts and ships them to Elasticsearch for the
**ForzaSearch** retrieval-augmented video-clip search frontend.

Bachelor thesis project. Benchmark match: **Chelsea 1-2 Liverpool, 2016-09-16**
(English Premier League) against the GOAL human ground-truth.

---

## Headline numbers (full pipeline)

| Halv | Raw V3 WER | Cleaned WER | Raw Entity-F1 | **Cleaned Entity-F1** | Δ F1 (rel) |
|---|---|---|---|---|---|
| 1 | 25.56% | 26.21% | 0.484 | **0.603** | **+24.5%** |
| 2 | 23.86% | 24.30% | 0.504 | **0.578** | **+14.7%** |

**Whisper engine vs SoccerNet bundled (re-transcription win):**

| Halv | SoccerNet stock WER | faster-whisper-v3 WER | Δ WER |
|---|---|---|---|
| Snitt | 27.32% | 24.71% | **−2.61 pp (−9.6% rel)** |

---

## Architecture

```
audio.mp3
   │
   ▼  whisper_runner.transcribe (faster-whisper large-v3)
   │      beam=5, no_speech_threshold=0.95,
   │      condition_on_previous_text=False, word_timestamps=True
   │
   ▼  {1,2}_asr_v3.json  (schema 2 with per-word probs)
   │
   ▼  pipeline/orchestrator.py:clean_match
   │
   ├── Step 0   detect_commentary_language        (langdetect)
   ├── Step 1   build_gazetteer                    (Labels-caption.json → name dict)
   ├── Step 2   filter_segments                    (alpha-ratio + langdetect ≥15w)
   ├── Step 3   deduplicate + Whisper-loop collapse (rapidfuzz, 3+ rule)
   ├── Stage 2A domain_normalize                   (regex compounds + disfluencies)
   ├── NER      extract_entities_batch             (spaCy + 3 heuristics)
   ├── Stage E  entity_corrector.correct_match     (TF-IDF + Qwen MCQ + xlm-r MLM veto)
   ├── Step L   llm_corrector.correct_match        (Confidence-gated GER, Qwen 1.5B)
   └── Step P   restore_punctuation_batch          (oliverguhr fullstop multilang)
        │
        ▼  cleaned_data/.../{1,2}_asr_*_cleaned.json  (schema-1 [start, end, text])
        │
        ▼  tools/export_to_frontend.py
        │
        ▼  frontend/forzasearch-final/matches/<id>/kamp.json
        │
        ▼  npm run ingest  (MiniLM-ONNX embeddings + ES bulk)
        │
        ▼  Elasticsearch hybrid BM25 + k-NN search
        │
        ▼  Mistral 7B (Ollama) RAG re-rank + answer
        │
        ▼  Next.js frontend → seek-to-segment HLS player
```

---

## Stage timings on Chelsea-Liverpool (single match, CPU)

| Stage | Wall (s) | Notes |
|---|---|---|
| Step 0  detect_language | 1.1 | langdetect on transcript sample |
| Step 1  build_gazetteer | 0.0 | parses Labels-caption.json |
| Step 2  hallucination_filter | 0.2 | alpha-ratio + langdetect ≥15 words |
| Step 3  deduplicate | 0.0 | rapidfuzz + 3+ collapse |
| Stage 2A domain_normalize | 0.1 | regex football compounds |
| NER     extract_entities | 7 | spaCy + heuristic + Rule 3 fuzz |
| Stage E entity_corrector | 52 | TF-IDF + Qwen MCQ + 79 cached mappings |
| Step L  llm_ger | 1452 | Qwen 1.5B + xlm-r MLM veto |
| Step P  punct_restore | 464 | oliverguhr fullstop multilang |
| **Total** | **~2245** | ~37 min wall on CPU |

---

## Quick start

### 1. Install
```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Multilingual extras:
# python -m spacy download sv_core_news_sm de_core_news_sm fr_core_news_sm
```

### 2. Run the pipeline on a single match
```bash
python run_pipeline.py --match "Chelsea 1 - 2 Liverpool"
```

Output:
* `cleaned_data/<league>/<season>/<match>/commentary_data/{1,2}_asr_cleaned.json`

### 3. Evaluate WER + Entity-F1 vs GOAL ground truth
```bash
python tools/evaluate_wer.py --match "Chelsea" --half 1
python tools/evaluate_wer.py --match "Chelsea" --half 2
```

### 4. Run all tests (226 unit + integration)
```bash
pytest tests/ -v
```

### 5. End-to-end with the ForzaSearch frontend
```powershell
python tools/export_to_frontend.py `
  --match "Chelsea 1 - 2 Liverpool" `
  --frontend frontend/forzasearch-final `
  --id chelsea-liverpool-2016 `
  --title "Chelsea 1-2 Liverpool" `
  --subtitle "Premier League 2016-09-16"

# Start ES container + ingest + dev server (Docker Desktop must be running)
pwsh tools/start_frontend_e2e.ps1
# Then open http://localhost:3000
```

---

## Project layout

```
pipeline/
  orchestrator.py         clean_match() — runs Step 0 → P with per-stage timing (505 lines)
  whisper_runner.py       faster-whisper transcription with lineup hotwords (199)
  loader.py               schema-1/2 ASR JSON parsing (170)
  config.py               all thresholds, paths, multilingual model maps (302)
  hallucination_filter.py Step 2: alpha-ratio + langdetect + repeat-cluster filter (194)
  deduplicator.py         Step 3: consecutive-segment dedup via rapidfuzz (64)
  domain_normalizer.py    Stage 2A: regex football compounds + disfluencies (126)
  ner_extractor.py        spaCy NER + 3 heuristics (lone-word, POS-filtered, gazetteer fuzz) (389)
  gazetteer.py            Labels-caption → variant→canonical dict + entity types (249)
  entity_corrector.py     Stage E: TF-IDF retrieve + Qwen MCQ + xlm-r MLM veto + gates (752)
  fuzzy_corrector.py      Validation gates (passes_conservative_gates) (202)
  llm_corrector.py        Step L: confidence-gated Qwen GER + MLM veto + drift guard (617)
  punct_restorer.py       Step P: oliverguhr fullstop multilang (204)
  report.py               CleaningResult → human-readable summary (181)

tools/
  evaluate_wer.py                 WER + Entity-F1 vs GOAL GT (legacy + windowed alignment)
  retranscribe_kb_whisper.py      One-off re-transcription via faster-whisper (CLI wrapper)
  export_to_frontend.py           Convert cleaned output → frontend kamp.json (schema-1)
  export_raw_to_frontend.py       Same but for raw V3 (for raw-vs-cleaned ablation)
  start_frontend_e2e.ps1          Boot ES + ingest + Next.js dev server
  generate_pipeline_walkthrough.py  Renders thesis doc from cleaning_metadata
  seed_validated_corrections.py   Scan raw V3 → propose new validated_cache mappings
  dump_detected_entities.py       Every NER detection to CSV for verification
  count_gt_dropped_segments.py    Classify hyp-only segments (real commentary vs garbage)
  count_entity_variants.py        Surface-form variant count per canonical
  analyze_entity_rejections.py    Why Stage E rejected each candidate
  compare_search_quality.py       A/B query test against ES (RAW vs CLEANED)
  compare_llm_answer_quality.py   Mistral RAG answer quality (RAW vs CLEANED indexes)
  compare_whisper_versions.py     SoccerNet bundled vs faster-whisper-v3 WER + F1
  download_soccernet_match.py     Download match audio from SoccerNet
  extract_audio_from_url.py       Extract audio from a video URL
  fetch_match_metadata.py         Pull lineup + match metadata
  insert_asr_results_into_docx.py Inject §4.2.1.7-9 results into bachelor.docx
  integrate_asr_into_thesis.py    Inject ASR pipeline references across chapters 1-3

tests/                  226 unit + integration tests, all green
data/
  validated_corrections.json    79 curated misspelling→canonical mappings (the
                                 'learned dictionary' that grows over runs)
cleaned_data/           Pipeline output (per-match, mirrors dataset structure)
evaluation_results/     WER + F1 markdown reports per half
frontend/forzasearch-final/   Next.js + ES + Mistral RAG search frontend
thesis/                 Walkthrough docs, diff examples, entity dumps, INDEX.md
.claude/                Claude Code project context (CLAUDE.md, rules, skills)
```

---

## Key design decisions

1. **No static word lists.** Filtering uses POS tags
   (`get_rejected_pos_tags(lang)`) and learned models, not hand-typed
   word lists.
2. **Multilingual by default.** Qwen 1.5B (MCQ + GER), xlm-roberta
   (MLM veto), oliverguhr fullstop (Step P), paraphrase-MiniLM
   (frontend embeddings) — all multilingual.
3. **Confidence-gated edits.** Step L only edits tokens with
   `avg_logprob > -0.3`. High-confidence tokens are passed through
   verbatim; the LLM never gets to "improve" already-correct text.
4. **Validated cache with consensus.** `data/validated_corrections.json`
   stores misspelling→canonical mappings; `MIN_CONSENSUS=1` allows
   single-match learning while strict gates (fuzz≥75 + dictionary veto)
   prevent common words from being mapped to entity names.
5. **Stage E veto pyramid.** TF-IDF retrieve → cosine shortcuts →
   frequency heuristic → MCQ pre-gates → Qwen MCQ → MLM veto →
   C1 fuzz floor → C2 length tolerance → dictionary veto. Each gate
   addresses a known failure mode of the previous one.
6. **Schema-1 output for the frontend.** Cleaned files emit
   `[start_time, end_time, text]` per segment — exactly what
   `frontend/.../ingest.ts` parses. Schema-2 internals (per-word probs,
   etc.) live only in the in-memory `Segment` dataclass while the
   pipeline runs.

---

## What lives where in the pipeline

The cleaning pipeline is organised as one orchestrator + 13 single-responsibility modules:

- **Tier 1 (cheap, deterministic):** `hallucination_filter.py`, `deduplicator.py`, `domain_normalizer.py`. Drop garbage, merge duplicates, fix obvious compound words. Sub-second per match.
- **Tier 2 (entity correction):** `gazetteer.py` builds the per-match name dict; `ner_extractor.py` finds candidate entities; `entity_corrector.py` runs TF-IDF retrieval + Qwen MCQ judge + xlm-roberta MLM veto + validation gates; `fuzzy_corrector.py` provides the gates.
- **Tier 3 (segment-level edits):** `llm_corrector.py` runs Confidence-gated GER over each segment with low-confidence tokens wrapped in `<>`. `punct_restorer.py` restores casing and punctuation.
- **Glue:** `loader.py` parses ASR JSON; `whisper_runner.py` re-transcribes audio when needed; `report.py` formats results; `config.py` centralises every threshold and model name.

---

## Thesis documentation

* `thesis/INDEX.md` — bundle navigation with quick-reference numbers
* `thesis/pipeline_detailed_walkthrough.md` — auto-generated per-stage walkthrough
* `thesis/whisper_versions_comparison.md` — SoccerNet vs faster-whisper-v3 numbers
* `thesis/search_quality_comparison.md` — RAW vs CLEANED retrieval A/B
* `thesis/lineup_query_expansion_test.md` — query-time expansion alternative
* `thesis/detected_entities.csv` + `thesis/detected_entities_summary.md` — NER verification
* `thesis/gt_dropped_segments.csv` — fair-WER discussion data

---

## Research backing

* **N-best retrieval-augmented entity correction**: Apple RAG-NEC, [arxiv:2409.06062](https://arxiv.org/abs/2409.06062), 2024 — 33-39% rel WER reduction on entity-heavy queries.
* **Confidence-gated GER**: [arxiv:2509.25048](https://arxiv.org/abs/2509.25048), 2025 — 68% rel WER reduction by gating LLM edits on Whisper logprob.
* **GER on Whisper**: [Whispering LLaMA, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.618.pdf); [Improving GER with LoRA, ACL Findings 2025](https://aclanthology.org/2025.findings-acl.125.pdf).
* **Multilingual punctuation restoration**: [oliverguhr/fullstop-punctuation-multilang-large](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large).
* **MLM veto via xlm-roberta**: pseudo-likelihood scoring per Wang et al. 2019.

---

## License & attribution

Bachelor thesis project, Høgskolen i Østfold, May 2026.
SoccerNet dataset © Inria/SoccerNet 2018-2024.
GOAL benchmark © THU-KEG, 2024.

---

## Status

| Component | State |
|---|---|
| Pipeline (Step 0 → P) | ✓ 226 / 226 tests green |
| Validated_corrections cache | ✓ 79 entries, consensus_min=1 |
| Multilingual support | ✓ EN/SV/DE/FR/ES/IT/PT/NL via per-language model maps |
| Frontend (ForzaSearch) | ✓ Indexed and searchable end-to-end (ES + Mistral) |
| Per-stage ablation | ✓ Documented in thesis/ |
