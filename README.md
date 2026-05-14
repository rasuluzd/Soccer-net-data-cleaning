# SoccerNet ASR Cleaning Pipeline

Multi-stage NLP pipeline that cleans Whisper ASR football commentary into
search-friendly transcripts and ships them to Elasticsearch for the
**ForzaSearch** retrieval-augmented video-clip search frontend.

Bachelor thesis project. Benchmark match: **Chelsea 1-2 Liverpool, 2016-09-16**
(English Premier League) against the GOAL human ground-truth.

---

## Headline numbers (V8, full pipeline)

| Halv | Raw V3 WER | Cleaned WER | Fair WER ⁱ | Raw Entity-F1 | **Cleaned Entity-F1** | Δ F1 (rel) |
|---|---|---|---|---|---|---|
| 1 | 25.56% | 26.21% | ~25.71% | 0.484 | **0.603** | **+24.5%** |
| 2 | 23.86% | 24.30% | ~23.80% | 0.504 | **0.578** | **+14.7%** |

ⁱ "Fair WER" subtracts ~0.5pp pga 3 ekte engelske kommentarsegmenter
GOAL-curators bevisst utelot. Se `tools/count_gt_dropped_segments.py`.

**Entity-F1 er hovedmetric** for downstream event-detection og søk —
WER er nesten flat fordi korreksjoner som er semantisk riktige
(`Davi → David`) telles som substitusjoner mot GT.

---

## Architecture

```
audio.mp3
   │
   ▼
[faster-whisper large-v3]  beam=5, T=0.0  →  1_asr_v3.json (schema-2)
                           beam=1, T=0.4  →  half1_T0.4.json
                                            │
                                            └→ time-aligned merge ──→
                                               1_asr_v3_nbest.json (schema-3)
   │
   ▼  pipeline/orchestrator.py:clean_match
   │
   ├── Step 0  Detect commentary language        (langdetect)
   ├── Step 1  Build gazetteer + entity types    (Labels-caption.json)
   ├── Step 2  Hallucination filter              (alpha-ratio + langdetect ≥15w)
   ├── Step 3  Deduplicate + Whisper-loop collapse (rapidfuzz, 3+ rule)
   ├── Step N  N-best entity-grounded reranker   (Apple RAG-NEC pattern)
   ├── Stage 2A Domain normalizer                (regex compounds + disfluencies)
   ├── NER     Entity extraction                 (spaCy + heuristics + gazetteer fuzz)
   ├── Stage E Validated entity corrector        (TF-IDF + Qwen MCQ + xlm-r MLM veto + gates)
   ├── Step L  Confidence-gated GER              (Qwen2.5-1.5B + xlm-r MLM veto + drift guard)
   └── Step P  Punctuation + casing restoration  (oliverguhr fullstop multilang)
        │
        ▼
   {1,2}_asr_v3_nbest_cleaned.json + es_chunks.json
        │
        ▼  tools/export_to_frontend.py
   frontend/forzasearch-final/matches/<id>/kamp.json
        │
        ▼  npm run ingest (MiniLM-ONNX + ES bulk)
   Elasticsearch forzasearch-windows index (BM25 + dense_vector)
        │
        ▼  Hybrid BM25 + k-NN search → Mistral 7B rerank → answer
   Frontend (Next.js) returns video clip with seek-to-segment HLS player
```

| Stage | Wall (s, full pipeline V8) | Notes |
|---|---|---|
| Step 0 detect_language | 1.1 | langdetect on transcript sample |
| Step 1 build_gazetteer | 0.0 | Labels-caption.json → name dict |
| Step 2 hallucination_filter | 0.2 | alpha-ratio + langdetect ≥15w gate |
| Step 3 deduplicate | 0.0 | rapidfuzz + 3+ collapse |
| Step N nbest_rerank | ~25 | Apple RAG-NEC, FAISS over gazetteer |
| Stage 2A domain_normalize | 0.1 | regex football compounds |
| NER extract_entities | 7 | spaCy + heuristic + Rule-3 gazetteer fuzz (NER_FLOOR=75) |
| Stage E entity_corrector | 52 | TF-IDF + Qwen MCQ + 79-entry validated_cache |
| Step L llm_ger | 1452 | Qwen 1.5B + xlm-r MLM veto |
| Step P punct_restore | 464 | oliverguhr fullstop multilang |
| **Total** | **~2245** | ~37 min wall on CPU (single match) |

---

## Quick start

### 1. Install Python deps + spaCy + Whisper models

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Multilingual extras (optional):
# python -m spacy download sv_core_news_sm de_core_news_sm fr_core_news_sm
```

### 2. Run the pipeline on a single match

```bash
# Default reads {1,2}_asr.json (schema-1 stock Whisper output)
python run_pipeline.py --match "Chelsea 1 - 2 Liverpool"

# To run on the n-best-enriched ASR (preferred):
ASR_INPUT_VARIANT=_v3_nbest python run_pipeline.py --match "Chelsea 1 - 2 Liverpool"
```

Output:
* `cleaned_data/<league>/<season>/<match>/commentary_data/{1,2}_asr*_cleaned.json`
* `cleaned_data/.../es_chunks.json` (12s rolling-window chunks for ES)

### 3. Evaluate WER + Entity-F1 vs GOAL human GT

```bash
python tools/evaluate_wer.py --match "Chelsea" --half 1
python tools/evaluate_wer.py --match "Chelsea" --half 2
# Add --aligned-only to skip hyp-only segments (fair WER):
python tools/evaluate_wer.py --match "Chelsea" --half 1 --aligned-only --alignment-mode windowed
```

### 4. Run all tests (244 unit/integration tests)

```bash
pytest tests/ -v
```

### 5. End-to-end with the ForzaSearch frontend

```powershell
# Convert pipeline output to frontend's kamp.json format
python tools/export_to_frontend.py `
  --match "Chelsea 1 - 2 Liverpool" `
  --frontend frontend/forzasearch-final `
  --id chelsea-liverpool-2016 `
  --title "Chelsea 1-2 Liverpool" `
  --subtitle "Premier League 2016-09-16"

# Start ES + ingest + dev server (Docker Desktop must be running)
pwsh tools/start_frontend_e2e.ps1

# Then open http://localhost:3000 and pick the match in the dropdown.
```

---

## Key design decisions

1. **No static word lists.** Filtering uses POS tags
   (`get_rejected_pos_tags(lang)`) and learned models.
2. **Multilingual by default.** Qwen 1.5B (MCQ + GER), xlm-roberta
   (MLM veto), oliverguhr fullstop (Step P), paraphrase-MiniLM (Step N
   FAISS) — all multilingual.
3. **Confidence-gated edits.** Step L only edits tokens with
   `avg_logprob > -0.3` per Confidence-Guided EC (arxiv:2509.25048).
4. **Validated cache with consensus.** `data/validated_corrections.json`
   stores misspelling→canonical mappings; `MIN_CONSENSUS=1` allows
   single-match learning while still requiring fuzz≥75 + dictionary veto
   gates so common words like `that`/`they`/`been` cannot become entities.
5. **N-best reranker** (Apple RAG-NEC, arxiv:2409.06062). Multi-signal:
   `entity_grounding + edit_distance + Whisper_confidence − length_penalty`.
   Hard cap `MAX_LENGTH_DIFF_WORDS=5` prevents structural drift from
   T=0.4 hallucinations.
6. **Stage E veto pyramid.** TF-IDF retrieve → cosine shortcuts →
   frequency heuristic → MCQ pre-gates → Qwen MCQ → MLM veto → C1 fuzz floor
   → C2 length tolerance → dictionary veto. Each gate addresses a known
   failure mode of the previous one.

---

## Project layout

```
pipeline/
  orchestrator.py         clean_match() — runs all stages + per-stage timing
  whisper_runner.py       faster-whisper transcription
  loader.py               schema-1/2 ASR JSON parsing
  config.py               all thresholds, model paths, multilingual maps
  hallucination_filter.py Tier 1: alpha-ratio + langdetect + repeat-cluster
  deduplicator.py         Tier 1: rapidfuzz + 3+ word collapse
  nbest_reranker.py       Step N: Apple RAG-NEC multi-signal scoring
  domain_normalizer.py    Stage 2A: football compounds + disfluencies
  ner_extractor.py        NER: spaCy + 3 heuristics
  gazetteer.py            Labels-caption → name dict + entity types
  entity_corrector.py     Stage E: TF-IDF + Qwen MCQ + xlm-r MLM veto
  fuzzy_corrector.py      Validation gates (passes_conservative_gates)
  llm_corrector.py        Step L: Qwen2.5-1.5B GER + MLM veto + drift guard
  punct_restorer.py       Step P: oliverguhr fullstop multilang
  temporal_chunker.py     12s rolling-window chunks for ES
  report.py               CleaningResult → human-readable summary

tools/
  build_nbest_chelsea.py        Re-transcribe T=0.4 + merge to schema-3 nbest
  transcribe_alt_for_nbest.py   Per-segment-progress wrapper
  evaluate_wer.py               WER + Entity-F1 vs GOAL GT
  generate_pipeline_walkthrough.py   Renders thesis doc from cleaning_metadata
  seed_validated_corrections.py Scans raw V3 → propose new validated_cache mappings
  dump_detected_entities.py     Dumps every NER detection to CSV for verification
  count_gt_dropped_segments.py  Counts hyp-only segments (real vs garbage)
  analyze_entity_rejections.py  Per-bucket breakdown of Stage E rejections
  export_to_frontend.py         Convert cleaned output → frontend kamp.json
  start_frontend_e2e.ps1        Boot ES + ingest + Next.js dev server

tests/                  244 unit + integration tests, all green
data/
  validated_corrections.json    79 curated misspelling→canonical mappings
cleaned_data/           Pipeline output (per-match, mirrors dataset structure)
evaluation_results/     WER + F1 markdown reports per half
frontend/forzasearch-final/   Next.js + ES + Mistral RAG search frontend
thesis/                 Walkthrough docs, diff examples, entity dumps
.claude/                Project context for Claude Code (CLAUDE.md, rules, skills)
```

---

## Thesis documentation

* `thesis/INDEX.md` — bundle index with quick-reference numbers
* `thesis/pipeline_detailed_walkthrough.md` — auto-generated per-stage walkthrough
* `thesis/detected_entities.csv` + `thesis/detected_entities_summary.md` — NER verification (766 entities, top-50 frequencies)
* `thesis/gt_dropped_segments.csv` — fair-WER discussion data (21 hyp-only segments classified)
* `thesis/diff_examples/` — concrete before/after diffs from V2-era runs
* `thesis_bundle_*.zip` (on Desktop) — ready-to-upload archive for Claude in Word

---

## Research backing

* **N-best entity reranking**: Apple RAG-NEC, [arxiv:2409.06062](https://arxiv.org/abs/2409.06062), 2024 — 33-39% rel WER reduction on entity-heavy queries.
* **Confidence-gated GER**: [arxiv:2509.25048](https://arxiv.org/abs/2509.25048), 2025 — 68% rel WER reduction by gating edits on Whisper logprob.
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
| Pipeline (Step 0 → P) | ✓ Production, 244/244 tests green |
| N-best reranker | ✓ Step N wired, multi-signal scoring + hard length cap |
| Validated_corrections cache | ✓ 79 entries, consensus_min=1 |
| Multilingual support | ✓ EN/SV/DE/FR/ES/IT/PT/NL via language-conditional models |
| Frontend (ForzaSearch) | ✓ Indexed and searchable end-to-end |
| Diarization (pyannote) | ◯ Implemented, off by default (slow on CPU) |
| Cross-match consensus promotion | ◯ Available (raise MIN_CONSENSUS to 2-3 once 5+ matches indexed) |
