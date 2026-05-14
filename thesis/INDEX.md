# Thesis Bundle — SoccerNet ASR Cleaning Pipeline

This bundle contains everything needed to write the ASR-cleaning section
of the thesis. Match used as benchmark: **Chelsea 1-2 Liverpool,
2016-09-16, Stamford Bridge** (English Premier League).

---

## Quick reference: latest measurements

**Configuration (production / V7 = V3 full):** Stage E + Step L (LLM GER)
+ Step P (Punctuation restorer), all enabled. validated_corrections.json
has 79 curated mappings (consensus_min=1).

| Halv | Raw V3 WER | Cleaned WER | Fair WER ⁱ | Raw Entity-F1 | Cleaned Entity-F1 | Δ F1 (rel) |
|---|---|---|---|---|---|---|
| 1 | 25.56% | 26.31% | ~25.81% | 0.484 | **0.605** | **+25.0%** |
| 2 | 23.86% | 24.43% | ~24.0% | 0.504 | **0.580** | **+15.1%** |

ⁱ Fair WER trekker fra ~0.5pp pga 3 ekte engelsk kommentarsegmenter
som GT (GOAL benchmark) bevisst utelot. Se `gt_dropped_segments.csv`.

**Per-stage timings (full pipeline V7, single match Chelsea-Liverpool):**

| Stage | Wall (s) | Hva |
|---|---|---|
| Step 0 detect_language | 1.1 | langdetect |
| Step 1 build_gazetteer | 0.0 | Labels-caption.json → name dict |
| Step 2 hallucination_filter | 0.2 | alpha-ratio + langdetect ≥15w gate |
| Step 3 deduplicate | 0.0 | rapidfuzz + 3+ collapse |
| Step N nbest_rerank | ~25 | Apple RAG-NEC pattern, FAISS |
| Stage 2A domain_normalize | 0.1 | regex compounds + disfluencies |
| NER extract_entities | 7-274* | spaCy + heuristics + gazetteer fuzz |
| Stage E entity_corrector | 56-380* | TF-IDF + Qwen MCQ + xlm-r MLM veto |
| Step L llm_ger | ~1450 | Qwen2.5-1.5B + xlm-r MLM veto |
| Step P punct_restore | ~464 | oliverguhr fullstop multilang |
| **Total** | **~2245** | ~37 min wall on CPU |

\* NER+Stage E times vary 4× depending on cache hits and detected entity count.

---

## Files in this bundle

### Core architecture & methodology

| File | Purpose |
|---|---|
| `INDEX.md` | This file |
| `pipeline_detailed_walkthrough.md` | Full per-stage walkthrough with diff examples (auto-generated from cleaning_metadata) |
| `thesis_statistics.md` | Earlier statistics doc (legacy; superseded by walkthrough) |

### Empirical data from this match

| File | Purpose |
|---|---|
| `pipeline_run_v7_full.log` | Stdout of full pipeline V7 = production config (E+L+P) |
| `pipeline_run_v6.log` | Ablation V6: Stage E + Step L only (no Step P) |
| `pipeline_run_v5.log` | Ablation V5: Stage E only |
| `detected_entities.csv` | All 766 entities NER detected, with source + duplicate flag |
| `detected_entities_summary.md` | Top-50 most-frequent entities with gazetteer match status |
| `gt_dropped_segments.csv` | 21 hyp-only segments (cleaned has, GT lacks) classified as real/garbage |
| `entity_rejection_analysis.md` | What Stage E rejected and why |

### Diff examples (V2 era — pre-fix)

| File | Purpose |
|---|---|
| `diff_examples/chelsea_liverpool_h1_diff.md` | Raw vs cleaned diffs H1 |
| `diff_examples/chelsea_liverpool_h2_diff.md` | Raw vs cleaned diffs H2 |
| `diff_examples/crystal_palace_h{1,2}_diff_*.md` | Crystal Palace match for cross-validation |

### Pipeline code (`pipeline/`)

The whole `pipeline/` directory is included. Key files:
- `orchestrator.py` — runs all stages with per-stage timing
- `nbest_reranker.py` — Step N (Apple RAG-NEC, multi-signal scoring)
- `entity_corrector.py` — Stage E (TF-IDF retrieve + Qwen MCQ + gates)
- `llm_corrector.py` — Step L (Confidence-gated GER)
- `punct_restorer.py` — Step P (oliverguhr fullstop)
- `hallucination_filter.py` — Tier 1 garbage filter
- `deduplicator.py` — segment dedup + Whisper-loop collapse
- `domain_normalizer.py` — regex normalization
- `ner_extractor.py` — spaCy + heuristics + Rule 3 gazetteer fuzz
- `whisper_runner.py` — faster-whisper transcription
- `gazetteer.py` — Labels-caption → name dict + entity types
- `temporal_chunker.py` — ES-ready rolling-window chunks
- `loader.py` — schema-1/2 ASR JSON parsing
- `config.py` — all thresholds + model paths

### Reproducibility tools (`tools/`)

| File | What it does |
|---|---|
| `build_nbest_chelsea.py` | Re-transcribes audio at T=0.4 + merges to schema-3 nbest |
| `transcribe_alt_for_nbest.py` | Per-segment-progress wrapper around faster-whisper |
| `evaluate_wer.py` | WER + Entity-F1 evaluator (windowed/legacy alignment, --aligned-only mode) |
| `generate_pipeline_walkthrough.py` | Renders per-stage thesis doc from cleaning_metadata |
| `seed_validated_corrections.py` | Scans raw V3 for high-fuzz misspellings → propose validated_cache mappings |
| `dump_detected_entities.py` | Dumps every NER-detected entity to CSV for verification |
| `count_gt_dropped_segments.py` | Counts hyp-only segments + classifies real vs garbage |
| `analyze_entity_rejections.py` | Per-rejection-bucket breakdown of why Stage E rejected candidates |

### Data caches & corrections

| File | Purpose |
|---|---|
| `data/validated_corrections.json` | 79 curated misspelling→canonical mappings (consensus_min=1) |
| `cleaned_data/.../{1,2}_asr_v3_nbest_cleaned.json` | Pipeline output for each half, with cleaning_metadata |
| `cleaned_data/.../es_chunks.json` | Elasticsearch-ready 12s rolling-window chunks |

### Source ASR data (raw V3 + n-best alternative)

| File | Purpose |
|---|---|
| `path/to/SoccerNet/.../{1,2}_asr_v3.json` | Raw faster-whisper output (T=0.0, schema-2) |
| `path/to/SoccerNet/.../{1,2}_asr_v3_nbest.json` | Same + n-best alternatives from T=0.4 pass |
| `path/to/SoccerNet/.../{1,2}_asr_corrected.json` | GOAL human GT (this is the WER reference) |
| `path/to/SoccerNet/.../Labels-caption.json` | Lineup, teams, referee — input to gazetteer |

### Evaluation results

| File | Purpose |
|---|---|
| `evaluation_results/2016-09-16*half1_v3_nbest_wer.md` | WER + F1 markdown per half |
| `evaluation_results/2016-09-16*half2_v3_nbest_wer.md` | Same for half 2 |

---

## Architecture summary (pipeline order)

```
Audio (MP3)
   │
   ▼
[Whisper large-v3]  beam=5, T=0.0  →  1_asr_v3.json (schema-2)
                    beam=1, T=0.4  →  half1_T0.4.json
                                      │
                                      └→ time-aligned merge ──→
                                         1_asr_v3_nbest.json (schema-3)
   │
   ▼  pipeline/orchestrator.py:clean_match
   │
   ├── Step 0  detect_commentary_language
   ├── Step 1  build_gazetteer (from Labels-caption.json)
   ├── Step 2  hallucination_filter
   ├── Step 3  deduplicate + Whisper-loop collapse
   ├── Step N  nbest entity-grounded reranker (Apple RAG-NEC)
   ├── Step 2A domain_normalizer (regex football compounds)
   ├── NER     extract_entities (spaCy + heuristics + gazetteer fuzz)
   ├── Step E  entity_corrector (TF-IDF + Qwen MCQ + xlm-r MLM veto + gates)
   ├── Step L  llm_corrector (Confidence-gated GER, Qwen 1.5B)
   └── Step P  punct_restorer (oliverguhr fullstop multilang)
        │
        ▼
   {1,2}_asr_v3_nbest_cleaned.json + es_chunks.json
```

## Key design decisions (for thesis discussion section)

1. **No static word lists.** All filtering uses POS tags or learned models — see `pipeline/config.py:REJECTED_POS_TAGS`.
2. **Multilingual by default.** Qwen 1.5B (MCQ + GER), xlm-roberta (MLM veto), oliverguhr (punct), paraphrase-MiniLM (FAISS) all multilingual.
3. **Confidence-gated edits.** Step L only edits tokens with `avg_logprob > -0.3` (Confidence-Guided EC, arxiv:2509.25048).
4. **Validated cache with consensus.** `validated_corrections.json` requires `MIN_CONSENSUS=1` match (relaxed from 3) to short-circuit MCQ — gives OLD-style aggressive correction.
5. **N-best reranker** (Apple RAG-NEC, arxiv:2409.06062). Multi-signal: entity_grounding + edit_distance + Whisper_confidence − length_penalty. Hard cap `MAX_LENGTH_DIFF_WORDS=5` prevents structural drift from T=0.4 hallucinations.
6. **Hard length cap** prevents the "1073 extra words" merger bug discovered in V2 ablation.

## What still needs work

- Cross-match validation cache promotion: currently consensus=1 means single-match wins; raise to 2 once we have 5+ matches indexed to reduce poison risk.
