# SoccerNet ASR Cleaning Pipeline

Multi-stage NLP pipeline that cleans Whisper ASR football commentary into
search-friendly transcripts and ships them to Elasticsearch for the
**ForzaSearch** retrieval-augmented video-clip search frontend.

Bachelor thesis project, Høgskolen i Østfold, May 2026. Benchmark match:
**Chelsea 1–2 Liverpool, 2016-09-16** (English Premier League) against the
GOAL human ground-truth.

---

## Headline numbers (full pipeline, stock Whisper input)

WER and Entity F1 on Chelsea-Liverpool (GOAL human GT), corpus-level after
time-range alignment (0.75s tolerance):

| Halv | Raw WER | Cleaned WER | Δ WER | Raw Entity-F1 | **Cleaned Entity-F1** | Δ F1 |
|---|---|---|---|---|---|---|
| 1 | 29.81% | **28.48%** | −1.33pp (−4.5% rel.) | 0.6203 | **0.6451** | +0.0248 |
| 2 | 24.84% | **23.54%** | −1.30pp (−5.2% rel.) | 0.5981 | **0.6084** | +0.0103 |

Pipeline is net-positive on both WER and Entity F1 vs raw SoccerNet-bundled
Whisper output, with no manual per-match tuning.

### Step L (LLM GER) ablation finding

A targeted ablation removing Step L from the pipeline produced essentially
identical WER numbers:

| Halv | raw | cleaned (no Step L) | cleaned (with Step L) |
|---|---|---|---|
| 1 WER | 29.71% | **28.16%** | 28.16% (Δ=+0.00pp) |
| 2 WER | 24.68% | **23.35%** | 23.37% (Δ=−0.02pp) |

Conclusion: on the Chelsea-Liverpool English benchmark, the confidence-gated
Qwen 1.5B GER stage produces 29 accepted edits but **does not improve WER**.
72 % of its edits are punctuation tweaks that Step P (oliverguhr fullstop)
performs without LLM cost; the remaining 8 word-edits split roughly 3 wins
to 5 misses for a net zero. The validation gates (length anomaly, editable
drift, MLM veto) prevented 49 bad LLM proposals from being applied — the
gates work, but the underlying LLM contribution on this benchmark is not
net-positive after gate filtering. Step L likely returns more on noisier
audio (Swedish KB-Whisper) or with sharper confidence signals from a
re-transcribed schema-2 input.

### Whisper engine — re-transcription with faster-whisper

Re-transcribing the same audio with `Systran/faster-whisper-large-v3`
(local, with lineup hotword biasing) instead of using SoccerNet's bundled
stock Whisper output gives a measurable WER win:

| Halv | SoccerNet stock (v2) WER | faster-whisper-v3 WER | Δ WER |
|---|---|---|---|
| 1 | 29.71% | **25.36%** | −4.35pp (−14.6% rel.) |
| 2 | 24.68% | **23.64%** | −1.04pp (−4.2% rel.) |

Trade-off: v3 has **higher entity precision but lower recall** (more
proposed entities → more false positives, but also recovers names like
*Anfield, Klopp, Lallana, Mignolet, Mourinho* that v2 missed). For
search-driven downstream use (ForzaSearch), the WER win and recall on
specific player names is what matters; v3 is the recommended ASR input.

---

## Architecture

```
audio.mp3
   │
   ▼  whisper_runner.transcribe (faster-whisper large-v3)
   │      beam=5, word_timestamps=True, condition_on_previous_text=True
   │      lineup-aware initial_prompt + hotwords from Labels-caption.json
   │
   ▼  {1,2}_asr_v3.json (dict-style segments with per-word probs)
   │
   ▼  pipeline/orchestrator.py:clean_match
   │
   ├── Step 0   detect_commentary_language        (langdetect)
   ├── Step 1   build_gazetteer                    (Labels-caption.json → name dict)
   ├── Step 2   filter_segments                    (alpha-ratio + langdetect ≥15 words)
   ├── Step 3   deduplicate + Whisper-loop collapse (rapidfuzz, 3+ rule)
   ├── Stage 2A domain_normalize                   (regex compounds + disfluencies)
   ├── NER      extract_entities_batch             (spaCy + 3 heuristics + Rule 3 gazetteer fuzz)
   ├── Stage E  entity_corrector.correct_match     (TF-IDF + Qwen 1.5B MCQ + xlm-r MLM veto)
   ├── Step L   llm_corrector.correct_match        (Confidence-gated Qwen GER + MLM veto + drift guard)
   └── Step P   restore_punctuation_batch          (oliverguhr fullstop multilang)
        │
        ▼  cleaned_data/.../{1,2}_asr_cleaned.json  (list-style [start, end, text])
        │
        ▼  tools/export_to_frontend.py
        │
        ▼  frontend/forzasearch-final/matches/<id>/kamp.json
        │
        ▼  npm run ingest  (MiniLM-ONNX embeddings + ES bulk)
        │
        ▼  Elasticsearch hybrid BM25 + k-NN search
        │
        ▼  Mistral 7B (Ollama) RAG re-rank + answer generation
        │
        ▼  Next.js frontend → seek-to-segment HLS player
```

---

## Stage timings on Chelsea-Liverpool (single match, CPU)

Measured on the May 2026 ablation run (Windows, Python 3.11, int8 inference):

| Stage | Wall (s) | Notes |
|---|---|---|
| Step 0  detect_language | 1.5 | langdetect on transcript sample |
| Step 1  build_gazetteer | 0.0 | parses Labels-caption.json (99 names) |
| Step 2  hallucination_filter | 0.5 | alpha-ratio + langdetect (≥15 words) |
| Step 3  deduplicate | 0.0 | rapidfuzz + 3-cluster collapse |
| Stage 2A domain_normalize | 0.2 | regex football compounds, disfluencies |
| NER     extract_entities | 200 | spaCy en_core_web_sm + 3 heuristics |
| Stage E entity_corrector | 39 | TF-IDF retrieve + Qwen MCQ + MLM veto + cache |
| Step L  llm_ger | 1007 | Qwen 1.5B GGUF q4_k_m + xlm-r MLM veto |
| Step P  punct_restore | 304 | oliverguhr fullstop multilang (first-load cost) |
| **Total** | **~1551** | ~26 min wall on CPU |
| Total without Step L | **~242** | ~4 min wall (no measurable WER loss) |

---

## Quick start

### 1. Install
```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Multilingual extras (Swedish, French — used by detect_commentary_language fallback):
# python -m spacy download sv_core_news_sm fr_core_news_sm
```

### 2. Run the pipeline on a single match
```bash
python run_pipeline.py --match "Chelsea 1 - 2 Liverpool"
```

Output:
* `cleaned_data/<league>/<season>/<match>/commentary_data/{1,2}_asr_cleaned.json`
* per-segment `cleaning_metadata` block with stage timings, telemetry, and
  the full corrections log

### 3. Evaluate WER + Entity-F1 vs GOAL ground truth
```bash
python tools/evaluate_wer.py --match "Chelsea" --half 1
python tools/evaluate_wer.py --match "Chelsea" --half 2

# Compare a non-default ASR variant against the same GT:
python tools/evaluate_wer.py --match "Chelsea" --half 1 --variant "_v3" --ablate
```

### 4. Run all tests (220 unit + integration)
```bash
pytest tests/ -v
```

### 5. Re-transcribe with faster-whisper (optional, slow on CPU)
```bash
PYTHONIOENCODING=utf-8 python tools/retranscribe_kb_whisper.py \
  --audio whisper_cache/audio/<match>/half1.mp3 \
  --labels path/to/.../Labels-caption.json \
  --output path/to/.../commentary_data/1_asr_v3.json \
  --language en --condition-on-previous
```

### 6. End-to-end with the ForzaSearch frontend
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
pipeline/                          (15 modules, single-responsibility)
  orchestrator.py         clean_match() — runs Step 0 → P with per-stage timing
  whisper_runner.py       faster-whisper transcription with lineup hotwords
  loader.py               ASR JSON parsing (list-style + dict-style schemas)
  config.py               all thresholds, paths, multilingual model maps
  hallucination_filter.py Step 2: alpha-ratio + langdetect + repeat-cluster filter
  deduplicator.py         Step 3: consecutive-segment dedup via rapidfuzz
  domain_normalizer.py    Stage 2A: regex football compounds + disfluencies
  ner_extractor.py        spaCy NER + 3 heuristics (lone-word, POS-filtered, gazetteer fuzz)
  gazetteer.py            Labels-caption → variant→canonical dict + entity-type map
  entity_corrector.py     Stage E: TF-IDF retrieve + Qwen MCQ + xlm-r MLM veto + gates
  fuzzy_corrector.py      Validation gates (passes_conservative_gates, C1 fuzz, C2 length)
  llm_corrector.py        Step L: confidence-gated Qwen GER + MLM veto + drift guard
  punct_restorer.py       Step P: oliverguhr fullstop multilang
  report.py               CleaningResult → human-readable summary

tools/
  evaluate_wer.py                 WER + Entity-F1 vs GOAL GT (legacy + windowed alignment)
  retranscribe_kb_whisper.py      CLI wrapper around whisper_runner.transcribe
  export_to_frontend.py           Convert cleaned output → frontend kamp.json (list schema)
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
  fetch_match_metadata.py         Pull lineup + match metadata
  insert_asr_results_into_docx.py Inject §4.2.1.7-9 results into bachelor.docx
  integrate_asr_into_thesis.py    Inject ASR pipeline references across chapters 1-3

tests/                  220 unit + integration tests, all green
data/
  validated_corrections.json    80 curated misspelling→canonical mappings (the
                                 'learned dictionary' that grows over runs)
cleaned_data/           Pipeline output (per-match, mirrors dataset structure)
evaluation_results/     WER + F1 markdown reports per half
frontend/forzasearch-final/   Next.js + ES + Mistral RAG search frontend
thesis/                 Walkthrough docs, diff examples, entity dumps
.claude/                Claude Code project context (CLAUDE.md, rules, skills)
```

---

## Key design decisions

1. **No static word lists.** Filtering uses POS tags
   (`get_rejected_pos_tags(lang)`) and learned dictionaries (pyspellchecker /
   pyenchant), never hand-typed exclusion lists. Scales to new languages by
   adding a per-language entry in `config.SPACY_MODELS` and `ASR_MODELS`.

2. **Multilingual by default.** The three load-bearing learned components are
   all multilingual: Qwen 1.5B (Stage E MCQ judge and Step L GER),
   xlm-roberta-base (MLM veto for both stages), oliverguhr fullstop
   (Step P punctuation). spaCy and faster-whisper use language-specific
   models routed through `pipeline.config.get_*_model(lang)`.

3. **Confidence-gated edits.** Step L only edits tokens with
   `avg_logprob ≤ -0.3` (paper default). High-confidence tokens are passed
   through verbatim; the LLM never gets to "improve" already-correct text.
   This implements the Confidence-Guided GER pattern from
   [arxiv:2509.25048](https://arxiv.org/abs/2509.25048).

4. **Validated cache with consensus.** `data/validated_corrections.json`
   stores misspelling→canonical mappings; `MIN_CONSENSUS=1` allows
   single-match learning while strict gates (fuzz≥75 + dictionary veto)
   prevent common words from being mapped to entity names. 80 entries
   currently.

5. **Stage E veto pyramid.** TF-IDF retrieve → cosine shortcuts →
   frequency heuristic → MCQ pre-gates → Qwen MCQ → MLM veto →
   C1 fuzz floor → C2 length tolerance → dictionary veto. Each gate
   addresses a known failure mode of the previous one.

6. **Simple JSON output for the frontend.** Cleaned files emit
   `[start_time, end_time, text]` per segment — exactly what
   `frontend/.../ingest.ts` parses. Per-word confidence metadata stays
   in-memory only during pipeline execution.

7. **Test-driven changes.** Every bug fix requires a regression test that
   fails before the fix and passes after (see
   `.claude/rules/00-dev-workflow.md`). 220 tests cover all 13 pipeline
   modules + multilingual variants + WER alignment.

---

## What lives where in the pipeline

The cleaning pipeline is organised as one orchestrator + 13 single-responsibility modules:

- **Tier 1 (cheap, deterministic):** `hallucination_filter.py`, `deduplicator.py`, `domain_normalizer.py`. Drop garbage, merge duplicates, fix obvious compound words. Sub-second per match.
- **Tier 2 (entity correction):** `gazetteer.py` builds the per-match name dict; `ner_extractor.py` finds candidate entities; `entity_corrector.py` runs TF-IDF retrieval + Qwen MCQ judge + xlm-roberta MLM veto + validation gates; `fuzzy_corrector.py` provides the gates.
- **Tier 3 (segment-level edits):** `llm_corrector.py` runs Confidence-gated GER over each segment with low-confidence tokens wrapped in `<>`. `punct_restorer.py` restores casing and punctuation.
- **Glue:** `loader.py` parses ASR JSON; `whisper_runner.py` re-transcribes audio when needed; `report.py` formats results; `config.py` centralises every threshold and model name.

---

## Test suite (220 tests)

Highlighted tests worth referencing in the thesis:

| Test | What it demonstrates |
|---|---|
| `test_hallucination_filter.py::test_short_english_football_phrase_kept` | Bug 3 — langdetect misclassified "get a good tackling, get a good passing" as Afrikaans; fixed by `MIN_WORDS_FOR_LANGDETECT=15` |
| `test_deduplicator.py::test_punctuation_variants_merge` | Whisper loops ("Sterling. Sterling! Sterling…") collapse across punctuation variants |
| `test_entity_corrector.py::test_mcq_min_token_len_gate_blocks_short_tokens` | The Kane→Mane defence — short-token gate (≥85 fuzz) blocks short-name swaps |
| `test_entity_corrector.py::test_validation_gate_blocks_dictionary_word_mcq_pick` | Dictionary veto stops Saturday→Sturridge style false positives |
| `test_entity_corrector.py::test_record_promotes_at_consensus_boundary` | Validated cache requires ≥3 independent matches before short-circuiting MCQ |
| `test_entity_corrector.py::test_mlm_veto_rejects_pick` | xlm-roberta MLM veto on MCQ proposals |
| `test_llm_corrector.py::test_wrap_low_confidence_with_word_probs` | Confidence-Guided GER — only low-logprob tokens get wrapped as editable |
| `test_llm_corrector.py::test_veto_rejects_when_mlm_prefers_original_strongly` | Step L MLM veto with `MLM_VETO_RATIO=1.5` |
| `test_punct_restorer.py::test_segment_metadata_preserved_when_text_changes` | Step P additive-only — per-token metadata survives |
| `test_multilingual.py::test_swedish_spacy_model` | Language-conditional model dispatch — same pipeline code for EN/SV |

---

## Thesis documentation

* `thesis/INDEX.md` — bundle navigation with quick-reference numbers
* `thesis/pipeline_detailed_walkthrough.md` — auto-generated per-stage walkthrough
* `thesis/whisper_versions_comparison.md` — SoccerNet vs faster-whisper-v3 numbers
* `thesis/search_quality_comparison.md` — RAW vs CLEANED retrieval A/B
* `thesis/lineup_query_expansion_test.md` — query-time expansion alternative
* `thesis/detected_entities.csv` + `thesis/detected_entities_summary.md` — NER verification
* `thesis/gt_dropped_segments.csv` — fair-WER discussion data
* `evaluation_results/2016-09-16_-_22-00_Chelsea_1_-_2_Liverpool_half*_wer.md` — per-half WER reports

---

## Research backing

* **N-best retrieval-augmented entity correction**: Apple RAG-NEC, [arxiv:2409.06062](https://arxiv.org/abs/2409.06062), 2024 — pattern used by Stage E (TF-IDF retrieval against per-match gazetteer).
* **Confidence-gated GER**: [arxiv:2509.25048](https://arxiv.org/abs/2509.25048), 2025 — Step L's logprob-gating directly implements this.
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
| Pipeline (Step 0 → P) | ✓ 220 / 220 tests green |
| Validated_corrections cache | ✓ 80 entries, consensus_min=1 |
| Multilingual support | ✓ EN / SV / NO / DA / FR / ES / IT / PT / NL via per-language model maps |
| Frontend (ForzaSearch) | ✓ Indexed and searchable end-to-end (ES + Mistral) |
| Per-stage ablation | ✓ Documented above (Step L = +0.00pp WER on Chelsea-Liverpool) |
| Whisper engine comparison | ✓ v3 wins WER by 4.35pp (half 1) / 1.04pp (half 2) over SoccerNet stock |
