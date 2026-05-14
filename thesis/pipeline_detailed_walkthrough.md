# Pipeline Detailed Walkthrough — Chelsea vs Liverpool 2016-09-16

*Generated automatically from `cleaning_metadata.stage_timings` and the cleaned-vs-raw segment diff. Source: `tools/generate_pipeline_walkthrough.py`.*

## Overview

- **Match:** 2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool
- **Detected language:** en
- **Halves analysed:** [1, 2]

## Per-Stage Wall Time (seconds)

| Stage | H1 | H2 | Total |
|---|---|---|---|
| step0_detect_language | 1.14 | 1.14 | 2.28 |
| step1_build_gazetteer | 0.00 | 0.00 | 0.00 |
| step2_hallucination_filter | 0.22 | 0.22 | 0.44 |
| step3_deduplicate | 0.01 | 0.01 | 0.02 |
| stepN_nbest_rerank | 25.04 | 25.04 | 50.08 |
| step2A_domain_normalize | 0.12 | 0.12 | 0.24 |
| stepNER_extract_entities | 6.64 | 6.64 | 13.28 |
| stepE_entity_corrector | 142.39 | 142.39 | 284.78 |
| stepL_llm_ger | 1067.89 | 1067.89 | 2135.78 |
| stepP_punct_restore | 187.28 | 187.28 | 374.57 |
| total_pipeline | 1430.74 | 1430.74 | 2861.47 |

## Per-Stage Detail

### Step 0: Language Detection

- **Module:** `pipeline/hallucination_filter.py:detect_commentary_language`
- **Model:** langdetect (port of Google language-detection)

Sample the first ~1000 characters of the transcript and predict the commentary language. The result is propagated to every downstream stage so each picks the right model (spaCy, Whisper, MLM veto).

**Per-half stats:**

| Half | Wall time (s) | Notes |
|---|---|---|
| 1 | 1.14 |  |
| 2 | 1.14 |  |

**Representative diffs:**

*(no changes recorded by this stage on this match — likely a no-op given the inputs)*

### Step 1: Build Gazetteer from Labels-caption.json

- **Module:** `pipeline/gazetteer.py:build_gazetteer`
- **Model:** —

Read the match's Labels-caption.json (lineup, teams, referee) and produce a name-variant dict (`gazetteer`) and typed entity map (player/team/referee/coach). This is the canonical name set that Stage E retrieval and the n-best reranker score candidates against.

**Per-half stats:**

| Half | Wall time (s) | Notes |
|---|---|---|
| 1 | 0.00 |  |
| 2 | 0.00 |  |

**Representative diffs:**

*(no changes recorded by this stage on this match — likely a no-op given the inputs)*

### Step 2: Hallucination Filter

- **Module:** `pipeline/hallucination_filter.py:filter_segments`
- **Model:** langdetect, regex

Remove segments that are obvious ASR garbage. Rules: empty, non-Latin script, alpha-ratio < 0.50, wrong language family, and clustered repeated single-word callouts. The langdetect gate only fires for segments ≥15 words to avoid misclassifying short football phrases.

**Per-half stats:**

| Half | Wall time (s) | Notes |
|---|---|---|
| 1 | 0.22 | removed 2 |
| 2 | 0.22 | removed 2 |

**Representative diffs:**

- H1 `[433]` removed `24` *(reason: low_alpha_ratio (0.00))*
- H1 `[482]` removed `10-2` *(reason: low_alpha_ratio (0.00))*
- H2 `[28]` removed `4-3` *(reason: low_alpha_ratio (0.00))*
- H2 `[767]` removed `21 and 35` *(reason: low_alpha_ratio (0.43))*

### Step 3: Deduplicate + Whisper-loop Collapse

- **Module:** `pipeline/deduplicator.py + pipeline.orchestrator._collapse_repeated_words`
- **Model:** rapidfuzz

Merge consecutive near-duplicate segments (rapidfuzz ratio ≥ 95) and collapse Whisper-loop word repetitions (3+ consecutive identical tokens → 1). Two-repeats are kept because GT has legitimate forms like "well, well" and "starry, starry night".

**Per-half stats:**

| Half | Wall time (s) | Notes |
|---|---|---|
| 1 | 0.01 | removed 0 duplicates |
| 2 | 0.01 | removed 1 duplicates |

**Representative diffs:**

- H2 `[450]` collapsed `Hazard`

### Step N: N-best Entity-Grounded Reranking

- **Module:** `pipeline/nbest_reranker.py:rerank_match`
- **Model:** sentence-transformers + FAISS

Apple RAG-NEC pattern (arxiv:2409.06062). For each segment with multiple beam hypotheses, score each candidate by sum of max-cosine to gazetteer canonical names (FAISS index over paraphrase-multilingual-MiniLM embeddings). A length-penalty blocks structurally distant hypotheses. Pass-through when n-best is missing or gazetteer is empty.

**Per-half stats:**

| Half | Wall time (s) | Notes |
|---|---|---|
| 1 | 25.04 | 0 of 0 re-picked |
| 2 | 25.04 | 0 of 0 re-picked |

**Representative diffs:**

*(no changes recorded by this stage on this match — likely a no-op given the inputs)*

### Stage 2A: Domain Normalizer

- **Module:** `pipeline/domain_normalizer.py:DomainNormalizer`
- **Model:** regex

Football-specific text normalization: disfluency removal (uh, um, eh), unambiguous compound merging (off side → offside, goal keeper → goalkeeper), repeated punctuation cleanup, whitespace normalization. The rule set is language-aware (English/Swedish/German). Empirically harmful rules ("half time → halftime", "line up → lineup") were removed after WER analysis vs GOAL human GT.

**Per-half stats:**

| Half | Wall time (s) | Notes |
|---|---|---|
| 1 | 0.12 |  |
| 2 | 0.12 |  |

**Representative diffs:**

*(no changes recorded by this stage on this match — likely a no-op given the inputs)*

### NER: Entity Extraction (spaCy + heuristics)

- **Module:** `pipeline/ner_extractor.py:extract_entities_batch`
- **Model:** en_core_web_sm (or sv/de/... per language)

Detect candidate entities for Stage E. Combines spaCy NER (PERSON/ORG/GPE/FAC) with three heuristics: (1) capitalised tokens that look like proper nouns, (2) POS-filtered tokens, (3) gazetteer fuzz-match — any token within fuzz.ratio≥65 of a gazetteer canonical (catches ASR mishearings whose surface is a real English word, e.g. "storage" → Sturridge).

**Per-half stats:**

| Half | Wall time (s) | Notes |
|---|---|---|
| 1 | 6.64 |  |
| 2 | 6.64 |  |

**Representative diffs:**

*(no changes recorded by this stage on this match — likely a no-op given the inputs)*

### Stage E: Validated Entity Corrector

- **Module:** `pipeline/entity_corrector.py:correct_match`
- **Model:** TF-IDF char-bigram + Qwen2.5-1.5B-Instruct (GGUF) + xlm-roberta

Replaces the legacy fuzzy/phonetic/context cascade. Per detected entity: (1) check validated cross-match cache, (2) TF-IDF char-bigram retrieve top-K=5 from gazetteer, (3) per-match decision cache, (4) frequency heuristic (≥5× in match → reject), (5) shortcut accept/reject by cosine, (6) MCQ judge with Qwen picking A/B/C/D=keep/E=unsure, (7) MLM veto via xlm-roberta pseudo-logprob, (8) validation gates (dictionary veto, fuzzy floor, length tolerance).

**Per-half stats:**

| Half | Wall time (s) | Notes |
|---|---|---|
| 1 | 142.39 | applied 9 entity corrections |
| 2 | 142.39 | applied 5 entity corrections |

**Representative diffs:**

- H1 `[21]` `Haspilicueta` → `Azpilicueta` *(method: mcq_judge, score 87.0)*
- H1 `[37]` `Jürgen Klopp` → `Jurgen Klopp` *(method: mcq_judge, score 86.1)*
- H1 `[48]` `Jürgen Klopp` → `Jurgen Klopp` *(method: per_match_cache, score 100.0)*
- H1 `[65]` `Davi` → `David` *(method: mcq_judge, score 88.9)*
- H1 `[141]` `Clyne` → `Clyne` *(method: mcq_judge, score 90.0)*
- H2 `[174]` `Terry` → `Terry` *(method: mcq_judge, score 90.0)*
- H2 `[267]` `Marcus Alonso` → `Marcos Alonso` *(method: mcq_judge, score 81.8)*
- H2 `[278]` `Terry` → `Terry` *(method: per_match_cache, score 100.0)*
- H2 `[439]` `Havanovic` → `Ivanovic` *(method: mcq_judge, score 82.4)*
- H2 `[569]` `Clyne` → `Clyne` *(method: per_match_cache, score 100.0)*

### Step L: Confidence-gated Generative Error Correction

- **Module:** `pipeline/llm_corrector.py:correct_match`
- **Model:** Qwen2.5-1.5B-Instruct Q4_K_M GGUF + xlm-roberta-base MLM

Confidence-Guided Error Correction (arxiv:2509.25048). Tokens with avg_logprob > -0.3 are kept verbatim; only low-confidence tokens are wrapped <token> in the prompt and may be edited. Match-context block (Teams/Players/Referee + prev/next segments) grounds the LLM. After the LLM proposes an edit, xlm-roberta vetos it if MLM(original)/MLM(proposed) ≥ 1.5. Step L respects entity_corrector decisions via frozen_word_indices so it does not re-touch canonical names.

**Per-half stats:**

| Half | Wall time (s) | Notes |
|---|---|---|
| 1 | 1067.89 | applied 27 GER edits (eligible 373/1524) |
| 2 | 1067.89 | applied 28 GER edits (eligible 373/1524) |

**Representative diffs:**

- H1 `[8]` `by Oscar` → `by by.`
- H1 `[37]` `for Jurgen Klopp he's the number one choice` → `for Jurgen Klopp, he's the number one choice.`
- H1 `[47]` `good away from home they're brilliant` → `they're away from home they're brilliant`
- H2 `[7]` `William looks like Oscars gone further forward` → `William looks like Oscar gone further forward`
- H2 `[13]` `Liverpool approach it in this situation as well` → `Liverpool approaches it in this situation as well.`
- H2 `[59]` `Matic as a` → `Matic as a.`

### Step P: Punctuation + Casing Restoration

- **Module:** `pipeline/punct_restorer.py:restore_punctuation_batch`
- **Model:** oliverguhr/fullstop-punctuation-multilang-large

Multilingual transformer that inserts (.,?!) and casing where Whisper dropped it. Conservative: only inserts where missing, never deletes existing punctuation. Important for downstream Elasticsearch tokenization and event extraction (NER works better on properly-cased text).

**Per-half stats:**

| Half | Wall time (s) | Notes |
|---|---|---|
| 1 | 187.28 | restyled 0 segments |
| 2 | 187.28 | restyled 0 segments |

**Representative diffs:**

*(no changes recorded by this stage on this match — likely a no-op given the inputs)*

## Summary

- Half 1 total cleaning time: **1430.7s**
- Half 2 total cleaning time: **1430.7s**
- Combined: **2861.5s**