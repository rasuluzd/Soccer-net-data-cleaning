---
paths:
  - "pipeline/llm_corrector.py"
  - "pipeline/punct_restorer.py"
---

# Step L + Step P: LLM GER and Punctuation Restoration

Replaces the legacy Tier 3 sentence-transformer cosine disambiguator
(May 2026 refactor). `pipeline/context_disambiguator.py` was deleted.

## Step L — Confidence-Gated Generative Error Correction

`pipeline/llm_corrector.py` runs AFTER Stage E. For each segment:

1. Read per-token Whisper avg-logprob from `Segment.words` (schema-2 only)
2. Mark tokens with `logprob ≤ LLM_LOGPROB_GATE (-0.3)` as low-confidence
3. Skip the segment if fewer than `LLM_MIN_TOKENS_TO_INVOKE=2` low-confidence tokens
4. Build a prompt with: typed match-context (Players / Teams / Referee / Venue / Coaches),
   ±`LLM_CTX_PREVIOUS_SEGMENTS=2` / `LLM_CTX_NEXT_SEGMENTS=1` neighbouring segments,
   and the segment with `<token>` markers around editable tokens
5. Ask Qwen2.5-1.5B-Instruct (GGUF q4_k_m via llama-cpp-python) to rewrite only the marked tokens
6. For each proposed edit, run xlm-roberta-base pseudo-log-likelihood:
   reject if `MLM(original) / MLM(proposed) ≥ MLM_VETO_RATIO=1.5`

Backbone is fully local. Graceful no-op if the GGUF file or `llama-cpp-python` is missing.

## Step P — Punctuation + Casing Restoration

`pipeline/punct_restorer.py` runs AFTER Step L:

- Model: `oliverguhr/fullstop-punctuation-multilang-large` (multilingual xlm-roberta backbone)
- Conservative: `PUNCT_PRESERVE_EXISTING=True` — only INSERTS missing punctuation and casing,
  never deletes or replaces what Whisper produced
- Per-segment batching, never crosses segment boundaries

## Key Directives

1. **Word-level confidence is required** for Step L — without `Segment.words` from schema-2 ASR, Step L treats every token as low-confidence (heavier LLM cost). Re-transcribe with `tools/retranscribe_kb_whisper.py` to get schema-2 output
2. **MLM veto is the safety net** — every LLM proposal must clear `MLM_VETO_RATIO=1.5`. Reusing the same xlm-r handle for Stage E MCQ veto is intentional (one model load, two consumers)
3. **Model via config only** — `LLM_MODEL_PATH`, `LLM_MODEL_REPO`, `LLM_MODEL_FILENAME`, `MLM_VETO_MODEL`, `PUNCT_MODEL` all in `pipeline/config.py`; never hardcode
4. **Prompt context matters** — the gazetteer-typed prompt block is what makes Qwen-1.5B pick correct names; do not trim it without re-evaluating WER + Entity-F1
5. **Step P preserves existing punctuation** — set `PUNCT_PRESERVE_EXISTING=False` only for ablation experiments, never as default

## Toggles

```
LLM_CORRECTION_ENABLED = True
MLM_VETO_ENABLED = True
PUNCT_RESTORATION_ENABLED = True
```

Disable for ablation comparisons; production defaults to all three on.
