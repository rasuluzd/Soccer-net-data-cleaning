# Step L + Step P: Acceptance Criteria

Reference for the `/diagnose` skill, post-May-2026 architecture. The legacy
Tier 3 sentence-transformer disambiguator was deleted; LLM-based correction
now happens in Step L (`pipeline/llm_corrector.py`) AFTER Stage E entity
correction. See `pipeline/config.py` for current values.

## Step L Decision Flow

For each segment with per-word probabilities (schema-2 ASR output):

1. **Confidence gating** — tokens with `logprob > LLM_LOGPROB_GATE=-0.3`
   are deemed acoustically confident and locked verbatim
2. **Edit budget** — skip the segment if it has fewer than
   `LLM_MIN_TOKENS_TO_INVOKE=2` low-confidence tokens
3. **Prompt build** — typed gazetteer block (Players / Teams / Referee /
   Venue / Coaches) + `LLM_CTX_PREVIOUS_SEGMENTS=2` previous segments +
   `LLM_CTX_NEXT_SEGMENTS=1` next segment + segment with `<token>` markers
4. **LLM call** — Qwen2.5-1.5B-Instruct (q4_k_m GGUF) at `LLM_TEMPERATURE=0.0`,
   `LLM_MAX_NEW_TOKENS=96`, `LLM_CTX_WINDOW=2048`
5. **MLM veto** — for each proposed edit, mask the position in context and
   compare xlm-roberta-base pseudo-log-likelihoods:
   reject if `lp(original) - lp(proposed) ≥ log(MLM_VETO_RATIO=1.5)`
6. **Apply** — only LLM proposals that touch wrapped tokens AND pass the
   MLM veto are written to the segment

## Step P (Punctuation Restorer) Behaviour

- Model: `oliverguhr/fullstop-punctuation-multilang-large`
- `PUNCT_PRESERVE_EXISTING=True` (always): the model is allowed to INSERT
  missing punctuation/casing but never to delete or replace what Whisper
  produced
- Per-segment batching; punctuation never crosses segment boundaries
- Lazy singleton load (~1.1 GB once per process); graceful no-op if
  `transformers` is unavailable

## Key Constants

```
LLM_CORRECTION_ENABLED          = True
LLM_MODEL_FILENAME              = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM_TEMPERATURE                 = 0.0
LLM_MAX_NEW_TOKENS              = 96
LLM_CTX_WINDOW                  = 2048
LLM_LOGPROB_GATE                = -0.3
LLM_MIN_TOKENS_TO_INVOKE        = 2
LLM_CTX_PREVIOUS_SEGMENTS       = 2
LLM_CTX_NEXT_SEGMENTS           = 1
MLM_VETO_ENABLED                = True
MLM_VETO_RATIO                  = 1.5
MLM_VETO_MODEL                  = "xlm-roberta-base"
PUNCT_RESTORATION_ENABLED       = True
PUNCT_MODEL                     = "oliverguhr/fullstop-punctuation-multilang-large"
PUNCT_PRESERVE_EXISTING         = True
```

## Key Details

- **Schema-2 ASR is required** for the confidence gate. Without per-word
  probabilities (`Segment.words`), Step L treats every token as low
  confidence, which spikes LLM cost. Re-transcribe with
  `tools/retranscribe_kb_whisper.py` to obtain schema-2 output
- **MLM veto is the safety net** for both Stage E MCQ picks and Step L
  edits. The same xlm-roberta-base handle is shared between
  `entity_corrector` and `llm_corrector`
- **Step L runs AFTER Stage E** — Stage E owns named-entity corrections
  via the gazetteer + Qwen MCQ judge; Step L handles residual segment-level
  drift (grammar, function words, low-confidence content words)
- **Model identifiers via config only** — `LLM_MODEL_FILENAME`,
  `MLM_VETO_MODEL`, `PUNCT_MODEL`. Never hardcode paths in pipeline modules
- **Step P preserves existing punctuation** by default — flip
  `PUNCT_PRESERVE_EXISTING=False` only when running ablation against the
  unrestricted oliverguhr behaviour
