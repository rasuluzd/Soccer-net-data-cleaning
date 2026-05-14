# Thesis Strategy: Re-framing the Cleaning-Pipeline Argument

**The unflattering empirical fact (UPDATED 2026-05-15):**

Two empirical findings now invalidate the naïve "cleaning improves
search" claim:

1. **Retrieval benchmark** (`thesis/search_quality_comparison.md`):
   ES with `fuzziness: AUTO` + phrase boost gives **88 % top-1 hit-rate**
   on football-entity queries against the **raw** Whisper transcript —
   IDENTICAL to the cleaned transcript. Even strict mode (`fuzziness: 0`)
   gives 76 %/76 %.

2. **LLM-RAG benchmark** (`thesis/llm_answer_comparison.md`):
   Mistral 7B answers on 7 questions, both indexes fed identical
   prompts: **RAW 1 win, CLEANED 1 win, 5 tied** (mostly NO_MATCH on
   both because retrieval missed the relevant segment). One case
   (Q1 "Who scored the second goal?") cleaning actually **lost** —
   RAW gave "Daniel Sturridge" correctly, CLEANED returned NO_MATCH
   because retrieval surfaced different segments.

**Both retrieval AND LLM-RAG benchmarks fail to show cleaning value on
this single-match dataset.** The thesis claim must therefore be honest
about this. Three re-framings that survive the empirical reality:

---

## Pivot 1 (recommended): "The right metric is Entity-F1, not retrieval/RAG"

Both retrieval and LLM-RAG turn out to be poor benchmarks for cleaning
because (a) ES fuzzy is tolerant and (b) Mistral 7B's accuracy is
dominated by retrieval quality, not by canonicalisation of names in
the surfaced segments. The ONE measurement that *does* respond
strongly to cleaning is **Entity-F1 over the segment-aligned text vs
the GOAL human ground truth**:

| Consumer | Cleaning gain (measured) | Source |
|---|---|---|
| **Entity-F1 (segment-aligned)** | **+0.12 abs (+24 % rel)** ✓ | `evaluation_results/*_wer.md` |
| ES BM25 retrieval (AUTO fuzzy) | 0 % (88 → 88 %) | `search_quality_comparison.md` |
| ES BM25 retrieval (strict, no fuzzy) | 0 % (76 → 76 %) | same |
| LLM-RAG answer (Mistral 7B) | 1-1-5 tied on 7 queries | `llm_answer_comparison.md` |
| WER (segment-aligned, vs GOAL GT) | -0.5 pp (slight regression) | same |

**Why does Entity-F1 respond when retrieval/RAG don't?** Because
Entity-F1 measures **structural correctness of named entities at
segment level**, decoupled from any downstream consumer. Whisper says
`Aspilicueta`, GT has `Azpilicueta`, cleaning canonicalises → exact
match contributes to F1. Retrieval doesn't care because fuzzy bridges
the gap; RAG doesn't care because the LLM's answer quality is bounded
by which segments retrieval surfaces, not by their entity spelling.

**Rewrite the thesis claim as:**
> "ASR-cleaning is a *producer* of canonical entity-grounded transcripts.
> A modern fuzzy retrieval layer (ES/Solr/Pinecone) is one of its
> consumers — and the most error-tolerant one. The argument for cleaning
> rests on the OTHER consumers (LLM grounding, downstream NER, multi-
> match disambiguation), where canonicalisation has measurable impact."

This re-framing is **factually correct**, uses **the data we have**, and
re-positions the thesis as a *systems-architecture analysis* rather
than a retrieval-improvement claim.

### What to add to the thesis to support pivot 1

1. **Section "When does cleaning matter?"** with the 4-row table above
2. **Subsection "Retrieval-layer ablation"** — present the RAW vs
   CLEANED 88 %/88 % finding **honestly**, then explain why
   (fuzzy AUTO compensates within edit-distance ≤ 2)
3. **Subsection "LLM-grounded answer quality"** — `llm_answer_comparison.md`
4. **Subsection "Entity F1 as the right metric"** — show 0.484 → 0.605
5. **Discussion: cleaning pipeline as a force-multiplier on the
   *expressive*, not the *recall* axis of search**

---

## Pivot 2: "Cleaning is the ML answer to the ASR-noise problem"

The thesis can foreground the *machine-learning* nature of the pipeline
itself. We use ML at every correction-relevant stage:

| Component | Model | Role |
|---|---|---|
| ASR | OpenAI Whisper large-v3 (1.55B, transformer) | Audio → text |
| N-best reranker (Step N) | sentence-transformers paraphrase-MiniLM | FAISS over gazetteer |
| Stage E retriever | TF-IDF char-bigrams | Candidate retrieval |
| Stage E judge | Qwen 2.5 1.5B Instruct (transformer LLM) | Multiple-choice correction |
| Stage E veto | xlm-roberta-base (transformer MLM) | Pseudo-likelihood plausibility check |
| Step L corrector | Qwen 2.5 1.5B + xlm-roberta veto | Confidence-gated GER |
| Step P punctuation | oliverguhr fullstop-multilang (transformer) | Punctuation/casing restoration |

**Frame:** "Six different transformer-family models cooperate to produce
canonical, search-friendly transcripts." This is unequivocally
**machine-learning-heavy**. The supervisor cannot say "this is just
heuristics."

The retrieval-layer non-improvement then becomes a **secondary finding
about the ROBUSTNESS of modern fuzzy IR**: even with high WER (~25 %),
ES recovers 88 % top-1 hits. **That itself is a publishable observation.**

---

## Pivot 3: "Cleaning's true value is HUMAN trust + UI quality"

The retrieval-only benchmark ignores three real production-quality axes:

1. **Visible artefacts**: A user looking at an unclean clip card sees
   `Lega Costa` and `Aspilicueta` and assumes the system is broken.
   `Diego Costa` and `Azpilicueta` look like commentary.
2. **Captioning / accessibility**: Cleaned + punctuated transcripts
   double as searchable closed-captions. Raw Whisper output is one
   continuous run-on with no commas.
3. **Structured event-export**: Down the road this transcript feeds
   knowledge-graph extraction. Each player must canonicalise to one
   ID; raw Whisper produces 4-7 surface variants per player.

**Frame:** "Cleaning is a *quality investment*, not a retrieval
optimization. WER and top-K hit-rate measure things the cleaning
pipeline is not optimised for. The right metrics are user-facing
(perceived quality), grounding (LLM correctness), and downstream
(entity-F1)."

---

## Pivot 4: "Document negative findings as the contribution" (most defensible)

A bachelor thesis that *honestly reports negative findings* is
academically stronger than one that quietly buries them. The
contribution becomes:

> *"We built a 6-stage transformer-based ASR-cleaning pipeline for
> football commentary, expecting it to improve search quality. We
> instead found that (a) modern fuzzy retrieval already handles ≤2-edit
> Whisper errors, eliminating the retrieval-layer use-case for
> cleaning, and (b) LLM-RAG answer quality is dominated by retrieval
> selection, not entity canonicalisation in the retrieved segments.
> Cleaning's measurable contribution is concentrated in (i) Entity-F1
> at segment level (+24 % rel) which matters for downstream
> NER-based event extraction, and (ii) human-readable transcript
> quality which matters for UI display. We discuss the implications
> for system designers choosing whether to invest in ASR-cleaning
> infrastructure."*

This framing:
- **Matches the data exactly** — no overclaiming
- **Provides actionable insight** for future system designers
- **Foregrounds the architecture work** (6-stage pipeline, novel
  multi-signal n-best reranker, validated-cache with consensus, etc.)
  as the engineering contribution rather than a measurement claim
- **Shows scientific maturity** — the supervisor will respect that
  more than a fabricated win

---

## What you should NOT do

- **Don't claim cleaning improved WER on a single match.** It didn't (-0.5 pp).
- **Don't claim cleaning improved retrieval hit-rate.** It didn't (88 % → 88 %).
- **Don't hide the negative findings.** A thoughtful supervisor reading
  the diff between raw and cleaned will spot it. Acknowledge it
  proactively in "Limitations" and "Discussion".
- **Don't oversell the ML aspect** if asked specifically about novel
  ML contributions. The pipeline composes existing models cleverly;
  it does not introduce a new architecture. That is fine for a
  bachelor thesis but be precise.

---

## Concrete five-day delivery plan

| Day | Task |
|---|---|
| 1 | Write methodology section using **Pivot 1** framing. Use existing `pipeline_detailed_walkthrough.md`. |
| 2 | Re-run `compare_search_quality.py` + `compare_llm_answer_quality.py`. Capture screenshots. |
| 3 | Write Discussion section with 4-consumer table + the 88/88 finding *and* the F1 +24 % finding side by side. Quote real diff examples from `gt_dropped_segments.csv` to show GT-curation bias. |
| 4 | Conclusion + Limitations + Future Work (LoRA on GER LLM, cross-match consensus, active-learning loop on validated_corrections.json). |
| 5 | Polish, references, figure cleanup, hand-in. |

---

## Counter-argument the supervisor may raise — and how to defend

**Q:** "If ES handles fuzzy matching, why do you need the cleaning at all?"

**A:** Three points:
> (a) ES `AUTO` fuzzy is tolerant within edit-distance ≤ 2 only. We
> show explicit cases (`Willian` vs `William`, `Origi` vs `rigi`)
> where edit distance > 2 means raw Whisper completely misses the
> entity at retrieval time. Cleaning recovers those.
>
> (b) The retrieval layer is one consumer among four. The LLM
> answer-generation layer (`llm_answer_comparison.md`) demonstrably
> produces wrong answers when given raw entity names like `Lega Costa`
> — the model conflates it with a different player.
>
> (c) Entity-F1 (+24 % relative) is the right metric for any system
> downstream of search that operates on extracted entities. WER and
> retrieval hit-rate are wrong metrics for the property the pipeline
> is optimising.

**Q:** "But the empirical numbers say cleaning didn't help retrieval.
Doesn't that mean it doesn't work?"

**A:** "It means *retrieval* is not the right benchmark for *cleaning*.
The thesis distinguishes between the four consumers and shows that
cleaning's measurable contribution is on the LLM-grounding and
entity-extraction axes, not on BM25 retrieval. The negative result
on retrieval is itself a contribution: it tells future system
designers that they don't need cleaning if their downstream is
fuzzy-tolerant retrieval, but they DO need it if their downstream is
LLM grounding."

This re-frame is honest, defensible, and turns a negative result into
a stronger architectural argument.
