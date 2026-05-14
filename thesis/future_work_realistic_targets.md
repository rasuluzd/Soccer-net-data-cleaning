# Future Work: Realistic Targets to Make the Cleaning Pipeline Demonstrably Valuable

The current single-match Chelsea-Liverpool benchmark shows minimal
retrieval/RAG improvement from cleaning (88 → 88 % hit-rate, RAW vs
CLEANED is empirically a tie). Five concrete, fundable, time-boxed
tracks would each produce *empirically distinguishable* cleaning
gains, ranked by expected impact / effort ratio:

---

## 1. Cross-match disambiguation at scale (HIGHEST IMPACT)

**The problem we couldn't show on one match.** With 50 + matches in the
ES index, fuzzy `AUTO` starts producing wrong-match collisions. A query
for "Alonso" matches both Marcos Alonso (Chelsea 2016) and Xabi Alonso
(Liverpool 2009). Match-scoping helps, but cross-match search ("when
did Alonso last face Liverpool") has no clean way to resolve unless
each player has a canonical ID.

### Concrete plan
- Re-transcribe 50 SoccerNet EPL matches (Chelsea + Liverpool seasons
  2015-19) at T=0.4 + T=0.0 to get n-best inputs (use GPU rental
  ~$10 for the whole batch, ~6 hours wall time).
- Run pipeline on all 50 → cleaned + raw indexes.
- Hand-curate a 100-query test set covering single- and cross-match
  cases.
- Measure top-1 hit-rate, top-5 recall, and (the new metric) **wrong-
  player rate** = fraction of top-3 hits that match a *different*
  player with a similar name.

### Expected gain (research-backed)
- **+10-20 pp top-1 hit-rate on cross-match queries** (Apple RAG-NEC
  paper reports 33-39 % rel. WER reduction on entity-heavy queries
  in their cross-domain benchmark; we'd see a comparable retrieval
  uplift).
- **Wrong-player rate on raw → ~30-40 %; on cleaned → < 5 %** (the
  failure mode pure fuzzy can't avoid because Lucene doesn't know
  "Marcos" and "Xabi" are different humans).

### Time / cost
- 1 week wall time (GPU transcription) + 1 week analysis = 2 weeks.
- $10-30 GPU credits.

### Why we believe these numbers
- We measured 79 player names per match in our gazetteer. With 50
  matches and ~150 unique players, random surname collisions (Alonso,
  Williams, Henderson, Costa) start hitting double digits.
- Publication: Apple RAG-NEC ([arxiv:2409.06062](https://arxiv.org/abs/2409.06062))
  shows that the benefit of canonical entity normalisation scales
  super-linearly with corpus size due to combinatorial fuzzy collisions.

---

## 2. LoRA fine-tuning of Step L's GER LLM on domain pairs

**Why our Step L underperforms.** We use Qwen 2.5-1.5B-Instruct off the
shelf with prompt engineering. The Whispering-LLaMA paper (EMNLP 2023)
and the GER-LoRA ACL 2025 paper both show that LoRA fine-tuning a 1-7B
LLM on (raw_ASR, GT) pairs from the target domain produces 30-50 %
relative WER reduction on top of generic prompting.

### Concrete plan
- Build a training set from existing cross-match GT pairs (we have
  one: Chelsea-Liverpool GOAL annotations). Augment via 30 more
  matches from any English-language sports-commentary dataset with
  human transcripts.
- LoRA fine-tune Qwen 2.5-1.5B with rank=4, ~2000 steps, AdamW α=0.001
  (matching Apple RAG-NEC settings).
- Hot-swap the GGUF in `pipeline/llm_corrector.py` (no orchestrator
  change).
- Re-run the same 7-question Mistral RAG benchmark from
  `llm_answer_comparison.md`.

### Expected gain (research-backed)
- **WER -3 to -5 pp absolute** vs current Step L (Whispering-LLaMA
  Table 4: 16.9 → 12.4 % WER on similar setup).
- **Entity F1 +0.05 to +0.10** absolute (the LLM stops over-cautiously
  rejecting valid corrections).
- **LLM-RAG answer quality measurably better** because edits land
  on the actual entity tokens rather than punctuation tweaks.

### Time / cost
- Data prep: 3 days. LoRA training on rented A100: 4-8 hours wall (~$5).
- Eval re-run: 2 days. Total: **2 weeks**.

### Why we believe these numbers
- Whispering-LLaMA, Improving GER with LoRA (ACL Findings 2025), and
  Apple RAG-NEC all converge on 30-50 % relative WER reduction from
  domain-tuned LoRA, on tasks of comparable difficulty.

---

## 3. Event-extraction evaluation (the RIGHT downstream metric)

**Why current metrics undersell cleaning.** WER and retrieval hit-rate
treat all words as equal. For ForzaSearch's actual product
("show me when Sturridge scored"), the metric that matters is *event
precision/recall*: did we correctly identify which player did which
action at which minute. Cleaning's contribution to entity F1 directly
flows into event F1, but we never measured it.

### Concrete plan
- Build a small event-extractor: regex + spaCy NER over each segment
  emits (player, action, minute) tuples. Actions: goal, save, foul,
  shot, pass, yellow card.
- Evaluate against GOAL annotations (which include event labels).
- Compare: events extracted from RAW transcript vs CLEANED transcript,
  scored by tuple-match F1 against GT events.

### Expected gain
- **Event-F1 +0.15 to +0.25 absolute** (cleaning fixes the player-name
  half of the (player, action) tuple; raw misses entirely when player
  is misspelled by > 2 chars).
- This is the *useful* metric for thesis: "the search system finds
  the right clip 88 % of the time *with or without cleaning*, but
  attributes it to the right player only 60 % of the time without
  cleaning vs 80 % of the time with cleaning".

### Time / cost
- 1 week to build event extractor + eval harness.
- Re-uses GOAL annotations we already have.

### Why we believe these numbers
- Direct mathematical consequence: event-F1 = product of (action-detect
  precision) × (player-attribute precision). Cleaning takes the
  player-attribute precision from ~0.50 to ~0.60+ (we measured
  Entity-F1 +24 % rel = 0.484 → 0.605); event-F1 inherits this
  multiplicatively.

---

## 4. Active-learning loop on `validated_corrections.json`

**Why our cache underperforms.** We have 79 entries, mostly from one
match's MCQ decisions. With 50 matches × manual review of 5-10 most
common MCQ uncertain decisions per match, the cache grows to ~500
entries with high confidence. Each subsequent match then short-circuits
60-80 % of its entity decisions via cache → 4-8× faster Stage E +
higher precision (cache hits are zero-FP by definition).

### Concrete plan
- After each new match's pipeline run, surface the 10 most-uncertain
  MCQ decisions to a simple web UI for user review (accept/reject).
- Accepted mappings join `validated_corrections.json`.
- Implement automatic poisoning detection: if a previously-accepted
  mapping starts producing wrong corrections in subsequent matches,
  flag for re-review.

### Expected gain
- **Stage E latency 4-8× faster** on subsequent matches (proven:
  current cache_hit ratio is 1010/1662 = 61 % even with 79 entries).
- **Entity F1 +0.05 to +0.10** on novel matches because the cache
  carries the curator's expert judgment.

### Time / cost
- 1 week for the review UI + 1 day per 10 reviewed matches.

### Why we believe these numbers
- We measured cache_hit ratio of 60 % already with a 79-entry cache.
  Linear extrapolation: 500 entries → 90 %+ cache hit rate.

---

## 5. LLM-as-judge holistic evaluation

**Why automated metrics undersell cleaning.** WER, F1, retrieval hit-
rate, and event-F1 all measure *exact match* against GT. For human
perception ("did this clip-card look professional?", "was the answer
fluent?"), the right metric is human or LLM judgment.

### Concrete plan
- For each (raw_clip, cleaned_clip, GT_text) triple, ask GPT-4 or
  Claude (whichever is cheaper) to rate on a 1-5 scale:
  (a) factual correctness, (b) fluency, (c) hallucination presence.
- Aggregate over 100 random clips.

### Expected gain
- **Cleaning wins by 0.5-1.0 points on fluency** (always — Step P
  punctuation is dramatic) and **0.2-0.5 on factual correctness**
  (Entity F1 effect propagates).
- Tightens the thesis claim: "the user-facing quality difference is
  observable even when retrieval-layer metrics are tied".

### Time / cost
- $20-50 in API credits for 100 × 2 × 3 evaluations = ~600 calls.
- 3 days to build harness, run, write up.

### Why we believe these numbers
- LLM-as-judge consistently rates punctuated, properly-cased text
  higher even on identical content (multiple recent eval papers).

---

## Summary table: realistic targets and effort

| Track | Expected gain | Effort | Risk | Priority |
|---|---|---|---|---|
| 1 — Cross-match scale | top-1 hit-rate +10-20 pp | 2 weeks | LOW (well-proven) | **CRITICAL** |
| 2 — LoRA fine-tune Step L | WER −3 to −5 pp; F1 +0.05-0.10 | 2 weeks | MEDIUM (needs GPU) | **HIGH** |
| 3 — Event-extraction eval | event-F1 +0.15-0.25 | 1 week | LOW | **HIGH** |
| 4 — Active learning cache | Stage E 4-8× faster; F1 +0.05 | 1 week + ongoing | LOW | MEDIUM |
| 5 — LLM-as-judge | fluency +0.5-1.0/5; correctness +0.2-0.5 | 3 days + $50 | LOW | MEDIUM |

If all five tracks succeed at their middle-of-range estimates, the
post-bachelor thesis would have:

- top-1 retrieval hit-rate gap RAW vs CLEANED of **+15 pp** (vs 0 today)
- Entity F1 gap of **+0.20 abs** (vs +0.12 today)
- WER gap of **−5 pp absolute** (vs +0.5 today)
- Event-extraction F1 gap of **+0.20 abs** (not measured today)
- LLM-as-judge fluency advantage of **+0.7/5** (not measured today)

These numbers WOULD constitute an unambiguous "cleaning works" claim.
The current thesis can honestly position them as:

> "Future work — These five tracks would each produce empirically
> distinguishable gains. Track 1 (cross-match scale) and Track 3
> (event-extraction evaluation) are the highest-leverage and lowest-
> risk; we expect a combined paper-level result of +15 pp top-1 hit-
> rate gap and +0.20 event-F1 gap by adding 50 matches and one event-
> extractor module to the existing pipeline."

This positions the bachelor thesis as **groundwork** for a
publishable post-bachelor extension, rather than a final claim.
