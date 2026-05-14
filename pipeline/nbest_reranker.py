"""
pipeline/nbest_reranker.py — Step N: N-best entity-grounded reranking.

Pattern: Apple RAG-NEC (Retrieval-Augmented Named Entity Correction),
arxiv:2409.06062 (2024). Adapted with explicit multi-signal scoring as
recommended by Confidence-Guided Error Correction (arxiv:2509.25048,
2025) and the LM-fusion literature: every alternative beam hypothesis is
scored on three orthogonal signals and a length penalty, then the
highest-scoring hypothesis is chosen.

Algorithm (per segment, when ``segment.nbest`` is populated):

  1. Build candidate set C = {primary 1-best} ∪ {nbest alternatives}.
  2. For each candidate c ∈ C compute four sub-scores:

        a) entity_grounding(c)  — Σ_t cosine(emb(t), emb(top canonical))
                                  over entity-shaped tokens t in c.
                                  Uses sentence-transformers + FAISS.
                                  This is Apple RAG-NEC's core signal:
                                  candidates whose proper-noun tokens
                                  embed close to real gazetteer names
                                  score higher.

        b) edit_distance_bonus(c) — for each entity-shaped token t,
                                  rapidfuzz.fuzz.ratio(t, top_canonical)/100,
                                  averaged. Catches near-typos that the
                                  semantic embedding under-rewards
                                  ("Stampford" vs "Stamford" — fuzz=92,
                                  cosine=0.94 → both signals add up).

        c) confidence_weight(c) — for the primary hypothesis, mean of
                                  per-word `prob` from Whisper
                                  (segment.words). For alternatives, a
                                  flat 0.5 (we don't have per-token
                                  probs for non-1-best hypotheses).
                                  This *biases toward the primary*
                                  unless an alternative has a clear
                                  entity-grounding win — implementing
                                  the "trust ASR confidence unless
                                  retrieval strongly disagrees" rule
                                  from Confidence-Guided EC.

        d) length_penalty(c)    — λ * |words(c) − words(primary)|.
                                  Discourages structurally distant
                                  alternatives (which would inflate WER
                                  even if they get one entity right).

  3. Combined score:
        S(c) = α·entity_grounding(c) + β·edit_distance_bonus(c)
             + γ·confidence_weight(c) − δ·length_penalty(c)

     Default weights chosen so that no single signal can override the
     others on its own: α=1.0, β=1.0, γ=0.3, δ=0.05.

  4. Pick c* = argmax_c S(c). Replace seg.text with c*. Drop
     `segment.words` (per-word alignment is invalidated by the swap).

The reranker is a *pass-through* when:
  - segment lacks ``nbest`` (schema-2 input from a single-temperature
    Whisper run); or
  - gazetteer is empty; or
  - sentence-transformers / faiss not installed.

Multilingual: uses ``get_context_model(language)`` →
``paraphrase-multilingual-MiniLM-L12-v2`` for non-English (50+
languages) and ``all-MiniLM-L6-v2`` for English.

Telemetry returned via ``rerank_match`` and accessible afterwards via
``get_last_telemetry()`` for the orchestrator to persist into
``cleaning_metadata.nbest_telemetry``.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Optional

from pipeline.loader import Segment


# ── Tunable scoring weights ──────────────────────────────────────────
# Calibrated against Chelsea-Liverpool 2016 Premier League match.
# α=1.0  — entity grounding (primary signal, Apple RAG-NEC)
# β=1.0  — edit-distance bonus (catches typo-class errors)
# γ=2.0  — Whisper confidence (strong bias toward the primary 1-best;
#          alternatives are sampled at temperature 0.4 and known to
#          hallucinate longer continuations than the audio supports)
# δ=0.30 — length penalty per word (HARD penalty so swapping in a
#          hypothesis 5+ words longer is essentially impossible)
W_ENTITY_GROUNDING = 1.0
W_EDIT_DISTANCE = 1.0
W_CONFIDENCE = 2.0
W_LENGTH_PENALTY = 0.30

# Replace primary with alternative only when the score beats by this
# absolute margin. Prevents tied-score thrashing.
MIN_REPLACEMENT_MARGIN = 0.30

# HARD length cap: reject any alternative that differs from the primary
# by more than this many words, regardless of score. Without this guard,
# T=0.4 alternatives that hallucinated extra clauses would inflate WER
# even when their entity grounding looks better. Empirically a 5-word
# diff is the upper bound for "same utterance, different transcription".
MAX_LENGTH_DIFF_WORDS = 5


_FAISS_INDEX_CACHE: dict = {}
_LAST_TELEMETRY: dict = {}


def _get_or_build_index(gazetteer: dict, language: str = "en"):
    """Return (faiss_index, canonicals_list, embedder) or None if unusable."""
    if not gazetteer:
        return None

    cache_key = (id(gazetteer), language)
    if cache_key in _FAISS_INDEX_CACHE:
        return _FAISS_INDEX_CACHE[cache_key]

    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    from pipeline.config import get_context_model

    canonicals = sorted({v for v in gazetteer.values() if v and len(v) >= 3})
    if not canonicals:
        return None

    embedder = SentenceTransformer(get_context_model(language))
    embs = embedder.encode(
        canonicals, normalize_embeddings=True, show_progress_bar=False,
    )
    embs = np.asarray(embs, dtype=np.float32)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    bundle = (index, canonicals, embedder)
    _FAISS_INDEX_CACHE[cache_key] = bundle
    return bundle


def _entity_shaped_tokens(text: str, min_len: int = 4) -> list[str]:
    """Heuristic: capitalised words ≥ min_len, possessives + punct stripped.

    The reranker only needs candidate-entity tokens; full NER is overkill
    here and would couple this module to spaCy. The capitalisation +
    length filter catches the vast majority of player/team mentions in
    football commentary.
    """
    out: list[str] = []
    for w in text.split():
        clean = w.strip(".,;:!?\"'()[]{}")
        if clean.endswith("'s"):
            clean = clean[:-2]
        elif clean.endswith("'"):
            clean = clean[:-1]
        if len(clean) >= min_len and clean[0].isupper():
            out.append(clean)
    return out


def _score_entity_grounding_and_edit(
    text: str, bundle,
) -> tuple[float, float]:
    """Compute (entity_grounding, edit_distance_bonus) for a candidate.

    entity_grounding = Σ_t max_cosine(emb(t), emb(canonical))
    edit_distance_bonus = mean_t fuzz.ratio(t, top_canonical)/100

    Both signals operate on the same entity-shaped tokens. Returns
    (0.0, 0.0) when the candidate has no entity-shaped tokens.
    """
    import numpy as np
    from rapidfuzz import fuzz

    index, canonicals, embedder = bundle
    tokens = _entity_shaped_tokens(text)
    if not tokens:
        return 0.0, 0.0

    embs = embedder.encode(
        tokens, normalize_embeddings=True, show_progress_bar=False,
    )
    embs = np.asarray(embs, dtype=np.float32)
    sims, idxs = index.search(embs, 1)

    grounding = float(sims[:, 0].sum())

    # Edit-distance signal: average rapidfuzz ratio over the matched canonical.
    fuzz_scores = [
        fuzz.ratio(tok, canonicals[idxs[i][0]]) / 100.0
        for i, tok in enumerate(tokens)
    ]
    edit_bonus = sum(fuzz_scores) / len(fuzz_scores) if fuzz_scores else 0.0
    return grounding, edit_bonus


def _confidence_weight(words: Optional[list[dict]]) -> float:
    """Mean per-word probability from faster-whisper schema-2 output.

    Returns 0.5 (neutral) when ``words`` is None (typical for n-best
    alternatives, which weren't produced with word_timestamps in the
    multi-temperature pass).
    """
    if not words:
        return 0.5
    probs = [float(w.get("prob", 0.5)) for w in words if "prob" in w]
    if not probs:
        return 0.5
    return sum(probs) / len(probs)


def _combined_score(
    text: str,
    is_primary: bool,
    words: Optional[list[dict]],
    primary_word_count: int,
    bundle,
) -> dict:
    """Compute the multi-signal combined score for one candidate."""
    grounding, edit_bonus = _score_entity_grounding_and_edit(text, bundle)
    conf = _confidence_weight(words) if is_primary else 0.5
    length_diff = abs(len(text.split()) - primary_word_count)

    combined = (
        W_ENTITY_GROUNDING * grounding
        + W_EDIT_DISTANCE * edit_bonus
        + W_CONFIDENCE * conf
        - W_LENGTH_PENALTY * length_diff
    )
    return {
        "text": text,
        "grounding": round(grounding, 4),
        "edit_bonus": round(edit_bonus, 4),
        "confidence": round(conf, 4),
        "length_diff": length_diff,
        "combined": round(combined, 4),
    }


def rerank_match(
    segments: list[Segment],
    gazetteer: dict,
    language: str = "en",
    length_penalty_per_word: float = None,  # legacy keyword, kept for API back-compat
) -> tuple[list[Segment], dict]:
    """Rerank n-best alternatives per segment by multi-signal scoring.

    Args:
        segments: Tier-1 filtered segments. Each may carry ``segment.nbest``
            (a list of alternative beam hypotheses); when missing,
            the segment passes through unchanged.
        gazetteer: name-variant dict from ``pipeline.gazetteer.build_gazetteer``.
            When empty, all segments pass through.
        language: detected match language (selects the embedder).
        length_penalty_per_word: deprecated; use module-level
            ``W_LENGTH_PENALTY`` instead. Kept for backwards-compatible
            test calls.

    Returns:
        ``(reranked_segments, telemetry)``.

        Telemetry keys:
          - ``segments_total``
          - ``segments_with_nbest``     (had at least one alternative)
          - ``segments_replaced``       (primary text replaced by an alternative)
          - ``examples``                (up to 10 before/after diffs with all sub-scores)
          - ``wall_time_sec``
          - ``pass_through_reason``     (set when the reranker bailed early)
    """
    global _LAST_TELEMETRY
    # length_penalty_per_word kept for back-compat but now lives in the module-level weight.
    _ = length_penalty_per_word

    bundle = _get_or_build_index(gazetteer, language) if gazetteer else None

    telemetry: dict = {
        "segments_total": len(segments),
        "segments_with_nbest": 0,
        "segments_replaced": 0,
        "examples": [],
        "wall_time_sec": 0.0,
        "pass_through_reason": None,
        "weights": {
            "entity_grounding": W_ENTITY_GROUNDING,
            "edit_distance": W_EDIT_DISTANCE,
            "confidence": W_CONFIDENCE,
            "length_penalty": W_LENGTH_PENALTY,
            "min_replacement_margin": MIN_REPLACEMENT_MARGIN,
        },
    }

    if bundle is None:
        telemetry["pass_through_reason"] = (
            "no_gazetteer" if not gazetteer else "embeddings_unavailable"
        )
        _LAST_TELEMETRY = telemetry
        return list(segments), telemetry

    t0 = time.perf_counter()
    out: list[Segment] = []

    for seg in segments:
        nbest: Optional[list[str]] = getattr(seg, "nbest", None)
        if not nbest or len(nbest) < 1:
            out.append(seg)
            continue

        telemetry["segments_with_nbest"] += 1

        primary_text = seg.text
        primary_word_count = max(1, len(primary_text.split()))

        # Score the primary first (with real per-word confidence).
        primary_scored = _combined_score(
            primary_text, is_primary=True, words=seg.words,
            primary_word_count=primary_word_count, bundle=bundle,
        )
        best_scored = primary_scored

        # Score every distinct alternative.
        seen: set[str] = {primary_text}
        all_scored = [primary_scored]
        for cand_text in nbest:
            if cand_text in seen:
                continue
            seen.add(cand_text)
            cand_word_count = len(cand_text.split())
            # HARD length cap — alternatives that drifted too far in length
            # are almost certainly hallucinations from the high-T pass.
            if abs(cand_word_count - primary_word_count) > MAX_LENGTH_DIFF_WORDS:
                continue
            cand_scored = _combined_score(
                cand_text, is_primary=False, words=None,
                primary_word_count=primary_word_count, bundle=bundle,
            )
            all_scored.append(cand_scored)
            if cand_scored["combined"] > best_scored["combined"] + MIN_REPLACEMENT_MARGIN:
                best_scored = cand_scored

        if best_scored["text"] != primary_text:
            telemetry["segments_replaced"] += 1
            if len(telemetry["examples"]) < 10:
                telemetry["examples"].append({
                    "segment_id": seg.segment_id,
                    "from": primary_text,
                    "to": best_scored["text"],
                    "primary_score": primary_scored,
                    "winning_score": best_scored,
                    "all_candidates": all_scored,
                })
            # Drop per-word probs because token alignment changes.
            out.append(replace(seg, text=best_scored["text"], words=None))
        else:
            out.append(seg)

    telemetry["wall_time_sec"] = round(time.perf_counter() - t0, 3)
    _LAST_TELEMETRY = telemetry
    return out, telemetry


def get_last_telemetry() -> dict:
    """Return telemetry from the most recent ``rerank_match`` call.

    Used by the orchestrator to persist Step N stats into the cleaned
    output's ``cleaning_metadata``.
    """
    return dict(_LAST_TELEMETRY)
