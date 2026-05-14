"""Tests for pipeline/nbest_reranker.py — Step N (Apple RAG-NEC pattern)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.loader import Segment  # noqa: E402
from pipeline.nbest_reranker import (  # noqa: E402
    _entity_shaped_tokens,
    rerank_match,
    get_last_telemetry,
)


def _seg(text, sid="0", nbest=None):
    return Segment(
        segment_id=sid, start_time=0.0, end_time=5.0,
        text=text, half=1, nbest=nbest,
    )


# ─── Token extraction ────────────────────────────────────────────────


class TestEntityShapedTokens:
    def test_capitalised_words_kept(self):
        toks = _entity_shaped_tokens("Sterling passes to Sturridge who scores")
        assert "Sterling" in toks
        assert "Sturridge" in toks
        assert "passes" not in toks  # lowercase

    def test_short_words_dropped(self):
        toks = _entity_shaped_tokens("De Bruyne to Tom")
        # Tom is 3 chars (< min_len=4) — dropped
        assert "Bruyne" in toks
        assert "Tom" not in toks

    def test_possessive_stripped(self):
        toks = _entity_shaped_tokens("Sterling's pass to Klopp's striker")
        assert "Sterling" in toks
        assert "Klopp" in toks

    def test_punctuation_stripped(self):
        toks = _entity_shaped_tokens("Sturridge! What a finish, Hazard.")
        assert "Sturridge" in toks
        assert "Hazard" in toks


# ─── Pass-through scenarios ──────────────────────────────────────────


class TestRerankPassThrough:
    """When the inputs make reranking impossible, return segments unchanged."""

    def test_empty_gazetteer_passes_through(self):
        segs = [_seg("Sterling shoots", nbest=["Sterling shoots", "Sterling shouts"])]
        out, tel = rerank_match(segs, gazetteer={})
        assert out[0].text == "Sterling shoots"
        assert tel["segments_replaced"] == 0
        assert tel["pass_through_reason"] == "no_gazetteer"

    def test_segment_without_nbest_passes_through(self):
        segs = [_seg("Sterling shoots", nbest=None)]
        gaz = {"sterling": "Sterling"}
        out, tel = rerank_match(segs, gazetteer=gaz)
        assert out[0].text == "Sterling shoots"
        assert tel["segments_with_nbest"] == 0

    def test_single_alternative_identical_to_primary(self):
        # 1 alternative that equals the primary → nothing to choose from
        segs = [_seg("Sterling shoots", nbest=["Sterling shoots"])]
        gaz = {"sterling": "Sterling"}
        out, tel = rerank_match(segs, gazetteer=gaz)
        assert out[0].text == "Sterling shoots"
        # We still count it as having an alternative; just no replacement
        assert tel["segments_replaced"] == 0


# ─── Real reranking (requires sentence-transformers + faiss) ─────────


class TestRerankWithEmbeddings:
    """These tests only run when the embedding stack is installed.
    They build a tiny FAISS index over a real gazetteer and verify
    the reranker picks the entity-richer hypothesis."""

    def setup_method(self):
        # Reset module-level FAISS cache so each test gets a fresh index.
        from pipeline import nbest_reranker as nr
        nr._FAISS_INDEX_CACHE.clear()
        try:
            import faiss  # noqa: F401
            import sentence_transformers  # noqa: F401
        except ImportError:
            pytest.skip("faiss / sentence-transformers not installed")

    def test_picks_entity_rich_hypothesis(self):
        """Given the primary text + a beam alternative, prefer the one
        containing a real player surname over one with a non-name lookalike."""
        gaz = {
            "sturridge": "Sturridge",
            "sterling": "Sterling",
            "henderson": "Henderson",
            "mignolet": "Mignolet",
        }
        # primary (wrong): "starridge" is gibberish → low entity score
        # nbest alt (correct): "Sturridge" matches gazetteer canonical
        segs = [_seg(
            "starridge shoots and scores",
            nbest=["Sturridge shoots and scores"],
        )]
        out, tel = rerank_match(segs, gazetteer=gaz)
        assert out[0].text == "Sturridge shoots and scores", (
            f"reranker should pick the entity-richer hypothesis; "
            f"got {out[0].text!r}, telemetry={tel}"
        )
        assert tel["segments_replaced"] == 1
        assert tel["examples"][0]["from"] == "starridge shoots and scores"
        assert tel["examples"][0]["to"] == "Sturridge shoots and scores"

    def test_length_penalty_blocks_drift(self):
        """A hypothesis that adds many extra words to the 1-best should
        be penalised even if it scores higher on entities (so we
        don't introduce structural WER drift)."""
        from pipeline import nbest_reranker as nr
        # Boost the length-penalty weight for this test only
        original_w = nr.W_LENGTH_PENALTY
        nr.W_LENGTH_PENALTY = 0.5
        try:
            gaz = {"sterling": "Sterling", "sturridge": "Sturridge"}
            segs = [_seg(
                "Sterling shoots",
                nbest=["Sterling Sturridge Henderson Mignolet shoots wide of post"],
            )]
            out, _ = rerank_match(segs, gazetteer=gaz)
            assert out[0].text == "Sterling shoots", (
                f"length penalty should block drift; got {out[0].text!r}"
            )
        finally:
            nr.W_LENGTH_PENALTY = original_w

    def test_telemetry_persists_via_get_last_telemetry(self):
        gaz = {"sterling": "Sterling"}
        segs = [_seg("Sterling shoots", nbest=["Sterling shoots", "Sterling shoots"])]
        rerank_match(segs, gazetteer=gaz)
        tel = get_last_telemetry()
        assert tel["segments_total"] == 1
        assert "wall_time_sec" in tel

    def test_batch_preserves_segment_ids(self):
        gaz = {"sterling": "Sterling"}
        segs = [
            _seg("Sterling shoots", sid="0", nbest=["Sterling shoots", "Stirling shoots"]),
            _seg("Sturridge wide", sid="1"),  # no nbest → pass-through
        ]
        out, _ = rerank_match(segs, gazetteer=gaz)
        assert [s.segment_id for s in out] == ["0", "1"]
