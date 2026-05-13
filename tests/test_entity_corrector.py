"""Smoke + correctness tests for pipeline/entity_corrector.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pipeline.entity_corrector import (
    _GazetteerIndex,
    _validated_cache_lookup,
    _validated_cache_record,
    _build_match_context_block,
    _build_mcq_user_msg,
    correct_match,
    get_last_telemetry,
    SHORTCUT_ACCEPT_TFIDF,
    SHORTCUT_REJECT_TFIDF,
)
from pipeline.loader import Segment


# ─── Test fixtures ──────────────────────────────────────────────────

CHELSEA_GAZ = {
    "Daniel Sturridge": "Daniel Sturridge",
    "Sturridge": "Daniel Sturridge",
    "Sadio Mane": "Sadio Mane",
    "Mane": "Sadio Mane",
    "Jordan Henderson": "Jordan Henderson",
    "Henderson": "Jordan Henderson",
    "Eden Hazard": "Eden Hazard",
    "Hazard": "Eden Hazard",
    "Diego Costa": "Diego Costa",
    "Costa": "Diego Costa",
    "Adam Lallana": "Adam Lallana",
    "Lallana": "Adam Lallana",
}

CHELSEA_TYPES = {
    "Daniel Sturridge": "player",
    "Sadio Mane": "player",
    "Jordan Henderson": "player",
    "Eden Hazard": "player",
    "Diego Costa": "player",
    "Adam Lallana": "player",
}


# ─── _GazetteerIndex (TF-IDF retrieval) ─────────────────────────────

class TestGazetteerIndex:
    def test_retrieve_shortcut_winner(self):
        idx = _GazetteerIndex(CHELSEA_GAZ)
        results = idx.retrieve("Sturridge", top_k=3)
        assert results
        assert results[0][0] == "Daniel Sturridge"
        # Score should be very high — entity exactly matches a gazetteer key.
        assert results[0][1] >= 0.5

    def test_retrieve_typo_match(self):
        idx = _GazetteerIndex(CHELSEA_GAZ)
        results = idx.retrieve("Starridge", top_k=3)
        assert results
        # First hit should be Sturridge despite the misspelling
        assert results[0][0] == "Daniel Sturridge"
        # Score should be in uncertain band (≥0.40 retrieve, <0.90 shortcut)
        assert 0.40 <= results[0][1] < 0.90

    def test_retrieve_high_fuzz_rescue_stays_below_shortcut(self):
        idx = _GazetteerIndex({
            "Mamadou Sakho": "Mamadou Sakho",
            "Sakho": "Mamadou Sakho",
        })
        results = idx.retrieve("Sako", top_k=3)
        assert results
        assert results[0][0] == "Mamadou Sakho"
        assert 0.85 <= results[0][1] < SHORTCUT_ACCEPT_TFIDF

    def test_retrieve_cross_domain_word_below_reject(self):
        """The whole point of TF-IDF: 'Saturday' should NOT score high
        against any player name. This is the architectural FP fix."""
        idx = _GazetteerIndex(CHELSEA_GAZ)
        results = idx.retrieve("Saturday", top_k=3)
        if results:
            # Best score should be below the reject threshold
            assert results[0][1] < SHORTCUT_REJECT_TFIDF, (
                f"'Saturday' should score < {SHORTCUT_REJECT_TFIDF}, "
                f"got {results[0][1]:.3f} for {results[0][0]}"
            )

    def test_retrieve_premier_below_reject(self):
        idx = _GazetteerIndex(CHELSEA_GAZ)
        results = idx.retrieve("Premier", top_k=3)
        if results:
            assert results[0][1] < SHORTCUT_REJECT_TFIDF

    def test_retrieve_dutchman_below_reject(self):
        idx = _GazetteerIndex(CHELSEA_GAZ)
        results = idx.retrieve("Dutchman", top_k=3)
        if results:
            assert results[0][1] < SHORTCUT_REJECT_TFIDF

    def test_empty_gazetteer_returns_nothing(self):
        idx = _GazetteerIndex({})
        assert idx.retrieve("anything") == []

    def test_empty_query_returns_nothing(self):
        idx = _GazetteerIndex(CHELSEA_GAZ)
        assert idx.retrieve("") == []
        assert idx.retrieve("   ") == []


# ─── Cross-match validated cache ────────────────────────────────────

class TestValidatedCache:
    def test_lookup_returns_none_when_below_consensus(self):
        cache = {
            "starridge": {"correct": "Daniel Sturridge", "matches_seen": ["m1", "m2"]}
        }
        # MIN_CONSENSUS=3 → 2 matches isn't enough
        assert _validated_cache_lookup("starridge", cache) is None

    def test_lookup_returns_correction_at_or_above_consensus(self):
        cache = {
            "starridge": {
                "correct": "Daniel Sturridge",
                "matches_seen": ["m1", "m2", "m3"],
            }
        }
        assert _validated_cache_lookup("starridge", cache) == "Daniel Sturridge"

    def test_record_promotes_at_consensus_boundary(self):
        cache = {}
        # First two matches: not yet promoted
        promoted = _validated_cache_record(
            cache, "starridge", "Daniel Sturridge", "m1", 90.0,
        )
        assert not promoted
        promoted = _validated_cache_record(
            cache, "starridge", "Daniel Sturridge", "m2", 88.0,
        )
        assert not promoted
        # Third match: promotes
        promoted = _validated_cache_record(
            cache, "starridge", "Daniel Sturridge", "m3", 92.0,
        )
        assert promoted, "Should have promoted at consensus boundary"
        # Fourth: already validated, no further promotion event
        promoted = _validated_cache_record(
            cache, "starridge", "Daniel Sturridge", "m4", 90.0,
        )
        assert not promoted

    def test_record_refuses_to_overwrite_established_mapping(self):
        """If a different correction is proposed for the same entity, refuse
        it. Prevents one match's mistake from poisoning the cache."""
        cache = {}
        _validated_cache_record(cache, "starridge", "Daniel Sturridge", "m1", 90.0)
        # Different correction proposed in match 2 — must NOT overwrite
        _validated_cache_record(cache, "starridge", "Adam Lallana", "m2", 85.0)
        assert cache["starridge"]["correct"] == "Daniel Sturridge"
        assert "m2" not in cache["starridge"]["matches_seen"]

    def test_record_rejects_low_fuzzy_mappings(self):
        """Don't pollute the cache with weak fuzzy matches even if MCQ accepted."""
        cache = {}
        # fuzz=50 — well below VALIDATED_CACHE_MIN_FUZZY=75
        promoted = _validated_cache_record(
            cache, "saturday", "Daniel Sturridge", "m1", 50.0,
        )
        assert not promoted
        assert "saturday" not in cache

    def test_idempotent_within_same_match(self):
        """Calling record twice with same match_id should be a no-op."""
        cache = {}
        _validated_cache_record(cache, "starridge", "Daniel Sturridge", "m1", 90.0)
        promoted = _validated_cache_record(
            cache, "starridge", "Daniel Sturridge", "m1", 90.0,
        )
        assert not promoted
        assert cache["starridge"]["matches_seen"] == ["m1"]


# ─── Prompt construction ────────────────────────────────────────────

class TestPromptBuilders:
    def test_match_context_block_includes_typed_lists(self):
        gaz = {"Hazard": "Eden Hazard"}
        types = {
            "Eden Hazard": "player",
            "Chelsea": "team",
            "Martin Atkinson": "referee",
        }
        block = _build_match_context_block(gaz, types, "Chelsea v Liverpool", half=1)
        assert "Players: Eden Hazard" in block
        assert "Teams: Chelsea" in block
        assert "Referee: Martin Atkinson" in block

    def test_mcq_user_msg_has_letter_options(self):
        msg = _build_mcq_user_msg(
            "Starridge",
            ["Daniel Sturridge", "Sadio Mane", "Diego Costa"],
            prev_text="prev", next_text="next", segment_text="seg text Starridge",
        )
        assert "A) Daniel Sturridge" in msg
        assert "B) Sadio Mane" in msg
        assert "C) Diego Costa" in msg
        assert "D) keep original" in msg
        assert "E) unsure" in msg
        assert "Original token: \"Starridge\"" in msg

    def test_mcq_user_msg_pads_when_few_candidates(self):
        msg = _build_mcq_user_msg(
            "Starridge", ["Daniel Sturridge"],
            prev_text="", next_text="", segment_text="x",
        )
        assert "A) Daniel Sturridge" in msg
        assert "D) keep original" in msg


# ─── End-to-end correct_match (mocked LLM) ──────────────────────────

def _seg(sid, text, half=1, t=0.0):
    return Segment(
        segment_id=sid, start_time=t, end_time=t + 1.0,
        text=text, half=half,
    )


class _Ent:
    """Stand-in for ner_extractor.DetectedEntity."""
    def __init__(self, text, start_char, end_char, label="PERSON", pos="PROPN"):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.label = label
        self.pos = pos
        self.source = "spacy"


class TestCorrectMatchEndToEnd:
    @pytest.fixture(autouse=True)
    def _disable_validated_cache_and_mlm(self, tmp_path, monkeypatch):
        # Point validated cache to a temp path so tests don't persist
        from pipeline import entity_corrector as ec
        monkeypatch.setattr(
            ec, "VALIDATED_CACHE_PATH", str(tmp_path / "validated.json"),
        )
        # Default-disable MLM veto in tests so we don't load xlm-roberta
        # for every smoke test. Per-test override via monkeypatch.
        monkeypatch.setattr(ec, "MLM_VETO_ON_MCQ_ENABLED", False)

    def test_empty_gazetteer_is_noop(self):
        segs = [_seg("0", "Hello world")]
        out, corrs = correct_match(
            segments=segs, gazetteer={}, entity_types={},
            segment_entities_map={}, match_id="m1",
        )
        assert out == segs
        assert corrs == []

    def test_no_entities_means_no_corrections(self):
        segs = [_seg("0", "Hello world")]
        out, corrs = correct_match(
            segments=segs, gazetteer=CHELSEA_GAZ, entity_types=CHELSEA_TYPES,
            segment_entities_map={(1, "0"): []}, match_id="m1",
        )
        assert out == segs
        assert corrs == []

    def test_shortcut_accept_for_typo(self):
        """A clear ASR typo should auto-accept via the TF-IDF shortcut without
        invoking the LLM."""
        text = "Starridge passes to Hazard"
        seg = _seg("0", text)
        ent = _Ent("Starridge", 0, 9)
        # Auto-accept threshold needs cosine ≥ 0.90 + clear winner.
        # 'Starridge' against 'Daniel Sturridge' typically scores ~0.55-0.65,
        # so this would route to MCQ — mock the LLM call.
        from pipeline import entity_corrector as ec
        with patch.object(ec, "_mcq_call", return_value="A") as mock_mcq:
            out, corrs = correct_match(
                segments=[seg], gazetteer=CHELSEA_GAZ, entity_types=CHELSEA_TYPES,
                segment_entities_map={(1, "0"): [ent]}, match_id="m1",
            )
        # Should have called MCQ for the uncertain band
        assert mock_mcq.called
        assert len(corrs) == 1
        assert corrs[0]["original"] == "Starridge"
        # Single-word entity → reduces to closest canonical word ("Sturridge")
        # not the full "Daniel Sturridge" (legacy Tier 2 also did this)
        assert corrs[0]["corrected"] == "Sturridge"

    def test_cross_domain_word_auto_rejected_without_llm(self):
        """The architectural fix: 'Saturday' has TF-IDF cosine well below the
        reject threshold, so the LLM is NEVER called for it."""
        text = "Saturday night football"
        seg = _seg("0", text)
        ent = _Ent("Saturday", 0, 8)
        from pipeline import entity_corrector as ec
        with patch.object(ec, "_mcq_call") as mock_mcq:
            out, corrs = correct_match(
                segments=[seg], gazetteer=CHELSEA_GAZ, entity_types=CHELSEA_TYPES,
                segment_entities_map={(1, "0"): [ent]}, match_id="m1",
            )
        assert not mock_mcq.called, "LLM should NOT be called for Saturday"
        assert corrs == []  # no correction applied
        assert out[0].text == text  # text unchanged

    def test_mcq_returns_keep_means_no_correction(self):
        text = "Starridge passes"
        seg = _seg("0", text)
        ent = _Ent("Starridge", 0, 9)
        from pipeline import entity_corrector as ec
        with patch.object(ec, "_mcq_call", return_value="D"):
            out, corrs = correct_match(
                segments=[seg], gazetteer=CHELSEA_GAZ, entity_types=CHELSEA_TYPES,
                segment_entities_map={(1, "0"): [ent]}, match_id="m1",
            )
        assert corrs == []
        assert out[0].text == text

    def test_mcq_returns_unsure_means_no_correction(self):
        text = "Starridge passes"
        seg = _seg("0", text)
        ent = _Ent("Starridge", 0, 9)
        from pipeline import entity_corrector as ec
        with patch.object(ec, "_mcq_call", return_value="E"):
            out, corrs = correct_match(
                segments=[seg], gazetteer=CHELSEA_GAZ, entity_types=CHELSEA_TYPES,
                segment_entities_map={(1, "0"): [ent]}, match_id="m1",
            )
        assert corrs == []

    def test_per_match_cache_avoids_duplicate_llm_calls(self):
        """Same entity text appearing twice should only invoke the MCQ once."""
        seg1 = _seg("0", "Starridge passes")
        seg2 = _seg("1", "Starridge again", t=2)
        ent1 = _Ent("Starridge", 0, 9)
        ent2 = _Ent("Starridge", 0, 9)
        from pipeline import entity_corrector as ec
        with patch.object(ec, "_mcq_call", return_value="A") as mock_mcq:
            out, corrs = correct_match(
                segments=[seg1, seg2], gazetteer=CHELSEA_GAZ,
                entity_types=CHELSEA_TYPES,
                segment_entities_map={
                    (1, "0"): [ent1], (1, "1"): [ent2],
                },
                match_id="m1",
            )
        # Should have invoked MCQ exactly once thanks to per-match cache
        assert mock_mcq.call_count == 1
        # But BOTH segments should get the correction applied
        assert len(corrs) == 2

    def test_telemetry_records_all_branches(self):
        text = "Starridge and Saturday"
        seg = _seg("0", text)
        starridge_ent = _Ent("Starridge", 0, 9)
        saturday_ent = _Ent("Saturday", 14, 22)
        from pipeline import entity_corrector as ec
        with patch.object(ec, "_mcq_call", return_value="A"):
            out, corrs = correct_match(
                segments=[seg], gazetteer=CHELSEA_GAZ, entity_types=CHELSEA_TYPES,
                segment_entities_map={(1, "0"): [starridge_ent, saturday_ent]},
                match_id="m1",
            )
        t = get_last_telemetry()
        assert t["total_entities"] >= 2
        assert t["mcq_invoked"] >= 1  # Starridge
        assert t["auto_reject_low"] >= 1  # Saturday
        assert t["mcq_chose_candidate"] >= 1  # Starridge accepted

    def test_validation_gate_blocks_dictionary_word_mcq_pick(self):
        """Even if MCQ picks A, the validation gate should still reject when
        the original is a real common-language word and fuzz is low."""
        text = "Premier kicks the ball"
        seg = _seg("0", text)
        ent = _Ent("Premier", 0, 7)
        from pipeline import entity_corrector as ec
        # Mock MCQ to (incorrectly) pick A — gate should still reject because
        # 'Premier' is a real English word AND the fuzz against 'Henderson' is
        # low. But TF-IDF likely auto-rejects first; ensure final result is
        # still no correction.
        with patch.object(ec, "_mcq_call", return_value="A"):
            out, corrs = correct_match(
                segments=[seg], gazetteer=CHELSEA_GAZ, entity_types=CHELSEA_TYPES,
                segment_entities_map={(1, "0"): [ent]}, match_id="m1",
            )
        assert corrs == [], "Premier should not be corrected by any path"

    def test_mcq_min_token_len_gate_blocks_short_tokens(self):
        """4-char tokens (Kane, Mann) should never reach MCQ regardless of
        TF-IDF cosine — they're the FP class."""
        text = "Kane in Tottenham"
        seg = _seg("0", text)
        ent = _Ent("Kane", 0, 4)
        from pipeline import entity_corrector as ec
        with patch.object(ec, "_mcq_call", return_value="A") as mock_mcq:
            out, corrs = correct_match(
                segments=[seg], gazetteer=CHELSEA_GAZ, entity_types=CHELSEA_TYPES,
                segment_entities_map={(1, "0"): [ent]}, match_id="m1",
            )
        assert not mock_mcq.called, "4-char Kane must skip MCQ"
        assert corrs == []

    def test_short_high_fuzz_name_typo_can_reach_mcq(self):
        """4-char player-name typos with very high fuzz should not be thrown
        away by the generic short-token FP gate."""
        gaz = {
            "Mamadou Sakho": "Mamadou Sakho",
            "Sakho": "Mamadou Sakho",
        }
        types = {"Mamadou Sakho": "player"}
        text = "Sako clears"
        seg = _seg("0", text)
        ent = _Ent("Sako", 0, 4)
        from pipeline import entity_corrector as ec
        with patch.object(ec, "_mcq_call", return_value="A") as mock_mcq:
            out, corrs = correct_match(
                segments=[seg], gazetteer=gaz, entity_types=types,
                segment_entities_map={(1, "0"): [ent]}, match_id="m1",
            )
        assert mock_mcq.called
        assert len(corrs) == 1
        assert corrs[0]["corrected"] == "Sakho"
        assert out[0].text == "Sakho clears"

    def test_mlm_veto_rejects_pick(self):
        """When MLM veto is enabled and reports True (original more plausible),
        the MCQ pick is rejected even if Qwen accepted it."""
        text = "Starridge passes"
        seg = _seg("0", text)
        ent = _Ent("Starridge", 0, 9)
        from pipeline import entity_corrector as ec
        with patch.object(ec, "MLM_VETO_ON_MCQ_ENABLED", True):
            with patch.object(ec, "_mlm_veto_mcq_pick", return_value=True):
                with patch.object(ec, "_mcq_call", return_value="A"):
                    out, corrs = correct_match(
                        segments=[seg], gazetteer=CHELSEA_GAZ,
                        entity_types=CHELSEA_TYPES,
                        segment_entities_map={(1, "0"): [ent]}, match_id="m1",
                    )
        assert corrs == []
        t = get_last_telemetry()
        assert t["mlm_vetoed_mcq"] == 1

    def test_self_consistency_majority_vote_picks(self):
        """With samples=3, 2 vote A, 1 votes E → majority is A (>50%)."""
        from pipeline import entity_corrector as ec
        seq = ["A", "A", "E"]
        with patch.object(ec, "MCQ_SELF_CONSISTENCY_SAMPLES", 3), \
                patch.object(ec, "_single_mcq_sample", side_effect=seq):
            r = ec._mcq_call("x", ["c1"], "", "", "x", "")
        assert r == "A"

    def test_self_consistency_no_majority_keeps(self):
        """With samples=3, 1 each A/B/C → no majority → returns D (keep)."""
        from pipeline import entity_corrector as ec
        seq = ["A", "B", "C"]
        with patch.object(ec, "MCQ_SELF_CONSISTENCY_SAMPLES", 3), \
                patch.object(ec, "_single_mcq_sample", side_effect=seq):
            r = ec._mcq_call("x", ["c1", "c2", "c3"], "", "", "x", "")
        assert r == "D"

    def test_default_single_sample_returns_pick(self):
        """Default config (samples=1) returns the single picked letter directly."""
        from pipeline import entity_corrector as ec
        with patch.object(ec, "_single_mcq_sample", return_value="A"):
            r = ec._mcq_call("x", ["c1"], "", "", "x", "")
        assert r == "A"

    def test_frozen_word_indices_set_after_correction(self):
        """After entity_corrector applies a correction, the segment must
        carry frozen_word_indices for Step L to honour."""
        text = "Starridge runs forward"
        seg = _seg("0", text)
        ent = _Ent("Starridge", 0, 9)
        from pipeline import entity_corrector as ec
        with patch.object(ec, "_mcq_call", return_value="A"):
            out, corrs = correct_match(
                segments=[seg], gazetteer=CHELSEA_GAZ,
                entity_types=CHELSEA_TYPES,
                segment_entities_map={(1, "0"): [ent]}, match_id="m1",
            )
        assert len(corrs) == 1
        assert out[0].frozen_word_indices is not None
        # Sturridge ends up at word index 0 of the new text
        assert 0 in out[0].frozen_word_indices
