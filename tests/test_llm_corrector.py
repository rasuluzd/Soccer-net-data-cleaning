"""Smoke tests for pipeline/llm_corrector.py.

Tests the contract of the confidence-gated GER module without invoking the
actual Qwen GGUF model (the model is large and tests must stay fast). The
real LLM path is exercised by the end-to-end pipeline run.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from pipeline.loader import Segment
import pipeline.llm_corrector as lc


def _seg(sid, text, half=1, words=None, nbest=None) -> Segment:
    return Segment(
        segment_id=sid, start_time=0.0, end_time=1.0, text=text, half=half,
        words=words, nbest=nbest,
    )


def _word(word, prob):
    return {"word": word, "start": 0.0, "end": 0.1, "prob": prob}


@pytest.fixture(autouse=True)
def _reset_singletons():
    lc._LLM = None
    lc._LLM_LOAD_ERROR = "mocked-no-model"  # block real download in tests
    lc._MLM_HANDLE = None
    yield
    lc._LLM = None
    lc._LLM_LOAD_ERROR = None
    lc._MLM_HANDLE = None


def test_disabled_in_config_is_noop():
    segs = [_seg("0", "hello world")]
    with patch("pipeline.llm_corrector.LLM_CORRECTION_ENABLED", False):
        out, corrections = lc.correct_match(segs, {}, {}, match_name="x")
    assert out == segs and corrections == []


def test_unavailable_llm_is_noop():
    """When the GGUF model can't be loaded, the corrector is a no-op."""
    segs = [_seg("0", "hello world", words=[_word("hello", 0.1), _word("world", 0.1)])]
    out, corrections = lc.correct_match(segs, {"Hazard": "Hazard"},
                                         {"Hazard": "player"}, match_name="x")
    assert out == segs and corrections == []


def test_wrap_low_confidence_with_word_probs():
    """Low-confidence words get <wrapped>, high-confidence stay verbatim."""
    seg = _seg("0", "hello Sturridge",
               words=[_word("hello", 0.99), _word("Sturridge", 0.05)])
    # log(0.99) ≈ -0.01 (above gate -0.3); log(0.05) ≈ -3.0 (below gate)
    wrapped, n, mask = lc._wrap_low_confidence_tokens(seg)
    assert "<Sturridge>" in wrapped
    assert "<hello>" not in wrapped
    assert n == 1


def test_wrap_falls_back_to_capitalized_when_no_word_probs():
    """Schema-1 input (no `words`) falls back to wrapping capitalised tokens."""
    seg = _seg("0", "hello Sturridge passes to Hazard")
    wrapped, n, mask = lc._wrap_low_confidence_tokens(seg)
    assert "<Sturridge>" in wrapped
    assert "<Hazard>" in wrapped
    assert "hello" in wrapped and "<hello>" not in wrapped
    assert n == 2


def test_skip_segment_below_min_tokens_to_invoke():
    """If too few low-confidence tokens, the LLM is not invoked."""
    seg = _seg("0", "all good", words=[_word("all", 0.99), _word("good", 0.99)])
    lc._LLM = object()  # pretend LLM is loaded
    lc._LLM_LOAD_ERROR = None
    with patch("pipeline.llm_corrector._llm_correct", lambda *_a, **_k: pytest.fail("LLM should not be called")):
        with patch("pipeline.llm_corrector.LLM_MIN_TOKENS_TO_INVOKE", 1):
            # 0 wrapped tokens → still skip
            out, corrections = lc.correct_match(
                [seg], {}, {}, match_name="x",
            )
    assert out == [seg]
    assert corrections == []


def test_clean_llm_output_strips_artefacts():
    assert lc._clean_llm_output("Corrected: hello") == "hello"
    assert lc._clean_llm_output('"hello world"') == "hello world"
    assert lc._clean_llm_output("Output: hello") == "hello"
    assert lc._clean_llm_output("  hello  ") == "hello"


def test_veto_disabled_passes_correction_through():
    with patch("pipeline.llm_corrector.MLM_VETO_ENABLED", False):
        result = lc._veto("Saturday is here", "Sturridge is here")
    assert result == "Sturridge is here"


def test_veto_rejects_when_mlm_prefers_original_strongly():
    """MLM votes original (-1.0) over proposed (-5.0) → reject the edit."""
    def fake_lp(_tokens, _idx, candidate):
        return -1.0 if candidate == "Saturday" else -5.0
    with patch("pipeline.llm_corrector._mlm_pseudo_logprob", side_effect=fake_lp):
        result = lc._veto("Saturday is here", "Sturridge is here")
    assert result == "Saturday is here"


def test_veto_accepts_when_mlm_prefers_proposed():
    """MLM votes proposed (-1.0) over original (-5.0) → accept the edit."""
    def fake_lp(_tokens, _idx, candidate):
        return -1.0 if candidate == "Sturridge" else -5.0
    with patch("pipeline.llm_corrector._mlm_pseudo_logprob", side_effect=fake_lp):
        result = lc._veto("Sterridge is here", "Sturridge is here")
    assert result == "Sturridge is here"


def test_build_context_prompt_includes_typed_lists():
    gaz = {"Hazard": "Hazard", "Chelsea": "Chelsea", "Atkinson": "Martin Atkinson"}
    types = {"Hazard": "player", "Chelsea": "team",
             "Martin Atkinson": "referee"}
    prompt = lc._build_context_prompt(
        gaz, types, prev_segments=[], next_segments=[],
        match_name="Chelsea v Liverpool", half=1,
    )
    assert "Players: Hazard" in prompt
    assert "Teams: Chelsea" in prompt
    assert "Referee: Martin Atkinson" in prompt


# ── Output-validation guards (catch Qwen-0.5B prompt-echo / over-rewrite) ──

def test_is_prompt_echo_detects_match_line():
    assert lc._is_prompt_echo("Match: Chelsea v Liverpool | Half: 1")
    assert lc._is_prompt_echo("Players: Hazard, Costa")
    assert lc._is_prompt_echo("Output: Sturridge")
    assert lc._is_prompt_echo("Input: hello")


def test_is_prompt_echo_passes_clean_output():
    assert not lc._is_prompt_echo("Sturridge passes to Hazard.")
    assert not lc._is_prompt_echo("And away we go.")
    assert not lc._is_prompt_echo("It's a starry night here at Stamford Bridge.")


def test_is_length_anomaly_flags_too_short_or_too_long():
    # GER should make the smallest possible edit — anything <50% or >150% is suspect.
    assert lc._is_length_anomaly("a b c d e f g h", "a b c")        # 3/8 = 0.375 < 0.5
    assert lc._is_length_anomaly("a b c", "a b c d e f g h")        # 8/3 = 2.67 > 1.5
    assert not lc._is_length_anomaly("a b c d e", "a b c d e")      # exact same
    assert not lc._is_length_anomaly("a b c d e", "x b c d e")      # word swap
    assert not lc._is_length_anomaly("a b c d", "a b c d e")        # +1 word


def test_clean_llm_output_takes_only_first_line():
    """Drop multi-line continuations the model emits after the corrected segment."""
    raw = "Sturridge passes to Hazard.\nInput: another seg\nOutput: another"
    assert lc._clean_llm_output(raw) == "Sturridge passes to Hazard."


# ── Hard constraint: only bracketed positions may change ──────────────

def test_editable_drift_rejects_change_at_non_editable_position():
    """Qwen 'Sturridge waiting' -> 'Sturridge IS waiting' must be rejected."""
    original = "Sturridge waiting on the right"
    # Pretend only "Sturridge" was bracketed
    mask = [True, False, False, False, False]
    assert lc._editable_drift(original, "Sturridge is waiting on the right", mask)


def test_editable_drift_rejects_length_change():
    """Any word-count change is treated as drift (non-editable insertion/deletion)."""
    original = "a b c"
    mask = [True, False, False]
    assert lc._editable_drift(original, "a b c d", mask)
    assert lc._editable_drift(original, "a b", mask)


def test_editable_drift_accepts_change_only_at_editable_position():
    """Replacing a bracketed token at a True position is allowed."""
    original = "Sterridge waiting on the right"
    mask = [True, False, False, False, False]
    assert not lc._editable_drift(original, "Sturridge waiting on the right", mask)


def test_editable_drift_tolerates_punctuation_only_change():
    """Trailing punctuation differences are normalised away."""
    original = "Sturridge passes to Hazard"
    mask = [False, False, False, True]
    assert not lc._editable_drift(original, "Sturridge passes to Hazard.", mask)
    # And changing the editable token while adding punct is fine
    assert not lc._editable_drift(original, "Sturridge passes to Hazard,", mask)


# ── Schema-1 fallback: POS-filter sentence-start common words ────────

def test_schema1_fallback_skips_sentence_start_pronoun():
    """'And', 'It's', 'He', 'The' at sentence start are common words, not
    entities. They must NOT be wrapped or the LLM is invited to rewrite
    grammar instead of fixing names."""
    seg = _seg("0", "And away we go.")
    wrapped, n, mask = lc._wrap_low_confidence_tokens(seg, gazetteer={}, language="en")
    assert "<And>" not in wrapped
    assert mask[0] is False, "sentence-start 'And' should not be editable"


def test_schema1_fallback_skips_multiple_sentence_start_commons():
    for sentence in ["He's running.", "It's a goal.", "They are out.",
                     "The keeper saved.", "Here's the kick."]:
        seg = _seg("0", sentence)
        wrapped, n, mask = lc._wrap_low_confidence_tokens(
            seg, gazetteer={}, language="en",
        )
        first_tok = sentence.split()[0]
        assert mask[0] is False, (
            f"sentence-start '{first_tok}' must be non-editable but mask is {mask}"
        )


def test_schema1_fallback_keeps_mid_sentence_capitalized_token():
    """Mid-sentence capitalised tokens are almost always entity names —
    Whisper rarely capitalises function words mid-sentence."""
    seg = _seg("0", "passes to Sturridge again")
    wrapped, n, mask = lc._wrap_low_confidence_tokens(
        seg, gazetteer={}, language="en",
    )
    assert "<Sturridge>" in wrapped
    assert mask[2] is True


def test_schema1_fallback_known_gazetteer_entry_always_editable():
    """A capitalised token that matches a gazetteer entry must stay
    editable even at sentence-start position. Otherwise the LLM never
    gets a chance to fix 'Davy Luiz' → 'David Luiz' when 'Davy' is
    also the first word."""
    seg = _seg("0", "Davy Luiz takes the free kick.")
    gaz = {"Davy Luiz": "David Luiz", "David Luiz": "David Luiz", "Davy": "David"}
    wrapped, n, mask = lc._wrap_low_confidence_tokens(
        seg, gazetteer=gaz, language="en",
    )
    assert mask[0] is True, f"gazetteer-known 'Davy' must be editable; mask={mask}"


def test_schema1_fallback_uses_default_when_pos_tagger_unavailable():
    """If spaCy isn't installed/can't load, _wrap... must NOT crash; it
    should degrade to wrapping every capitalised token (the prior
    behaviour). Tests safety, not quality."""
    seg = _seg("0", "And Sturridge runs.")
    with patch("pipeline.llm_corrector._pos_tags_for", return_value={}):
        wrapped, n, mask = lc._wrap_low_confidence_tokens(
            seg, gazetteer={}, language="en",
        )
    # With no POS info, sentence-start cap stays editable (degraded mode).
    assert mask[0] is True


# ── Telemetry counters ───────────────────────────────────────────────

def test_telemetry_disabled_records_total_segments():
    segs = [_seg("0", "a"), _seg("1", "b")]
    with patch("pipeline.llm_corrector.LLM_CORRECTION_ENABLED", False):
        lc.correct_match(segs, {}, {}, match_name="x")
    t = lc.get_last_telemetry()
    assert t["total_segments"] == 2
    assert t["accepted"] == 0
    assert t.get("disabled_reason")


def test_telemetry_unavailable_llm_records_total_segments():
    segs = [_seg("0", "Hi", words=[_word("Hi", 0.99)])]
    lc.correct_match(segs, {}, {}, match_name="x")
    t = lc.get_last_telemetry()
    assert t["total_segments"] == 1
    assert t["accepted"] == 0


def test_telemetry_format_has_all_required_keys():
    """Lock the schema so downstream consumers (eval reports, dashboards)
    can rely on a stable shape."""
    segs = [_seg("0", "Hi")]
    with patch("pipeline.llm_corrector.LLM_CORRECTION_ENABLED", False):
        lc.correct_match(segs, {}, {}, match_name="x")
    t = lc.get_last_telemetry()
    for required in (
        "total_segments", "eligible_segments", "total_wrapped_tokens",
        "skipped_below_min", "llm_call_failed", "rejected_prompt_echo",
        "rejected_length_anomaly", "rejected_editable_drift",
        "mlm_rolled_back", "no_op", "accepted", "examples",
    ):
        assert required in t, f"missing telemetry field: {required}"
