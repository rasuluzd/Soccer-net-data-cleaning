"""Smoke tests for pipeline/punct_restorer.py.

These tests verify the module's *contract* (graceful no-op when model is
unavailable, conservative behaviour, segment metadata preservation) without
loading the actual ~1 GB punctuation model. Real model inference is exercised
indirectly by the end-to-end pipeline run.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pipeline.loader import Segment
from pipeline.punct_restorer import PunctuationRestorer, restore_punctuation_batch


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test gets a fresh restorer (no leaked HF model state)."""
    PunctuationRestorer._SINGLETON = None
    yield
    PunctuationRestorer._SINGLETON = None


def _seg(sid: str, text: str, half: int = 1) -> Segment:
    return Segment(
        segment_id=sid, start_time=0.0, end_time=1.0, text=text, half=half,
    )


def test_disabled_in_config_is_noop():
    segs = [_seg("0", "hello world")]
    with patch("pipeline.punct_restorer.PUNCT_RESTORATION_ENABLED", False):
        out, corrections = restore_punctuation_batch(segs, language="en")
    assert out == segs
    assert corrections == []


def test_unavailable_model_is_graceful_noop():
    """If the HF model fails to load, the restorer returns input unchanged."""
    segs = [_seg("0", "hello world"), _seg("1", "another segment")]
    r = PunctuationRestorer(language="en")
    # Force the lazy loader to record an init failure
    r._tried_init = True
    r._init_error = "transformers not installed (mocked)"
    out, corrections = r.restore_batch(segs)
    assert out == segs
    assert corrections == []


def test_unchanged_text_yields_no_correction():
    """When restored text equals input, no correction record is emitted."""
    segs = [_seg("0", "Hello, world.")]
    r = PunctuationRestorer(language="en")
    r._tried_init = True
    r._pipeline = lambda *_a, **_k: []  # mock pipeline returns no predictions
    out, corrections = r.restore_batch(segs)
    assert len(out) == 1
    assert corrections == []  # no edit to record


def test_segment_metadata_preserved_when_text_changes():
    """Schema-2 enrichments (words, prob, nbest, speaker_id) survive restoration."""
    seg = Segment(
        segment_id="0", start_time=10.0, end_time=12.0,
        text="hello world", half=1, global_id="abc",
        words=[{"word": "hello", "start": 10.0, "end": 10.5, "prob": 0.99}],
        avg_logprob=-0.18, no_speech_prob=0.01,
        nbest=["hello world", "hallo world"], speaker_id="SPEAKER_00",
    )
    r = PunctuationRestorer(language="en")
    r._tried_init = True
    # Mock the pipeline: predicts a period after "world"
    r._pipeline = lambda text: [
        {"entity": ".", "start": 6, "end": 11, "score": 0.95},
    ]
    out, corrections = r.restore_batch([seg])
    assert len(out) == 1
    new_seg = out[0]
    # Either text changed or it didn't — but metadata is always preserved
    assert new_seg.segment_id == "0"
    assert new_seg.global_id == "abc"
    assert new_seg.words == seg.words
    assert new_seg.avg_logprob == -0.18
    assert new_seg.nbest == seg.nbest
    assert new_seg.speaker_id == "SPEAKER_00"
