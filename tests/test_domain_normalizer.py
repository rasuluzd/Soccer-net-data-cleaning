"""Tests for pipeline/domain_normalizer.py — Stage 2A."""

import pytest
from pipeline.domain_normalizer import DomainNormalizer
from pipeline.loader import Segment


def _seg(text, sid="0"):
    return Segment(segment_id=sid, start_time=0.0, end_time=5.0, text=text, half=1)


class TestDisfluencyRemoval:
    """Filler words (uh, um, eh) should be removed."""

    def test_removes_uh(self):
        norm = DomainNormalizer("en")
        result, corr = norm.normalize_segment("Sterling uh passes the ball")
        assert "uh" not in result
        assert "Sterling" in result
        assert len(corr) == 1
        assert corr[0]["method"] == "normalization"

    def test_removes_multiple_fillers(self):
        norm = DomainNormalizer("en")
        result, corr = norm.normalize_segment("um he shoots eh and scores")
        assert "um" not in result
        assert "eh" not in result
        assert len(corr) == 2

    def test_preserves_clean_text(self):
        norm = DomainNormalizer("en")
        result, corr = norm.normalize_segment("brilliant goal from Sterling")
        assert result == "brilliant goal from Sterling"
        assert len(corr) == 0

    def test_filler_at_end(self):
        norm = DomainNormalizer("en")
        result, corr = norm.normalize_segment("great tackle um")
        assert result.strip() == "great tackle"

    def test_case_insensitive(self):
        norm = DomainNormalizer("en")
        result, _ = norm.normalize_segment("UH what a save UM")
        assert "UH" not in result
        assert "UM" not in result


class TestCompoundMerging:
    """Split football compounds should be merged."""

    def test_offside(self):
        norm = DomainNormalizer("en")
        result, corr = norm.normalize_segment("he was off side")
        assert "offside" in result
        assert len(corr) == 1

    def test_goalkeeper(self):
        norm = DomainNormalizer("en")
        result, _ = norm.normalize_segment("the goal keeper saves it")
        assert "goalkeeper" in result

    def test_halftime(self):
        norm = DomainNormalizer("en")
        result, _ = norm.normalize_segment("half time whistle blows")
        assert "halftime" in result

    def test_already_correct_not_doubled(self):
        norm = DomainNormalizer("en")
        result, corr = norm.normalize_segment("offside called by the referee")
        assert result == "offside called by the referee"
        assert len(corr) == 0

    def test_case_insensitive_compound(self):
        norm = DomainNormalizer("en")
        result, _ = norm.normalize_segment("Off Side!")
        assert "offside" in result.lower() or "Offside" in result


class TestSwedishNormalization:
    """Swedish football compounds."""

    def test_swedish_offside(self):
        norm = DomainNormalizer("sv")
        result, _ = norm.normalize_segment("det var av sida")
        assert "avside" in result

    def test_swedish_halftime(self):
        norm = DomainNormalizer("sv")
        result, _ = norm.normalize_segment("halv tid paus")
        assert "halvtid" in result


class TestGermanNormalization:
    """German football compounds."""

    def test_german_offside(self):
        norm = DomainNormalizer("de")
        result, _ = norm.normalize_segment("das war ab seits")
        assert "Abseits" in result


class TestPunctuationCleanup:
    """Repeated punctuation and extra spaces."""

    def test_repeated_dots(self):
        norm = DomainNormalizer("en")
        result, _ = norm.normalize_segment("he shoots... and scores!!")
        assert "..." not in result
        assert "!!" not in result

    def test_extra_spaces(self):
        norm = DomainNormalizer("en")
        result, _ = norm.normalize_segment("Sterling   passes   the ball")
        assert "   " not in result
        assert "Sterling passes the ball" == result


class TestBatchNormalization:
    """Batch processing preserves segment metadata."""

    def test_batch_preserves_ids(self):
        norm = DomainNormalizer("en")
        segments = [
            _seg("off side call", "0"),
            _seg("great goal", "1"),
        ]
        corrected, corrections = norm.normalize_batch(segments)
        assert len(corrected) == 2
        assert corrected[0].segment_id == "0"
        assert corrected[1].segment_id == "1"
        assert "offside" in corrected[0].text
        assert corrected[1].text == "great goal"  # unchanged

    def test_batch_tracks_segment_ids_in_corrections(self):
        norm = DomainNormalizer("en")
        segments = [_seg("off side uh", "42")]
        _, corrections = norm.normalize_batch(segments)
        assert all(c["segment_id"] == "42" for c in corrections)

    def test_batch_preserves_timestamps(self):
        norm = DomainNormalizer("en")
        seg = Segment(segment_id="0", start_time=10.5, end_time=15.3, text="off side", half=2)
        corrected, _ = norm.normalize_batch([seg])
        assert corrected[0].start_time == 10.5
        assert corrected[0].end_time == 15.3
        assert corrected[0].half == 2

    def test_batch_preserves_confidence_metadata_when_unchanged(self):
        norm = DomainNormalizer("en")
        seg = Segment(
            segment_id="0",
            start_time=0.0,
            end_time=5.0,
            text="clean football text",
            half=1,
            words=[{"word": "clean", "prob": 0.93}],
            avg_logprob=-0.1,
        )
        corrected, corrections = norm.normalize_batch([seg])
        assert corrections == []
        assert corrected[0].words == seg.words
        assert corrected[0].avg_logprob == -0.1

    def test_batch_drops_word_metadata_when_text_changes(self):
        norm = DomainNormalizer("en")
        seg = Segment(
            segment_id="0",
            start_time=0.0,
            end_time=5.0,
            text="off side call",
            half=1,
            words=[{"word": "off", "prob": 0.93}],
            avg_logprob=-0.1,
        )
        corrected, corrections = norm.normalize_batch([seg])
        assert corrections
        assert corrected[0].text == "offside call"
        assert corrected[0].words is None
        assert corrected[0].avg_logprob == -0.1
