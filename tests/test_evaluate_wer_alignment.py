"""Regression tests for time-range alignment in tools/evaluate_wer.py.

The prior implementation matched segments by segment_id, which silently
produced wrong WER numbers when ground-truth curation dropped hallucinated
segments and re-numbered the remaining ones. These tests guard the fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.evaluate_wer import (  # noqa: E402
    TimedSegment,
    align_by_time,
    align_by_window,
    evaluate,
)


class TestAlignByTime:
    def test_identical_timings_pair_up(self):
        ref = [TimedSegment(0.0, 5.0, "a"), TimedSegment(5.0, 10.0, "b")]
        hyp = [TimedSegment(0.0, 5.0, "a"), TimedSegment(5.0, 10.0, "b")]
        pairs = align_by_time(ref, hyp)
        assert len(pairs) == 2
        assert pairs[0] == (ref[0], hyp[0])
        assert pairs[1] == (ref[1], hyp[1])

    def test_dropped_hyp_segment_pairs_correctly(self):
        """This is the critical case: GT dropped one segment and re-numbered.
        The prior ID-match approach compared ref[2] to hyp[2] even though
        hyp[2] was audio-wise equivalent to ref[3]. Time alignment fixes it.
        """
        ref = [
            TimedSegment(0.0, 5.0, "hello world"),
            TimedSegment(5.0, 10.0, "the match begins"),
            TimedSegment(15.0, 20.0, "goal scored"),
        ]
        hyp = [
            TimedSegment(0.0, 5.0, "hello world"),
            TimedSegment(5.0, 10.0, "the match begins"),
            TimedSegment(10.0, 15.0, "hallucinated noise"),  # not in GT
            TimedSegment(15.0, 20.0, "goal scored"),
        ]
        pairs = align_by_time(ref, hyp)
        # Pair 0: ref[0] ↔ hyp[0]
        assert pairs[0] == (ref[0], hyp[0])
        # Pair 1: ref[1] ↔ hyp[1]
        assert pairs[1] == (ref[1], hyp[1])
        # Pair 2: hyp[2] is a hallucination (no ref counterpart)
        assert pairs[2] == (None, hyp[2])
        # Pair 3: ref[2] ↔ hyp[3] — this is the key pairing that ID-match breaks
        assert pairs[3] == (ref[2], hyp[3])

    def test_dropped_ref_segment_emits_none(self):
        """When the hypothesis has a segment with no GT counterpart in time,
        the alignment emits (None, hyp_segment)."""
        ref = [TimedSegment(0.0, 5.0, "a"), TimedSegment(10.0, 15.0, "c")]
        hyp = [
            TimedSegment(0.0, 5.0, "a"),
            TimedSegment(5.0, 10.0, "b"),  # hyp-only
            TimedSegment(10.0, 15.0, "c"),
        ]
        pairs = align_by_time(ref, hyp)
        assert pairs[1] == (None, hyp[1])
        assert pairs[2] == (ref[1], hyp[2])

    def test_close_midpoints_within_tolerance_pair(self):
        """Segments with small timing drift should still pair within tolerance."""
        ref = [TimedSegment(0.0, 5.0, "a")]
        hyp = [TimedSegment(0.2, 5.5, "a")]  # 0.35s midpoint drift, within 0.75 tol
        pairs = align_by_time(ref, hyp, tolerance=0.75)
        assert pairs == [(ref[0], hyp[0])]

    def test_disjoint_segments_do_not_pair(self):
        """Segments with no time overlap and large midpoint gap should not pair."""
        ref = [TimedSegment(0.0, 2.0, "a")]
        hyp = [TimedSegment(20.0, 25.0, "z")]
        pairs = align_by_time(ref, hyp, tolerance=0.5)
        assert pairs == [(ref[0], None), (None, hyp[0])]

    def test_empty_inputs(self):
        assert align_by_time([], []) == []
        only_ref = align_by_time([TimedSegment(0, 5, "x")], [])
        assert only_ref == [(TimedSegment(0, 5, "x"), None)]
        only_hyp = align_by_time([], [TimedSegment(0, 5, "x")])
        assert only_hyp == [(None, TimedSegment(0, 5, "x"))]


class TestEvaluateUsesTimeAlignment:
    def test_evaluate_legacy_default_ignores_id_shift(self):
        """Legacy 1-to-1 alignment (default): extra hyp segment in middle
        counts as hyp_only, the actual content pairs by time."""
        ref = [
            TimedSegment(0.0, 5.0, "hello world"),
            TimedSegment(5.0, 10.0, "goal scored by Sterling"),
        ]
        hyp = [
            TimedSegment(0.0, 5.0, "hello world"),
            TimedSegment(5.0, 8.0, "hallu"),
            TimedSegment(8.0, 12.0, "goal scored by Sterling"),
        ]
        m = evaluate("test", ref, hyp)  # default = legacy
        assert m.aligned_pairs == 2
        assert m.hyp_only == 1
        assert 15 <= m.wer <= 25

    def test_evaluate_windowed_concatenates_overlapping_hyp(self):
        """Windowed (opt-in) alignment: hyp[1] AND hyp[2] both overlap
        ref[1], so they get concatenated under that GT window. No hyp_only
        is generated for segmentation differences."""
        ref = [
            TimedSegment(0.0, 5.0, "hello world"),
            TimedSegment(5.0, 10.0, "goal scored by Sterling"),
        ]
        hyp = [
            TimedSegment(0.0, 5.0, "hello world"),
            TimedSegment(5.0, 8.0, "hallu"),
            TimedSegment(8.0, 12.0, "goal scored by Sterling"),
        ]
        m = evaluate("test", ref, hyp, alignment_mode="windowed")
        assert m.aligned_pairs == 2
        assert m.hyp_only == 0  # both extras concat under ref[1] now
        assert 10 <= m.wer <= 25

    def test_evaluate_with_fallback_fills_dropped_hyp(self):
        """When hypothesis drops a GT segment, the fallback provides text."""
        ref = [TimedSegment(0, 5, "perfect match"), TimedSegment(5, 10, "second part")]
        hyp = [TimedSegment(0, 5, "perfect match")]  # missing segment 2
        fallback = [TimedSegment(0, 5, "perfect match"), TimedSegment(5, 10, "second partz")]
        m = evaluate("test", ref, hyp, fallback=fallback)
        # Because fallback has slightly different text ("partz"), we still
        # measure an error on the dropped segment — but a very small one.
        assert m.ref_only == 1
        assert m.wer > 0


class TestAlignByWindow:
    """Many-to-one alignment for per-segment debugging.

    GOAL human GT segments are 5-15s; raw Whisper segments are 2-5s. So
    one GT window typically spans 2-4 hyp segments. ``align_by_window``
    bundles all overlapping hyp segments under their GT window so the
    diff tool compares the full GT sentence to the concatenation of the
    raw fragments that cover it.
    """

    def test_one_gt_window_collects_multiple_hyp(self):
        ref = [TimedSegment(0.0, 12.0, "long ground truth sentence covering many fragments")]
        hyp = [
            TimedSegment(0.0, 4.0, "long ground truth"),
            TimedSegment(4.0, 8.0, "sentence covering"),
            TimedSegment(8.0, 12.0, "many fragments"),
        ]
        groups = align_by_window(ref, hyp)
        assert len(groups) == 1
        gt, bucket = groups[0]
        assert gt is ref[0]
        assert bucket == hyp  # all three hyp fragments collected

    def test_partial_overlap_below_threshold_excluded(self):
        ref = [TimedSegment(10.0, 15.0, "x")]
        hyp = [
            TimedSegment(9.95, 10.05, "edge"),       # 0.05s overlap < 0.25 threshold
            TimedSegment(10.0, 15.0, "real"),
        ]
        groups = align_by_window(ref, hyp, overlap_min_s=0.25)
        assert len(groups) == 2  # one for ref[0], one trailing (None, leftovers)
        gt, bucket = groups[0]
        assert gt is ref[0]
        assert bucket == [hyp[1]]
        leftover_gt, leftover_hyp = groups[1]
        assert leftover_gt is None
        assert leftover_hyp == [hyp[0]]

    def test_hyp_outside_any_window_is_leftover(self):
        ref = [TimedSegment(0.0, 5.0, "a")]
        hyp = [
            TimedSegment(0.0, 5.0, "a"),
            TimedSegment(20.0, 25.0, "hallucinated"),
        ]
        groups = align_by_window(ref, hyp)
        assert groups[0][0] is ref[0]
        assert groups[0][1] == [hyp[0]]
        assert groups[-1] == (None, [hyp[1]])

    def test_empty_hyp_yields_empty_buckets(self):
        ref = [TimedSegment(0.0, 5.0, "a"), TimedSegment(5.0, 10.0, "b")]
        groups = align_by_window(ref, [])
        assert len(groups) == 2
        assert groups[0] == (ref[0], [])
        assert groups[1] == (ref[1], [])

    def test_chelsea_style_3to1_grouping(self):
        """Reproduces the Chelsea half-1 pattern: GT 230 segs, raw 612 segs.

        Tests that a GT window like the [t=35.7-53.9s] case collects all 5
        raw segments inside it, not just the closest-by-midpoint one.
        """
        ref = [TimedSegment(35.7, 53.9, "long GT span")]
        hyp = [
            TimedSegment(35.7, 39.0, "as expected"),
            TimedSegment(39.0, 42.5, "Liverpool straight out"),
            TimedSegment(42.5, 46.0, "of the blocks"),
            TimedSegment(46.0, 50.0, "half cleared by Oscar"),
            TimedSegment(50.0, 53.9, "trying to get across"),
        ]
        groups = align_by_window(ref, hyp)
        assert len(groups) == 1
        assert len(groups[0][1]) == 5
