"""Regression tests for cleaned-output metadata in orchestrator._write_cleaned_output.

The bug being fixed: SOTA corrections from half 1 were leaking into the
half-2 output file (and vice versa) because the previous filter combined
``c.get('half') == half_num`` with an OR clause that fell back to a
segment_id-only lookup. Segment IDs collide across halves (both halves
use "0", "1", ...), so the OR always matched.

These tests pin the new half-only filter behaviour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.loader import MatchData, Segment  # noqa: E402
from pipeline.orchestrator import _write_cleaned_output  # noqa: E402


def _seg(sid: str, half: int, text: str = "x", t: float = 0.0) -> Segment:
    return Segment(
        segment_id=sid, start_time=t, end_time=t + 1.0,
        text=text, half=half,
    )


def _make_match(tmp_path: Path) -> MatchData:
    """Build a minimal on-disk match layout that satisfies _write_cleaned_output."""
    league = "england_epl"
    season = "2015-2016"
    match_name = "test-match"
    match_dir = tmp_path / "ds" / "caption-2023" / league / season / match_name
    (match_dir / "commentary_data").mkdir(parents=True)

    segments = [
        _seg("0", 1, "half-1 seg-0", t=0),
        _seg("1", 1, "half-1 seg-1", t=2),
        _seg("0", 2, "half-2 seg-0", t=2700),  # collides with half-1 seg id "0"
        _seg("1", 2, "half-2 seg-1", t=2702),  # collides with half-1 seg id "1"
    ]
    return MatchData(
        match_dir=match_dir, match_name=match_name,
        league=league, season=season, segments=segments,
    )


def test_sota_corrections_filtered_by_half_only(tmp_path, monkeypatch):
    """Corrections tagged half:1 must only appear in 1_asr_cleaned.json,
    even when their segment_id collides with a half-2 segment id."""
    from pipeline import orchestrator as orch

    cleaned_root = tmp_path / "cleaned"
    monkeypatch.setattr(orch, "CLEANED_OUTPUT_DIR", cleaned_root)

    match = _make_match(tmp_path)
    sota = [
        {"segment_id": "0", "half": 1, "original": "half-1 seg-0",
         "corrected": "fixed-1-0", "method": "llm_ger", "stage": "L"},
        {"segment_id": "0", "half": 2, "original": "half-2 seg-0",
         "corrected": "fixed-2-0", "method": "llm_ger", "stage": "L"},
        {"segment_id": "1", "half": 2, "original": "half-2 seg-1",
         "corrected": "fixed-2-1", "method": "llm_ger", "stage": "L"},
    ]
    _write_cleaned_output(
        match, match.segments, [], [], [],
        sota_corrections=sota, llm_telemetry={"accepted": 3},
    )

    h1 = json.load(open(
        cleaned_root / "caption-2023" / "england_epl" / "2015-2016"
        / "test-match" / "commentary_data" / "1_asr_cleaned.json",
        encoding="utf-8",
    ))
    h2 = json.load(open(
        cleaned_root / "caption-2023" / "england_epl" / "2015-2016"
        / "test-match" / "commentary_data" / "2_asr_cleaned.json",
        encoding="utf-8",
    ))

    h1_corr = h1["cleaning_metadata"]["sota_corrections"]
    h2_corr = h2["cleaning_metadata"]["sota_corrections"]

    assert all(c["half"] == 1 for c in h1_corr), (
        f"half-1 file leaked corrections from other halves: {h1_corr}"
    )
    assert all(c["half"] == 2 for c in h2_corr), (
        f"half-2 file leaked corrections from other halves: {h2_corr}"
    )
    assert {c["corrected"] for c in h1_corr} == {"fixed-1-0"}
    assert {c["corrected"] for c in h2_corr} == {"fixed-2-0", "fixed-2-1"}


def test_llm_telemetry_persisted_to_metadata(tmp_path, monkeypatch):
    from pipeline import orchestrator as orch
    cleaned_root = tmp_path / "cleaned"
    monkeypatch.setattr(orch, "CLEANED_OUTPUT_DIR", cleaned_root)

    match = _make_match(tmp_path)
    telem = {
        "total_segments": 4,
        "eligible_segments": 2,
        "accepted": 1,
        "rejected_editable_drift": 1,
    }
    _write_cleaned_output(
        match, match.segments, [], [], [],
        sota_corrections=[], llm_telemetry=telem,
    )
    h1 = json.load(open(
        cleaned_root / "caption-2023" / "england_epl" / "2015-2016"
        / "test-match" / "commentary_data" / "1_asr_cleaned.json",
        encoding="utf-8",
    ))
    assert h1["cleaning_metadata"]["llm_telemetry"] == telem
