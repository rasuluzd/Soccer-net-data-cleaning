"""Tests for pipeline.whisper_runner — initial_prompt and hotwords builders.

These cover the prompt builders + verify the hotwords param is plumbed
through to ``faster_whisper.WhisperModel.transcribe``. The transcribe
function itself is mocked end-to-end so no actual ASR runs in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.whisper_runner import (  # noqa: E402
    _ordered_lineup_names,
    build_hotwords_string,
    build_initial_prompt,
    transcribe,
)


def _write_labels(tmp_path: Path) -> Path:
    """Minimal valid Labels-caption.json with two players + a coach."""
    payload = {
        "teams": ["Chelsea", "Liverpool"],
        "lineup": {
            "home": {
                "players": [
                    {"short_name": "Costa", "long_name": "Diego Costa"},
                    {"short_name": "Hazard", "long_name": "Eden Hazard"},
                ],
                "coach": [{"long_name": "Antonio Conte"}],
            },
            "away": {
                "players": [
                    {"short_name": "Sturridge", "long_name": "Daniel Sturridge"},
                ],
                "coach": [],
            },
        },
    }
    p = tmp_path / "Labels-caption.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


class TestOrderedLineupNames:
    def test_missing_file_returns_empty(self, tmp_path):
        assert _ordered_lineup_names(tmp_path / "nope.json") == []

    def test_orders_teams_then_long_then_short(self, tmp_path):
        names = _ordered_lineup_names(_write_labels(tmp_path))
        # Teams come first
        assert names[0] == "Chelsea"
        assert names[1] == "Liverpool"
        # Long names before short names (so short names survive truncation at the end)
        assert names.index("Diego Costa") < names.index("Costa")
        assert names.index("Daniel Sturridge") < names.index("Sturridge")
        # Coach long_name appears among long_names
        assert "Antonio Conte" in names

    def test_dedupes_repeats(self, tmp_path):
        # Build labels where home long_name == away short_name to check dedup
        p = tmp_path / "Labels-caption.json"
        p.write_text(json.dumps({
            "teams": ["A", "A"],  # duplicate team
            "lineup": {
                "home": {"players": [{"short_name": "X", "long_name": "X"}], "coach": []},
                "away": {"players": [], "coach": []},
            },
        }), encoding="utf-8")
        names = _ordered_lineup_names(p)
        assert names.count("A") == 1
        assert names.count("X") == 1


class TestBuildInitialPrompt:
    def test_includes_swedish_header(self, tmp_path):
        s = build_initial_prompt(_write_labels(tmp_path))
        assert s.startswith("Fotbollsmatch. Namn: ")
        assert s.endswith(".")
        assert "Costa" in s and "Sturridge" in s

    def test_missing_labels_returns_empty(self, tmp_path):
        assert build_initial_prompt(tmp_path / "nope.json") == ""

    def test_max_names_truncates(self, tmp_path):
        s = build_initial_prompt(_write_labels(tmp_path), max_names=2)
        # Only 2 names included → "Fotbollsmatch. Namn: Chelsea, Liverpool."
        assert s.count(",") == 1


class TestBuildHotwordsString:
    def test_no_header_no_period(self, tmp_path):
        s = build_hotwords_string(_write_labels(tmp_path))
        # Plain comma-separated name list — no "Fotbollsmatch", no trailing "."
        assert not s.startswith("Fotbollsmatch")
        assert not s.endswith(".")
        assert "Chelsea" in s and "Costa" in s

    def test_missing_labels_returns_empty(self, tmp_path):
        assert build_hotwords_string(tmp_path / "nope.json") == ""

    def test_uses_same_ordering_as_initial_prompt(self, tmp_path):
        labels = _write_labels(tmp_path)
        # Both builders read from _ordered_lineup_names so the order matches
        names_in_prompt = build_initial_prompt(labels).removeprefix("Fotbollsmatch. Namn: ").rstrip(".")
        names_in_hot = build_hotwords_string(labels)
        assert names_in_prompt == names_in_hot


class TestTranscribePlumbsHotwords:
    """Verify transcribe() forwards both prompt and hotwords to faster-whisper."""

    def test_hotwords_passed_through(self, tmp_path):
        captured: dict = {}

        class _FakeInfo:
            language = "en"
            language_probability = 0.99

        def _fake_transcribe(self_model, audio_str, **kwargs):
            captured["kwargs"] = kwargs
            return iter([]), _FakeInfo()

        with patch("faster_whisper.WhisperModel") as MockModel:
            instance = MagicMock()
            instance.transcribe.side_effect = lambda *a, **k: _fake_transcribe(instance, *a, **k)
            MockModel.return_value = instance

            audio = tmp_path / "fake.mp3"
            audio.write_bytes(b"")
            out = tmp_path / "out.json"
            transcribe(
                audio_path=audio, output_path=out,
                initial_prompt="Football match.",
                hotwords="Costa, Hazard, Sturridge",
                language="en", model_name="tiny", device="cpu",
            )

        kw = captured["kwargs"]
        assert kw["initial_prompt"] == "Football match."
        assert kw["hotwords"] == "Costa, Hazard, Sturridge"

    def test_empty_hotwords_passes_none(self, tmp_path):
        captured: dict = {}

        class _FakeInfo:
            language = "en"
            language_probability = 0.99

        def _fake(self_model, *a, **k):
            captured["kwargs"] = k
            return iter([]), _FakeInfo()

        with patch("faster_whisper.WhisperModel") as MockModel:
            inst = MagicMock()
            inst.transcribe.side_effect = lambda *a, **k: _fake(inst, *a, **k)
            MockModel.return_value = inst
            audio = tmp_path / "fake.mp3"
            audio.write_bytes(b"")
            out = tmp_path / "out.json"
            transcribe(
                audio_path=audio, output_path=out,
                initial_prompt="", hotwords="",
                language="en", model_name="tiny", device="cpu",
            )

        kw = captured["kwargs"]
        # Empty strings convert to None so faster-whisper sees "no biasing"
        assert kw["initial_prompt"] is None
        assert kw["hotwords"] is None
