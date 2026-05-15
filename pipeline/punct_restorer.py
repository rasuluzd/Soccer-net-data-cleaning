"""Step P: insert punctuation + casing using oliverguhr/fullstop-punctuation-multilang-large.

Conservative — only inserts where missing, never deletes existing
punctuation or casing. Lazy singleton load. No-op when transformers is missing."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Optional

from pipeline.config import (
    PUNCT_MODEL,
    PUNCT_PRESERVE_EXISTING,
    PUNCT_RESTORATION_ENABLED,
)
from pipeline.loader import Segment


# The classifier emits one of: 0 (none), . , ? - :
_PUNCT_TOKENS = {".", ",", "?", "-", ":"}


@dataclass
class PunctuationCorrection:
    segment_id: str
    half: int
    before: str
    after: str
    method: str = "punct_restoration"


class PunctuationRestorer:
    """Lazy-loaded wrapper around the fullstop punctuation model.
    Returns unchanged segments + empty correction list when unavailable."""

    _SINGLETON: "Optional[PunctuationRestorer]" = None

    def __init__(self, language: str = "en", model_name: str = PUNCT_MODEL):
        self.language = language
        self.model_name = model_name
        self._pipeline = None
        self._init_error: Optional[str] = None
        self._tried_init = False

    # ── Lazy init ────────────────────────────────────────────────────

    @classmethod
    def get(cls, language: str = "en") -> "PunctuationRestorer":
        if cls._SINGLETON is None:
            cls._SINGLETON = cls(language=language)
        return cls._SINGLETON

    def _ensure_loaded(self) -> bool:
        if self._tried_init:
            return self._pipeline is not None
        self._tried_init = True
        try:
            from transformers import pipeline
            print(
                f"  [punct] loading {self.model_name} (first load downloads ~1.1 GB) ...",
                file=sys.stderr,
            )
            self._pipeline = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="none",
            )
            return True
        except ImportError as e:
            self._init_error = f"transformers not installed: {e}"
        except Exception as e:
            self._init_error = f"failed to load {self.model_name}: {e}"
        print(
            f"  [punct] DISABLED — {self._init_error}; segments pass through unchanged",
            file=sys.stderr,
        )
        return False

    @property
    def is_available(self) -> bool:
        return self._ensure_loaded()

    # ── Restoration ──────────────────────────────────────────────────

    def restore_text(self, text: str) -> str:
        """Insert punctuation + casing in one string. Only inserts where
        missing; never overwrites existing punctuation or casing."""
        if not self.is_available:
            return text
        text = text.strip()
        if not text:
            return text
        try:
            preds = self._pipeline(text)
        except Exception:
            return text

        # Walk subword predictions back to whitespace tokens so we can
        # insert punctuation between words, not inside subwords.
        words = text.split()
        if not words:
            return text

        word_punct: list[str] = [""] * len(words)
        word_should_capitalize: list[bool] = [False] * len(words)
        if word_should_capitalize:
            word_should_capitalize[0] = True
        # Last subword's prediction wins for each whitespace word.
        char_pos = 0
        word_idx = 0
        word_starts = []
        word_ends = []
        for w in words:
            start = text.find(w, char_pos)
            if start < 0:
                start = char_pos
            word_starts.append(start)
            word_ends.append(start + len(w))
            char_pos = start + len(w)
        for pred in preds:
            label = pred.get("entity", "0")
            if label == "0" or label not in _PUNCT_TOKENS:
                continue
            tok_start = pred.get("start")
            if tok_start is None:
                continue
            for i, e in enumerate(word_ends):
                if tok_start < e:
                    word_punct[i] = label
                    break

        out_parts: list[str] = []
        capitalize_next = True
        for i, w in enumerate(words):
            piece = w
            if capitalize_next and not PUNCT_PRESERVE_EXISTING:
                piece = piece[:1].upper() + piece[1:]
            elif capitalize_next and not (piece[:1].isupper()):
                # Only capitalise if it isn't already cased differently.
                piece = piece[:1].upper() + piece[1:]
            out_parts.append(piece)
            punct = word_punct[i]
            # Don't insert if the word already ends in punctuation.
            if punct and not piece.endswith(tuple(_PUNCT_TOKENS) + ("!", ";")):
                out_parts.append(punct)
                capitalize_next = punct in {".", "?"}
            else:
                capitalize_next = piece.endswith((".", "?", "!"))

        # Single-space join, then collapse "word ." -> "word.".
        rejoined = " ".join(out_parts)
        rejoined = re.sub(r"\s+([.,?!:;])", r"\1", rejoined)
        return rejoined

    # ── Batch API ────────────────────────────────────────────────────

    def restore_batch(
        self, segments: list[Segment]
    ) -> tuple[list[Segment], list[dict]]:
        """Per-segment restore. No-op when the model isn't loaded."""
        if not self.is_available:
            return segments, []
        out: list[Segment] = []
        corrections: list[dict] = []
        for seg in segments:
            new_text = self.restore_text(seg.text)
            if new_text != seg.text:
                corrections.append({
                    "segment_id": seg.segment_id,
                    "half": seg.half,
                    "original": seg.text,
                    "corrected": new_text,
                    "score": 100.0,
                    "method": "punct_restoration",
                    "stage": "P",
                })
                out.append(Segment(
                    segment_id=seg.segment_id,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    text=new_text,
                    half=seg.half,
                    global_id=seg.global_id,
                    words=seg.words,
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                    nbest=seg.nbest,
                    speaker_id=seg.speaker_id,
                ))
            else:
                out.append(seg)
        return out, corrections


def restore_punctuation_batch(
    segments: list[Segment],
    language: str = "en",
) -> tuple[list[Segment], list[dict]]:
    """Convenience wrapper. Returns segments unchanged when Step P is off."""
    if not PUNCT_RESTORATION_ENABLED:
        return segments, []
    return PunctuationRestorer.get(language).restore_batch(segments)
