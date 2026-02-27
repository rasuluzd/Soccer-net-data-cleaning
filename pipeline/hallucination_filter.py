"""
Hallucination & Garbage Filter — removes invalid ASR segments.

Whisper sometimes produces garbled output: non-Latin characters, wrong-language
text, single-word noise, or nonsensical fragments. This module detects and flags
these segments so they can be excluded from the cleaned output.
"""

import re
from typing import Optional

from pipeline.config import HALLUCINATION_MIN_ALPHA_RATIO, MIN_SEGMENT_WORD_COUNT
from pipeline.loader import Segment

# Try to import langdetect; if not installed, skip language detection
try:
    from langdetect import detect as detect_language
    from langdetect.lang_detect_exception import LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


# Regex pattern matching non-Latin scripts (CJK, Arabic, Cyrillic, etc.)
# These should NOT appear in English soccer commentary
NON_LATIN_PATTERN = re.compile(
    r"[\u0400-\u04FF"   # Cyrillic
    r"\u0600-\u06FF"    # Arabic
    r"\u3000-\u9FFF"    # CJK (Chinese, Japanese, Korean)
    r"\uAC00-\uD7AF"    # Korean Hangul
    r"\u3040-\u309F"    # Japanese Hiragana
    r"\u30A0-\u30FF]",  # Japanese Katakana
    re.UNICODE,
)


def compute_alpha_ratio(text: str) -> float:
    """
    Calculate the ratio of ASCII alphabetic characters to total characters.

    A low ratio indicates the text contains a lot of non-English characters,
    numbers, or symbols — a hallmark of Whisper hallucinations.

    Args:
        text: the segment text

    Returns:
        Ratio between 0.0 and 1.0
    """
    if not text:
        return 0.0
    alpha_count = sum(1 for c in text if c.isascii() and c.isalpha())
    # Don't count spaces/punctuation in the denominator — focus on "content" chars
    content_chars = sum(1 for c in text if not c.isspace())
    if content_chars == 0:
        return 0.0
    return alpha_count / content_chars


def has_non_latin_characters(text: str) -> bool:
    """Check if the text contains any non-Latin script characters."""
    return bool(NON_LATIN_PATTERN.search(text))


def is_likely_english(text: str) -> bool:
    """
    Use language detection to check if the text is likely English.

    Returns True if text is English or if detection is uncertain.
    Returns False only if we're confident it's NOT English.
    """
    if not HAS_LANGDETECT:
        return True  # can't check, assume OK

    # langdetect needs a LOT of text to be reliable on sports commentary.
    # Short phrases like "stunning goal" get misclassified.
    if len(text.split()) < 8:
        return True  # too short to detect reliably, assume OK

    try:
        lang = detect_language(text)
        # Accept English or closely-related languages (langdetect sometimes
        # misidentifies English sports commentary as Scottish Gaelic, etc.)
        return lang in ("en", "sco", "cy")
    except LangDetectException:
        return True  # detection failed, assume OK


def filter_segment(segment: Segment) -> tuple[bool, Optional[str]]:
    """
    Check a single segment and determine if it should be kept or removed.

    Args:
        segment: the Segment to evaluate

    Returns:
        Tuple of (is_valid, reason).
        - is_valid=True, reason=None → keep the segment
        - is_valid=False, reason="..." → remove the segment, reason explains why
    """
    text = segment.text.strip()

    # ── Rule 1: Empty or whitespace-only ─────────────────────────────
    if not text:
        return False, "empty_segment"

    # ── Rule 2: Contains non-Latin characters ────────────────────────
    if has_non_latin_characters(text):
        return False, f"non_latin_characters"

    # ── Rule 3: Low ASCII-alpha ratio ────────────────────────────────
    alpha_ratio = compute_alpha_ratio(text)
    if alpha_ratio < HALLUCINATION_MIN_ALPHA_RATIO:
        return False, f"low_alpha_ratio ({alpha_ratio:.2f})"

    # ── Rule 4: Too few words ────────────────────────────────────────
    word_count = len(text.split())
    if word_count < MIN_SEGMENT_WORD_COUNT:
        return False, f"too_few_words ({word_count})"

    # ── Rule 5: Non-English language detected ────────────────────────
    if not is_likely_english(text):
        return False, "non_english_detected"

    # ── All checks passed ────────────────────────────────────────────
    return True, None


def filter_segments(segments: list[Segment]) -> tuple[list[Segment], list[dict]]:
    """
    Filter a list of segments, removing hallucinated/garbage entries.

    Args:
        segments: list of Segment objects to filter

    Returns:
        Tuple of:
        - kept: list of valid Segment objects
        - removed: list of dicts with { segment, reason } for logging
    """
    kept = []
    removed = []

    for seg in segments:
        is_valid, reason = filter_segment(seg)
        if is_valid:
            kept.append(seg)
        else:
            removed.append({
                "segment_id": seg.segment_id,
                "half": seg.half,
                "start_time": seg.start_time,
                "text": seg.text[:80],  # truncate for readability
                "reason": reason,
            })

    return kept, removed
