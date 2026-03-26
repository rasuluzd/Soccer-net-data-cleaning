"""
Hallucination & Garbage Filter — removes invalid ASR segments.

Whisper sometimes produces garbled output: non-Latin characters, wrong-language
text, single-word noise, or nonsensical fragments. This module detects and flags
these segments so they can be excluded from the cleaned output.
"""

import re
from typing import Optional

from pipeline.config import (
    HALLUCINATION_MIN_ALPHA_RATIO,
    MIN_SEGMENT_WORD_COUNT,
    LANGUAGE_FAMILIES,
)
from pipeline.loader import Segment

# Try to import langdetect; if not installed, skip language detection
try:
    from langdetect import detect as detect_language
    from langdetect.lang_detect_exception import LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


# Regex pattern matching non-Latin scripts (CJK, Arabic, Cyrillic, etc.)
# These should NOT appear in Latin-script soccer commentary
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
    Calculate the ratio of alphabetic characters to total content characters.

    A low ratio indicates the text contains a lot of numbers or symbols
    — a hallmark of Whisper hallucinations.

    Uses c.isalpha() (not c.isascii()) so accented Latin characters
    (å, ä, ö, é, ü, etc.) are correctly counted. Non-Latin scripts
    are already rejected by Rule 2 (has_non_latin_characters) which
    runs before this check.

    Args:
        text: the segment text

    Returns:
        Ratio between 0.0 and 1.0
    """
    if not text:
        return 0.0
    # Count Latin-script alphabetic characters (ASCII + accented Latin).
    # CJK/Arabic/Cyrillic are excluded by the Unicode range check (< U+0250
    # covers Basic Latin, Latin-1 Supplement, Latin Extended-A/B).
    # Non-Latin scripts are already caught by Rule 2 (has_non_latin_characters)
    # but this is defensive.
    alpha_count = sum(1 for c in text if c.isalpha() and c < '\u0250')
    # Don't count spaces/punctuation in the denominator — focus on "content" chars
    content_chars = sum(1 for c in text if not c.isspace())
    if content_chars == 0:
        return 0.0
    return alpha_count / content_chars


def has_non_latin_characters(text: str) -> bool:
    """Check if the text contains any non-Latin script characters."""
    return bool(NON_LATIN_PATTERN.search(text))


def detect_commentary_language(segments: list[Segment], sample_size: int = 20) -> str:
    """
    Detect the primary language of commentary from a sample of segments.

    Samples the longest segments (more text = more reliable detection),
    concatenates them, and runs langdetect.

    Args:
        segments: list of Segment objects from a match
        sample_size: max number of segments to sample

    Returns:
        ISO 639-1 language code (e.g., 'en', 'sv', 'de'). Defaults to 'en'.
    """
    if not HAS_LANGDETECT or not segments:
        return "en"

    # Pick segments with the most words for reliable detection
    candidates = sorted(segments, key=lambda s: len(s.text.split()), reverse=True)
    sample_texts = [s.text for s in candidates[:sample_size] if len(s.text.split()) >= 5]

    if not sample_texts:
        return "en"

    combined = " ".join(sample_texts)
    try:
        lang = detect_language(combined)
        # Map to supported language family key
        for family_key, family_set in LANGUAGE_FAMILIES.items():
            if lang in family_set:
                return family_key
        # Unknown language — return as-is (will use "default" multilingual models)
        return lang
    except LangDetectException:
        return "en"


def is_valid_commentary(text: str, expected_lang: str = "en") -> bool:
    """
    Check if text matches the expected commentary language family.

    Replaces the old is_likely_english() — now supports any language.
    When expected_lang="en", behavior is identical to the original.

    Args:
        text: the segment text
        expected_lang: the detected match language (e.g., 'en', 'sv', 'de')

    Returns:
        True if text matches the expected language family (or detection is uncertain).
    """
    if not HAS_LANGDETECT:
        return True

    # langdetect needs enough text to be reliable on sports commentary.
    if len(text.split()) < 8:
        return True

    try:
        lang = detect_language(text)
        accepted = LANGUAGE_FAMILIES.get(expected_lang, {expected_lang})
        return lang in accepted
    except LangDetectException:
        return True


def filter_segment(
    segment: Segment,
    expected_lang: str = "en",
) -> tuple[bool, Optional[str]]:
    """
    Check a single segment and determine if it should be kept or removed.

    Args:
        segment: the Segment to evaluate
        expected_lang: the detected match language

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
        return False, "non_latin_characters"

    # ── Rule 3: Low alpha ratio ──────────────────────────────────────
    alpha_ratio = compute_alpha_ratio(text)
    if alpha_ratio < HALLUCINATION_MIN_ALPHA_RATIO:
        return False, f"low_alpha_ratio ({alpha_ratio:.2f})"

    # ── Rule 4: Too few words ────────────────────────────────────────
    word_count = len(text.split())
    if word_count < MIN_SEGMENT_WORD_COUNT:
        return False, f"too_few_words ({word_count})"

    # ── Rule 5: Wrong language detected ──────────────────────────────
    if not is_valid_commentary(text, expected_lang):
        return False, "wrong_language_detected"

    # ── All checks passed ────────────────────────────────────────────
    return True, None


def filter_segments(
    segments: list[Segment],
    expected_lang: str = "en",
) -> tuple[list[Segment], list[dict]]:
    """
    Filter a list of segments, removing hallucinated/garbage entries.

    Args:
        segments: list of Segment objects to filter
        expected_lang: the detected match language

    Returns:
        Tuple of:
        - kept: list of valid Segment objects
        - removed: list of dicts with { segment, reason } for logging
    """
    kept = []
    removed = []

    for seg in segments:
        is_valid, reason = filter_segment(seg, expected_lang=expected_lang)
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
