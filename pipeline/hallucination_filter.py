"""Drop hallucinated/garbage segments: empty text, non-Latin scripts,
all-symbols garbage, wrong language, repeated single-name loops."""

import re
from typing import Optional

from pipeline.config import (
    HALLUCINATION_MIN_ALPHA_RATIO,
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


# Non-Latin script chars (CJK / Arabic / Cyrillic / Korean / Japanese).
# These don't appear in any of our supported commentary languages.
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
    """Ratio of Latin-script alpha chars to non-space chars (0.0..1.0).
    Low ratio = lots of numbers/symbols, typically a hallucination."""
    if not text:
        return 0.0
    # < U+0250 covers Basic Latin + Latin-1 + Latin Extended A/B (incl. å/ä/ö/é/ü).
    alpha_count = sum(1 for c in text if c.isalpha() and c < '\u0250')
    content_chars = sum(1 for c in text if not c.isspace())
    if content_chars == 0:
        return 0.0
    return alpha_count / content_chars


def has_non_latin_characters(text: str) -> bool:
    return bool(NON_LATIN_PATTERN.search(text))


def detect_commentary_language(segments: list[Segment], sample_size: int = 20) -> str:
    """Run langdetect on the longest segments and return an ISO 639-1 code.
    Falls back to 'en' if langdetect isn't installed or can't decide."""
    if not HAS_LANGDETECT or not segments:
        return "en"

    candidates = sorted(segments, key=lambda s: len(s.text.split()), reverse=True)
    sample_texts = [s.text for s in candidates[:sample_size] if len(s.text.split()) >= 5]

    if not sample_texts:
        return "en"

    combined = " ".join(sample_texts)
    try:
        lang = detect_language(combined)
        for family_key, family_set in LANGUAGE_FAMILIES.items():
            if lang in family_set:
                return family_key
        return lang
    except LangDetectException:
        return "en"


def is_valid_commentary(text: str, expected_lang: str = "en") -> bool:
    """True if text could plausibly be in expected_lang's family."""
    if not HAS_LANGDETECT:
        return True

    # langdetect mis-labels short football phrases (e.g. "get a good tackling,
    # get a good passing" -> Afrikaans). Only run on long-enough text.
    if len(text.split()) < 15:
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
    """Returns (keep?, reason). reason is None when the segment is kept."""
    text = segment.text.strip()

    if not text:
        return False, "empty_segment"

    if has_non_latin_characters(text):
        return False, "non_latin_characters"

    alpha_ratio = compute_alpha_ratio(text)
    if alpha_ratio < HALLUCINATION_MIN_ALPHA_RATIO:
        return False, f"low_alpha_ratio ({alpha_ratio:.2f})"

    if not is_valid_commentary(text, expected_lang):
        return False, "wrong_language_detected"

    return True, None


def find_repeated_name_hallucinations(
    segments: list[Segment],
    min_cluster_size: int = 3,
    cluster_window_s: float = 60.0,
) -> set[tuple[int, str]]:
    """Catch Whisper getting stuck on a name. Flags >=3 identical single-word
    segments within cluster_window_s. Other occurrences of that name
    elsewhere in the match stay untouched."""
    from collections import defaultdict

    by_token: dict[str, list[Segment]] = defaultdict(list)
    for seg in segments:
        stripped = seg.text.strip().rstrip(".,!?;:").strip()
        words = stripped.split()
        if len(words) != 1:
            continue
        w = words[0]
        if not w[:1].isupper() or w.isupper():
            continue
        if not any(c.isalpha() for c in w):
            continue
        by_token[w.lower()].append(seg)

    # Key by (half, segment_id) — segment_id alone isn't unique across halves.
    bad_ids: set[tuple[int, str]] = set()
    for token, occurrences in by_token.items():
        if len(occurrences) < min_cluster_size:
            continue
        occs = sorted(occurrences, key=lambda s: s.start_time)
        i = 0
        while i < len(occs):
            j = i
            # Extend cluster while next is within the window of the first.
            while j + 1 < len(occs) and (
                occs[j + 1].start_time - occs[i].start_time <= cluster_window_s
            ):
                j += 1
            cluster = occs[i : j + 1]
            if len(cluster) >= min_cluster_size:
                for s in cluster:
                    bad_ids.add((s.half, s.segment_id))
            i = j + 1
    return bad_ids


def filter_segments(
    segments: list[Segment],
    expected_lang: str = "en",
) -> tuple[list[Segment], list[dict]]:
    """Returns (kept, removed). removed entries carry a reason string."""
    repeated_name_ids = find_repeated_name_hallucinations(segments)

    kept = []
    removed = []

    for seg in segments:
        if (seg.half, seg.segment_id) in repeated_name_ids:
            removed.append({
                "segment_id": seg.segment_id,
                "half": seg.half,
                "start_time": seg.start_time,
                "text": seg.text[:80],
                "reason": "hallucination_repeated_name_segment",
            })
            continue

        is_valid, reason = filter_segment(seg, expected_lang=expected_lang)
        if is_valid:
            kept.append(seg)
        else:
            removed.append({
                "segment_id": seg.segment_id,
                "half": seg.half,
                "start_time": seg.start_time,
                "text": seg.text[:80],
                "reason": reason,
            })

    return kept, removed
