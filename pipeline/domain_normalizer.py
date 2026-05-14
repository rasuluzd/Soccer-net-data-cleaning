"""
Domain Normalizer — football-specific text normalization for ASR output.

Stage 2A of the pipeline. Handles unambiguous ASR artifacts:
- Disfluency removal (uh, um, eh — filler words from speech)
- Football compound merging (off side → offside)
- Extra whitespace collapse
- Repeated punctuation cleanup

Language-aware: patterns are loaded per detected language.
No external dependencies — pure regex.
"""

import re
from dataclasses import replace

from pipeline.loader import Segment


# ─── Per-language football compound corrections ──────────────────────
# These fix Whisper word-boundary errors where known compounds
# are split into separate words. Only unambiguous patterns.
FOOTBALL_COMPOUNDS = {
    "en": {
        # Removed: "half time"→"halftime" and "line up"→"lineup". Empirical
        # WER analysis on Chelsea-Liverpool V3 vs GOAL human GT showed both
        # rules HURT WER — the GT consistently uses the two-word forms
        # ("at half time", "lineup at the back" with "line up" as verb).
        # See .debug/find_harmful_corrections.py.
        "off side": "offside",
        "on side": "onside",
        "goal keeper": "goalkeeper",
        "goal keeping": "goalkeeping",
        "full time": "fulltime",
        "kick off": "kick-off",
        "throw in": "throw-in",
        "break away": "breakaway",
        "counter attack": "counterattack",
        "over time": "overtime",
        "play maker": "playmaker",
        "mid field": "midfield",
        "mid fielder": "midfielder",
        "back heel": "backheel",
        "cross bar": "crossbar",
        "goal line": "goal line",       # already correct — kept for completeness
        "penalty area": "penalty area",  # already correct
    },
    "sv": {
        "av sida": "avside",
        "halv tid": "halvtid",
        "mål vakt": "målvakt",
        "mitt fält": "mittfält",
        "fri spark": "frispark",
        "hörn spark": "hörnspark",
        "straff spark": "straffspark",
    },
    "de": {
        "ab seits": "Abseits",
        "halb zeit": "Halbzeit",
        "tor wart": "Torwart",
        "frei stoß": "Freistoß",
        "eck ball": "Eckball",
        "mittel feld": "Mittelfeld",
        "nach spiel zeit": "Nachspielzeit",
    },
}

# ─── Disfluency patterns (language-agnostic filler words) ────────────
# Whisper sometimes transcribes speech disfluencies. These are
# universal across languages and never carry meaning.
DISFLUENCY_PATTERN = re.compile(
    r"\b(uh|um|eh|er|ah|hm|hmm|uhm|erm)\b",
    re.IGNORECASE,
)

# ─── Repeated punctuation cleanup ────────────────────────────────────
REPEATED_PUNCT = re.compile(r"([.!?,;:])\1+")
MULTI_SPACE = re.compile(r"\s{2,}")


class DomainNormalizer:
    """Football-specific text normalization for ASR segments."""

    def __init__(self, language: str = "en"):
        self.language = language
        self.compounds = FOOTBALL_COMPOUNDS.get(language, {})

    def normalize_segment(self, text: str) -> tuple[str, list[dict]]:
        """
        Normalize a single segment's text.

        Returns:
            Tuple of (corrected_text, list of correction dicts)
        """
        corrections = []
        original = text

        # 1. Remove disfluencies (uh, um, eh)
        cleaned = DISFLUENCY_PATTERN.sub("", text)
        if cleaned != text:
            # Track each removed disfluency
            for match in DISFLUENCY_PATTERN.finditer(text):
                corrections.append({
                    "original": match.group(),
                    "corrected": "",
                    "method": "normalization",
                    "stage": "2A",
                    "position": match.start(),
                    "confidence": 1.0,
                })
            text = cleaned

        # 2. Merge football compounds (off side → offside)
        for wrong, right in self.compounds.items():
            if wrong == right:
                continue  # skip identity mappings
            pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                corrections.append({
                    "original": match.group(),
                    "corrected": right,
                    "method": "normalization",
                    "stage": "2A",
                    "position": match.start(),
                    "confidence": 1.0,
                })
            text = pattern.sub(right, text)

        # 3. Clean repeated punctuation (... → ., !! → !)
        text = REPEATED_PUNCT.sub(r"\1", text)

        # 4. Collapse multiple spaces
        text = MULTI_SPACE.sub(" ", text).strip()

        return text, corrections

    def normalize_batch(
        self, segments: list[Segment],
    ) -> tuple[list[Segment], list[dict]]:
        """
        Normalize a batch of segments.

        Returns:
            Tuple of (corrected_segments, all_corrections)
        """
        corrected = []
        all_corrections = []

        for seg in segments:
            new_text, corrections = self.normalize_segment(seg.text)
            for c in corrections:
                c["segment_id"] = seg.segment_id
            all_corrections.extend(corrections)

            # Preserve schema-2 metadata when the text is unchanged. If this
            # stage removes/merges tokens, the per-word confidence alignment is
            # no longer reliable, so drop only the word-level list for that
            # segment and keep the segment-level metadata.
            if new_text == seg.text:
                corrected.append(replace(seg))
            else:
                corrected.append(replace(seg, text=new_text, words=None))

        return corrected, all_corrections
