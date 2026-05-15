"""Football-specific text cleanup: disfluencies, compound merges, whitespace.
Pure regex, no deps. Patterns are per-language."""

import re
from dataclasses import replace

from pipeline.loader import Segment


# Word-boundary fixes for known football compounds Whisper splits.
FOOTBALL_COMPOUNDS = {
    "en": {
        # NOTE: do NOT add "half time"->"halftime" or "line up"->"lineup".
        # GT uses the two-word forms; merging them hurts WER.
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

# Filler words. Same set works across languages.
DISFLUENCY_PATTERN = re.compile(
    r"\b(uh|um|eh|er|ah|hm|hmm|uhm|erm)\b",
    re.IGNORECASE,
)

REPEATED_PUNCT = re.compile(r"([.!?,;:])\1+")
MULTI_SPACE = re.compile(r"\s{2,}")


class DomainNormalizer:
    def __init__(self, language: str = "en"):
        self.language = language
        self.compounds = FOOTBALL_COMPOUNDS.get(language, {})

    def normalize_segment(self, text: str) -> tuple[str, list[dict]]:
        corrections = []
        original = text

        # Remove disfluencies (uh, um, eh).
        cleaned = DISFLUENCY_PATTERN.sub("", text)
        if cleaned != text:
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

        # Merge football compounds.
        for wrong, right in self.compounds.items():
            if wrong == right:
                continue
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

        # Collapse repeated punctuation and extra whitespace.
        text = REPEATED_PUNCT.sub(r"\1", text)
        text = MULTI_SPACE.sub(" ", text).strip()

        return text, corrections

    def normalize_batch(
        self, segments: list[Segment],
    ) -> tuple[list[Segment], list[dict]]:
        corrected = []
        all_corrections = []

        for seg in segments:
            new_text, corrections = self.normalize_segment(seg.text)
            for c in corrections:
                c["segment_id"] = seg.segment_id
            all_corrections.extend(corrections)

            # Drop word-level confs only when text changed (alignment broken).
            if new_text == seg.text:
                corrected.append(replace(seg))
            else:
                corrected.append(replace(seg, text=new_text, words=None))

        return corrected, all_corrections
