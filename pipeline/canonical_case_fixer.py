"""
pipeline/canonical_case_fixer.py — lightweight regex-based casing
restoration for canonical entity names.

Replaces Step P (oliverguhr full punctuation transformer, 8 min/match)
with a deterministic regex pass that only fixes the casing of
known-canonical names from the gazetteer. Does NOT add punctuation,
does NOT case general English words — only entity names.

Why this exists: Step P spent 8 minutes per match restoring full
punctuation + casing for all 1500 segments, but the entity-F1 metric
(case-sensitive) only cares about the ~50 canonical names. A
lookup-table fix on the canonicals gives most of the F1 gain at
~50ms cost per match (rather than 8 min).

Usage:
    fixer = CanonicalCaseFixer(gazetteer)
    fixed_text = fixer.restore(lowercase_text)

Trade-offs vs Step P:
- WIN: 10 000× faster (50 ms vs 8 min)
- WIN: zero ML dependency
- LOSS: doesn't add commas, periods, sentence boundaries
- LOSS: doesn't case proper-noun-like English words outside the gazetteer
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class CanonicalCaseFixer:
    gazetteer: dict[str, str]  # variant_lower → canonical (with proper case)

    def __post_init__(self):
        # Build a single regex that matches any canonical (case-insensitive)
        # and a lookup from lowered-form to the canonical form.
        canonicals = sorted({v for v in self.gazetteer.values() if v},
                            key=len, reverse=True)
        self._canon_lookup = {c.lower(): c for c in canonicals}
        # Word-boundary regex matching any canonical ignore-case
        if canonicals:
            patt = "|".join(re.escape(c) for c in canonicals)
            self._regex = re.compile(rf"\b({patt})\b", re.IGNORECASE)
        else:
            self._regex = None

    def restore(self, text: str) -> str:
        """Replace any canonical occurrence (case-insensitive) with its
        canonical-cased form. Leaves all other text untouched."""
        if not self._regex:
            return text
        def _sub(m: re.Match) -> str:
            return self._canon_lookup.get(m.group(0).lower(), m.group(0))
        return self._regex.sub(_sub, text)

    def restore_batch(self, segments: list) -> tuple[list, list[dict]]:
        """Apply restore() to each segment's text. Mirrors the API of
        pipeline/punct_restorer.py for drop-in replacement."""
        from dataclasses import replace
        out = []
        corrections: list[dict] = []
        for seg in segments:
            new_text = self.restore(seg.text)
            if new_text != seg.text:
                corrections.append({
                    "segment_id": seg.segment_id,
                    "half": seg.half,
                    "original": seg.text,
                    "corrected": new_text,
                    "method": "canonical_case_fix",
                    "stage": "P_lite",
                })
                out.append(replace(seg, text=new_text))
            else:
                out.append(seg)
        return out, corrections
