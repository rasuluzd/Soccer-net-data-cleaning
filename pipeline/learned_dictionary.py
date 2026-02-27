"""
Self-Learning Correction Dictionary — accumulates corrections across matches.

When the pipeline makes corrections, this module records them. Over time,
corrections that appear consistently (across multiple matches/segments)
get higher confidence and are applied instantly without needing fuzzy matching.

This is the key to scalability: as you process more matches, the pipeline
gets faster and more accurate because it "remembers" past corrections.
"""

import json
from pathlib import Path
from typing import Optional

from pipeline.config import LEARNED_CORRECTIONS_PATH
from pipeline.fuzzy_corrector import Correction


def load_learned_dictionary() -> dict[str, dict]:
    """
    Load the learned corrections from disk.

    Returns:
        Dict mapping lowercase misspelling -> {
            "correct": canonical name,
            "confidence": float (0-1),
            "seen_count": int,
            "fuzzy_score_avg": float,
        }
    """
    if not LEARNED_CORRECTIONS_PATH.exists():
        return {}

    with open(LEARNED_CORRECTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_learned_dictionary(dictionary: dict[str, dict]) -> None:
    """Save the learned corrections dictionary to disk."""
    LEARNED_CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEARNED_CORRECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=2, ensure_ascii=False)


def update_learned_dictionary(corrections: list[Correction]) -> dict[str, dict]:
    """
    Update the learned dictionary with new corrections from the current run.

    Each correction increments the seen_count and updates the running average
    of fuzzy scores. Confidence increases with more sightings.

    Args:
        corrections: list of Correction objects from the current pipeline run

    Returns:
        The updated dictionary (also saved to disk)
    """
    dictionary = load_learned_dictionary()

    for corr in corrections:
        key = corr.original.lower()
        if key in dictionary:
            # Update existing entry
            entry = dictionary[key]
            entry["seen_count"] += 1
            # Running average of fuzzy scores
            n = entry["seen_count"]
            entry["fuzzy_score_avg"] = (
                (entry["fuzzy_score_avg"] * (n - 1) + corr.combined_score) / n
            )
            # Confidence grows with more sightings (asymptotically approaches 1.0)
            entry["confidence"] = min(0.99, 1.0 - (1.0 / (entry["seen_count"] + 1)))
        else:
            # New entry
            dictionary[key] = {
                "correct": corr.corrected,
                "confidence": 0.5,    # initial confidence
                "seen_count": 1,
                "fuzzy_score_avg": corr.combined_score,
            }

    save_learned_dictionary(dictionary)
    return dictionary


def lookup_learned(entity_text: str) -> Optional[str]:
    """
    Check if a misspelling has been seen before and has high enough confidence.

    High-confidence entries (seen 2+ times) bypass fuzzy matching entirely,
    providing instant correction.

    Args:
        entity_text: the entity text to look up

    Returns:
        The canonical name if found with sufficient confidence, else None.
    """
    dictionary = load_learned_dictionary()
    key = entity_text.lower()

    if key in dictionary:
        entry = dictionary[key]
        # Require at least 2 sightings and 0.6 confidence for instant correction
        if entry["seen_count"] >= 2 and entry["confidence"] >= 0.6:
            return entry["correct"]

    return None
