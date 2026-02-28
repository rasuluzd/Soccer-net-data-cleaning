"""
Pipeline Orchestrator — runs all cleaning steps on each match.

This is the central coordinator that ties together all pipeline components
in the correct order. It processes each match independently, allowing
scaling to hundreds of matches across multiple seasons.

Processing order per match:
    1. Load ASR JSONs + Labels-caption.json
    2. Build gazetteer for this match
    3. Check learned dictionary for instant corrections
    4. Filter hallucinated/garbage segments
    5. De-duplicate segments
    6. Run NER to detect entities
    7. Fuzzy+phonetic correction against gazetteer
    8. Update learned dictionary
    9. Write cleaned JSON output
    10. Return cleaning statistics
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.config import CLEANED_OUTPUT_DIR
from pipeline.loader import MatchData, Segment, discover_matches
from pipeline.gazetteer import build_gazetteer
from pipeline.hallucination_filter import filter_segments
from pipeline.deduplicator import deduplicate_segments
from pipeline.ner_extractor import extract_entities
from pipeline.fuzzy_corrector import correct_segment_text, Correction, COMMON_WORDS_EXCLUDE
from pipeline.learned_dictionary import (
    lookup_learned,
    update_learned_dictionary,
)


@dataclass
class CleaningResult:
    """Results from cleaning a single match."""
    match_name: str
    original_segment_count: int
    hallucinations_removed: int
    duplicates_removed: int
    segments_after_cleaning: int
    entities_detected: int
    entities_corrected: int
    corrections: list[dict]
    removed_hallucinations: list[dict]
    removed_duplicates: list[dict]


def clean_match(
    match: MatchData,
    dry_run: bool = False,
) -> CleaningResult:
    """
    Run the full cleaning pipeline on a single match.

    Args:
        match: the MatchData object to clean
        dry_run: if True, don't write output files (just report what would change)

    Returns:
        CleaningResult with all statistics
    """
    print(f"\n{'='*70}")
    print(f"Processing: {match.match_name}")
    print(f"  League: {match.league} | Season: {match.season}")
    print(f"  Segments: {len(match.segments)} | Labels: {'Yes' if match.labels else 'No'}")
    print(f"{'='*70}")

    original_count = len(match.segments)

    # ── Step 1: Build gazetteer ──────────────────────────────────────
    gazetteer = build_gazetteer(match.labels)
    print(f"  Gazetteer: {len(gazetteer)} name entries")

    # ── Step 2: Filter hallucinations ────────────────────────────────
    valid_segments, removed_hallucinations = filter_segments(match.segments)
    print(f"  Hallucinations removed: {len(removed_hallucinations)}")

    # ── Step 3: De-duplicate ─────────────────────────────────────────
    deduped_segments, removed_duplicates = deduplicate_segments(valid_segments)
    print(f"  Duplicates removed: {len(removed_duplicates)}")

    # ── Step 4: NER + Correction ─────────────────────────────────────
    all_corrections: list[Correction] = []
    corrected_segments: list[Segment] = []
    total_entities = 0

    for seg in deduped_segments:
        # First check the learned dictionary for instant corrections
        text = seg.text

        # Detect entities
        entities = extract_entities(seg)
        total_entities += len(entities)

        # Try learned dictionary first for each entity
        learned_applied = False
        for entity in entities:
            entity_text = entity.text.strip()

            # Strip possessive before lookup
            trailing_possessive = ""
            if entity_text.endswith("'s") or entity_text.endswith("\u2019s"):
                trailing_possessive = entity_text[-2:]
                entity_text = entity_text[:-2]

            # Strip punctuation
            while entity_text and entity_text[-1] in ".,!?;:":
                entity_text = entity_text[:-1]

            if not entity_text:
                continue

            # Skip common English words — they should never be corrected
            if entity_text.lower() in COMMON_WORDS_EXCLUDE:
                continue
            # Skip very short entities
            if len(entity_text.strip()) <= 2:
                continue

            learned_correction = lookup_learned(entity_text)
            if learned_correction:
                # Apply single-word logic: if entity is 1 word,
                # don't replace with full name
                entity_word_count = len(entity_text.split())
                if entity_word_count == 1 and " " in learned_correction:
                    # Extract just the surname from the full name
                    corrected_name = learned_correction.split()[-1]
                else:
                    corrected_name = learned_correction

                # Re-append possessive if present
                replacement = corrected_name + trailing_possessive

                text = text.replace(entity.text, replacement)
                all_corrections.append(Correction(
                    original=entity.text,
                    corrected=corrected_name,
                    combined_score=100.0,
                    fuzzy_score=100.0,
                    phonetic_match=True,
                    context_match=False,
                    segment_id=seg.segment_id,
                    method="learned_dictionary",
                ))
                learned_applied = True

        # If learned dict didn't handle everything, do fuzzy matching
        if entities and not learned_applied:
            corrected_text, corrections = correct_segment_text(
                text=text,
                entities=entities,
                gazetteer=gazetteer,
                segment_id=seg.segment_id,
            )
            text = corrected_text
            all_corrections.extend(corrections)

        corrected_segments.append(Segment(
            segment_id=seg.segment_id,
            start_time=seg.start_time,
            end_time=seg.end_time,
            text=text,
            half=seg.half,
        ))

    print(f"  Entities detected: {total_entities}")
    print(f"  Entities corrected: {len(all_corrections)}")

    # ── Step 5: Update learned dictionary ────────────────────────────
    if all_corrections:
        update_learned_dictionary(all_corrections)

    # ── Step 6: Write output ─────────────────────────────────────────
    if not dry_run:
        _write_cleaned_output(match, corrected_segments, all_corrections,
                              removed_hallucinations, removed_duplicates)

    # ── Build result ─────────────────────────────────────────────────
    return CleaningResult(
        match_name=match.match_name,
        original_segment_count=original_count,
        hallucinations_removed=len(removed_hallucinations),
        duplicates_removed=len(removed_duplicates),
        segments_after_cleaning=len(corrected_segments),
        entities_detected=total_entities,
        entities_corrected=len(all_corrections),
        corrections=[
            {
                "segment_id": c.segment_id,
                "original": c.original,
                "corrected": c.corrected,
                "score": round(c.combined_score, 1),
                "method": c.method,
            }
            for c in all_corrections
        ],
        removed_hallucinations=removed_hallucinations,
        removed_duplicates=removed_duplicates,
    )


def _write_cleaned_output(
    match: MatchData,
    segments: list[Segment],
    corrections: list[Correction],
    removed_hallucinations: list[dict],
    removed_duplicates: list[dict],
) -> None:
    """Write cleaned JSON files mirroring the original directory structure."""
    # Create output directory mirroring the match structure
    relative_path = match.match_dir.relative_to(match.match_dir.parents[3])
    output_dir = CLEANED_OUTPUT_DIR / relative_path / "commentary_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split segments by half
    for half_num in [1, 2]:
        half_segments = [s for s in segments if s.half == half_num]
        if not half_segments:
            continue

        # Build the output JSON in the same format as the input
        output_data = {
            "segments": {
                s.segment_id: [s.start_time, s.end_time, s.text]
                for s in half_segments
            },
            "cleaning_metadata": {
                "original_segment_count": len([
                    s for s in match.segments if s.half == half_num
                ]),
                "cleaned_segment_count": len(half_segments),
                "corrections": [
                    {
                        "segment_id": c.segment_id,
                        "original": c.original,
                        "corrected": c.corrected,
                        "score": round(c.combined_score, 1),
                        "method": c.method,
                    }
                    for c in corrections
                    if any(s.segment_id == c.segment_id and s.half == half_num
                           for s in segments)
                ],
            },
        }

        output_path = output_dir / f"{half_num}_asr_cleaned.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"  Output written to: {output_dir}")


def run_pipeline(
    match_filter: str | None = None,
    dry_run: bool = False,
) -> list[CleaningResult]:
    """
    Run the full pipeline on all discovered matches.

    Args:
        match_filter: optional substring to filter matches by name
                     (e.g., "Manchester City" to process only that match)
        dry_run: if True, just preview changes without writing

    Returns:
        List of CleaningResult objects, one per match
    """
    print("Soccer ASR Data Cleaning Pipeline")
    print("=" * 70)
    print(f"Mode: {'DRY RUN (no files written)' if dry_run else 'LIVE'}")
    print()

    # Discover all matches
    matches = discover_matches()
    print(f"Discovered {len(matches)} match(es) with ASR data")

    # Apply filter if provided
    if match_filter:
        matches = [m for m in matches if match_filter.lower() in m.match_name.lower()]
        print(f"After filter '{match_filter}': {len(matches)} match(es)")

    if not matches:
        print("No matches to process!")
        return []

    # Process each match
    results = []
    for match in matches:
        result = clean_match(match, dry_run=dry_run)
        results.append(result)

    return results
