"""
Pipeline Orchestrator — runs all cleaning steps on each match.

This is the central coordinator that ties together all pipeline components
in the correct order. It processes each match independently, allowing
scaling to hundreds of matches across multiple seasons.

Processing order per match:
    1. Load ASR JSONs + Labels-caption.json
    2. Build gazetteer for this match (+ Wikidata enrichment if enabled)
    3. Check learned dictionary for instant corrections
    4. Filter hallucinated/garbage segments
    5. De-duplicate segments
    6. Run NER to detect entities
    7. Fuzzy+phonetic correction against gazetteer (Tier 2)
    8. Contextual disambiguation for unresolved entities (Tier 3)
    9. Update learned dictionary
    10. Write cleaned JSON output
    11. Return cleaning statistics
"""

import json
from dataclasses import dataclass

from pipeline.config import CLEANED_OUTPUT_DIR
from pipeline.loader import MatchData, Segment, discover_matches
from pipeline.gazetteer import build_gazetteer, get_team_words
from pipeline.hallucination_filter import filter_segments
from pipeline.deduplicator import deduplicate_segments
from pipeline.ner_extractor import extract_entities
from pipeline.fuzzy_corrector import (
    correct_segment_text,
    Correction,
    COMMON_WORDS_EXCLUDE,
    extract_entity_core,
    extract_and_rebuild_entity,
)
from pipeline.learned_dictionary import (
    lookup_learned,
    update_learned_dictionary,
    load_learned_dictionary,
)
from pipeline.wikidata_enrichment import enrich_gazetteer
from pipeline.context_disambiguator import (
    batch_disambiguate,
)


def _collapse_repeated_words(text: str) -> str:
    """Collapse immediately repeated words: 'Zaha Zaha dribbles' → 'Zaha dribbles'."""
    words = text.split()
    if len(words) < 2:
        return text
    result = [words[0]]
    for w in words[1:]:
        if w.lower() != result[-1].lower():
            result.append(w)
    return " ".join(result)


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
    enrich_wikidata: bool = False,
    max_tier: int = 3,
) -> CleaningResult:
    """
    Run the full cleaning pipeline on a single match.

    Args:
        match: the MatchData object to clean
        dry_run: if True, don't write output files (just report what would change)
        enrich_wikidata: if True, expand gazetteer with Wikidata EPL players
        max_tier: maximum tier to run (2=fuzzy only, 3=fuzzy+context)

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
    gazetteer, entity_types = build_gazetteer(match.labels)
    team_words = get_team_words(entity_types, gazetteer)
    print(f"  Gazetteer: {len(gazetteer)} name entries")
    print(f"  Entity types: {len(entity_types)} typed | Team words: {team_words}")

    # ── Step 1b: Wikidata enrichment (optional) ───────────────────────
    if enrich_wikidata:
        # Extract season years from the match path
        season = match.season  # e.g. "2014-2015"
        try:
            parts = season.split("-")
            year_start = int(parts[0])
            year_end = int(parts[1]) if len(parts) > 1 else year_start + 1
        except (ValueError, IndexError):
            year_start, year_end = 2014, 2016
        gazetteer = enrich_gazetteer(gazetteer, year_start, year_end)

    # ── Step 2: Filter hallucinations ────────────────────────────────
    valid_segments, removed_hallucinations = filter_segments(match.segments)
    print(f"  Hallucinations removed: {len(removed_hallucinations)}")

    # ── Step 3: De-duplicate ─────────────────────────────────────────
    deduped_segments, removed_duplicates = deduplicate_segments(valid_segments)

    for i, seg in enumerate(deduped_segments):
        collapsed = _collapse_repeated_words(seg.text)
        if collapsed != seg.text:
            deduped_segments[i] = Segment(
                segment_id=seg.segment_id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=collapsed,
                half=seg.half,
            )
    print(f"  Duplicates removed: {len(removed_duplicates)}")

    # ── Step 4: NER + Correction ─────────────────────────────────────
    all_corrections: list[Correction] = []
    uncertain_corrections: list[Correction] = []  # Tier 2 corrections needing Tier 3 validation
    corrected_segments: list[Segment] = []
    total_entities = 0

    saved_entities_for_step5 = []

    learned_dict=load_learned_dictionary()  # Load once to avoid repeated disk I/O

    for seg in deduped_segments:
        # First check the learned dictionary for instant corrections
        text = seg.text

        # Detect entities
        entities = extract_entities(seg)
        total_entities += len(entities)

        saved_entities_for_step5.append(entities)

        # Try learned dictionary first for each entity
        learned_applied = False

        sorted_entities = sorted(entities, key=lambda e: e.start_char, reverse=True)

        for entity in sorted_entities:
            entity_text = extract_entity_core(entity.text)

            if not entity_text:
                continue

            # Skip common English words — they should never be corrected
            if entity_text.lower() in COMMON_WORDS_EXCLUDE:
                continue
            # Skip very short entities
            if len(entity_text.strip()) <= 2:
                continue

            learned_correction = lookup_learned(entity_text, learned_dict)
            if learned_correction:
                # Apply single-word logic: if entity is 1 word,
                # don't replace with full name
                entity_word_count = len(entity_text.split())
                if entity_word_count == 1 and " " in learned_correction:
                    # Extract just the surname from the full name
                    corrected_name = learned_correction.split()[-1]
                else:
                    corrected_name = learned_correction

                replacement = extract_and_rebuild_entity(entity.text, corrected_name)

                before = text[:entity.start_char]
                after = text[entity.end_char:]
                text = before + replacement + after

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
                entity_types=entity_types,
                team_words=team_words,
            )
            # Split corrections by confidence band:
            # - "accepted" (≥75): apply immediately
            # - "uncertain" (48-74): hold for Tier 3 context validation
            accepted = [c for c in corrections if c.confidence_band == "accepted"]
            uncertain = [c for c in corrections if c.confidence_band == "uncertain"]

            if accepted:
                # Apply only high-confidence corrections to the text
                # Re-run correction with only accepted entities
                text = corrected_text  # Use the corrected text (has all corrections)
                all_corrections.extend(accepted)
            else:
                # No accepted corrections — keep original text
                pass

            # Store uncertain corrections for potential Tier 3 routing
            uncertain_corrections.extend(uncertain)

        corrected_segments.append(Segment(
            segment_id=seg.segment_id,
            start_time=seg.start_time,
            end_time=seg.end_time,
            text=text,
            half=seg.half,
        ))

    print(f"  Entities detected (Tier 2): {total_entities}")
    print(f"  Entities corrected (Tier 2 accepted): {len(all_corrections)}")
    print(f"  Uncertain corrections (routed to Tier 3): {len(uncertain_corrections)}")

    # ── Step 5: Tier 3 — Contextual disambiguation ────────────────────
    tier3_corrections = 0
    if max_tier >= 3:
        # Collect unresolved entities (detected by NER but not corrected)
        # We re-run NER on corrected segments to find remaining entities
        unresolved = []
        corrected_set = {
            (c.segment_id, c.original) for c in all_corrections
        }
        lower_gazetteer = {k.lower() for k in gazetteer}
        team_names = [
            canon for canon, etype in entity_types.items()
            if etype == "team"
        ]

        for seg_idx, (seg, entities) in enumerate(zip(corrected_segments, saved_entities_for_step5)):
            for entity in entities:
                entity_text = extract_entity_core(entity.text)
                if not entity_text:
                    continue
                # ── Strict filtering for Tier 3 ──
                # Only PERSON entities should be disambiguated
                if entity.label != "PERSON":
                    continue
                # Skip common words on the exclusion list
                if entity_text.lower() in COMMON_WORDS_EXCLUDE:
                    continue
                # Must start with uppercase (names always do)
                if not entity_text[0].isupper():
                    continue
                # Skip possessives
                # Skip if already in gazetteer (no correction needed)
                # Strip trailing punctuation first — spaCy often
                # includes sentence-ending punctuation in the entity span
                # e.g. "Joel Ward." should match "Joel Ward" in gazetteer
                entity_cleaned = entity_text
                if entity_cleaned in gazetteer or entity_cleaned.lower() in lower_gazetteer:
                    continue
                # Skip if Tier 2 already corrected this
                if (seg.segment_id, entity_text) in corrected_set:
                    continue
                # ── Entity-type-aware filtering ──
                # Skip entities whose text matches a team/venue word.
                # Fixes: "Palace"→player, "Wickham Palace"→player
                entity_lower_words = {w.lower() for w in entity_cleaned.split()}
                if entity_lower_words & team_words:
                    continue
                # Skip entities that contain a full team name as substring
                # Fixes: "West Ham United" being sent for correction
                entity_lower = entity_cleaned.lower()
                if any(tn.lower() in entity_lower for tn in team_names):
                    continue
                # Skip multi-word entities where each word is a known name
                # (e.g. "Ivanovic Zouma" — two adjacent correct names)
                entity_words = entity_cleaned.split()
                if len(entity_words) >= 2:
                    words_in_gaz = sum(
                        1 for w in entity_words
                        if w.lower() in lower_gazetteer
                    )
                    if words_in_gaz >= 2:
                        continue

                # This entity is a plausible name that Tier 2 couldn't
                # resolve — send to Tier 3 for contextual disambiguation
                unresolved.append({
                    "text": entity_text,
                    "segment_id": seg.segment_id,
                    "segment_idx": seg_idx,
                })

        # Also route uncertain Tier 2 corrections to Tier 3.
        # These scored 48-74 and need context validation.
        for uc in uncertain_corrections:
            # Find the segment index for this correction
            seg_idx_for_uc = None
            for idx, seg in enumerate(corrected_segments):
                if seg.segment_id == uc.segment_id:
                    seg_idx_for_uc = idx
                    break
            if seg_idx_for_uc is not None:
                unresolved.append({
                    "text": uc.original,
                    "segment_id": uc.segment_id,
                    "segment_idx": seg_idx_for_uc,
                    "proposed_correction": uc.corrected,
                    "proposed_score": uc.combined_score,
                })

        if unresolved:
            print(f"  Unresolved entities for Tier 3: {len(unresolved)}")
            results_t3 = batch_disambiguate(
                unresolved_entities=unresolved,
                all_segments=corrected_segments,
                gazetteer=gazetteer,
                labels=match.labels,
                entity_types=entity_types,
            )

            # Apply Tier 3 corrections
            for r in results_t3:
                # Find and update the segment
                for i, seg in enumerate(corrected_segments):
                    if seg.segment_id == r.segment_id:
                        new_text = seg.text.replace(r.entity_text, r.corrected)
                        corrected_segments[i] = Segment(
                            segment_id=seg.segment_id,
                            start_time=seg.start_time,
                            end_time=seg.end_time,
                            text=new_text,
                            half=seg.half,
                        )
                        all_corrections.append(Correction(
                            original=r.entity_text,
                            corrected=r.corrected,
                            combined_score=round(r.similarity * 100, 1),
                            fuzzy_score=0.0,
                            phonetic_match=False,
                            context_match=True,
                            segment_id=r.segment_id,
                            method=f"context_similarity({r.similarity:.2f})",
                        ))
                        tier3_corrections += 1
                        break

            print(f"  Tier 3 corrections: {tier3_corrections}")
        else:
            print("  Tier 3: no unresolved entities")

    # ── Step 6: Update learned dictionary ────────────────────────────
    # Only save Tier 1/2 corrections — Tier 3 context_similarity
    # corrections are less certain and could poison future runs.
    # Also exclude corrections that target team/venue names to
    # prevent cross-match contamination (e.g. City ↔ United).
    tier12_corrections = [
        c for c in all_corrections
        if not c.method.startswith("context_similarity")
    ]
    if tier12_corrections:
        update_learned_dictionary(tier12_corrections, entity_types=entity_types, gazetteer=gazetteer)

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
    enrich_wikidata: bool = False,
    max_tier: int = 3,
) -> list[CleaningResult]:
    """
    Run the full pipeline on all discovered matches.

    Args:
        match_filter: optional substring to filter matches by name
                     (e.g., "Manchester City" to process only that match)
        dry_run: if True, just preview changes without writing
        enrich_wikidata: if True, expand gazetteer with Wikidata EPL data
        max_tier: max correction tier (2=fuzzy+phonetic, 3=+context AI)

    Returns:
        List of CleaningResult objects, one per match
    """
    print("Soccer ASR Data Cleaning Pipeline")
    print("=" * 70)
    mode_parts = []
    if dry_run:
        mode_parts.append("DRY RUN (no files written)")
    else:
        mode_parts.append("LIVE")
    mode_parts.append(f"Tier {max_tier}")
    if enrich_wikidata:
        mode_parts.append("+ Wikidata enrichment")
    print(f"Mode: {' | '.join(mode_parts)}")
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
        result = clean_match(
            match,
            dry_run=dry_run,
            enrich_wikidata=enrich_wikidata,
            max_tier=max_tier,
        )
        results.append(result)

    return results
