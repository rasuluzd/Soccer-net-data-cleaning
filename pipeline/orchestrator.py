"""
Pipeline Orchestrator — runs all cleaning steps on each match.

Processing order per match (post-May-2026 architecture):
    1. Load ASR JSONs + Labels-caption.json
    2. Build gazetteer + entity-type map from labels
    3. Detect language (langdetect on full transcript sample)
    4. Stage 1: Hallucination filter
    5. Stage 1: Deduplicator
    6. Stage 2A: Domain normalizer (scores, times)
    7. NER + heuristic entity extraction
    8. Stage E: ValidatedEntityCorrector  ← TF-IDF retrieve + Qwen MCQ judge
    9. Step L: Confidence-gated GER (Qwen + MLM veto + drift guard)
    10. Step P: Punctuation + casing restoration
    11. Write cleaned JSON output (schema-2 enrichments + telemetry)
    12. Generate temporal chunks for ES indexing

The legacy heuristic cascade (Tier 2 fuzzy/phonetic/context, Tier 3 cosine
disambiguator, mT5/BERT span-infill, Ollama Mistral rewriter, MCQ
validator) was replaced by Stage E + Step L. See plans/check-all-files-
and-peppy-lemur.md for rationale.
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

from pipeline.config import (
    CLEANED_OUTPUT_DIR, MAX_WORKERS,
    ENTITY_CORRECTION_ENABLED,
    DOMAIN_NORMALIZATION_ENABLED,
)
from pipeline.loader import MatchData, Segment, discover_matches
from pipeline.gazetteer import build_gazetteer
from pipeline.hallucination_filter import filter_segments, detect_commentary_language
from pipeline.deduplicator import deduplicate_segments
from pipeline.ner_extractor import extract_entities, extract_entities_batch
from pipeline.fuzzy_corrector import (
    Correction,
    extract_entity_core,
    extract_and_rebuild_entity,
    passes_conservative_gates,
)
from pipeline.entity_corrector import correct_match as entity_correct_match
from pipeline.entity_corrector import get_last_telemetry as entity_last_telemetry
from pipeline.domain_normalizer import DomainNormalizer
from pipeline.llm_corrector import correct_match as llm_correct_match
from pipeline.llm_corrector import get_last_telemetry as llm_last_telemetry
from pipeline.punct_restorer import restore_punctuation_batch


def generate_match_id(league: str, season: str, match_name: str) -> str:
    """Composite match identifier (replaces deleted temporal_chunker)."""
    safe = match_name.replace(" ", "_").replace("/", "_")
    return f"{league}_{season}_{safe}"


def generate_segment_global_id(match_id: str, half: int, segment_id: str) -> str:
    return f"{match_id}__h{half}__s{segment_id}"


def _collapse_repeated_words(text: str) -> str:
    """Collapse Whisper-loop word repetitions (3+ consecutive identical) ONLY.

    Why 3+ and not 2: GT often has legitimate 2-repetitions ("well, well",
    "starry, starry night") that lose comma in Whisper output. Collapsing
    on 2 destroys those. Whisper LOOPING produces 3+ ("Zaha Zaha Zaha
    Zaha"), which we flatten to a single token.
    """
    words = text.split()
    if len(words) < 3:
        return text
    out: list[str] = []
    i = 0
    while i < len(words):
        run = 1
        while i + run < len(words) and words[i + run].lower() == words[i].lower():
            run += 1
        if run >= 3:
            out.append(words[i])  # collapse Whisper-loop to single token
        else:
            out.extend(words[i:i + run])  # keep run-of-1 or legitimate run-of-2
        i += run
    return " ".join(out)


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
    tier12_corrections: list = field(default_factory=list)  # Raw Correction objects for learned dict merge
    # Stage 2: general text correction tracking
    text_corrections: list[dict] = field(default_factory=list)
    # Stage 3.5: XLM-R detection flags (signals, not corrections — tracked
    # separately so they don't inflate the "Text corrections" report total).
    flagged_words_count: int = 0
    correction_breakdown: dict = field(default_factory=lambda: {
        "normalization": 0,
        "spell_check": 0,
        "grammar": 0,
        "entity": 0,
        "neural": 0,
        "llm": 0,
    })


def clean_match(
    match: MatchData,
    dry_run: bool = False,
    max_tier: int = 3,
    learned_dict: dict | None = None,
) -> CleaningResult:
    """
    Run the full cleaning pipeline on a single match.

    Args:
        match: the MatchData object to clean
        dry_run: if True, don't write output files (just report what would change)
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
    stage_timings: dict[str, float] = {}

    def _t(label: str, fn):
        t0 = time.perf_counter()
        result = fn()
        stage_timings[label] = round(time.perf_counter() - t0, 3)
        return result

    # ── Step 0: Detect commentary language ───────────────────────────
    detected_lang = _t("step0_detect_language",
                       lambda: detect_commentary_language(match.segments))
    print(f"  Detected language: {detected_lang} ({stage_timings['step0_detect_language']:.2f}s)")

    # ── Step 1: Build gazetteer ──────────────────────────────────────
    gazetteer, entity_types = _t("step1_build_gazetteer",
                                 lambda: build_gazetteer(match.labels))
    print(f"  Gazetteer: {len(gazetteer)} name entries  Entity types: "
          f"{len(entity_types)} typed ({stage_timings['step1_build_gazetteer']:.2f}s)")


    # ── Step 2: Filter hallucinations ────────────────────────────────
    valid_segments, removed_hallucinations = _t(
        "step2_hallucination_filter",
        lambda: filter_segments(match.segments, expected_lang=detected_lang),
    )
    print(f"  Hallucinations removed: {len(removed_hallucinations)} "
          f"({stage_timings['step2_hallucination_filter']:.2f}s)")

    # ── Step 3: De-duplicate ─────────────────────────────────────────
    def _dedup_with_collapse():
        out, removed = deduplicate_segments(valid_segments)
        for i, seg in enumerate(out):
            collapsed = _collapse_repeated_words(seg.text)
            if collapsed != seg.text:
                # Word-level confidence no longer aligns after deleting
                # repeated tokens, but all other schema-2 metadata remains.
                out[i] = replace(seg, text=collapsed, words=None)
        return out, removed
    deduped_segments, removed_duplicates = _t("step3_deduplicate", _dedup_with_collapse)
    print(f"  Duplicates removed: {len(removed_duplicates)} "
          f"({stage_timings['step3_deduplicate']:.2f}s)")

    # entity_corrector builds its own match-wide token frequency for the
    # frequency heuristic; nothing to precompute here.

    # ── Stage 2: General text correction (spell-check, grammar, normalization)
    # This seam is where new correction modules plug in.
    # Each module receives segments and returns corrected segments + corrections.
    stage2_corrections: list[dict] = []

    if DOMAIN_NORMALIZATION_ENABLED:
        def _normalize():
            n = DomainNormalizer(detected_lang)
            return n.normalize_batch(deduped_segments)
        deduped_segments, norm_corrections = _t("step2A_domain_normalize", _normalize)
        for c in norm_corrections:
            c["match"] = match.match_name
        stage2_corrections.extend(norm_corrections)
        if norm_corrections:
            print(f"  Stage 2A normalization: {len(norm_corrections)} corrections "
                  f"({stage_timings['step2A_domain_normalize']:.2f}s)")

    # NOTE: Stage 2B (pyspellchecker) + Stage 2C (LanguageTool grammar) were
    # removed in the May 2026 architectural refactor. Both fired ~5 correc-
    # tions per match (mostly punctuation) and were superseded by Step P
    # (oliverguhr punctuation restorer) plus Step L (Qwen confidence-gated
    # GER which catches grammar drift via the editable-mask constraint).

    if stage2_corrections:
        print(f"  Stage 2 text corrections: {len(stage2_corrections)}")

    # ── Stage E: Validated Entity Correction ─────────────────────────
    # Replaces the legacy Tier 2 (fuzzy/phonetic/context) + Tier 3 (cosine
    # disambiguator) cascade. Architecture:
    #   1. TF-IDF char-bigram retrieval over gazetteer (no Metaphone)
    #   2. Auto-accept on cosine ≥0.90 + clear winner
    #   3. Auto-reject on cosine <0.40
    #   4. MCQ judge (Qwen GGUF) for the uncertain 0.40-0.89 band
    #   5. Two-layer cache: per-match decisions + cross-match validated
    #      cache (3-match consensus required) — see entity_corrector.py
    all_corrections: list[Correction] = []
    corrected_segments: list[Segment] = []
    total_entities = 0
    entity_telemetry: dict = {}

    if not ENTITY_CORRECTION_ENABLED:
        corrected_segments = list(deduped_segments)
        saved_entities_for_step5 = [[] for _ in deduped_segments]
        print("  Entity correction: DISABLED (config.ENTITY_CORRECTION_ENABLED=False)")
    else:
        saved_entities_for_step5 = []

        print("  Extracting entities in batch mode...")
        segment_entities_map = _t(
            "stepNER_extract_entities",
            lambda: extract_entities_batch(
                deduped_segments, language=detected_lang, gazetteer=gazetteer,
            ),
        )
        for seg in deduped_segments:
            saved_entities_for_step5.append(
                segment_entities_map.get((seg.half, seg.segment_id), [])
            )
            total_entities += len(segment_entities_map.get((seg.half, seg.segment_id), []))

        # Run the new Validated Entity Corrector (TF-IDF + MCQ judge)
        match_id = generate_match_id(match.league, match.season, match.match_name)
        corrected_segments, entity_corrections_dicts = _t(
            "stepE_entity_corrector",
            lambda: entity_correct_match(
                segments=deduped_segments,
                gazetteer=gazetteer,
                entity_types=entity_types,
                segment_entities_map=segment_entities_map,
                match_id=match_id,
                match_name=match.match_name,
                language=detected_lang,
            ),
        )
        entity_telemetry = entity_last_telemetry()

        # Convert dicts → Correction records for downstream metadata writers.
        for d in entity_corrections_dicts:
            d["match"] = match.match_name
            all_corrections.append(Correction(
                original=d["original"],
                corrected=d["corrected"],
                combined_score=d["score"],
                fuzzy_score=0.0,
                phonetic_match=False,
                context_match=d.get("method") in ("mcq_judge", "validated_cache"),
                segment_id=d["segment_id"],
                half=d["half"],
                method=d.get("method", "entity_corrector"),
            ))

        print(f"  Entities detected: {total_entities} "
              f"(NER {stage_timings.get('stepNER_extract_entities', 0):.2f}s)")
        print(f"  Stage E entity corrections applied: {len(all_corrections)} "
              f"({stage_timings.get('stepE_entity_corrector', 0):.2f}s)")

    # NOTE: Stage 3.5 (XLM-R error detection) + Stage 3.7 (LLM MCQ validator) +
    # Stage 4 (mT5 / BERT masked-LM) + Stage 5 (Ollama generative rewriter)
    # were all removed in the May 2026 cleanup. They were either gated off,
    # produced 0 net corrections, or have been superseded by the SOTA Step L
    # (pipeline/llm_corrector.py — confidence-gated GER with Qwen + MLM veto).
    # See plans/check-all-files-and-peppy-lemur.md.
    flagged_words_count = 0  # kept in CleaningResult for backward-compat reports

    # ── Step 6: Collect Tier 1/2 corrections for learned dict ────────
    # Only Tier 1/2 corrections — Tier 3 context_similarity
    # corrections are less certain and could poison future runs.
    # Also exclude corrections that target team/venue names to
    # prevent cross-match contamination (e.g. City ↔ United).
    tier12_corrections = [
        c for c in all_corrections
        if not c.method.startswith("context_similarity")
    ]
    # NOTE: We do NOT update the learned dictionary here.
    # Corrections are returned to the caller for batch update
    # after all matches complete (enables safe parallel execution).

    # ── SOTA Refactor — Steps R, L, P (see plan + module docstrings) ─
    # These run AFTER the legacy correction stages so that legacy can be
    # re-enabled via config flags for ablation comparison without code
    # changes. With defaults (legacy stages = False), legacy is a no-op
    # and `corrected_segments` here is the dedup+text-clean output.
    sota_corrections: list[dict] = []
    # NOTE: Step R (n-best entity rerank) was removed in May 2026 — it was
    # a no-op since the Whisper output schema-1 doesn't carry n-best
    # alternatives. The architectural slot is preserved in entity_corrector
    # which now handles all entity-correction work.

    # Step L: Confidence-gated GER (Qwen2.5-0.5B + MLM veto).
    corrected_segments, llm_corrections = _t(
        "stepL_llm_ger",
        lambda: llm_correct_match(
            corrected_segments, gazetteer, entity_types,
            match_name=match.match_name, language=detected_lang,
        ),
    )
    llm_telemetry = llm_last_telemetry()
    if llm_corrections:
        for c in llm_corrections:
            c["match"] = match.match_name
        sota_corrections.extend(llm_corrections)
        print(f"  Step L LLM GER: {len(llm_corrections)} corrections "
              f"({stage_timings['stepL_llm_ger']:.2f}s)")

    # Step P: Punctuation + casing restoration (search-friendly output).
    corrected_segments, punct_corrections = _t(
        "stepP_punct_restore",
        lambda: restore_punctuation_batch(corrected_segments, language=detected_lang),
    )
    if punct_corrections:
        for c in punct_corrections:
            c["match"] = match.match_name
        sota_corrections.extend(punct_corrections)
        print(f"  Step P punct restoration: {len(punct_corrections)} segments restyled "
              f"({stage_timings['stepP_punct_restore']:.2f}s)")

    # ── Stage timings summary ────────────────────────────────────────
    stage_timings["total_pipeline"] = round(sum(stage_timings.values()), 3)
    print(f"\n  ── Per-stage timings (s) ──")
    for k, v in stage_timings.items():
        print(f"    {k:32s} {v:8.2f}")

    # ── Step 7: Write output ─────────────────────────────────────────
    if not dry_run:
        _write_cleaned_output(match, corrected_segments, all_corrections,
                              removed_hallucinations, removed_duplicates,
                              sota_corrections=sota_corrections,
                              llm_telemetry=llm_telemetry,
                              stage_timings=stage_timings,
                              entity_telemetry=entity_telemetry)

    # ── Build result ─────────────────────────────────────────────────
    # Deduplicate corrections for the report
    unique_corrections = []
    seen_corrections = set()
    for c in all_corrections:
        key = (c.segment_id, c.original, c.corrected)
        if key not in seen_corrections:
            seen_corrections.add(key)
            unique_corrections.append({
                "segment_id": c.segment_id,
                "original": c.original,
                "corrected": c.corrected,
                "score": round(c.combined_score, 1),
                "method": c.method,
                "stage": "3",  # Entity correction stage
            })

    # Build correction breakdown by stage
    breakdown = {
        "normalization": 0, "spell_check": 0, "grammar": 0,
        "entity": len(unique_corrections), "neural": 0, "llm": 0,
    }
    for sc in stage2_corrections:
        method = sc.get("method", "")
        if method in breakdown:
            breakdown[method] += 1

    return CleaningResult(
        match_name=match.match_name,
        original_segment_count=original_count,
        hallucinations_removed=len(removed_hallucinations),
        duplicates_removed=len(removed_duplicates),
        segments_after_cleaning=len(corrected_segments),
        entities_detected=total_entities,
        entities_corrected=len(unique_corrections),
        corrections=unique_corrections,
        removed_hallucinations=removed_hallucinations,
        removed_duplicates=removed_duplicates,
        tier12_corrections=tier12_corrections,
        text_corrections=stage2_corrections,
        flagged_words_count=flagged_words_count,
        correction_breakdown=breakdown,
    )


def _write_cleaned_output(
    match: MatchData,
    segments: list[Segment],
    corrections: list[Correction],
    removed_hallucinations: list[dict],
    removed_duplicates: list[dict],
    sota_corrections: list[dict] | None = None,
    llm_telemetry: dict | None = None,
    stage_timings: dict | None = None,
    entity_telemetry: dict | None = None,
) -> None:
    """Write cleaned JSON files mirroring the original directory structure.

    Output is schema-1 compatible (start_time, end_time, text per segment)
    matching what the ForzaSearch frontend's ingest.ts actually consumes.
    Schema-2 fields (per-word probs, nbest, no_speech_prob) are persisted
    in the same file but not extracted by the frontend; they remain
    available for any future consumer that wants confidence-aware search.
    """
    # Create output directory mirroring the match structure
    relative_path = match.match_dir.relative_to(match.match_dir.parents[3])
    output_dir = CLEANED_OUTPUT_DIR / relative_path / "commentary_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate composite match ID for globally unique segment IDs
    match_id = generate_match_id(match.league, match.season, match.match_name)

    # Assign global IDs to segments
    for seg in segments:
        seg.global_id = generate_segment_global_id(
            match_id, seg.half, seg.segment_id,
        )

    # Split segments by half
    for half_num in [1, 2]:
        half_segments = [s for s in segments if s.half == half_num]
        if not half_segments:
            continue

        def _seg_payload(s: Segment) -> dict:
            # Schema-1 compatible payload. Frontend's ingest.ts reads only
            # start, end, text. Schema-2 enrichments (words.prob, nbest)
            # were stripped out — no consumer was using them.
            return {
                "global_id": s.global_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "text": s.text,
            }

        # Half-aware slice of SOTA corrections (R/L/P stages).
        # Filter on the half tag ONLY — segment_id is not unique across halves
        # (both halves use "0", "1", "2", ...), so any segment_id-based fallback
        # silently leaks cross-half corrections into the wrong file.
        sota_for_half = [
            c for c in (sota_corrections or [])
            if c.get("half") == half_num
        ]

        # Build the output JSON with global IDs
        output_data = {
            "segments": {s.segment_id: _seg_payload(s) for s in half_segments},
            "cleaning_metadata": {
                "match_id": match_id,
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
                        "stage": "3",
                    }
                    for c in corrections
                    if c.half == half_num
                ],
                "sota_corrections": sota_for_half,
                "llm_telemetry": llm_telemetry or {},
                "stage_timings": stage_timings or {},
                "entity_telemetry": entity_telemetry or {},
                "removed_hallucinations": [
                    r for r in (removed_hallucinations or [])
                    if r.get("half") == half_num
                ],
                "removed_duplicates": [
                    r for r in (removed_duplicates or [])
                    if r.get("half") == half_num
                ],
            },
        }

        from pipeline.config import ASR_INPUT_VARIANT
        output_path = output_dir / f"{half_num}_asr{ASR_INPUT_VARIANT}_cleaned.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

    # NOTE: temporal chunk output (es_chunks.json) was removed in May 2026.
    # The frontend's ingest.ts builds its own segment windows from kamp.json
    # and never consumed our pre-built chunks.
    print(f"  Output written to: {output_dir}")


def _clean_match_wrapper(args: tuple) -> CleaningResult:
    """Wrapper for ProcessPoolExecutor.map() — unpacks the args tuple."""
    match, dry_run, max_tier, learned_dict = args
    return clean_match(
        match,
        dry_run=dry_run,
        max_tier=max_tier,
        learned_dict=learned_dict,
    )


def run_pipeline(
    match_filter: str | None = None,
    dry_run: bool = False,
    max_tier: int = 3,
    workers: int | None = None,
) -> list[CleaningResult]:
    """
    Run the full pipeline on all discovered matches.

    Args:
        match_filter: optional substring to filter matches by name
                     (e.g., "Manchester City" to process only that match)
        dry_run: if True, just preview changes without writing
        max_tier: max correction tier (2=fuzzy+phonetic, 3=+context AI)
        workers: number of parallel worker processes.
                 None/0 = auto-detect (use config MAX_WORKERS or CPU count).
                 1 = sequential (no parallelism).

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

    # Resolve worker count.
    # With en_core_web_sm (~12MB), model loading is fast (~2-5s per process).
    # The sentence-transformer (~80MB) is the main startup cost.
    # For very small batches, sequential avoids process spawn overhead.
    PARALLEL_THRESHOLD = 3  # Matches needed before parallelism helps
    if workers is None or workers == 0:
        if MAX_WORKERS > 0:
            effective_workers = MAX_WORKERS
        elif len(matches) > PARALLEL_THRESHOLD:
            cpu_count = os.cpu_count() or 1
            effective_workers = max(2, cpu_count // 2)
        else:
            effective_workers = 1  # Sequential for small batches
    else:
        effective_workers = workers
    # Don't spawn more workers than matches
    effective_workers = min(effective_workers, len(matches))

    # Legacy learned_dictionary removed — entity_corrector now owns the
    # validated cross-match cache (per-match decision cache + safe-cache
    # requiring 3-match consensus).
    learned_dict: dict = {}

    start_time = time.time()

    # ── Process matches ──────────────────────────────────────────────
    if effective_workers <= 1:
        # Sequential mode
        print(f"Processing {len(matches)} match(es) sequentially...")
        results = []
        for match in matches:
            result = clean_match(
                match,
                dry_run=dry_run,
                max_tier=max_tier,
                learned_dict=learned_dict,
            )
            results.append(result)
    else:
        # Parallel mode
        print(f"Processing {len(matches)} match(es) in parallel ({effective_workers} workers)...")
        args_list = [
            (match, dry_run, max_tier, learned_dict)
            for match in matches
        ]
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            # No initializer — models load lazily via singletons.
            # This is faster for small batches because model loading
            # in one worker overlaps with computation in others.
        ) as pool:
            results = list(pool.map(_clean_match_wrapper, args_list))

    elapsed = time.time() - start_time
    print(f"\nAll matches processed in {elapsed:.1f}s")

    # ── Batch-update learned dictionary with corrections from all matches ──
    all_tier12 = []
    # Collect entity_types and gazetteers for validation
    # (re-build for each match's corrections is not feasible here;
    #  we pass None to skip per-match validation in batch mode)
    for result in results:
        all_tier12.extend(result.tier12_corrections)

    # The validated cross-match cache (entity_corrector owns it) updates
    # itself in-process per match; nothing to batch-write here.

    return results
