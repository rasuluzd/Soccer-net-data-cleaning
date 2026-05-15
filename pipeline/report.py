"""Build a human-readable summary report from CleaningResult objects."""

import sys

from pipeline.orchestrator import CleaningResult


def generate_report(results: list[CleaningResult]) -> str:
    """Format a multi-match report. Returns the rendered string."""
    lines = []
    lines.append("=" * 70)
    lines.append("  SOCCER ASR DATA CLEANING — REPORT")
    lines.append("=" * 70)
    lines.append("")

    # ── Overall Summary ──────────────────────────────────────────────
    total_original = sum(r.original_segment_count for r in results)
    total_after = sum(r.segments_after_cleaning for r in results)
    total_hallucinations = sum(r.hallucinations_removed for r in results)
    total_duplicates = sum(r.duplicates_removed for r in results)
    total_entities = sum(r.entities_detected for r in results)
    total_corrections = sum(r.entities_corrected for r in results)
    total_text_corrections = sum(len(r.text_corrections) for r in results)
    # XLM-R flags are signals not corrections — kept separate so the report
    # total isn't inflated by detections that never became edits.
    total_flagged_words = sum(
        getattr(r, "flagged_words_count", 0) for r in results
    )

    total_breakdown = {
        "normalization": 0, "spell_check": 0, "grammar": 0,
        "entity": 0, "neural": 0, "llm": 0,
    }
    for r in results:
        for key in total_breakdown:
            total_breakdown[key] += r.correction_breakdown.get(key, 0)

    lines.append("OVERALL SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Matches processed:     {len(results)}")
    lines.append(f"  Total segments (raw):  {total_original}")
    lines.append(f"  Hallucinations removed:{total_hallucinations}")
    lines.append(f"  Duplicates removed:    {total_duplicates}")
    lines.append(f"  Segments retained:     {total_after}")
    lines.append(f"  Entities detected:     {total_entities}")
    lines.append(f"  Entities corrected:    {total_corrections}")
    lines.append(f"  Text corrections:      {total_text_corrections}")
    if total_flagged_words > 0:
        lines.append(f"  XLM-R flagged words:   {total_flagged_words} (detection signals, not corrections)")
    lines.append(f"  Retention rate:        {total_after/total_original*100:.1f}%" if total_original else "")
    lines.append("")

    # ── Correction Breakdown by Stage ────────────────────────────────
    active_stages = {k: v for k, v in total_breakdown.items() if v > 0}
    if active_stages:
        lines.append("CORRECTION BREAKDOWN BY STAGE")
        lines.append("-" * 40)
        # Stage labels are kept for legacy report-format compatibility.
        stage_labels = {
            "normalization": "Stage 2A (Domain Normalization)",
            "spell_check":   "Stage 2B (Spell-Check)",
            "grammar":       "Stage 2C (Grammar)",
            "entity":        "Stage 3  (Entity Names)",
            "neural":        "Stage 4  (Neural Infill)",
            "llm":           "Stage 5  (LLM Quality Pass)",
        }
        for key, count in total_breakdown.items():
            label = stage_labels.get(key, key)
            lines.append(f"  {label:<40} {count:>5}")
        total_all = sum(total_breakdown.values())
        lines.append(f"  {'TOTAL':<40} {total_all:>5}")
        lines.append("")

    # ── Per-Match Table ──────────────────────────────────────────────
    lines.append("PER-MATCH BREAKDOWN")
    lines.append("-" * 90)
    lines.append(f"  {'Match':<50} {'Raw':>5} {'Kept':>5} {'Hallu':>5} {'Dupe':>5} {'Fixed':>5}")
    lines.append("  " + "-" * 80)

    for r in results:
        short_name = r.match_name[:48]
        lines.append(
            f"  {short_name:<50} {r.original_segment_count:>5} "
            f"{r.segments_after_cleaning:>5} {r.hallucinations_removed:>5} "
            f"{r.duplicates_removed:>5} {r.entities_corrected:>5}"
        )
    lines.append("")

    # ── Entity Corrections ───────────────────────────────────────────
    all_corrections = []
    for r in results:
        for c in r.corrections:
            c["match"] = r.match_name
            all_corrections.append(c)

    if all_corrections:
        lines.append("ENTITY CORRECTIONS")
        lines.append("-" * 90)

        high_conf = [c for c in all_corrections if c["score"] >= 80]
        medium_conf = [c for c in all_corrections if 70 <= c["score"] < 80]
        low_conf = [c for c in all_corrections if c["score"] < 70]

        if high_conf:
            lines.append(f"\n  HIGH CONFIDENCE (score >= 80) — {len(high_conf)} corrections:")
            for c in sorted(high_conf, key=lambda x: -x["score"]):
                lines.append(
                    f"    [{c['score']:5.1f}] \"{c['original']}\" -> \"{c['corrected']}\""
                    f"  ({c['method']})"
                    f"  [seg {c['segment_id']} | {c['match']}]"
                )

        if medium_conf:
            lines.append(f"\n  MEDIUM CONFIDENCE (70 <= score < 80) — {len(medium_conf)} corrections:")
            for c in sorted(medium_conf, key=lambda x: -x["score"]):
                lines.append(
                    f"    [{c['score']:5.1f}] \"{c['original']}\" -> \"{c['corrected']}\""
                    f"  ({c['method']})"
                    f"  [seg {c['segment_id']} | {c['match']}]"
                )

        if low_conf:
            lines.append(f"\n  ! LOW CONFIDENCE (score < 70) — REVIEW THESE — {len(low_conf)} corrections:")
            for c in sorted(low_conf, key=lambda x: -x["score"]):
                lines.append(
                    f"    [{c['score']:5.1f}] \"{c['original']}\" -> \"{c['corrected']}\""
                    f"  ({c['method']})"
                    f"  [seg {c['segment_id']} | {c['match']}]"
                    f"  <- REVIEW"
                )

        lines.append("")

    # ── Hallucination Examples ───────────────────────────────────────
    lines.append("HALLUCINATION EXAMPLES (removed segments)")
    lines.append("-" * 70)
    example_count = 0
    for r in results:
        for h in r.removed_hallucinations[:5]:
            lines.append(
                f"  [{h['reason']}] \"{h['text']}\""
            )
            example_count += 1
        if example_count >= 15:
            lines.append("  ... (truncated, see full output files)")
            break
    lines.append("")

    # ── Duplicate Examples ───────────────────────────────────────────
    lines.append("DUPLICATE EXAMPLES (removed segments)")
    lines.append("-" * 70)
    example_count = 0
    for r in results:
        for d in r.removed_duplicates[:5]:
            lines.append(
                f"  Seg #{d['segment_id']} (dup of #{d['duplicate_of']}, "
                f"sim={d['similarity']}): \"{d['text']}\""
            )
            example_count += 1
        if example_count >= 10:
            lines.append("  ... (truncated)")
            break
    lines.append("")

    lines.append("=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


def print_report(results: list[CleaningResult]) -> None:
    report = generate_report(results)
    sys.stdout.buffer.write((report + "\n").encode("utf-8"))


def save_report(results: list[CleaningResult], filepath: str = "cleaning_report.txt") -> None:
    report = generate_report(results)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {filepath}")
