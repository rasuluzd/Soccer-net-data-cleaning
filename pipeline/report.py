"""
Cleaning Report Generator — produces human-readable statistics.

After the pipeline runs, this module creates a summary report showing:
- Total segments processed vs. retained
- Number and examples of hallucinations removed
- Number of duplicates removed
- Full list of entity corrections with scores and methods
- Per-match summary table
- Low-confidence corrections flagged for manual review
"""

from pipeline.orchestrator import CleaningResult


def generate_report(results: list[CleaningResult]) -> str:
    """
    Generate a comprehensive cleaning report from pipeline results.

    Args:
        results: list of CleaningResult objects (one per match)

    Returns:
        Formatted report string
    """
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

    lines.append("OVERALL SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Matches processed:     {len(results)}")
    lines.append(f"  Total segments (raw):  {total_original}")
    lines.append(f"  Hallucinations removed:{total_hallucinations}")
    lines.append(f"  Duplicates removed:    {total_duplicates}")
    lines.append(f"  Segments retained:     {total_after}")
    lines.append(f"  Entities detected:     {total_entities}")
    lines.append(f"  Entities corrected:    {total_corrections}")
    lines.append(f"  Retention rate:        {total_after/total_original*100:.1f}%" if total_original else "")
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

        # Group by confidence level
        high_conf = [c for c in all_corrections if c["score"] >= 80]
        medium_conf = [c for c in all_corrections if 70 <= c["score"] < 80]
        low_conf = [c for c in all_corrections if c["score"] < 70]

        if high_conf:
            lines.append(f"\n  HIGH CONFIDENCE (score ≥ 80) — {len(high_conf)} corrections:")
            for c in sorted(high_conf, key=lambda x: -x["score"]):
                lines.append(
                    f"    [{c['score']:5.1f}] \"{c['original']}\" → \"{c['corrected']}\""
                    f"  ({c['method']})"
                )

        if medium_conf:
            lines.append(f"\n  MEDIUM CONFIDENCE (70 ≤ score < 80) — {len(medium_conf)} corrections:")
            for c in sorted(medium_conf, key=lambda x: -x["score"]):
                lines.append(
                    f"    [{c['score']:5.1f}] \"{c['original']}\" → \"{c['corrected']}\""
                    f"  ({c['method']})"
                )

        if low_conf:
            lines.append(f"\n  ⚠ LOW CONFIDENCE (score < 70) — REVIEW THESE — {len(low_conf)} corrections:")
            for c in sorted(low_conf, key=lambda x: -x["score"]):
                lines.append(
                    f"    [{c['score']:5.1f}] \"{c['original']}\" → \"{c['corrected']}\""
                    f"  ({c['method']}) ← REVIEW"
                )

        lines.append("")

    # ── Hallucination Examples ───────────────────────────────────────
    lines.append("HALLUCINATION EXAMPLES (removed segments)")
    lines.append("-" * 70)
    example_count = 0
    for r in results:
        for h in r.removed_hallucinations[:5]:  # show up to 5 per match
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
    """Generate and print the cleaning report."""
    report = generate_report(results)
    print(report)


def save_report(results: list[CleaningResult], filepath: str = "cleaning_report.txt") -> None:
    """Generate and save the cleaning report to a file."""
    report = generate_report(results)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {filepath}")
