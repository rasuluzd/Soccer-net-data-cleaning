"""
Per-segment diff viewer — shows raw Whisper vs cleaned vs ground truth
side-by-side for each segment in a match. Designed for thesis case studies.

Uses TIME-RANGE alignment (shared with tools/evaluate_wer.py) so GT segments
are correctly paired with raw/cleaned segments even when curation dropped
segments and re-numbered.

Classifies each GT segment's paired raw/cleaned pair as:
    GOOD      — raw was wrong, cleaned now matches GT
    HARMFUL   — raw was right, cleaned introduced an error
    PARTIAL   — cleaned is closer to GT than raw, but not identical
    MISSED    — raw was wrong, cleaned is still wrong
    CLEAN     — raw == cleaned == GT (no work needed)
    DROPPED   — GT has a segment that raw/cleaned don't (possibly OK if
                raw hallucinated and pipeline filtered correctly, but here
                we treat it as lost content)

Usage:
    python tools/per_segment_diff.py --match "AIK"
    python tools/per_segment_diff.py --match "AIK" --filter HARMFUL
    python tools/per_segment_diff.py --match "AIK" --filter GOOD --limit 20
    python tools/per_segment_diff.py --match "AIK" --output-md diff.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.evaluate_wer import (  # noqa: E402
    TimedSegment,
    align_by_time,
    align_by_window,
    load_timed_segments,
    normalize_for_wer,
    resolve_match_paths,
)


def classify(raw: str, cleaned: str, gt: str) -> str:
    """Classify a segment based on whether cleaning improved it."""
    r = normalize_for_wer(raw)
    c = normalize_for_wer(cleaned)
    g = normalize_for_wer(gt)

    if not g:
        return "EMPTY_GT"
    if r == g and c == g:
        return "CLEAN"
    if r != g and c == g:
        return "GOOD"
    if r == g and c != g:
        return "HARMFUL"
    if r != c and _edit_distance(c, g) < _edit_distance(r, g):
        return "PARTIAL"
    return "MISSED"


def _edit_distance(a: str, b: str) -> int:
    """Word-level Levenshtein distance."""
    aw, bw = a.split(), b.split()
    if not aw:
        return len(bw)
    if not bw:
        return len(aw)
    prev = list(range(len(bw) + 1))
    for i, ac in enumerate(aw, 1):
        cur = [i]
        for j, bc in enumerate(bw, 1):
            cost = 0 if ac == bc else 1
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def render_segment(
    start: float,
    end: float,
    raw: str,
    cleaned: str,
    gt: str,
    label: str,
) -> str:
    """Format one diff block."""
    return (
        f"[t={start:.1f}-{end:.1f}s] [{label}]\n"
        f"  RAW      : {raw}\n"
        f"  CLEANED  : {cleaned}\n"
        f"  GT       : {gt}\n"
    )


def run(
    match_substring: str,
    half: int,
    filter_label: str | None,
    limit: int,
    output_md: Path | None,
) -> None:
    paths = resolve_match_paths(match_substring, half=half)
    print(f"Match: {paths['match_name']}\n")

    if not paths["ground_truth"].exists():
        print(f"ERROR: no ground truth at {paths['ground_truth']}", file=sys.stderr)
        sys.exit(2)

    gt_segs = load_timed_segments(paths["ground_truth"])
    raw_segs = load_timed_segments(paths["raw"])
    cleaned_segs = (
        load_timed_segments(paths["cleaned"])
        if paths["cleaned"].exists() else []
    )

    # Many-to-one alignment: each GT window collects all hyp segments
    # whose time spans overlap it. GOAL GT is much coarser than Whisper
    # output (Chelsea half-1: 230 GT vs 612 raw), so per-segment diff
    # only makes sense after aggregating hyp text within each GT window.
    raw_groups = align_by_window(gt_segs, raw_segs)
    raw_by_gt = {id(g): hs for g, hs in raw_groups if g is not None}
    raw_hallu_count = sum(len(hs) for g, hs in raw_groups if g is None)

    if cleaned_segs:
        cleaned_groups = align_by_window(gt_segs, cleaned_segs)
        cleaned_by_gt = {id(g): hs for g, hs in cleaned_groups if g is not None}
        cleaned_hallu_count = sum(
            len(hs) for g, hs in cleaned_groups if g is None
        )
    else:
        cleaned_by_gt = {}
        cleaned_hallu_count = 0

    counts: dict[str, int] = {}
    rendered_blocks: list[str] = []
    shown = 0

    for g in gt_segs:
        raw_bucket = raw_by_gt.get(id(g), [])
        cleaned_bucket = cleaned_by_gt.get(id(g), [])

        raw_text = " ".join(s.text for s in raw_bucket).strip()
        cleaned_text = " ".join(s.text for s in cleaned_bucket).strip()
        if not cleaned_text:
            cleaned_text = raw_text  # cleaning pipeline produced no overlap
        gt_text = g.text

        if not raw_bucket:
            # GT has a window raw doesn't cover — lost content
            label = "DROPPED"
        else:
            label = classify(raw_text, cleaned_text, gt_text)

        counts[label] = counts.get(label, 0) + 1

        if filter_label is not None and label != filter_label:
            continue
        if shown >= limit:
            continue
        rendered_blocks.append(render_segment(
            g.start, g.end, raw_text, cleaned_text, gt_text, label,
        ))
        shown += 1

    print("Summary (by GT segment):")
    total = sum(counts.values())
    for k in ("CLEAN", "GOOD", "PARTIAL", "MISSED", "HARMFUL", "DROPPED", "EMPTY_GT"):
        pct = (counts.get(k, 0) / total * 100) if total else 0
        print(f"  {k:<10} {counts.get(k, 0):>5}  ({pct:5.1f}%)")
    print()
    print(f"  Raw segments without GT counterpart (hallucinations):     {raw_hallu_count}")
    print(f"  Cleaned segments without GT counterpart (hallucinations): {cleaned_hallu_count}")
    print()

    if rendered_blocks:
        print("Examples:\n")
        print("\n".join(rendered_blocks))

    if output_md is not None:
        lines = [f"# Per-segment diff — {paths['match_name']} (half {half})\n"]
        lines.append("## Summary\n")
        for k in ("CLEAN", "GOOD", "PARTIAL", "MISSED", "HARMFUL", "DROPPED", "EMPTY_GT"):
            pct = (counts.get(k, 0) / total * 100) if total else 0
            lines.append(f"- **{k}**: {counts.get(k, 0)} ({pct:.1f}%)")
        lines.append(f"\n_Raw hallucinations: {raw_hallu_count}, "
                     f"cleaned hallucinations: {cleaned_hallu_count}_\n")
        lines.append("## Examples\n")
        for blk in rendered_blocks:
            lines.append("```")
            lines.append(blk.rstrip())
            lines.append("```\n")
        output_md.parent.mkdir(parents=True, exist_ok=True)
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nSaved: {output_md}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--match", required=True)
    p.add_argument("--half", type=int, default=1, choices=[1, 2])
    p.add_argument(
        "--filter",
        choices=["GOOD", "HARMFUL", "PARTIAL", "MISSED", "CLEAN", "DROPPED", "EMPTY_GT"],
        default=None,
        help="Only show segments with this classification",
    )
    p.add_argument("--limit", type=int, default=50, help="Max segments to show")
    p.add_argument("--output-md", type=Path, default=None, help="Save diff to markdown")
    args = p.parse_args()
    run(args.match, args.half, args.filter, args.limit, args.output_md)


if __name__ == "__main__":
    main()
