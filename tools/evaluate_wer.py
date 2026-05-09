"""
Evaluate WER/CER/Entity-F1 of the cleaning pipeline against ground truth.

Uses TIME-RANGE alignment (not segment_id) to pair raw / cleaned / GT
segments. This is necessary because ground-truth curation typically
drops hallucinated segments and re-numbers the remaining ones, so
segment_id matching silently compares non-corresponding content.

Compares three corpora:
    1. Raw Whisper output     — 1_asr.json
    2. Cleaned pipeline output — 1_asr_cleaned.json (from cleaned_data/)
    3. Gold-standard reference — 1_asr_corrected.json

Produces a markdown table + JSON artifact showing:
    - WER (word error rate), CER (character error rate)
    - Entity-level precision/recall/F1
    - Non-entity WER (isolates common-word cleanup from entity correction)
    - Improvement % of cleaned vs raw

Usage:
    python tools/evaluate_wer.py --match "AIK"
    python tools/evaluate_wer.py --match "AIK" --half 2

Requires: jiwer
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Put repo root on sys.path so we can import the pipeline when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jiwer import wer, cer, process_words  # type: ignore
from pipeline.config import ASR_INPUT_VARIANT, CLEANED_OUTPUT_DIR, DATASET_ROOT


# ─── Text normalization for fair comparison ──────────────────────────

_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "—–…""''„‟")


def normalize_for_wer(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().translate(_PUNCT_TABLE)
    return " ".join(text.split())


# ─── Loading time-stamped segments ───────────────────────────────────

@dataclass
class TimedSegment:
    """A single segment with start/end time and text.

    Raw ASR JSON uses [start, end, text] list format; cleaned pipeline
    output uses {"start_time": ..., "end_time": ..., "text": ...} dict
    format. This class unifies both.
    """
    start: float
    end: float
    text: str


def load_timed_segments(filepath: Path) -> list[TimedSegment]:
    """Load segments from an ASR/corrected/cleaned JSON file.

    Returns segments sorted by start time. Handles both the raw list
    format and the cleaned dict format.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: list[TimedSegment] = []
    for _sid, value in data.get("segments", {}).items():
        if isinstance(value, list) and len(value) >= 3:
            out.append(TimedSegment(
                start=float(value[0]),
                end=float(value[1]),
                text=str(value[2]),
            ))
        elif isinstance(value, dict):
            out.append(TimedSegment(
                start=float(value.get("start_time", 0.0)),
                end=float(value.get("end_time", 0.0)),
                text=str(value.get("text", "")),
            ))
    out.sort(key=lambda s: s.start)
    return out


# ─── Time-range alignment ────────────────────────────────────────────

# Tolerance for pairing segments by midpoint. Commentary segments are
# typically 2-5s long; 0.75s absorbs boundary drift from dropped segments
# but is tight enough never to pair two distinct commentary events.
TIME_TOLERANCE_S = 0.75


def align_by_window(
    reference: list[TimedSegment],
    hypothesis: list[TimedSegment],
    overlap_min_s: float = 0.25,
) -> list[tuple[TimedSegment, list[TimedSegment]]]:
    """Many-to-one alignment: group all hyp segments overlapping each ref window.

    GOAL human-curated GT segments are typically 5-15s long while raw
    Whisper segments are 2-5s long, so a single GT segment usually
    spans 2-4 hyp segments. ``align_by_time`` pairs one-to-one and
    drops the rest as ``None`` — fine for corpus WER (because the
    join-then-WER step still aggregates everything) but misleading for
    per-segment debugging because ``GOOD``/``MISSED``/``HARMFUL``
    classification ends up comparing fragments to whole sentences.

    This helper returns one entry per reference segment with the FULL
    list of hypothesis segments whose time ranges overlap by at least
    ``overlap_min_s``. Hypothesis segments not overlapping any
    reference are appended as a final ``(None, [hyp...])`` entry to
    surface hallucinations for downstream reporting.
    """
    pairs: list[tuple[TimedSegment, list[TimedSegment]]] = []
    matched_hyp_ids: set[int] = set()
    for r in reference:
        bucket: list[TimedSegment] = []
        for h in hypothesis:
            overlap = min(r.end, h.end) - max(r.start, h.start)
            if overlap >= overlap_min_s:
                bucket.append(h)
                matched_hyp_ids.add(id(h))
        pairs.append((r, bucket))
    leftovers = [h for h in hypothesis if id(h) not in matched_hyp_ids]
    if leftovers:
        pairs.append((None, leftovers))  # type: ignore[arg-type]
    return pairs


def align_by_time(
    reference: list[TimedSegment],
    hypothesis: list[TimedSegment],
    tolerance: float = TIME_TOLERANCE_S,
) -> list[tuple[TimedSegment | None, TimedSegment | None]]:
    """Pair reference/hypothesis segments by overlapping time range.

    Two-pointer greedy walk: when segments are time-aligned (midpoints
    within tolerance OR time spans overlap) they pair; otherwise the
    side with the earlier end time advances and emits a None-pairing.

    This handles the common case where ground-truth curation drops
    hallucinated segments and re-numbers: the re-numbering is invisible
    because we match by time, not by ID.

    Returns a list of (ref_segment, hyp_segment) pairs. Either side can
    be None when a segment has no counterpart within tolerance.
    """
    pairs: list[tuple[TimedSegment | None, TimedSegment | None]] = []
    i = j = 0
    while i < len(reference) and j < len(hypothesis):
        r, h = reference[i], hypothesis[j]
        r_mid = (r.start + r.end) / 2
        h_mid = (h.start + h.end) / 2
        spans_overlap = r.start < h.end and h.start < r.end
        if abs(r_mid - h_mid) <= tolerance or spans_overlap:
            pairs.append((r, h))
            i += 1
            j += 1
        elif r.end <= h.start:
            # Reference segment ends before hypothesis starts → hypothesis
            # dropped this ref segment (or ref is ahead and needs to catch up).
            pairs.append((r, None))
            i += 1
        else:
            # Hypothesis segment has no reference counterpart (extra/hallucination).
            pairs.append((None, h))
            j += 1
    # Flush tails
    while i < len(reference):
        pairs.append((reference[i], None))
        i += 1
    while j < len(hypothesis):
        pairs.append((None, hypothesis[j]))
        j += 1
    return pairs


# ─── Match path resolution ───────────────────────────────────────────

def resolve_match_paths(
    match_substring: str,
    half: int = 1,
    variant: str = "",
) -> dict[str, Path]:
    """Find the match directory by substring (direct filesystem scan).

    Supports both layouts:
        match_dir/1_asr.json                (flat layout, e.g. AIK test match)
        match_dir/commentary_data/1_asr.json (standard SoccerNet layout)

    ``variant`` selects an alternate raw file (e.g. ``"_kb"`` for the
    KB-Whisper output ``1_asr_kb.json``). The cleaned-output path is
    suffixed with the same variant so stock and KB-Whisper runs don't
    collide. Falls back to stock ``1_asr.json`` if the variant is missing.
    """
    needle = match_substring.lower()
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    for raw_stock_path in DATASET_ROOT.rglob(f"{half}_asr.json"):
        match_parent = raw_stock_path.parent
        is_standard_layout = match_parent.name == "commentary_data"
        match_dir = match_parent.parent if is_standard_layout else match_parent

        if needle not in match_dir.name.lower():
            continue

        gt_path = match_parent / f"{half}_asr_corrected.json"

        # Variant raw path (e.g. 1_asr_kb.json) — fall back to stock if absent.
        raw_variant_path = match_parent / f"{half}_asr{variant}.json"
        raw_path = raw_variant_path if (variant and raw_variant_path.exists()) else raw_stock_path

        try:
            rel = match_dir.relative_to(DATASET_ROOT.parent)
        except ValueError:
            rel = Path(match_dir.name)

        cleaned_path = (
            CLEANED_OUTPUT_DIR / rel / "commentary_data"
            / f"{half}_asr{variant}_cleaned.json"
        )

        return {
            "match_name": match_dir.name,
            "match_dir": match_dir,
            "raw": raw_path,
            "raw_stock": raw_stock_path,
            "ground_truth": gt_path,
            "cleaned": cleaned_path,
            "variant": variant,
        }
    raise FileNotFoundError(f"No match found matching '{match_substring}'")


# ─── Metrics ─────────────────────────────────────────────────────────

@dataclass
class Metrics:
    label: str
    wer: float
    cer: float
    non_entity_wer: float
    total_reference_words: int
    total_hypothesis_words: int
    substitutions: int
    insertions: int
    deletions: int
    hits: int
    entity_precision: float
    entity_recall: float
    entity_f1: float
    # Alignment stats — how many GT segments had a hyp counterpart
    aligned_pairs: int = 0
    ref_only: int = 0     # GT segments without hyp match (pipeline dropped or missed)
    hyp_only: int = 0     # Hyp segments without GT match (hallucinations, insertions)


def compute_wer_cer(reference: list[str], hypothesis: list[str]) -> tuple:
    """Compute aggregated WER, CER, and edit counts across all segment pairs."""
    ref = [normalize_for_wer(t) for t in reference]
    hyp = [normalize_for_wer(t) for t in hypothesis]

    ref_full = " ".join(ref)
    hyp_full = " ".join(hyp)

    if not ref_full.strip():
        return 0.0, 0.0, 0, 0, 0, 0, 0, 0

    word_wer = wer(ref_full, hyp_full)
    char_cer = cer(ref_full, hyp_full)

    measures = process_words(ref_full, hyp_full)
    subs = measures.substitutions
    ins = measures.insertions
    dels = measures.deletions
    hits = measures.hits

    ref_words = len(ref_full.split())
    hyp_words = len(hyp_full.split())

    return word_wer, char_cer, ref_words, hyp_words, subs, ins, dels, hits


def compute_non_entity_wer(
    reference_texts: list[str],
    hypothesis_texts: list[str],
) -> float:
    """WER computed only on non-proper-noun (lowercase) tokens.

    Isolates common-word cleanup (what the neural/LLM stages target)
    from proper-noun correction (what the gazetteer-based Tier 2/3 targets).
    """
    def strip_entities(text: str) -> str:
        tokens = []
        for raw in text.split():
            w = raw.strip(string.punctuation + "—–…""''„‟")
            if w and w[0].isupper() and any(c.islower() for c in w):
                continue  # proper noun — skip
            if not w:
                continue
            tokens.append(w.lower().translate(_PUNCT_TABLE))
        return " ".join(t for t in tokens if t)

    ref_full = " ".join(strip_entities(t) for t in reference_texts)
    hyp_full = " ".join(strip_entities(t) for t in hypothesis_texts)
    if not ref_full.strip():
        return 0.0
    return wer(ref_full, hyp_full)


def extract_entity_tokens(text: str) -> set[str]:
    """Capitalized-word heuristic entity set (lightweight, no spaCy load)."""
    entities: set[str] = set()
    for raw in text.split():
        w = raw.strip(string.punctuation + "—–…""''„‟")
        if len(w) >= 3 and w[0].isupper() and not w.isupper():
            entities.add(w.lower())
    return entities


def compute_entity_f1(
    reference_texts: list[str],
    hypothesis_texts: list[str],
) -> tuple[float, float, float]:
    """Precision/recall/F1 for entity-like (Capitalized) tokens."""
    ref_entities: set[str] = set()
    hyp_entities: set[str] = set()
    for t in reference_texts:
        ref_entities.update(extract_entity_tokens(t))
    for t in hypothesis_texts:
        hyp_entities.update(extract_entity_tokens(t))

    if not hyp_entities and not ref_entities:
        return 1.0, 1.0, 1.0
    if not hyp_entities:
        return 0.0, 0.0, 0.0
    if not ref_entities:
        return 0.0, 1.0, 0.0

    tp = len(ref_entities & hyp_entities)
    precision = tp / len(hyp_entities) if hyp_entities else 0.0
    recall = tp / len(ref_entities) if ref_entities else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate(
    label: str,
    reference: list[TimedSegment],
    hypothesis: list[TimedSegment],
    fallback: list[TimedSegment] | None = None,
    alignment_mode: str = "legacy",
) -> Metrics:
    """Evaluate hypothesis against reference.

    ``alignment_mode``:

      • ``"legacy"`` (default): one-to-one greedy time alignment. Matches
        the numbers an online WER tool would produce when given the
        joined corpus directly — useful for external verification.
      • ``"windowed"``: for each GT segment, concatenate ALL hypothesis
        segments whose time spans overlap it, then compare the joined
        text to the GT text. More rigorous (no jiwer-DP "drift" of
        hyp_only words) but produces ~9pp higher absolute WER. Use
        when comparing against papers that report strict alignment.

    When a reference segment has no hypothesis counterpart AND a
    fallback is provided (typically the raw Whisper output), the
    fallback's overlapping text is used so dropped cleaned segments
    don't hide content from the eval.

    Hypothesis segments that overlap NO GT window (true hallucinations
    rather than segmentation artefacts) are still counted as insertions.
    """
    if alignment_mode == "legacy":
        return _evaluate_legacy(label, reference, hypothesis, fallback)
    return _evaluate_windowed(label, reference, hypothesis, fallback)


def _evaluate_windowed(
    label: str,
    reference: list[TimedSegment],
    hypothesis: list[TimedSegment],
    fallback: list[TimedSegment] | None,
) -> Metrics:
    """Many-to-one alignment: concatenate hyp segs overlapping each GT window."""
    groups = align_by_window(reference, hypothesis)
    fallback_groups = align_by_window(reference, fallback) if fallback is not None else []
    fallback_text_by_ref_id: dict[int, str] = {}
    for ref, fbs in fallback_groups:
        if ref is not None and fbs:
            fallback_text_by_ref_id[id(ref)] = " ".join(s.text for s in fbs)

    ref_texts: list[str] = []
    hyp_texts: list[str] = []
    aligned_pairs = 0
    ref_only = 0
    hyp_only = 0

    for ref, hyps in groups:
        if ref is None:
            for h in hyps:
                ref_texts.append("")
                hyp_texts.append(h.text)
                hyp_only += 1
            continue
        if not hyps:
            ref_texts.append(ref.text)
            hyp_texts.append(fallback_text_by_ref_id.get(id(ref), ""))
            ref_only += 1
            continue
        ref_texts.append(ref.text)
        hyp_texts.append(" ".join(s.text for s in hyps))
        aligned_pairs += 1
    return _metrics_from_pairs(label, ref_texts, hyp_texts, aligned_pairs, ref_only, hyp_only)


def _evaluate_legacy(
    label: str,
    reference: list[TimedSegment],
    hypothesis: list[TimedSegment],
    fallback: list[TimedSegment] | None,
) -> Metrics:
    """Original 1-to-1 greedy time alignment (kept for ablation)."""
    ref_hyp_pairs = align_by_time(reference, hypothesis)
    fallback_by_ref_id: dict[int, str] = {}
    if fallback is not None:
        fb_pairs = align_by_time(reference, fallback)
        for r, h in fb_pairs:
            if r is not None and h is not None:
                fallback_by_ref_id[id(r)] = h.text

    ref_texts: list[str] = []
    hyp_texts: list[str] = []
    aligned_pairs = 0
    ref_only = 0
    hyp_only = 0

    for r, h in ref_hyp_pairs:
        if r is None and h is None:
            continue
        if r is None:
            ref_texts.append("")
            hyp_texts.append(h.text)
            hyp_only += 1
            continue
        if h is None:
            ref_texts.append(r.text)
            fb_text = fallback_by_ref_id.get(id(r), "")
            hyp_texts.append(fb_text)
            ref_only += 1
            continue
        ref_texts.append(r.text)
        hyp_texts.append(h.text)
        aligned_pairs += 1
    return _metrics_from_pairs(label, ref_texts, hyp_texts, aligned_pairs, ref_only, hyp_only)


def _metrics_from_pairs(
    label: str,
    ref_texts: list[str],
    hyp_texts: list[str],
    aligned_pairs: int,
    ref_only: int,
    hyp_only: int,
) -> Metrics:
    """Compute WER/CER/Entity-F1 etc. from already-aligned text pairs."""

    word_wer, char_cer, ref_w, hyp_w, subs, ins, dels, hits = compute_wer_cer(
        ref_texts, hyp_texts
    )
    non_entity = compute_non_entity_wer(ref_texts, hyp_texts)
    p, r_score, f = compute_entity_f1(ref_texts, hyp_texts)

    return Metrics(
        label=label,
        wer=round(word_wer * 100, 2),
        cer=round(char_cer * 100, 2),
        non_entity_wer=round(non_entity * 100, 2),
        total_reference_words=ref_w,
        total_hypothesis_words=hyp_w,
        substitutions=subs,
        insertions=ins,
        deletions=dels,
        hits=hits,
        entity_precision=round(p, 4),
        entity_recall=round(r_score, 4),
        entity_f1=round(f, 4),
        aligned_pairs=aligned_pairs,
        ref_only=ref_only,
        hyp_only=hyp_only,
    )


# ─── Report rendering ────────────────────────────────────────────────

def render_table(rows: list[Metrics]) -> str:
    """Render a markdown-friendly ablation table."""
    header = (
        "| Configuration                   | WER (%) | Non-ent WER (%) | CER (%) | Entity F1 | Sub/Ins/Del | Align(ok/ref/hyp) |"
    )
    sep = (
        "|---------------------------------|---------|-----------------|---------|-----------|-------------|-------------------|"
    )
    lines = [header, sep]
    for m in rows:
        lines.append(
            f"| {m.label:<31} | "
            f"{m.wer:>7.2f} | {m.non_entity_wer:>15.2f} | {m.cer:>7.2f} | "
            f"{m.entity_f1:>9.4f} | "
            f"{m.substitutions:>4}/{m.insertions:>3}/{m.deletions:>3} | "
            f"{m.aligned_pairs:>4}/{m.ref_only:>3}/{m.hyp_only:>3}      |"
        )

    if len(rows) >= 2:
        baseline = rows[0]
        best = min(rows[1:], key=lambda m: m.wer)
        wer_delta = baseline.wer - best.wer
        wer_rel = (wer_delta / baseline.wer * 100) if baseline.wer else 0.0
        ne_delta = baseline.non_entity_wer - best.non_entity_wer
        cer_delta = baseline.cer - best.cer
        f1_delta = best.entity_f1 - baseline.entity_f1
        lines.append(sep)
        lines.append(
            f"| {'Absolute improvement':<31} | "
            f"{wer_delta:>+7.2f} | {ne_delta:>+15.2f} | {cer_delta:>+7.2f} | "
            f"{f1_delta:>+9.4f} | —         | —                 |"
        )
        lines.append(
            f"| {'Relative WER improvement':<31} | "
            f"{wer_rel:>+6.1f}% |            —    |    —    |     —     | —         | —                 |"
        )
    return "\n".join(lines)


# ─── Main entry ──────────────────────────────────────────────────────

def run(
    match_substring: str,
    half: int = 1,
    output_dir: Path | None = None,
    variant: str = "",
    ablate: bool = False,
    alignment_mode: str = "windowed",
) -> dict:
    """Evaluate one match-half against its ground truth.

    When ``ablate`` is True, both the stock and the variant raw/cleaned
    paths are evaluated together so the markdown report shows the full
    KB-Whisper-vs-stock-Whisper comparison in one table.
    """
    paths = resolve_match_paths(match_substring, half=half, variant=variant)
    print(f"Match: {paths['match_name']}")
    print(f"  Raw:         {paths['raw']}")
    print(f"  Ground truth:{paths['ground_truth']}")
    print(f"  Cleaned:     {paths['cleaned']}")
    print(f"  (time-range alignment, tolerance={TIME_TOLERANCE_S}s)")

    if not paths["ground_truth"].exists():
        print(f"ERROR: no ground truth at {paths['ground_truth']}", file=sys.stderr)
        sys.exit(2)

    gt_segs = load_timed_segments(paths["ground_truth"])
    results: list[Metrics] = []

    if ablate:
        # Stock raw + cleaned (existing baseline) -----------------------
        stock_raw_segs = load_timed_segments(paths["raw_stock"])
        results.append(evaluate("Stock Whisper raw", gt_segs, stock_raw_segs, alignment_mode=alignment_mode))
        stock_cleaned_paths = resolve_match_paths(match_substring, half=half, variant="")
        if stock_cleaned_paths["cleaned"].exists():
            stock_cleaned_segs = load_timed_segments(stock_cleaned_paths["cleaned"])
            results.append(evaluate(
                "Stock Whisper + pipeline", gt_segs, stock_cleaned_segs,
                fallback=stock_raw_segs, alignment_mode=alignment_mode,
            ))
        # Variant raw + cleaned (e.g. KB-Whisper) -----------------------
        if variant and paths["raw"] != paths["raw_stock"]:
            kb_raw_segs = load_timed_segments(paths["raw"])
            label_raw = f"{variant.lstrip('_').upper() or 'VARIANT'}-Whisper raw"
            label_cleaned = f"{variant.lstrip('_').upper() or 'VARIANT'}-Whisper + pipeline"
            results.append(evaluate(label_raw, gt_segs, kb_raw_segs, alignment_mode=alignment_mode))
            if paths["cleaned"].exists():
                kb_cleaned_segs = load_timed_segments(paths["cleaned"])
                results.append(evaluate(
                    label_cleaned, gt_segs, kb_cleaned_segs, fallback=kb_raw_segs,
                    alignment_mode=alignment_mode,
                ))
    else:
        raw_segs = load_timed_segments(paths["raw"])
        print(f"  Raw: {len(raw_segs)} segs | GT: {len(gt_segs)} segs")
        label = "Raw Whisper (baseline)" if not variant else f"{variant.lstrip('_').upper()}-Whisper raw"
        results.append(evaluate(label, gt_segs, raw_segs, alignment_mode=alignment_mode))

        if paths["cleaned"].exists():
            cleaned_segs = load_timed_segments(paths["cleaned"])
            print(f"  Cleaned: {len(cleaned_segs)} segs")
            results.append(evaluate(
                "Pipeline cleaned output", gt_segs, cleaned_segs, fallback=raw_segs,
                alignment_mode=alignment_mode,
            ))
        else:
            print(
                f"WARNING: no cleaned output yet. Run the pipeline first:\n"
                f"  python run_pipeline.py --match \"{match_substring}\"",
                file=sys.stderr,
            )

    table = render_table(results)
    print()
    print(table)
    print()

    payload = {
        "match": paths["match_name"],
        "half": half,
        "variant": variant,
        "ablate": ablate,
        "tolerance_s": TIME_TOLERANCE_S,
        "results": [asdict(m) for m in results],
    }

    if output_dir is None:
        output_dir = _REPO_ROOT / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = paths["match_name"].replace(" ", "_").replace("/", "_")
    suffix = f"_{variant.lstrip('_')}" if variant else ""
    if ablate:
        suffix = f"{suffix}_ablation"
    json_path = output_dir / f"{slug}_half{half}{suffix}_wer.json"
    md_path = output_dir / f"{slug}_half{half}{suffix}_wer.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# WER Evaluation — {paths['match_name']} (half {half})\n\n")
        f.write(f"_Time-range alignment, tolerance {TIME_TOLERANCE_S}s_\n\n")
        f.write(table + "\n")

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--match", required=True, help="Substring of the match name")
    p.add_argument("--half", type=int, default=1, choices=[1, 2], help="Which half to evaluate")
    p.add_argument("--output-dir", type=Path, default=None, help="Where to save reports")
    p.add_argument(
        "--variant",
        type=str,
        default=ASR_INPUT_VARIANT,
        help='Variant suffix for the raw ASR file (e.g. "_kb" reads 1_asr_kb.json). '
             'Default reads ASR_INPUT_VARIANT env var.',
    )
    p.add_argument(
        "--ablate",
        action="store_true",
        help="Evaluate both stock and variant (raw + cleaned) in a single 4-row table.",
    )
    p.add_argument(
        "--alignment-mode",
        choices=["legacy", "windowed"],
        default="legacy",
        help='"legacy" (default): one-to-one greedy time alignment, matches '
             'online WER tool numbers when given the joined corpus. '
             '"windowed": each GT window concatenates all overlapping '
             'cleaned segments — stricter (~9pp higher absolute WER).',
    )
    args = p.parse_args()
    run(args.match, half=args.half, output_dir=args.output_dir,
        variant=args.variant, ablate=args.ablate,
        alignment_mode=args.alignment_mode)


if __name__ == "__main__":
    main()
