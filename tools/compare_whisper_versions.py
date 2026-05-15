"""Compare SoccerNet's bundled Whisper transcript vs our faster-whisper-v3.

Two ASR transcripts of the same audio file (Chelsea 1-2 Liverpool 2016)
exist side-by-side in commentary_data/:

  - 1_asr.json        (schema-1 list format, what SoccerNet ships)
                      Likely produced by stock OpenAI Whisper (medium
                      or large) and frozen into the SoccerNet-Echoes
                      release. We have NO control over which model /
                      decoding parameters they used.

  - 1_asr_v3.json     (schema-2 dict format, our regeneration via
                      Systran/faster-whisper-large-v3 with beam=5,
                      word_timestamps=True, no_speech_threshold=0.95)

Both transcripts target the same audio. Both are evaluated against the
GOAL human-annotated ground truth (1_asr_corrected.json) via jiwer
WER and an entity-aware F1 over the segment-aligned text.

This isolates the impact of "which Whisper engine + decoding params"
from the impact of the cleaning pipeline. The cleaning pipeline can
then be ablated separately on top of either input.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Seg:
    start: float
    end: float
    text: str


def load_segments(path: Path) -> list[Seg]:
    """Load any of: schema-1 list, schema-2 dict, GOAL list-of-dicts."""
    if not path.exists():
        return []
    raw = json.load(open(path, encoding="utf-8"))

    if isinstance(raw, list):
        # GOAL: [{offset, duration, commentary}, ...]
        return [
            Seg(float(s["offset"]),
                float(s["offset"] + s["duration"]),
                s.get("commentary", ""))
            for s in raw
        ]

    segs = raw.get("segments", {})
    out = []
    for sid, v in segs.items():
        if isinstance(v, list) and len(v) >= 3:
            out.append(Seg(float(v[0]), float(v[1]), str(v[2])))
        elif isinstance(v, dict):
            out.append(Seg(
                float(v.get("start", 0.0)),
                float(v.get("end", 0.0)),
                str(v.get("text", "")),
            ))
    return sorted(out, key=lambda s: s.start)


def time_aligned_pairs(ref: list[Seg], hyp: list[Seg],
                       mode: str = "legacy", tolerance: float = 0.75
                       ) -> list[tuple[Seg, str]]:
    """Pair reference segments with hypothesis segments.

    mode='legacy'   — 1-to-1 greedy time alignment by midpoint distance
                      (matches tools/evaluate_wer.py default behaviour).
    mode='windowed' — concatenate ALL hyp segs that overlap each GT window
                      (stricter; gives ~9pp higher WER).
    """
    if mode == "windowed":
        out = []
        for r in ref:
            joined = " ".join(
                h.text for h in hyp
                if h.start < r.end and h.end > r.start
            )
            out.append((r, joined))
        return out

    # legacy: greedy 1-to-1 by midpoint proximity
    pairs: list[tuple[Seg, str]] = []
    used_hyp: set[int] = set()
    for r in ref:
        r_mid = (r.start + r.end) / 2
        best_idx = -1
        best_dist = float("inf")
        for i, h in enumerate(hyp):
            if i in used_hyp:
                continue
            h_mid = (h.start + h.end) / 2
            d = abs(r_mid - h_mid)
            if d < best_dist and d <= tolerance + (h.end - h.start) / 2:
                best_dist = d
                best_idx = i
        if best_idx >= 0:
            used_hyp.add(best_idx)
            pairs.append((r, hyp[best_idx].text))
        else:
            pairs.append((r, ""))
    return pairs


def compute_wer(ref_texts: list[str], hyp_texts: list[str]) -> dict:
    """Use jiwer to compute WER + counts.

    Pre-normalise manually so we can pass plain strings to process_words
    (cleaner than chaining jiwer's Compose, which has gotchas about
    where ReduceToList* must sit in the chain).
    """
    import jiwer
    import re

    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s']", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    ref_proc = [norm(r) for r in ref_texts]
    hyp_proc = [norm(h) for h in hyp_texts]
    # Filter out empty pairs (jiwer requires non-empty refs)
    pairs = [(r, h) for r, h in zip(ref_proc, hyp_proc) if r]
    if not pairs:
        return {"wer": 0.0, "substitutions": 0, "deletions": 0,
                "insertions": 0, "hits": 0, "ref_words": 0}
    refs, hyps = zip(*pairs)
    out = jiwer.process_words(list(refs), list(hyps))
    return {
        "wer": out.wer,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hits": out.hits,
        "ref_words": sum(len(r.split()) for r in refs),
    }


def entity_f1(ref_texts: list[str], hyp_texts: list[str],
              gazetteer_canonicals: set[str]) -> dict:
    """Token-level F1 restricted to gazetteer canonicals (case-insensitive)."""
    canon_lower = {c.lower() for c in gazetteer_canonicals}

    def extract_entities(text: str) -> list[str]:
        out = []
        for w in text.split():
            clean = w.strip(".,;:!?\"'()[]{}").rstrip("'s").rstrip("'")
            if clean.lower() in canon_lower:
                out.append(clean.lower())
        return out

    tp = fp = fn = 0
    for r, h in zip(ref_texts, hyp_texts):
        ref_ents = extract_entities(r)
        hyp_ents = extract_entities(h)
        ref_set = set(ref_ents)
        hyp_set = set(hyp_ents)
        tp += len(ref_set & hyp_set)
        fp += len(hyp_set - ref_set)
        fn += len(ref_set - hyp_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--match", required=True)
    p.add_argument("--raw-root", type=Path,
                   default=Path("path/to/SoccerNet/caption-2023"))
    p.add_argument("--out", type=Path,
                   default=Path("thesis/whisper_versions_comparison.md"))
    p.add_argument("--alignment", choices=["legacy", "windowed"],
                   default="legacy",
                   help="Match tools/evaluate_wer.py default (legacy)")
    args = p.parse_args()

    # Find match dir
    match_dir = None
    for league in args.raw_root.iterdir():
        if league.is_dir():
            for s in league.iterdir():
                if s.is_dir():
                    for m in s.iterdir():
                        if m.is_dir() and args.match.lower() in m.name.lower():
                            match_dir = m
    if match_dir is None:
        print(f"match not found", file=sys.stderr); return 1
    print(f"Match: {match_dir.name}")

    # Build gazetteer for entity-F1
    from pipeline.gazetteer import build_gazetteer
    labels = json.load(open(match_dir / "Labels-caption.json", encoding="utf-8"))
    gaz, _ = build_gazetteer(labels)
    canonicals = {v for v in gaz.values() if v}
    canonical_words = set()
    for c in canonicals:
        for w in c.split():
            if len(w) >= 4:
                canonical_words.add(w)

    md = ["# Whisper Version Comparison: SoccerNet bundled vs faster-whisper-v3",
          "",
          f"Match: **{match_dir.name}**",
          "",
          "Both ASR transcripts cover the same audio. The difference is",
          "**which Whisper engine + decoding parameters were used to",
          "produce the raw transcript** — not the cleaning pipeline.",
          "",
          "| Track | Engine + parameters |",
          "|---|---|",
          "| **1_asr.json** | SoccerNet bundled — most likely stock OpenAI "
          "Whisper (medium or large), default decoding parameters, frozen "
          "in the SoccerNet-Echoes release. Schema-1 list format. |",
          "| **1_asr_v3.json** | Our regeneration via "
          "Systran/faster-whisper-large-v3 with beam=5, word_timestamps=True, "
          "no_speech_threshold=0.95, condition_on_previous_text=False, "
          "Q4_K_M int8 quantisation on CPU. Schema-2 dict format with per-"
          "word probabilities. |",
          "",
          "Both are evaluated against `1_asr_corrected.json` (GOAL human-",
          "annotated ground truth, scored with jiwer + a custom entity-F1).",
          "",
          "---",
          "",
          "## Results per half",
          ""]

    overall = {"soccernet": {"wer_num": 0, "wer_den": 0,
                              "ent_tp": 0, "ent_fp": 0, "ent_fn": 0},
               "fasterv3":  {"wer_num": 0, "wer_den": 0,
                              "ent_tp": 0, "ent_fp": 0, "ent_fn": 0}}

    for half in (1, 2):
        gt_path = match_dir / "commentary_data" / f"{half}_asr_corrected.json"
        soc_path = match_dir / "commentary_data" / f"{half}_asr.json"
        v3_path = match_dir / "commentary_data" / f"{half}_asr_v3.json"

        if not gt_path.exists():
            md.append(f"### Half {half}: GT not found, skipping"); continue
        gt = load_segments(gt_path)
        soc = load_segments(soc_path) if soc_path.exists() else []
        v3 = load_segments(v3_path) if v3_path.exists() else []

        md.append(f"### Half {half} ({len(gt)} GT segments)")
        md.append("")
        md.append("| Track | Segments | WER | Sub/Ins/Del | Entity-F1 | Entity P/R |")
        md.append("|---|---|---|---|---|---|")

        for label, segs, key in (
            ("SoccerNet bundled", soc, "soccernet"),
            ("faster-whisper-v3", v3,  "fasterv3"),
        ):
            if not segs:
                md.append(f"| {label} | — | (file missing) | — | — | — |")
                continue

            pairs = time_aligned_pairs(gt, segs, mode=args.alignment)
            ref_texts = [r.text for r, _ in pairs]
            hyp_texts = [h for _, h in pairs]
            wer_stats = compute_wer(ref_texts, hyp_texts)
            f1_stats = entity_f1(ref_texts, hyp_texts, canonical_words)

            md.append(
                f"| {label} | {len(segs)} | "
                f"**{wer_stats['wer']*100:.2f} %** | "
                f"{wer_stats['substitutions']}/"
                f"{wer_stats['insertions']}/"
                f"{wer_stats['deletions']} | "
                f"**{f1_stats['f1']:.3f}** | "
                f"{f1_stats['precision']:.2f} / {f1_stats['recall']:.2f} |"
            )

            # Aggregate
            total_errs = (wer_stats['substitutions']
                          + wer_stats['insertions']
                          + wer_stats['deletions'])
            overall[key]["wer_num"] += total_errs
            overall[key]["wer_den"] += wer_stats['ref_words']
            overall[key]["ent_tp"] += f1_stats['tp']
            overall[key]["ent_fp"] += f1_stats['fp']
            overall[key]["ent_fn"] += f1_stats['fn']
        md.append("")

    # Combined / corpus-level summary
    md.append("---")
    md.append("")
    md.append("## Combined (both halves)")
    md.append("")
    md.append("| Track | Corpus WER | Entity-F1 | Entity P / R |")
    md.append("|---|---|---|---|")
    for key, label in (("soccernet", "SoccerNet bundled"),
                       ("fasterv3",  "faster-whisper-v3")):
        o = overall[key]
        wer = o["wer_num"] / o["wer_den"] if o["wer_den"] else 0
        p = o["ent_tp"] / max(1, o["ent_tp"] + o["ent_fp"])
        r = o["ent_tp"] / max(1, o["ent_tp"] + o["ent_fn"])
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        md.append(f"| **{label}** | **{wer*100:.2f} %** | **{f1:.3f}** | "
                  f"{p:.2f} / {r:.2f} |")

    # Compute deltas
    soc_wer = overall["soccernet"]["wer_num"] / max(1, overall["soccernet"]["wer_den"])
    v3_wer  = overall["fasterv3"]["wer_num"] / max(1, overall["fasterv3"]["wer_den"])
    soc_p = overall["soccernet"]["ent_tp"] / max(1, overall["soccernet"]["ent_tp"] + overall["soccernet"]["ent_fp"])
    soc_r = overall["soccernet"]["ent_tp"] / max(1, overall["soccernet"]["ent_tp"] + overall["soccernet"]["ent_fn"])
    soc_f = 2*soc_p*soc_r/(soc_p+soc_r) if (soc_p+soc_r) else 0
    v3_p = overall["fasterv3"]["ent_tp"] / max(1, overall["fasterv3"]["ent_tp"] + overall["fasterv3"]["ent_fp"])
    v3_r = overall["fasterv3"]["ent_tp"] / max(1, overall["fasterv3"]["ent_tp"] + overall["fasterv3"]["ent_fn"])
    v3_f = 2*v3_p*v3_r/(v3_p+v3_r) if (v3_p+v3_r) else 0

    md.append("")
    md.append(f"- **WER delta**: {(v3_wer - soc_wer)*100:+.2f} pp "
              f"({(v3_wer - soc_wer)/soc_wer*100:+.1f} % rel) "
              f"— faster-whisper-v3 vs SoccerNet bundled")
    md.append(f"- **Entity-F1 delta**: {(v3_f - soc_f):+.3f} abs "
              f"({(v3_f - soc_f)/soc_f*100:+.1f} % rel)")
    md.append("")

    md.append("## Interpretation")
    md.append("")
    md.append("If WER goes DOWN (negative delta) with faster-whisper-v3, "
              "the regeneration improved transcription quality before our "
              "cleaning pipeline ever runs — meaning some of the gain we "
              "attribute to our pipeline is actually the better Whisper "
              "engine, not the post-processing.")
    md.append("")
    md.append("If Entity-F1 goes UP, the larger model + better decoding "
              "parameters resolved more named entities correctly. This "
              "matters when comparing our cleaning pipeline against the "
              "raw input it actually ingests (`1_asr_v3.json`), not against "
              "the SoccerNet-bundled output.")
    md.append("")
    md.append("**Takeaway for thesis**: report both the Whisper-engine "
              "delta and the cleaning-pipeline delta separately so the "
              "two effects don't get conflated.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
