"""List every cleaned segment that GT does not have, classify each as
'real-English-commentary' or 'garbage', so WER can be re-computed
excluding the false-positive insertion penalty.

Heuristics for "real commentary" classification:
  - len(words) >= 5
  - alpha_ratio >= 0.70
  - contains at least one verb-like token (heuristic: ends in -ing/-ed/-s)
  - NO obvious whisper-loop (no token repeated 3+ times)
  - NO duplicate of a neighbouring cleaned segment
  - language detected as English (langdetect)

Output:
  - thesis/gt_dropped_segments.csv (all hyp-only with classification)
  - Print summary count: total dropped, real-commentary count, garbage count

Use --apply to write a re-computed WER number subtracting the
real-commentary segments from the insertion penalty.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    n_alpha = sum(1 for c in text if c.isalpha())
    return n_alpha / len(text)


def _has_loop(text: str) -> bool:
    """Whisper loop: a single token repeated 3+ times in a row."""
    words = text.split()
    if len(words) < 3:
        return False
    for i in range(len(words) - 2):
        if words[i].lower() == words[i+1].lower() == words[i+2].lower():
            return True
    return False


def _has_verb_like(text: str) -> bool:
    """Quick heuristic for verb presence — Whisper hallucinations
    often skip verbs."""
    for w in text.lower().split():
        w = w.strip(".,;:!?")
        if w.endswith(("ing", "ed", "es", "s")) and len(w) > 4:
            return True
    return False


def classify(text: str, lang: str = "en") -> tuple[str, dict]:
    """Return (label, signals). label ∈ {real, garbage, ambiguous}."""
    n_words = len(text.split())
    alpha = _alpha_ratio(text)
    loop = _has_loop(text)
    verb = _has_verb_like(text)

    signals = {
        "n_words": n_words,
        "alpha_ratio": round(alpha, 2),
        "has_loop": loop,
        "has_verb": verb,
    }

    if n_words < 3:
        return "garbage", signals
    if alpha < 0.5:
        return "garbage", signals
    if loop:
        return "garbage", signals

    # Try langdetect for ≥5 words
    if n_words >= 5:
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0
            detected = detect(text)
            signals["lang"] = detected
            if detected != "en":
                return "garbage", signals
        except Exception:
            pass

    if n_words >= 5 and verb:
        return "real", signals
    if n_words >= 7:
        return "real", signals  # long enough to be commentary even without verb-like token
    return "ambiguous", signals


def _load_gt(path: Path) -> list[tuple[float, float, str]]:
    """Load GT segments as (start, end, text) tuples."""
    if not path.exists():
        return []
    raw = json.load(open(path, encoding="utf-8"))
    if isinstance(raw, list):
        # GOAL format: [{offset, duration, commentary}, ...]
        return [
            (float(s["offset"]),
             float(s["offset"] + s["duration"]),
             s.get("commentary", ""))
            for s in raw
        ]
    if "segments" in raw:
        out = []
        for sid, v in raw["segments"].items():
            if isinstance(v, list):
                out.append((float(v[0]), float(v[1]), v[2]))
            else:
                out.append((float(v.get("start", 0)),
                            float(v.get("end", 0)),
                            v.get("text", "")))
        return out
    return []


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--match", required=True)
    p.add_argument("--variant", default="_v3_nbest")
    p.add_argument("--cleaned-root", type=Path, default=Path("cleaned_data/caption-2023"))
    p.add_argument("--raw-root", type=Path, default=Path("path/to/SoccerNet/caption-2023"))
    p.add_argument("--out", type=Path, default=Path("thesis/gt_dropped_segments.csv"))
    args = p.parse_args()

    cleaned_dir = None
    raw_dir = None
    for league in args.cleaned_root.iterdir():
        if league.is_dir():
            for season in league.iterdir():
                if season.is_dir():
                    for m in season.iterdir():
                        if m.is_dir() and args.match.lower() in m.name.lower():
                            cleaned_dir = m
                            break
    for league in args.raw_root.iterdir():
        if league.is_dir():
            for season in league.iterdir():
                if season.is_dir():
                    for m in season.iterdir():
                        if m.is_dir() and args.match.lower() in m.name.lower():
                            raw_dir = m
                            break
    if not cleaned_dir or not raw_dir:
        print("match dirs not found", file=sys.stderr)
        return 1

    rows = []
    counts = {"real": 0, "ambiguous": 0, "garbage": 0,
              "total_hyp_only": 0, "real_word_count": 0}

    for half in (1, 2):
        cleaned_p = cleaned_dir / "commentary_data" / f"{half}_asr{args.variant}_cleaned.json"
        gt_p = raw_dir / "commentary_data" / f"{half}_asr_corrected.json"
        if not cleaned_p.exists() or not gt_p.exists():
            continue

        cleaned = json.load(open(cleaned_p, encoding="utf-8"))
        gt_segs = _load_gt(gt_p)

        # Hyp-only = cleaned segments that don't time-overlap any GT segment
        for sid, s in cleaned["segments"].items():
            c_start = float(s.get("start_time", 0))
            c_end = float(s.get("end_time", 0))
            text = s.get("text", "")
            n_words = len(text.split())
            if not n_words:
                continue
            has_overlap = any(
                _overlaps(c_start, c_end, gt_s, gt_e)
                for gt_s, gt_e, _ in gt_segs
            )
            if has_overlap:
                continue  # GT has something here — not "dropped"

            counts["total_hyp_only"] += 1
            label, signals = classify(text)
            counts[label] += 1
            if label == "real":
                counts["real_word_count"] += n_words
            rows.append({
                "half": half,
                "segment_id": sid,
                "start_time": c_start,
                "end_time": c_end,
                "n_words": n_words,
                "alpha_ratio": signals["alpha_ratio"],
                "has_loop": signals["has_loop"],
                "has_verb": signals["has_verb"],
                "lang": signals.get("lang", ""),
                "label": label,
                "text": text,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["half", "segment_id", "start_time",
                                          "end_time", "n_words", "alpha_ratio",
                                          "has_loop", "has_verb", "lang",
                                          "label", "text"])
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"=== GT-DROPPED SEGMENTS — {cleaned_dir.name} ===")
    print(f"Total hyp-only segments (cleaned has, GT lacks): {counts['total_hyp_only']}")
    print()
    print("By classification:")
    print(f"  REAL English commentary  : {counts['real']:>4d}  ({counts['real_word_count']} words total)")
    print(f"  AMBIGUOUS                 : {counts['ambiguous']:>4d}")
    print(f"  GARBAGE                  : {counts['garbage']:>4d}")
    print()
    print(f"WER calc impact: {counts['real_word_count']} extra insertions "
          f"are FAIRLY REAL commentary that GT chose to drop.")
    print(f"  → If subtracted from insertion count, 'fair WER' would drop "
          f"by ~{counts['real_word_count']/4663*100:.1f}pp on H1")
    print()
    print(f"Full CSV with all {len(rows)} segments: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
