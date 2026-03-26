#!/usr/bin/env python3
"""
Transcribe a full football match broadcast into ASR JSON files for the pipeline.

Downloads the video, uses ffmpeg to cut out each half, then runs Whisper on
each half separately. Whisper only sees the match commentary — pre-match,
halftime, and post-match content are cut out before transcription starts.

Forzasys/Allsvenskan broadcasts include pre-match show, halftime, and post-match,
so you must specify where each half starts and ends in the broadcast.

Usage:
    python transcribe_match.py "https://api.forzasys.com/.../Manifest.m3u8" \\
        --league sweden_allsvenskan --season 2024-2025 \\
        --match "2025-11-09 Malmö FF 2 - 1 GAIS" \\
        --h1-start 5:00  --h1-end 51:06 \\
        --h2-start 1:07:05 --h2-end 1:59:38 \\
        --language sv

    # Local audio file (skips download):
    python transcribe_match.py "C:/path/to/audio.mp3" \\
        --league ... --h1-start 5:00 --h1-end 51:06 ...

    # Don't know the timestamps yet? Transcribe first, inspect, then split:
    python transcribe_match.py "https://..." --transcribe-only -o full.json --language sv
    # Open full.json, find where each half starts/ends, then:
    python transcribe_match.py full.json --from-json \\
        --h1-start 5:00 --h1-end 51:06 --h2-start 1:07:05 --h2-end 1:59:38 \\
        --league sweden_allsvenskan --season 2024-2025 --match "..."

Prerequisites:
    pip install openai-whisper yt-dlp
    # ffmpeg:
    #   Windows: winget install ffmpeg   (or choco install ffmpeg)
    #   Mac:     brew install ffmpeg
    #   Linux:   sudo apt install ffmpeg

Resuming after a crash:
    Trimmed half audio files are kept in whisper_cache/<cache-key>/.
    Re-run the same command — already-trimmed files are reused automatically.
    Delete whisper_cache/ to start from scratch.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_WHISPER_MODEL = "medium"
DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parent / "path" / "to" / "SoccerNet" / "caption-2023"
)
CACHE_DIR = Path(__file__).resolve().parent / "whisper_cache"


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def parse_timestamp(ts: str) -> float:
    """Parse '5:00', '51:06', or '1:07:05' into seconds."""
    parts = ts.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(ts)
    except ValueError:
        print(f"Error: invalid timestamp '{ts}'. Use MM:SS or H:MM:SS.")
        sys.exit(1)


def format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ── Dependency check ──────────────────────────────────────────────────────────

def check_dependencies():
    missing = []
    try:
        import whisper  # noqa: F401
    except ImportError:
        missing.append("openai-whisper")
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        missing.append("yt-dlp")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("WARNING: ffmpeg not found.")
        print("  Windows: winget install ffmpeg   Mac: brew install ffmpeg\n")
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print(f"Install: pip install {' '.join(missing)}")
        sys.exit(1)


# ── Audio helpers ─────────────────────────────────────────────────────────────

def download_audio(url: str, out_dir: Path) -> Path:
    """Download URL as MP3 using yt-dlp."""
    print(f"  Downloading audio from:\n    {url}")
    cmd = [
        "yt-dlp",
        "--extract-audio", "--audio-format", "mp3",
        "--audio-quality", "3",
        "--no-playlist", "--hls-prefer-native", "--no-check-certificates",
        "-o", str(out_dir / "audio.%(ext)s"),
        url,
    ]
    subprocess.run(cmd, check=True)
    for ext in (".mp3", ".m4a", ".opus", ".wav"):
        p = out_dir / f"audio{ext}"
        if p.exists():
            print(f"  Audio: {p}  ({p.stat().st_size / 1024 / 1024:.1f} MB)")
            return p
    print("Error: yt-dlp produced no audio file.")
    sys.exit(1)


def trim_audio(src: Path, dst: Path, start_ts: str, end_ts: str) -> None:
    """
    Cut src to [start_ts, end_ts] and write to dst.

    Uses output-side seeking (-ss after -i) for timestamp accuracy.
    The resulting file starts at 0:00 so Whisper timestamps are
    automatically relative to the half start.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ss", start_ts,
        "-to", end_ts,
        "-c", "copy",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr.decode()}")
        sys.exit(1)
    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"  Trimmed: {dst.name}  ({start_ts} → {end_ts}, {size_mb:.1f} MB)")


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: str | None,
    label: str = "",
) -> list[dict]:
    """
    Run Whisper on a single audio file.

    verbose=True shows [00:01.000 --> 00:04.500]  text lines so you can
    see Whisper's progress in real time.
    """
    import whisper

    print(f"  Loading Whisper '{model_name}' ...")
    model = whisper.load_model(model_name)

    opts: dict = {"verbose": True}
    if language:
        opts["language"] = language
        print(f"  Language: {language} (forced)")
    else:
        print("  Language: auto-detect")

    size_mb = audio_path.stat().st_size / 1024 / 1024
    print(f"  Transcribing {label}{audio_path.name} ({size_mb:.1f} MB) — this takes a while on CPU ...")

    try:
        result = model.transcribe(str(audio_path), **opts)
    except Exception:
        print("\n[ERROR] Whisper crashed. Full traceback:")
        traceback.print_exc()
        sys.exit(1)

    segments = result.get("segments", [])
    detected = result.get("language", "?")
    print(f"  Done. Detected language: {detected},  segments: {len(segments)}")
    return segments


# ── Output helpers ────────────────────────────────────────────────────────────

def segments_to_asr(whisper_segments: list[dict]) -> dict:
    """Convert Whisper segment list → SoccerNet pipeline format."""
    out: dict[str, list] = {}
    idx = 0
    for seg in whisper_segments:
        text = seg["text"].strip()
        if not text:
            continue
        out[str(idx)] = [round(seg["start"], 2), round(seg["end"], 2), text]
        idx += 1
    return {"segments": out}


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"  Written: {path}  ({len(data.get('segments', {}))} segments)")


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://", "ftp://"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe football match broadcast → SoccerNet ASR JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source", help="URL or local audio/JSON file")

    # Half boundaries in the broadcast (MM:SS or H:MM:SS)
    parser.add_argument("--h1-start", metavar="TIME", help="First half start in broadcast, e.g. 5:00")
    parser.add_argument("--h1-end",   metavar="TIME", help="First half end in broadcast, e.g. 51:06")
    parser.add_argument("--h2-start", metavar="TIME", help="Second half start, e.g. 1:07:05")
    parser.add_argument("--h2-end",   metavar="TIME", help="Second half end, e.g. 1:59:38")

    # Output location
    parser.add_argument("--league",  help="League folder, e.g. sweden_allsvenskan")
    parser.add_argument("--season",  help="Season folder, e.g. 2024-2025")
    parser.add_argument("--match",   help="Match folder name")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)

    # Special modes
    parser.add_argument("--transcribe-only", action="store_true",
                        help="Transcribe full audio without splitting. "
                             "Use when you don't know half timestamps yet.")
    parser.add_argument("--from-json", action="store_true",
                        help="Source is a JSON from --transcribe-only. "
                             "Splits into halves without re-transcribing.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path for --transcribe-only (default: full_transcription.json)")
    parser.add_argument("--cache-key", default=None,
                        help="Cache folder name under whisper_cache/ for resuming")

    # Whisper
    parser.add_argument("--model", default=DEFAULT_WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help=f"Whisper model (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--language", default=None,
                        help="Force language, e.g. 'sv', 'en', 'de'")

    args = parser.parse_args()
    check_dependencies()

    cache_key = args.cache_key or Path(args.source).stem.replace(" ", "_")[:40]
    work_dir = CACHE_DIR / cache_key
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── --transcribe-only: no splitting, just dump everything ────────
    if args.transcribe_only:
        if is_url(args.source):
            tmpdir = Path(tempfile.mkdtemp(prefix="whisper_"))
            audio_path = download_audio(args.source, tmpdir)
        else:
            audio_path = Path(args.source)

        segs = transcribe_audio(audio_path, args.model, args.language)
        full = {"language": args.language or "auto", "segments": [
            {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
            for s in segs if s["text"].strip()
        ]}
        out = args.output or Path("full_transcription.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(full, f, indent=4, ensure_ascii=False)
        total = len(full["segments"])
        duration = full["segments"][-1]["end"] if total else 0
        print(f"\nSaved: {out}  ({total} segments, {format_ts(duration)})")
        print(f"\nNow find the half boundaries, then run:")
        print(f"  python transcribe_match.py {out} --from-json \\")
        print(f"      --h1-start MM:SS --h1-end MM:SS --h2-start H:MM:SS --h2-end H:MM:SS \\")
        print(f"      --league LEAGUE --season SEASON --match \"MATCH NAME\"")
        return

    # ── Both normal and --from-json modes need half timestamps ───────
    if not all([args.h1_start, args.h1_end, args.h2_start, args.h2_end]):
        parser.error("--h1-start, --h1-end, --h2-start, --h2-end are all required\n"
                     "       (use --transcribe-only if you need to find the timestamps first)")
    if not all([args.league, args.season, args.match]):
        parser.error("--league, --season, and --match are required")

    commentary_dir = (
        args.dataset_root / args.league / args.season / args.match / "commentary_data"
    )

    # ── --from-json: split a previous full transcription ─────────────
    if args.from_json:
        json_path = Path(args.source)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        raw_segs = data["segments"]  # list of {start, end, text}

        def _split_json(segs, t_start, t_end, offset):
            out: dict[str, list] = {}
            idx = 0
            for s in segs:
                mid = (s["start"] + s["end"]) / 2
                if t_start <= mid <= t_end:
                    text = s["text"].strip()
                    if text:
                        out[str(idx)] = [
                            round(s["start"] - offset, 2),
                            round(s["end"] - offset, 2),
                            text,
                        ]
                        idx += 1
            return {"segments": out}

        h1s = parse_timestamp(args.h1_start)
        h1e = parse_timestamp(args.h1_end)
        h2s = parse_timestamp(args.h2_start)
        h2e = parse_timestamp(args.h2_end)

        write_json(_split_json(raw_segs, h1s, h1e, h1s), commentary_dir / "1_asr.json")
        write_json(_split_json(raw_segs, h2s, h2e, h2s), commentary_dir / "2_asr.json")
        print(f"\nDone! → {commentary_dir}")
        print(f'Run: python run_pipeline.py --match "{args.match}" --dry-run')
        return

    # ── Normal mode: download → trim → transcribe ─────────────────────

    # Step 1: get full audio
    if is_url(args.source):
        full_audio = work_dir / "audio.mp3"
        if full_audio.exists():
            print(f"  Reusing cached download: {full_audio} ({full_audio.stat().st_size/1024/1024:.1f} MB)")
        else:
            download_audio(args.source, work_dir)
            # rename whatever yt-dlp produced to audio.mp3
            for ext in (".mp3", ".m4a", ".opus", ".wav"):
                p = work_dir / f"audio{ext}"
                if p.exists() and p != full_audio:
                    p.rename(full_audio)
                    break
    else:
        full_audio = Path(args.source)
        if not full_audio.exists():
            print(f"Error: file not found: {full_audio}")
            sys.exit(1)

    halves = [
        (1, args.h1_start, args.h1_end, commentary_dir / "1_asr.json"),
        (2, args.h2_start, args.h2_end, commentary_dir / "2_asr.json"),
    ]

    for half_num, t_start, t_end, out_path in halves:
        print(f"\n{'─'*60}")
        print(f"  Half {half_num}:  {t_start} → {t_end}")
        print(f"{'─'*60}")

        # Step 2: trim to this half (cached)
        trimmed = work_dir / f"half{half_num}.mp3"
        if trimmed.exists():
            print(f"  Reusing trimmed audio: {trimmed.name} ({trimmed.stat().st_size/1024/1024:.1f} MB)")
        else:
            trim_audio(full_audio, trimmed, t_start, t_end)

        # Step 3: transcribe — Whisper sees timestamps from 0:00
        segs = transcribe_audio(trimmed, args.model, args.language, label=f"Half {half_num} — ")

        # Step 4: save
        write_json(segments_to_asr(segs), out_path)

    print(f"\n{'='*60}")
    print(f"Done! Files written to:\n  {commentary_dir}")
    print(f"\nRun cleaning pipeline:")
    print(f'  python run_pipeline.py --match "{args.match}" --dry-run')


if __name__ == "__main__":
    main()
