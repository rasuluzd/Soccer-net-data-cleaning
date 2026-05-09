"""
Extract audio segments from an HLS (m3u8) URL into local MP3 files using
the ffmpeg binary bundled with imageio-ffmpeg (no system ffmpeg required,
no admin needed).

Usage:
    python tools/extract_audio_from_url.py \\
        --url "https://api.forzasys.com/.../Manifest.m3u8" \\
        --start 00:05:00 --end 00:53:42 \\
        --output whisper_cache/audio/half1.mp3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg


def hms_to_seconds(t: str) -> float:
    parts = [float(p) for p in t.split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"unsupported timestamp: {t}")


def extract(url: str, start: str, end: str, output: Path,
            sample_rate: int = 16000, channels: int = 1, bitrate: str = "64k") -> None:
    duration_s = hms_to_seconds(end) - hms_to_seconds(start)
    if duration_s <= 0:
        raise ValueError(f"end {end} <= start {start}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg, "-y",
        "-ss", start,        # fast input seek
        "-i", url,
        "-t", f"{duration_s:.3f}",
        "-vn",                # drop video
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        "-loglevel", "warning",
        "-stats",
        str(output),
    ]
    print(f"[extract] {start} ->{end} ({duration_s:.1f}s) ->{output}")
    print(f"[ffmpeg] {' '.join(cmd[:1])} ... {output.name}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)
    if output.exists():
        size_mb = output.stat().st_size / 1e6
        print(f"[done] wrote {output} ({size_mb:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--start", required=True, help="HH:MM:SS or seconds")
    p.add_argument("--end", required=True, help="HH:MM:SS or seconds")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--bitrate", type=str, default="64k")
    args = p.parse_args()
    extract(args.url, args.start, args.end, args.output,
            args.sample_rate, args.channels, args.bitrate)


if __name__ == "__main__":
    main()
