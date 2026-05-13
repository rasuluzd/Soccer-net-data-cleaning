"""Download a single SoccerNet match (video) to ``path/to/SoccerNet/``.

Defaults to the Chelsea–Liverpool 2016-09-16 match at 224p (~600MB per
halftime — enough to extract audio for re-transcription).

Usage:
    # Most secure: set the password as env var, then run
    set SOCCERNET_PASSWORD=...                     # PowerShell: $env:SOCCERNET_PASSWORD = "..."
    python tools/download_soccernet_match.py

    # Override game (anything else from the SoccerNet catalogue):
    python tools/download_soccernet_match.py \\
        --game "england_epl/2014-2015/2015-02-22 - 19-15 Southampton 0 - 2 Liverpool"

    # Higher resolution (720p ≈ 2GB/half, full HD ≈ 4-6GB/half — overkill
    # for audio extraction but useful if you also want video frames):
    python tools/download_soccernet_match.py --files 1_720p.mkv 2_720p.mkv
    python tools/download_soccernet_match.py --files 1.mkv 2.mkv

    # Audio extraction after download:
    python tools/extract_audio_from_url.py \\
        --url "path/to/SoccerNet/.../1_224p.mkv" \\
        --start 0:00 --end 0:55:00 \\
        --output whisper_cache/audio/chelsea/half1.mp3
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_GAME = (
    "england_epl/2016-2017/"
    "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool"
)
DEFAULT_FILES = ["1_224p.mkv", "2_224p.mkv"]
DEFAULT_LOCAL_DIR = _REPO_ROOT / "path" / "to" / "SoccerNet"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--game", default=DEFAULT_GAME,
                   help=f"Game directory under SoccerNet (default: {DEFAULT_GAME})")
    p.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                   help=f"Files to download (default: {' '.join(DEFAULT_FILES)})")
    p.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR,
                   help=f"Where to save (default: {DEFAULT_LOCAL_DIR})")
    p.add_argument("--password", default=None,
                   help="SoccerNet NDA password (else read from $SOCCERNET_PASSWORD or prompted)")
    args = p.parse_args()

    pw = args.password or os.environ.get("SOCCERNET_PASSWORD") or ""
    if not pw:
        pw = getpass.getpass("SoccerNet password: ").strip()
    if not pw:
        print("ERROR: no password provided", file=sys.stderr)
        return 2

    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        print("ERROR: SoccerNet not installed. Run: pip install SoccerNet", file=sys.stderr)
        return 2

    args.local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Local dir : {args.local_dir}")
    print(f"Game      : {args.game}")
    print(f"Files     : {args.files}")
    print()

    dl = SoccerNetDownloader(LocalDirectory=str(args.local_dir))
    dl.password = pw
    try:
        dl.downloadGame(game=args.game, files=args.files)
    except Exception as e:
        print(f"\nERROR: download failed: {e}", file=sys.stderr)
        # Common causes: wrong password, game path typo, network issue
        return 1

    print("\nDone. Verify the files exist:")
    target = args.local_dir / "caption-2023" / args.game
    legacy = args.local_dir / args.game
    for d in (target, legacy):
        if d.exists():
            print(f"  {d}")
            for f in sorted(d.iterdir()):
                size_mb = f.stat().st_size / 1024 / 1024 if f.is_file() else 0
                print(f"    {f.name}  ({size_mb:.1f} MB)" if size_mb else f"    {f.name}/")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
