"""Convert pipeline cleaned output (schema-2 dict) to frontend's
kamp.json (schema-1 list with 'Part 2' marker), then update
matches/registry.json so the frontend can ingest + serve the match.

Pipeline output:
  cleaned_data/.../1_asr_v3_nbest_cleaned.json   (schema-2 dict)
  cleaned_data/.../2_asr_v3_nbest_cleaned.json

Frontend input:
  frontend/forzasearch-final/matches/<id>/kamp.json
  Format:
      Part 1
      {"segments": {"0": [start, end, "text"], ...}}
      Part 2
      {"segments": {"0": [start, end, "text"], ...}}

Usage:
  python tools/export_to_frontend.py \\
      --match "Chelsea 1 - 2 Liverpool" \\
      --frontend frontend/forzasearch-final \\
      --id chelsea-liverpool-2016 \\
      --title "Chelsea 1-2 Liverpool" \\
      --subtitle "Premier League 2016-09-16"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_match(name: str, root: Path) -> Path | None:
    for league in root.iterdir():
        if not league.is_dir():
            continue
        for season in league.iterdir():
            if not season.is_dir():
                continue
            for m in season.iterdir():
                if m.is_dir() and name.lower() in m.name.lower():
                    return m
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--match", required=True)
    p.add_argument("--cleaned-root", type=Path, default=Path("cleaned_data/caption-2023"))
    p.add_argument("--variant", default="_v3_nbest")
    p.add_argument("--frontend", type=Path, required=True,
                   help="Path to frontend/forzasearch-final")
    p.add_argument("--id", required=True, help="match id slug for registry/folder")
    p.add_argument("--title", required=True)
    p.add_argument("--subtitle", default="")
    p.add_argument("--video-url", default=None,
                   help="Optional HLS .m3u8 URL for video playback")
    args = p.parse_args()

    cleaned_dir = _find_match(args.match, args.cleaned_root)
    if not cleaned_dir:
        print(f"cleaned dir not found for {args.match!r}", file=sys.stderr)
        return 1

    frontend_match_dir = args.frontend / "matches" / args.id
    frontend_match_dir.mkdir(parents=True, exist_ok=True)
    kamp_path = frontend_match_dir / "kamp.json"

    parts: list[str] = []
    for half in (1, 2):
        cp = cleaned_dir / "commentary_data" / f"{half}_asr{args.variant}_cleaned.json"
        if not cp.exists():
            print(f"  [skip half {half}] {cp} missing", file=sys.stderr)
            continue
        d = json.load(open(cp, encoding="utf-8"))
        # Convert dict-of-{global_id, start_time, end_time, text, ...}
        # to schema-1 dict-of-[start, end, text]
        out_segs = {}
        for sid, seg in d["segments"].items():
            text = seg.get("text", "")
            start = seg.get("start_time", 0.0)
            end = seg.get("end_time", 0.0)
            out_segs[sid] = [round(start, 2), round(end, 2), text]
        block = json.dumps({"segments": out_segs}, indent=4, ensure_ascii=False)
        parts.append(f"Part {half}\n\n{block}")

    kamp_path.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"  Wrote {kamp_path} ({sum(p.count(chr(10))+1 for p in parts)} lines)")

    # Update registry.json
    reg_path = args.frontend / "matches" / "registry.json"
    if not reg_path.exists():
        registry = {"matches": []}
    else:
        registry = json.load(open(reg_path, encoding="utf-8"))

    matches = registry.setdefault("matches", [])
    # Replace existing entry with same id, or append
    new_entry = {
        "id": args.id,
        "title": args.title,
        "subtitle": args.subtitle,
        "folder": args.id,
    }
    if args.video_url:
        new_entry["video"] = {
            "type": "hls",
            "parts": {"1": args.video_url},
        }

    matches[:] = [m for m in matches if m.get("id") != args.id] + [new_entry]
    json.dump(registry, open(reg_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"  Updated {reg_path} ({len(matches)} entries)")

    print()
    print("Next steps:")
    print("  1. Make sure Elasticsearch is running:")
    print("       docker run -d --name es -p 9200:9200 \\\\")
    print("         -e \"discovery.type=single-node\" \\\\")
    print("         -e \"xpack.security.enabled=false\" \\\\")
    print("         docker.elastic.co/elasticsearch/elasticsearch:8.17.0")
    print("  2. cd frontend/forzasearch-final && npm run ingest -- --force")
    print("  3. cd frontend/forzasearch-final && npm run dev")
    print("  4. Open http://localhost:3000 and pick the match in the dropdown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
