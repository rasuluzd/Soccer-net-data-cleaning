"""Export RAW (un-cleaned) Whisper output as a separate frontend match.

Mirror of tools/export_to_frontend.py but reads the *raw* schema-2/3 ASR
JSON instead of cleaned_data/. Lets us index raw vs cleaned side-by-side
in Elasticsearch and compare search quality directly.

Usage:
  python tools/export_raw_to_frontend.py \\
      --match "Chelsea 1 - 2 Liverpool" \\
      --variant _v3_nbest \\
      --frontend frontend/forzasearch-final \\
      --id chelsea-liverpool-2016-RAW \\
      --title "Chelsea 1-2 Liverpool (RAW Whisper)" \\
      --subtitle "Premier League 2016-09-16 — uncleaned"
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
    p.add_argument("--raw-root", type=Path,
                   default=Path("path/to/SoccerNet/caption-2023"))
    p.add_argument("--variant", default="_v3_nbest")
    p.add_argument("--frontend", type=Path, required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--subtitle", default="")
    args = p.parse_args()

    raw_dir = _find_match(args.match, args.raw_root)
    if not raw_dir:
        print(f"raw dir not found", file=sys.stderr)
        return 1

    out_dir = args.frontend / "matches" / args.id
    out_dir.mkdir(parents=True, exist_ok=True)
    kamp_path = out_dir / "kamp.json"

    parts: list[str] = []
    for half in (1, 2):
        rp = raw_dir / "commentary_data" / f"{half}_asr{args.variant}.json"
        if not rp.exists():
            print(f"  [skip half {half}] {rp} missing", file=sys.stderr)
            continue
        d = json.load(open(rp, encoding="utf-8"))
        out = {}
        for sid, seg in d["segments"].items():
            text = seg.get("text") if isinstance(seg, dict) else (seg[2] if len(seg) >= 3 else "")
            start = seg.get("start") if isinstance(seg, dict) else seg[0]
            end = seg.get("end") if isinstance(seg, dict) else seg[1]
            out[sid] = [round(float(start), 2), round(float(end), 2), str(text)]
        block = json.dumps({"segments": out}, indent=4, ensure_ascii=False)
        parts.append(f"Part {half}\n\n{block}")

    kamp_path.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"  Wrote RAW {kamp_path} ({sum(p.count(chr(10))+1 for p in parts)} lines)")

    reg_path = args.frontend / "matches" / "registry.json"
    registry = json.load(open(reg_path, encoding="utf-8"))
    matches = registry.setdefault("matches", [])
    new_entry = {
        "id": args.id, "title": args.title, "subtitle": args.subtitle,
        "folder": args.id,
    }
    matches[:] = [m for m in matches if m.get("id") != args.id] + [new_entry]
    json.dump(registry, open(reg_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"  Updated {reg_path} ({len(matches)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
