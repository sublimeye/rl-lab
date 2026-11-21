"""Prints the numbers quoted in post 10 from the exported site data.

uv run experiments/010-drone/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).parent.parent.parent / "site" / "public" / "data" / "010-drone"


def main() -> None:
    ev = json.loads((DATA / "eval.json").read_text())
    print(f"test courses: {ev['seeds']}")
    print("\nfinal pilots, landed of 200 (crashed/water/lost/timeout), mean time, touchdown speed,")
    print("mean stick movement per update, share of stick readings at an end")
    for label, rows in ev["table"].items():
        print(label)
        for w, s in rows.items():
            fails = f"{s['crashed']}/{s['water']}/{s['lost']}/{s['timeout']}"
            print(
                f"  {w:<6} {s['landed']:3d} ({fails})  {s['time']} s  {s['touch']} m/s"
                f"  move {s['change']:.3f}  pinned {s['pinned']:.0%}"
                f"  hard landings {s.get('hard')} at {s.get('hard_vy')} m/s down"
            )
    print("\ncurves (calm, landed of 200 per seed; minutes since the start of training)")
    for group, runs in ev["curves"].items():
        print(group)
        for run, pts in runs.items():
            cells = [f"{p['steps'] / 1e6:g}M:{p['landed']}@{p['seconds'] / 60:.1f}m" for p in pts]
            print(f"  {run}: " + "  ".join(cells))
    tr = json.loads((DATA / "traces.json").read_text())
    print(f"\ntraces on course {tr['seed']}")
    for p in tr["pilots"]:
        print(
            f"  {p['label']:<28} {p['outcome']} {p['time']} s  move {p['change']:.3f}"
            f"  pinned {p['pinned']:.0%}"
        )
    print("\nfiles")
    for f in sorted(DATA.glob("*.json")):
        print(f"  {f.name}: {f.stat().st_size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
