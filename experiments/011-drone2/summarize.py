"""Prints the numbers quoted in post 11 from the exported site data.

uv run experiments/011-drone2/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).parent.parent.parent / "site" / "public" / "data" / "011-drone2"


def line(label: str, s: dict) -> str:
    fails = " ".join(f"{k} {s[k]}" for k in ("crashed", "water", "lost", "timeout", "bird"))
    return (
        f"  {label:<14} landed {s['landed']:3d}/{s['n']}  {fails}  time {s['time']}"
        f"  touch {s['touch']}  move {s['change']}  pinned {s['pinned']}"
        f"\n  {'':<14} by birds 0-3 {s['by_birds']}  pad moving {s['pad_moving']}"
        f" still {s['pad_still']}  battery <15% {s['low_battery']}"
    )


def main() -> None:
    ev = json.loads((DATA / "eval.json").read_text())
    print("raw simulator speed (steps/s)")
    for k, v in ev["bench"].items():
        print(f"  {k:<48} {v:>10,.0f}")
    print("\ntraining runs (post 10's task, 5M steps)")
    for run, s in ev["speed"].items():
        print(
            f"  {run:<11} {s['sim']:<6} envs {s['envs']:<5} threads {s['threads']}"
            f"  {s['steps']:>11,} steps in {s['seconds'] / 60:.2f} min"
            f"  ({s['steps'] / s['seconds']:,.0f} steps/s)"
        )
    print("\ncurves: landed of 200 at each checkpoint (steps, minutes)")
    for group, runs in ev["curves"].items():
        for run, pts in runs.items():
            cells = [f"{p['steps'] / 1e6:g}M:{p['landed']}@{p['seconds'] / 60:.2f}" for p in pts]
            print(f"  {group}/{run}: " + "  ".join(cells))
    print("\nfinal pilots on the 200 test courses (rough weather is the training condition)")
    for label, rows in ev["table"].items():
        print(f"{label} ({rows['task']})")
        for w in ("rough", "calm"):
            print(line(w, rows[w]))
    print("\nblindfold: the exact-position pilot with its bird inputs replaced by 'no bird'")
    for run, b in ev["blindfold"].items():
        s, x = b["seeing"], b["blindfolded"]
        print(
            f"  {run:<10} seeing: landed {s['landed']} bird {s['bird']}"
            f"   blindfolded: landed {x['landed']} bird {x['bird']}"
        )
        print(f"  {'':<10} warning before each hit (s, from 3 m): {b['warnings']}")
    print("\nCAPS terms at the end of training [temporal, spatial]")
    for run, v in ev["caps"].items():
        print(f"  {run}: {v}")
    tr = json.loads((DATA / "traces.json").read_text())
    print(f"\ntraces on course {tr['seed']}")
    for p in tr["pilots"]:
        print(
            f"  {p['label']:<26} {p['outcome']} {p['time']} s  move {p['change']}"
            f"  pinned {p['pinned']}"
        )
    print("\nfiles")
    for f in sorted(DATA.glob("*.json")):
        print(f"  {f.name}: {f.stat().st_size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
