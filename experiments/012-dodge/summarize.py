"""Prints the numbers quoted in post 12 from the exported site data.

uv run experiments/012-dodge/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).parent.parent.parent / "site" / "public" / "data" / "012-dodge"
ARENAS = ("arena", "bird", "aimed", "rock", "missile")


def main() -> None:
    ev = json.loads((DATA / "eval.json").read_text())
    print("training speed (school task, 3 runs side by side, 3 threads each)")
    for run, s in ev["speed"].items():
        print(
            f"  {run:<9} envs {s['envs']:<4} epochs {s['epochs']:<3} warm {bool(s['warm'])!s:<5}"
            f" {s['steps']:>11,} steps in {s['seconds'] / 60:.1f} min"
        )
    print("\ncurves: landed (courses) or held (arenas) of 200 at each checkpoint [steps M, min]")
    for group, runs in ev["curves"].items():
        for run, pts in runs.items():
            cells = [
                f"{p['steps'] / 1e6:g}M:{p['landed']}@{(p['seconds'] or 0) / 60:.1f}" for p in pts
            ]
            print(f"  {group}/{run}: " + "  ".join(cells))
    print("\nthe scripted hover pilot (ignores threats): held of 200")
    print("  " + "  ".join(f"{k} {v}" for k, v in ev["hover"].items()))
    print("\nfinal pilots (test courses in rough weather; arenas; blindfold = threats hidden)")
    for run, r in ev["table"].items():
        m = r["mix"]
        print(
            f"{run} ({r['task']}): landed {m['landed']} (blindfolded {m['blind']['landed']})"
            f"  hits {m['hits']} (blind {m['blind']['hits']}): bird {m['bird']} rock {m['rock']}"
            f" missile {m['missile']}  crashed {m['crashed']}  no threats: landed"
            f" {r['none']['landed']}  move {m['change']}"
        )
        cells = []
        for a in ARENAS:
            s = r[a]
            blind = f" (blind {s['blind']['held']})" if "blind" in s else ""
            cells.append(f"{a} {s['held']}{blind}")
        print("  held: " + "  ".join(cells))
        for a in ("bird", "aimed", "rock", "missile"):
            e = r[a]["escape"]
            esc = " ".join(f"{k}s {v}" for k, v in e.items() if k != "hits")
            print(f"  escape test, {a}: of {e['hits']} hits avoidable with notice {esc}")
    print("\nshare of arena flying time within 1.5 m of the point")
    for run, h in ev.get("hold", {}).items():
        print(f"  {run}: " + "  ".join(f"{k} {v:.0%}" for k, v in h.items()))
    print("\ncurriculum levels [steps, level or missile top speed]")
    for run, lv in ev["levels"].items():
        print(f"  {run}: {lv[-1] if lv else None} ({len(lv)} changes)")
    print("\nfiles")
    for f in sorted(DATA.glob("*.json")):
        print(f"  {f.name}: {f.stat().st_size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
