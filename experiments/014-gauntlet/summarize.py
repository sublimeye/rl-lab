"""Prints the numbers quoted in post 14 from the exported site data.

uv run experiments/014-gauntlet/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).parent.parent.parent / "site" / "public" / "data" / "014-gauntlet"


def enc(s: dict) -> str:
    return "  ".join(
        f"{k} {v['hit']}/{v['met']} ({v['hit'] / max(v['met'], 1):.0%})"
        for k, v in s["encounters"].items()
    )


def main() -> None:
    ev = json.loads((DATA / "eval.json").read_text())
    print("final pilots and earlier recipes (200 test courses each)")
    for run, r in ev["table"].items():
        print(f"{run} ({r['ckpt']})")
        for test, s in r.items():
            if test == "ckpt":
                continue
            print(
                f"  {test:<14} landed {s['landed']:>3}  {s['outcomes']}"
                f"  frog fails {s['frog_fail']}  time {s['time']}  hit/met: {enc(s)}"
            )
    print("\ncurves: gauntlet landed of 200 per checkpoint [steps M: landed @ minutes]")
    for group, runs in ev["curves"].items():
        for run, pts in runs.items():
            cells = [
                f"{p['steps'] / 1e6:g}M:{p['landed']}@{(p['seconds'] or 0) / 60:.0f}" for p in pts
            ]
            print(f"  {group}/{run}: " + "  ".join(cells))
    print("\ndials: when each reached the test's difficulty (steps), and the last level")
    for run, lv in ev["levels"].items():
        done = {}
        for steps, levels in lv:
            for k, v in levels.items():
                if v >= 1.0 and k not in done:
                    done[k] = steps
        print(f"  {run}: last {lv[-1][1]}  full at {done}  ({ev['seconds'][run]})")
    d = ev["diagnostics"]
    print("\ndodge drills (short courses, pad in sight): hits / encounters met, landed of 400")
    for k, kinds in d["drills"].items():
        cells = [f"{n} {v['hit']}/{v['met']} ({v['hit'] / v['met']:.0%})" for n, v in kinds.items()]
        print(f"  {k}: " + "  ".join(cells))
    print("\nfrog drills (200 courses each)")
    for k, tests in d["frogs"].items():
        for t, s in tests.items():
            print(f"  {k} / {t}: landed {s['landed']}  {s['outcomes']}  time {s['time']}")
    print("\nflock only: hits / met, sighted vs blindfolded")
    for p in d["flock"]:
        s, b = p["sighted"], p["blind"]
        print(
            f"  {p['steps'] / 1e6:5.1f}M  sighted {s['hit']}/{s['met']}"
            f" ({s['hit'] / max(s['met'], 1):.0%})  blind {b['hit']}/{b['met']}"
            f" ({b['hit'] / max(b['met'], 1):.0%})"
        )
    sc = d["scripted"]
    c = sc["clear"]
    print(f"\nthe cruise pilot (threats ignored): clear landed {c['landed']}/{c['of']}")
    for n in ("bird", "two birds", "missile"):
        v = sc[n]
        print(
            f"  {n} at every waypoint: hit {v['hit']}/{v['of']},"
            f" escapable with {v['lead']} s: {v['escapable']}/{v['hit']}"
        )
    print("\nthe hand-written rule (inputs only), one encounter kind per course, no frogs")
    for n, v in d["rule"].items():
        print(f"  {n}: {v['hit']}/{v['met']} ({v['hit'] / v['met']:.1%})  landed {v['landed']}")
    print(f"\nthe reflex on {d['reflex']['final']}")
    for k in ("pilot", "pilot + reflex"):
        for t, s in d["reflex"][k].items():
            print(f"  {k} / {t}: landed {s['landed']}  {s['outcomes']}  hit/met: {enc(s)}")
    print("\nfiles")
    for f in sorted(DATA.glob("*.json")):
        print(f"  {f.name}: {f.stat().st_size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
