"""Prints the numbers quoted in post 13 from the exported site data.

uv run experiments/013-frogs/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).parent.parent.parent / "site" / "public" / "data" / "013-frogs"


def main() -> None:
    ev = json.loads((DATA / "eval.json").read_text())
    print(f"tests: 200 courses each; the beat switch: {ev['switch']} (after strike n, x ratio)")
    print("entry bins (s to the next strike when first within reach):",
          ev["table"][next(iter(ev["table"]))]["frogs"]["entry_bins"])  # fmt: skip
    for run, r in ev["table"].items():
        c = r["config"]
        print(
            f"\n{run} (task {r['task']}, countdown head {c.get('aux')},"
            f" ramp {c.get('levels', False)}, lstm {c.get('lstm', False)}):"
            f" {r['steps']:,} steps in {(r['seconds'] or 0) / 60:.1f} min"
        )
        for test in ("frogs", "switch"):
            s = r[test]
            print(
                f"  {test:<6} landed {s['landed']:3d}/200  frog {s['frog']:3d}"
                f"  crashed {s['crashed']}  timeout {s['timeout']}  water {s['water']}"
                f"  time {s['time']} s  hover {s['hover']} s  entries {s['entry_hist']}"
            )
        print(f"  probe (mean error, s): {r['probe']}")
        lv = ev["levels"].get(run, [])
        if len(lv) > 1:
            print(f"  ramp: {lv[-1]} after {len(lv) - 1} steps up")
    pc = ev.get("phase_check")
    if pc:
        print("\nscripted straight-flying pilot, one frog (2-3 s beat): hit rate by when it")
        print("first came within reach (s to the next strike)")
        for (a, b), (hit, n) in zip(pc["bins"], pc["rows"], strict=True):
            print(f"  {a:.1f}-{b:.1f} s: hit {hit} of {n}")
    print("\ncurves: landed of 200 at each checkpoint [steps M, min]")
    for group, runs in ev["curves"].items():
        for run, pts in runs.items():
            cells = [
                f"{p['steps'] / 1e6:g}M:{p['landed']}@{(p['seconds'] or 0) / 60:.1f}" for p in pts
            ]
            print(f"  {group}/{run}: " + "  ".join(cells))
    print("\nfiles")
    for f in sorted(DATA.glob("*.json")):
        print(f"  {f.name}: {f.stat().st_size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
