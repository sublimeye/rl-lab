"""Print every number quoted in post 009 from the exported eval JSON.

uv run experiments/009-beaver/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent.parent.parent / "site" / "public" / "data" / "009-beaver"


def pct(a: int, n: int) -> str:
    return f"{a}/{n} ({a / n:.0%})"


def show(tag: str) -> None:
    p = OUT / f"eval-{tag}.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    print(f"\n=== {tag} ===  rivers: {d['rivers']}")
    bots = ["route", "dodge", "wander"]
    for b, row in d["table"].items():
        dry = sum(row[r]["dry"] for r in bots)
        n = sum(row[r]["n"] for r in bots)
        cells = "  ".join(f"{r}:{row[r]['dry']}" for r in d["rivers"])
        print(f"{b:16s} bots {pct(dry, n):16s} {cells}")
    for b, row in d["table"].items():
        for r in d["rivers"]:
            c = row[r]
            if c.get("logs"):
                print(
                    f"  {b:14s} vs {r:10s} dams/game {c['dams']:.2f} wet {c['logWet']:.0%} "
                    f"fresh {c['logFresh']:.0%} fromSpring {c['logFromSpring']:.1f} "
                    f"fromDen {c['logFromDen']:.1f} round {c['logRound']:.1f} "
                    f"cut {c['logCut']:.1f} floodRound {c['floodRound']} dug {c['dug']}"
                )
    for run, rows in d.get("curves", {}).items():
        line = []
        for r in rows:
            dry = sum(v["dry"] for v in r["vs"].values())
            n = sum(v["n"] for v in r["vs"].values())
            line.append(f"{r['steps'] / 1e6:g}M:{dry / n:.0%}")
        print(f"curve {run}: " + " ".join(line))


def main() -> None:
    for tag in ("round1", "round2", "exploit"):
        show(tag)
    for p in sorted(OUT.glob("*.json")):
        print(f"{p.name}: {p.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
