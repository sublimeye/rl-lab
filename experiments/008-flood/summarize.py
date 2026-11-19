"""Print the numbers quoted in post 8 from the exported eval.json.

uv run experiments/008-flood/summarize.py round1
uv run experiments/008-flood/summarize.py round2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "site" / "public" / "data" / HERE.name


def pct(a: int, n: int) -> str:
    return f"{a}/{n} ({a / n:.0%})"


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "round1"
    ev = json.loads((DATA / f"eval-{tag}.json").read_text())
    print(f"{tag}: house wall {ev['world']['houseWall']}")
    print("baselines (200 test maps):")
    for k, b in ev["baselines"].items():
        print(f"  {k:8s} dry {pct(b['dry'], b['n'])}  flooded at {b['floodT']}", end="")
        print(f"  grains {b['dropped']}")

    print("\nper run, per checkpoint: dry, grains dropped, grains dug")
    finals = []
    for r, d in ev["runs"].items():
        mins = (d["seconds"] or 0) / 60
        print(f"  {r} (trained in {mins:.0f} min)")
        for c in d["checkpoints"]:
            print(
                f"    {c['steps']:>10,}  dry {pct(c['dry'], c['n']):>14s}  "
                f"top action {pct(c['best']['dry'], c['n']):>14s}  "
                f"drop {c['dropped']:6.1f}  dug {c['dug']:6.1f}  floodT {c['floodT']}"
            )
        finals.append(d["checkpoints"][-1])

    print("\nfinal checkpoints, dry episodes: where and how high is the dam")
    for r, c in zip(ev["runs"], finals, strict=True):
        dist = np.array(c["damFromHouse"])
        over = np.array(c["damOverPeak"])
        added = np.array(c["damAdded"])
        peak = np.array(c["peak"])
        print(
            f"  {r}: crest {np.median(dist):.0f} columns from the house (median; "
            f"25-75% {np.percentile(dist, 25):.0f}-{np.percentile(dist, 75):.0f}), "
            f"crest above high tide: median {np.median(over):.2f}, min {over.min():.2f}, "
            f"max {over.max():.2f}; grains on the crest column: median {np.median(added):.0f}; "
            f"corr(peak, crest height) {np.corrcoef(peak, peak + over)[0, 1]:.2f}"
        )
    agree = [
        sum(a == b == "d" for a, b in zip(finals[0]["outcomes"], c["outcomes"], strict=True))
        for c in finals
    ]
    print(
        "  maps dry for every seed:",
        sum(all(c["outcomes"][i] == "d" for c in finals) for i in range(ev["episodes"])),
        agree,
    )
    builder = ev["baselines"]["builder"]["outcomes"]
    for r, c in zip(ev["runs"], finals, strict=True):
        net_only = sum(a == "d" and b == "f" for a, b in zip(c["outcomes"], builder, strict=True))
        b_only = sum(a == "f" and b == "d" for a, b in zip(c["outcomes"], builder, strict=True))
        print(f"  {r}: dry where the builder floods {net_only}; ", end="")
        print(f"floods where the builder is dry {b_only}")

    print("\ntides lower and higher than training (final checkpoints):")
    for name, t in ev["tides"].items():
        row = ", ".join(
            f"{k} {pct(v['dry'], v['n'])} crest {v['damOverFloor']}"
            f" (tide {v['peakOverFloor']}) above the floor"
            for k, v in t.items()
            if k != "range"
        )
        print(f"  {name} {t['range']}: {row}")

    if "transfer" in ev:
        print("\nnetworks from the other round, scored in this world:")
        for r, s in ev["transfer"].items():
            print(f"  {r}: dry {pct(s['dry'], s['n'])}, flooded at {s['floodT']}")


if __name__ == "__main__":
    main()
