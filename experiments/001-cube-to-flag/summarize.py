"""Print the numbers quoted in the long-training report (post 002).

uv run experiments/001-cube-to-flag/summarize.py
"""

from __future__ import annotations

import json
from statistics import mean

from export import OUT

KEYS = ("usual_best", "usual_sampled", "anywhere_best", "anywhere_sampled")


def at(run: dict, steps: int) -> dict | None:
    return next((c for c in run["checkpoints"] if c["steps"] == steps), None)


def table(group: str, points: list[int]) -> None:
    data = json.loads((OUT / f"{group}-eval.json").read_text())
    runs = data["runs"]
    print(f"\n== {group}: {len(runs)} seeds, {data['episodes']} episodes per check ==")
    for r in runs:
        print(
            f"{r['name']}: first 99% at {r['firstHit']:,}, stays capable from "
            f"{r['capableAt'] or 'never'}, "
            f"anywhere-capable from {r['capableAtAnywhere'] or 'never'}"
        )
    header = "steps".rjust(10) + "".join(k.rjust(18) for k in KEYS) + "  sec(usual)  coins"
    print(header)
    for p in points:
        cs = [at(r, p) for r in runs]
        if any(c is None for c in cs):
            continue
        row = f"{p:>10,}"
        for k in KEYS:
            vals = [c[k]["success"] for c in cs]
            row += f"   {mean(vals):5.1%} [{min(vals):4.0%}]".rjust(18)
        secs = [c["usual_best"]["secondsToFlag"] for c in cs if c["usual_best"]["secondsToFlag"]]
        coins = [c["usual_best"]["coins"] for c in cs]
        row += f"  {mean(secs):6.2f}" if secs else "     n/a"
        row += f"   {mean(coins):4.2f}"
        print(row)

    # Stability: after each seed first hits 99%, how many later checks fall below it?
    print("dips below 99% (usual, best action) after the first 99%, per training window:")
    for lo, hi in ((0, 1_000_000), (1_000_000, 5_000_000), (5_000_000, 10**9)):
        dips, checks = 0, 0
        for r in runs:
            for c in r["checkpoints"]:
                if c["steps"] > r["firstHit"] and lo <= c["steps"] < hi:
                    checks += 1
                    dips += c["usual_best"]["success"] < 0.99
        if checks:
            worst = min(
                c["usual_best"]["success"]
                for r in runs
                for c in r["checkpoints"]
                if c["steps"] > r["firstHit"] and lo <= c["steps"] < hi
            )
            print(f"  {lo:>10,} to {hi:>11,}: {dips}/{checks} checks dip, worst {worst:.0%}")


if __name__ == "__main__":
    pts = [0, 100_000, 200_000, 300_000, 500_000, 1_000_000, 2_000_000, 3_000_000]
    table("long", pts + [5_000_000, 7_500_000, 10_000_000])
    table("rand", pts)
