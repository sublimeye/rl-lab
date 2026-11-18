"""Print the numbers quoted in post 004 from site/public/data/004-tag-selfplay/eval.json.

uv run experiments/003-tag/summarize_selfplay.py
"""

from __future__ import annotations

import json

from evaluate_selfplay import OUT


def main() -> None:
    data = json.loads((OUT / "eval.json").read_text())
    print("fresh chaser trained against one frozen runner: escapes / mean seconds, 200 starts")
    print(f"{'run':14s} {'runner':28s} {'ratio':>5s} {'picks':>7s}  by exploiter steps")
    for x in sorted(data["exploits"], key=lambda x: (x["target"], x["ratio"], x["name"])):
        cells = "  ".join(
            f"{c['steps'] // 1000}k:{c['escaped']:.0%}/{c['seconds']:.1f}s"
            for c in x["checkpoints"]
        )
        picks = "sample" if x["sample"] else "best"
        print(f"{x['name']:14s} {x['target']:28s} {x['ratio']:5.2f} {picks:>7s}  {cells}")
    print("\nper run: first and last round, escapes (best action, 90% chasers)")
    tests = ["pursuit-90", "intercept-90", "keyboard-90", "chaser-s1", "chaser-s2", "chaser-s3"]
    for r in data["runs"]:
        for e in (r["rounds"][0], r["rounds"][-1]):
            cells = "  ".join(f"{t}={e[t]['escaped']:.0%}" for t in tests)
            print(f"{r['name']:10s} round {e['round']:2d}  {cells}")


if __name__ == "__main__":
    main()
