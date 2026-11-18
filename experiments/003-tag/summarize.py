"""Print the numbers quoted in post 003 from site/public/data/003-tag/eval.json.

uv run experiments/003-tag/summarize.py
"""

from __future__ import annotations

import json
from statistics import mean

from evaluate import OUT, TESTS


def main() -> None:
    data = json.loads((OUT / "eval.json").read_text())
    tests = list(TESTS)
    print("escape rate (survived 20 s), best action, 200 episodes per test\n")
    print("baselines:")
    for kind, res in data["baselines"].items():
        cells = "  ".join(f"{t}={res[t]['escaped']:.0%}/{res[t]['seconds']:.1f}s" for t in tests)
        print(f"  {kind:7s} {cells}")

    groups: dict[str, list[dict]] = {}
    for r in data["runs"]:
        key = r["chasers"] if r["rewards"] == "default" else f"{r['chasers']}-{r['rewards']}"
        groups.setdefault(key, []).append(r)

    for key, runs in groups.items():
        print(f"\n== {key}: {len(runs)} seed(s), train time {[r['trainSeconds'] for r in runs]} s")
        steps = [c["steps"] for c in runs[0]["checkpoints"]]
        print("steps".rjust(10) + "".join(t.rjust(15) for t in tests))
        for i, s in enumerate(steps):
            row = f"{s:>10,}"
            for t in tests:
                vals = [r["checkpoints"][i][t]["escaped"] for r in runs]
                row += f"{mean(vals):6.0%} [{min(vals):4.0%}]".rjust(15)
            print(row)
        final = [r["checkpoints"][-1] for r in runs]
        print("final, mean seconds before caught (all episodes):")
        print("  " + "  ".join(f"{t}={mean(c[t]['seconds'] for c in final):.1f}" for t in tests))


if __name__ == "__main__":
    main()
