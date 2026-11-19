"""Print every number quoted in post 006 from site/public/data/006-tag-seek/eval.json.

uv run experiments/003-tag/summarize_seek.py
"""

from __future__ import annotations

import json
from collections import defaultdict

from evaluate_seek import OUT, RUNNERS

NAMES = {
    "xray": "sees through walls",
    "none": "sight only",
    "last": "+ last sighting",
    "map": "+ possibility map",
}


def pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def rng(vals: list[float], f=pct) -> str:
    lo, hi = min(vals), max(vals)
    return f(lo) if f(lo) == f(hi) else f"{f(lo)}-{f(hi)}"


def main() -> None:
    d = json.loads((OUT / "eval.json").read_text())
    print("== scripted chasers (caught / mean time to first sight / seen share)")
    for b in d["bots"]:
        print(
            f"  {b['kind']:9s} vs {b['runner']:6s} @{b['ratio']:.1f}: {pct(b['caught'])}"
            f"  first sight {b['firstSeenMean']:.1f}s  seen {pct(b['seenShare'])}"
            f"  found {pct(b['found'])}"
        )
    by = defaultdict(list)
    for r in d["runs"]:
        by[r["sight"]].append(r)
    for ratio in (0.9, 1.0):
        print(f"== trained chasers at 10M steps, speed {ratio:.0%} (range over seeds)")
        for s in ("xray", "none", "last", "map"):
            cells = []
            for rn in RUNNERS:
                ev = [
                    e
                    for r in by[s]
                    for e in r["evals"]
                    if e["steps"] == 10_000_000 and e["runner"] == rn and e["ratio"] == ratio
                ]
                if not ev:
                    continue
                txt = rng([e["caught"] for e in ev])
                if rn == "still":
                    first = rng([e["firstSeenMean"] for e in ev], lambda v: f"{v:.1f}s")
                    txt += f" (first sight {first})"
                else:
                    txt += f" (seen {rng([e['seenShare'] for e in ev])})"
                cells.append(f"{rn}: {txt}")
            print(f"  {NAMES[s]:20s} " + " | ".join(cells))
    print("== learning curve, caught vs each runner at 90% (mean of seeds)")
    for s in ("xray", "none", "last", "map"):
        rows = defaultdict(list)
        for r in by[s]:
            for e in r["evals"]:
                if e["ratio"] == 0.9:
                    rows[(e["steps"], e["runner"])].append(e["caught"])
        steps = sorted({k[0] for k in rows})
        line = " ".join(
            f"{st // 1000}k:"
            + "/".join(pct(sum(rows[(st, rn)]) / len(rows[(st, rn)])) for rn in RUNNERS)
            for st in steps
        )
        print(f"  {NAMES[s]:20s} {line}")
    print("== by map kind, 10M, 90%, caught vs sp40 / hider (mean of seeds)")
    for s in ("xray", "none", "last", "map"):
        fams = defaultdict(list)
        for r in by[s]:
            for e in r["evals"]:
                if e["steps"] == 10_000_000 and e["ratio"] == 0.9 and e["runner"] != "still":
                    for f, v in e["byFamily"].items():
                        fams[(f, e["runner"])].append(v)
        print(
            f"  {NAMES[s]:20s} "
            + " ".join(
                f"{f}:{pct(sum(fams[(f, 'sp40')]) / 2)}/{pct(sum(fams[(f, 'hider')]) / 2)}"
                for f in ("pillars", "blocks", "walls", "maze", "mixed")
            )
        )
    print("== training time")
    for r in d["runs"]:
        print(
            f"  {r['name']}: {r['minutes']} min,",
            f"caught while training {r['training'][-1]['caught']}",
        )


if __name__ == "__main__":
    main()
