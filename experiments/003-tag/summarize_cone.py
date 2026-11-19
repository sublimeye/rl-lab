"""Print every number quoted in post 007 from site/public/data/007-tag-cone/eval.json.

uv run experiments/003-tag/summarize_cone.py
"""

from __future__ import annotations

import json
from collections import defaultdict

from evaluate_cone import ANY, FINAL, OUT, SPECIALISTS, WIDE, ckpt
from evaluate_seek import RUNNERS


def pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def rng(vals: list[float], f=pct) -> str:
    lo, hi = min(vals), max(vals)
    return f(lo) if f(lo) == f(hi) else f"{f(lo)}-{f(hi)}"


def main() -> None:
    d = json.loads((OUT / "eval.json").read_text())
    R = {(r["who"], r["runner"], r["setting"]): r for r in d["results"]}

    def row(whos: list[str], setting: str) -> str:
        cells = []
        for rn in RUNNERS:
            ev = [R[(w, rn, setting)] for w in whos if (w, rn, setting) in R]
            if not ev:
                continue
            txt = rng([e["caught"] for e in ev])
            if rn == "still":
                first = rng([e["firstSeenMean"] for e in ev], lambda v: f"{v:.1f}s")
                txt += f" (first sight {first}"
                txt += ", found " + rng([e["found"] for e in ev]) + ")"
            else:
                txt += f" (seen {rng([e['seenShare'] for e in ev])})"
            cells.append(f"{rn}: {txt}")
        got = [R[(w, rn, setting)] for w in whos for rn in RUNNERS if (w, rn, setting) in R]
        if got:
            turn = sum(e["turnPerSecond"] for e in got) / len(got)
            aside = sum(e["lookAside"] for e in got) / len(got)
            cells.append(f"turns {turn:.0f} deg/s, looks aside {pct(aside)}")
        return " | ".join(cells)

    anyc = [ckpt(r) for r in ANY]
    print("== at HUMAN and at post 6's all-round settings, 90% speed (range over seeds)")
    for name, whos in (
        ("any settings (2 seeds)", anyc),
        ("any, 128x128", [ckpt(r) for r in WIDE]),
        ("human only (2 seeds)", [ckpt(r) for r in SPECIALISTS]),
        ("post 6 seeker", ["post6"]),
        ("search bot", ["bot:search"]),
    ):
        for s in ("human", "allround"):
            line = row(whos, s)
            if line:
                print(f"  {name:24s} @{s:8s} {line}")
    print("== one setting at a time (any-settings seeker, range over seeds)")
    for s, v in d["settings"].items():
        print(f"  {s:9s} {v}  {row(anyc, s)}")
    print("== learning curve at HUMAN, caught vs each runner (mean of seeds)")
    for name, runs in (("any", ANY), ("human", SPECIALISTS)):
        line = []
        steps = sorted(
            {
                int(w.split("_")[-1])
                for (w, _, s) in R
                if s == "human" and any(w.startswith(r + "/") for r in runs)
            }
        )
        for st in steps:
            vals = []
            for rn in RUNNERS:
                keys = [(f"{r}/ckpt_{st:08d}", rn, "human") for r in runs]
                ev = [R[k]["caught"] for k in keys if k in R]
                vals.append(pct(sum(ev) / len(ev)) if ev else "-")
            line.append(f"{st // 1_000_000}M:" + "/".join(vals))
        print(f"  {name:6s} " + " ".join(line))
    print("== by map kind at HUMAN, caught vs sp40 / hider (any-settings, mean of seeds)")
    for who, label in ((anyc, "any"), (["post6"], "post6")):
        fams = defaultdict(list)
        for w in who:
            for rn in ("sp40", "hider"):
                for f, v in R[(w, rn, "human")]["byFamily"].items():
                    fams[(f, rn)].append(v)
        print(
            f"  {label:6s} "
            + " ".join(
                f"{f}:{pct(sum(fams[(f, 'sp40')]) / len(fams[(f, 'sp40')]))}/"
                f"{pct(sum(fams[(f, 'hider')]) / len(fams[(f, 'hider')]))}"
                for f in ("pillars", "blocks", "walls", "maze", "mixed")
            )
        )
    print("== training")
    for r in d["runs"]:
        last = r["training"][-1]["caught"]
        print(f"  {r['name']}: {r['minutes']} min, caught while training {last}")
    print("final steps", FINAL)


if __name__ == "__main__":
    main()
