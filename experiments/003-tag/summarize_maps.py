"""Print the numbers quoted in post 005 from site/public/data/005-tag-maps/eval.json.

uv run experiments/003-tag/summarize_maps.py
"""

from __future__ import annotations

import json

from evaluate_maps import OUT


def pct(r: dict) -> str:
    return f"{r['escaped']:.0%}"


def main() -> None:
    d = json.loads((OUT / "eval.json").read_text())
    print(f"{d['testMaps']} test maps, chaser at {d['ratio']:.0%}; escapes")
    print("\nstep 1 learning curves vs intercept-90 (best action), by steps:")
    for r in d["runs"]:
        pts = " ".join(f"{c['steps'] // 1000}k:{pct(c['intercept-90'])}" for c in r["checkpoints"])
        print(f"  {r['name']:16s} {r['arch']} {r['trainSeconds'] / 60:.0f} min  {pts}")
    print("\nfinal runners vs scripted bots (pursuit / intercept / keyboard):")
    for f in d["finals"]:
        for mode in ("best", "sample"):
            row = []
            for where in ("maps", "classic"):
                row.append(" / ".join(pct(v) for v in f[mode][where].values()))
            fam = f[mode]["maps"]["intercept-90"]["byFamily"]
            fams = " ".join(f"{k}:{v:.0%}" for k, v in fam.items())
            print(f"  {f['path']:34s} {mode:6s} maps {row[0]:16s} room {row[1]:16s} {fams}")
    print("\nself-play rounds (with randomness):")
    for r in d["selfplay"]:
        for x in r.get("rounds", []):
            cells = " ".join(f"{k}:{pct(v)}" for k, v in x.items() if k != "round")
            print(f"  {r['name']} r{x['round']:02d} {cells}")
        last = r["log"]["rounds"][-1]
        print(
            f"  {r['name']} log: {last['minutes']} min, last chaser catch rate "
            f"{last['chaserCatchRate']:.0%}"
        )
    print("\nexploiters (escapes / mean seconds by exploiter steps):")
    for x in sorted(d["exploits"], key=lambda x: (x["target"], x["ratio"], x["name"])):
        cells = " ".join(
            f"{c['steps'] // 1000}k:{c['escaped']:.0%}/{c['seconds']:.1f}s"
            for c in x["checkpoints"]
        )
        fam = x["checkpoints"][-1]["byFamily"]
        fams = " ".join(f"{k}:{v:.0%}" for k, v in fam.items())
        print(
            f"  {x['name']:12s} {x['target']:24s} {x['ratio']:.0%} {cells}\n"
            f"      last by kind: {fams}"
        )


if __name__ == "__main__":
    main()
