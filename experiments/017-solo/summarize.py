"""Prints the numbers post 17 quotes, from site/public/data/017-solo/results.json.

uv run experiments/017-solo/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).parent.parent.parent / "site" / "public" / "data" / "017-solo"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    r = json.loads((DATA / "results.json").read_text())
    d = r["diagnosis"]
    p0 = d["planner_round"]
    print("== the planner's own flights (round 0)")
    print(f"  {p0['flights']} flights, {p0['states']:,} moments; its move is the move made last "
          f"{pct(p0['label_is_last_move'])}; moves weighed {p0['tried_moves_mean']:.1f}, safe "
          f"(median) {p0['safe_moves_median']:.0f}")  # fmt: skip
    print("== each round's network on the next round's flights (GPU run, then CPU run)")
    for run in ("gpu", "cpu"):
        rows = d["on_policy"][run]
        for k, label in (("accuracy", "planner's move"), ("accuracy_change", "when it changes"),
                         ("fatal", "fatal picks"), ("fatal_urgent", "fatal, urgent threat"),
                         ("fatal_frog", "fatal, frog near"), ("fatal_calm", "fatal, calm"),
                         ("keeps_last", "keeps its last move")):  # fmt: skip
            print(f"  {run} {label:22s}: " + "  ".join(pct(o[k]) for o in rows))
        o = rows[-1]
        print(f"  {run} round 5, failed flights: first fatal move, seconds before the end "
              f"(quartiles, n): {o['first_fatal_s']}")  # fmt: skip
        print(f"  {run} round 5: no safe move left {o['lost_s']} s before the end; fatal decisions "
              f"per failed flight {o['fatal_per_failed_flight']}")  # fmt: skip
        print(f"  {run} top-k holds a safe move, rounds 1 and 5: {rows[0]['top_k_safe']} "
              f"{o['top_k_safe']}")  # fmt: skip
    print("== post 16's network alone, by course (200 test courses each)")
    for k, v in d["post16_net"].items():
        print(f"  {k:15s}: {v['landed']:3d}  {v['outcomes']}")
    print("  with its danger head as a veto:", {k: v["landed"] for k, v in d["veto"].items()})
    print("== the planner deciding twice (nudged by the sensors' noise)")
    s = r["sensitivity"]
    print(f"  {s['flights']} flights, landed {s['landed']}")
    for k, v in s["by"].items():
        print(f"  {k:9s}: {v['n']:5d} decisions; another move {pct(v['flip'] / v['n'])}; another "
              f"kind {pct(v['kind'] / v['n'])}; fatal for the real planner "
              f"{pct(v['fatal'] / v['n'])}")  # fmt: skip
    print("== refits on post 16's saved flights (197 flights held out; 200 dev courses)")
    for name, f in r["refits"].items():
        a = f["args"]
        tag = f"{a['loss']}{' no-prev' if a['no_prev'] else ''}{' no-air' if a['no_air'] else ''}"
        tag += f"{' reg' if a['dropout'] else ''} steps {a['steps']} share {a['share']}"
        print(f"  {name:12s} ({tag}): {f['flights']} flights; planner's move fitted "
              f"{pct(f['fitted']['accuracy'])} unseen {pct(f['held_out']['accuracy'])}; fatal "
              f"fitted {pct(f['fitted']['fatal'])} unseen {pct(f['held_out']['fatal'])} (urgent "
              f"{pct(f['fitted']['fatal_urgent'])} / {pct(f['held_out']['fatal_urgent'])}); dev "
              f"{f['dev'].get('landed', 0)}/200"
              + (f"; solo_env 1000 dev {f['solo_dev_hold1'].get('landed', 0)}"
                 if "solo_dev_hold1" in f else ""))  # fmt: skip
    print("== practice")
    for t in r["trainings"]:
        c = t["config"]
        print(f"  {t['run']}: {t['minutes']} min, {t['decisions']:,} decisions "
              f"({t['decisions'] * 0.05 / 86400:.0f} days of flight); init {c.get('init')}, "
              f"bc {c.get('bc')}, lr {c.get('lr')}")  # fmt: skip
        for h in t["checks"]:
            secs = h["sampled_outcomes"].get("seconds")
            print(f"     {h['minutes']:5.1f} min {h['decisions'] / 1e6:6.1f}M: top {h['top']:3d} "
                  f"sampled {h['sampled']:3d} of {t['dev_n']}"
                  + (f", {secs} s a landing" if secs else "")
                  + (f", open pads {h['open']}/200" if h.get("open") is not None else "")
                  )  # fmt: skip
        if t.get("final_dev"):
            print(f"     1000 dev courses at the end: {t['final_dev']}")
    print("== alone on the test and fresh courses, and one kind of course at a time")
    for name, ev in r["evals"].items():
        for mode, tests in ev.items():
            for k, v in tests.items():
                hit = (
                    f"; hit by {sum(v['hit_by'])} of {max(v['met'][1:])}"
                    if k in ("bird", "two birds", "missile", "bird + missile", "flock")
                    else ""
                )
                print(f"  {name:9s} {mode:7s} {k:15s}: {v['landed']:3d}/200 {v['outcomes']}{hit}")
    print("== networks compared on dev courses (solo_eval.py --dev)")
    for name, ev in r.get("dev", {}).items():
        print(f"  {name}: " + "; ".join(
            f"{mode} {t['dev']['landed']}/{t['dev']['n']}" for mode, t in ev.items()))  # fmt: skip
    print("== the same courses again, rewinds, speed (rewind.py)")
    for name, w in r.get("rewinds", {}).items():
        print(
            f"  -- {name}: a move held {w.get('hold', 1)} update(s), "
            f"temperature {w.get('temp', 1.0)}"
        )
        if "again" in w:
            a = w["again"]
            print(
                f"     {a.get('n', 1000)} courses x {a.get('reps', 8)}: landed "
                f"{pct(a['landed_share'])} {a['outcomes']}"
            )
            print(
                f"     courses by landings {a['courses_by_landings']}; if all alike "
                f"{a['if_all_courses_alike']}; frogs {a['frog']}; hit by {a['hit_by']}"
            )
        if "failures_by_course_landings" in w:
            print(f"     failures by the course's landings: {w['failures_by_course_landings']}")
        for k, v in w.get("temps", {}).items():
            print(f"     temperature {k}: {v.get('landed', 0)}")
        for k, v in w.get("holds", {}).items():
            print(f"     a move held {k} updates, no retraining: {v.get('landed', 0)}")
        if "rewind" in w:
            for g, o in w["rewind"]["by_outcome"].items():
                print(
                    f"     rewound {w['rewind']['back_s']} s, {g} ({o['flights']} flights): "
                    f"alive {o['drawn']}; under 10% left {o['drawn_doomed']}"
                )
        for rule, o in w.get("speed", {}).items():
            print(
                f"     speed, {rule}: landed {o['landed']}/{o['n']}; mean {o['speed_mean']} m/s, "
                f"over 10 m/s {pct(o['over_10'])}, over 12 {pct(o['over_12'])}, the sprint "
                f"{pct(o['sprint_share'])} of moves, {o['seconds_landed']} s a landing"
            )
        if "planner" in w:
            print(
                f"     planner on its {w['planner']['courses']} hardest courses: "
                f"{w['planner']['estimated']}; assuming still air {w['planner']['nominal']}"
            )
        for back, o in w.get("moves", {}).get("by_back", {}).items():
            print(f"     every next move tried, {back} s back ({o['states']} moments): {o}")
    print("== second tries (retry.py)")
    for name, rows in r.get("retries", {}).items():
        for row in rows:
            if row["pass"] == 0:
                print(f"  {name} start: dev {row['dev']}; open pads {row['dev_open']['landed']}")
                continue
            print(
                f"  {name} pass {row['pass']}: {row['courses']} courses, {row['failed_flights']} "
                f"failed, {row['moments']} moments; as flown {row['alive_as_flown']}; held-out "
                f"{row['held_before']} -> {row['held_after']}; dev {row['dev']['landed']}; "
                f"{row['minutes']} min"
            )
    for f in ("solo", "hurry", "copy"):
        if not (DATA / f"{f}-flights.json").exists():
            continue
        fl = json.loads((DATA / f"{f}-flights.json").read_text())["flights"]
        n = sum(x["outcome"] == "landed" for x in fl)
        out = {}
        for x in fl:
            out[x["outcome"]] = out.get(x["outcome"], 0) + 1
        print(f"== recorded {f} flights: landed {n} of {len(fl)} {out}")


if __name__ == "__main__":
    main()
