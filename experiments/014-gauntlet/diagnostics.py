"""The side experiments behind post 14, as data (runs/diagnostics.json).

uv run experiments/014-gauntlet/diagnostics.py [--final final-s1]

- drills: the dodge-drill pilots (short courses, the pad in sight from the start) on 400
  courses per encounter kind: without and with closest-approach inputs, and with the two
  shaping terms tried;
- frog drills: the hovering fix (a cost per second in the air) and the drill pilot without it;
- flock: a pilot trained on crossing flocks alone, checkpoint by checkpoint, sighted and blind;
- scripted: a cruise pilot that ignores every threat, and the escape test on its hits (are the
  threats dangerous, and avoidable?);
- rule: a hand-written dodge rule reading only the pilot's inputs (does the input carry what's
  needed?) and the final pilot with the reflex added.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
from reflex import Reflex

from evaluate import RUNS, TESTS, Net, checkpoints, fly, load, summary
from gauntlet_env import ENCOUNTERS, FULL, K_SEE, N_BASE, frame_width
from train import DODGE, FROGS, TASKS

DRILL_SEEDS = range(30_000, 30_400)
FROG_SEEDS = range(40_000, 40_200)
DRILL_RUNS = {
    "without closest approach": "ab-dodge",
    "with closest approach": "ab-dodge-predict2",
    "+ near-miss cost": "ab-dodge-predict-near",
    "+ danger shaping": "ab-danger",
}
FROG_RUNS = {
    "no cost for time": "ab-frogs",
    "cost for time": "ab-frogs-time",
    "post 13's reward": "ab-frogs13",
}


def only(feat, k):
    mix = [0.0] * len(ENCOUNTERS)
    mix[k] = 1.0
    return replace(feat, mix=tuple(mix))


def per_kind(pilot, feat, seeds, hide=()):
    out = {}
    for k in range(1, len(ENCOUNTERS)):
        s = summary(fly(pilot, only(feat, k), seeds=seeds, hide=hide))
        e = s["encounters"].get(ENCOUNTERS[k], {"met": 0, "hit": 0})
        out[ENCOUNTERS[k]] = {"landed": s["landed"], "of": len(seeds), **e}
    return out


def last(run):
    return Net(load(checkpoints(run)[-1]))


def rule_pilot():
    """Fly at 7 m/s and 7 m up, land when over the pad, and climb or dive 3 m away from any
    pursuer predicted to pass within 1.6 m in the next 1.6 s. Inputs only, frame 0."""
    w = frame_width(replace(FULL, predict=True))

    def pilot(obs):
        o = obs[:, :w].astype(float)
        n = len(o)
        b = o[:, :N_BASE]
        sl = o[:, N_BASE : N_BASE + K_SEE * 10].reshape(n, K_SEE, 10)
        pdx, pdy, vx, vy, y = b[:, 0] * 10, b[:, 1] * 10, b[:, 2] * 5, b[:, 3] * 5, b[:, 10] * 10
        seen, d = b[:, 14] > 0.5, b[:, 17]
        want_vx = np.where(seen & (np.abs(pdx) < 6), np.clip(0.8 * pdx, -2.5, 2.5), d * 7)
        cruise = np.where(seen, np.maximum(7, y + pdy + 1.5), 7.0)
        want_y = np.where(seen & (np.abs(pdx) < 1.0), y + pdy - 0.5, cruise)
        t, mx, my = sl[..., 7] * 3, sl[..., 8] * 5, sl[..., 9] * 5
        bad = (sl[..., 4] > 0.5) & (t < 1.6) & (np.hypot(mx, my) < 1.6)
        k = np.argmax(bad, 1)
        away = np.where(my[np.arange(n), k] > 0, -1.0, 1.0)
        want_y = np.where(bad.any(1), y + away * 3, want_y)
        want_vy = np.clip(1.5 * (want_y - y), -3, 3)
        tilt = np.clip(0.35 * (want_vx - vx), -1, 1)
        thr = np.clip(0.05 + 0.6 * (want_vy - vy), -1, 1)
        return np.stack([thr, tilt], 1)

    pilot.predict = True
    return pilot


def scripted_checks(seeds=range(20_000, 20_200), lead=0.5):
    """The cruise pilot (7 m up at 7 m/s, threats ignored) on clear courses and on courses with
    one encounter kind at every waypoint, and the escape test on its bird and missile hits."""
    from gauntlet_env import BIRD, CLEAR, FLYING, LANDED, MISSILE, ROUGH, Fleet, make_course
    from scripted import cruise_pilot, escapable, fly_snapshots

    def fleet(feat):
        f = Fleet(len(seeds), feat=feat)
        for i, s in enumerate(seeds):
            f.set_course(i, make_course(s, ROUGH, feat, f.ph))
        return f

    f = fleet(CLEAR)
    for _ in range(1300):
        if not (f.outcome == FLYING).any():
            break
        f.step(cruise_pilot(f))
    out = {"clear": {"landed": int((f.outcome == LANDED).sum()), "of": len(seeds)}}
    for k, kind in ((1, BIRD), (2, BIRD), (3, MISSILE)):
        f = fleet(only(CLEAR, k))
        snaps = fly_snapshots(f, cruise_pilot)
        hit = np.nonzero(f.outcome == kind)[0]
        ok = escapable(snaps, f, hit, lead=lead) if len(hit) else np.zeros(0, bool)
        out[ENCOUNTERS[k]] = {
            "of": len(seeds),
            "hit": int(len(hit)),
            "escapable": int(ok.sum()),
            "lead": lead,
        }
        print(ENCOUNTERS[k], out[ENCOUNTERS[k]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="final-s1")
    ap.add_argument("--scripted", action="store_true", help="add the scripted checks only")
    args = ap.parse_args()
    if args.scripted:
        path = RUNS / "diagnostics.json"
        out = json.loads(path.read_text())
        out["scripted"] = scripted_checks()
        path.write_text(json.dumps(out))
        return
    out: dict = {}
    print("dodge drills")
    out["drills"] = {k: per_kind(last(r), DODGE, DRILL_SEEDS) for k, r in DRILL_RUNS.items()}
    print("frog drills")
    frog_tests = {
        "frog drill, slow beat": replace(FROGS, frog_period=(6.0, 7.0)),
        "frog drill, test beat": FROGS,
        "pad frog only, slow beat": replace(FROGS, route_frog=0.0, frog_period=(6.0, 7.0)),
    }
    out["frogs"] = {}
    for k, r in {**FROG_RUNS, "drill mix, no cost for time": "drills-s1"}.items():
        net = Net(load(RUNS / r / "ckpt_020000000.zip")) if r != "drills-s1" else last(r)
        out["frogs"][k] = {t: summary(fly(net, f, seeds=FROG_SEEDS)) for t, f in frog_tests.items()}
    print("flock only")
    flock = TASKS["flock-only"]
    pts = []
    for c in checkpoints("diag-flock"):
        net = Net(load(c))
        s = summary(fly(net, flock, seeds=DRILL_SEEDS))["encounters"]["flock"]
        b = summary(fly(net, flock, seeds=DRILL_SEEDS, hide=("threats",)))["encounters"]["flock"]
        pts.append({"steps": int(c.stem.split("_")[1]), "sighted": s, "blind": b})
    out["flock"] = pts
    print("scripted")
    out["scripted"] = scripted_checks()
    print("rule and reflex")
    rule = rule_pilot()
    out["rule"] = per_kind(rule, replace(FULL, route_frog=0.0, guard=0.0), range(20_000, 20_200))
    final = last(args.final)
    out["reflex"] = {
        "final": args.final,
        "pilot": {t: summary(fly(final, TESTS[t][0])) for t in ("gauntlet", "flock", "missile")},
        "pilot + reflex": {
            t: summary(fly(Reflex(final), TESTS[t][0])) for t in ("gauntlet", "flock", "missile")
        },
    }
    (RUNS / "diagnostics.json").write_text(json.dumps(out))
    for k, v in out.items():
        print(k, json.dumps(v)[:600])


if __name__ == "__main__":
    main()
