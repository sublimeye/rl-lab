"""Why a solo pilot still fails (post 17): its flights flown again.

Nothing is trained here. Measurements of a saved network, on courses used for nothing else
(50000 up); each part adds to runs/rewind-<name>.json:
- again: every course flown several times, with different draws of its moves and of the
  sensors' noise. If every course fails now and then it is the pilot's dice; if the same
  courses fail every time it is the course.
- temps: its moves drawn at other temperatures (0: always the top move).
- rewind: each failed flight wound back 0.5 to 8 s before its end and flown on K times from
  there. How often it is then still alive 4 s after the moment it died says when the flight
  was lost: at the last move, or seconds earlier.
- speed: how fast it flies and which moves it makes, and the same network with its sprint (the
  full-throttle, full-tilt jink) taken off the menu.
- planner: post 16's fair planner on the courses the network landed at most once (slow: about
  12 s a course on 10 cores), with its air estimate and told to assume still air.
- moves: each failed flight wound back 1, 2 and 3 s, and from there every move on offer tried
  as the next one (held 0.3 s, 8 copies each, the network flying on as usual). Does the next
  move decide whether it lives? The best-looking move is picked on 4 of the copies and scored
  on the other 4, so luck doesn't flatter it.
- holds: the same network made to hold each move 1, 2 or 4 updates, with no retraining.
- trace: one flight on one course (--trace-seed), update by update: where the drone and each
  bird or missile in sight are, and the network's three most probable moves.
--hold and --temp say how the pilot under test flies: how many updates it holds a move, and
whether its moves are drawn (1) or always its top one (0).

uv run experiments/017-solo/rewind.py --net runs/b1/last.npz --name b1
uv run experiments/017-solo/rewind.py --net runs/k2/last.npz --name k2 --hold 4 --temp 0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import solo_env as se  # noqa: E402
from solo_net import load  # noqa: E402

import expert  # noqa: E402
from gauntlet_env import ENCOUNTERS, FLYING, LANDED, OUTCOMES, SIGHT  # noqa: E402

DIAG = 50_000
CAP = 1300
BACK = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)  # seconds before the end
AFTER = 80  # updates past the original end a rewound flight must survive (4 s)
SNAP = 10  # a snapshot every this many updates
CHUNK = 25
SPRINT = expert.move_id(expert.JINK, 1.0, 1.0)
KIND = ("cruise", "jink", "land", "wait", "level")
# the sprint taken away: never; unless a bird or missile is in sight; above 10 m/s; always
RULES = ("as trained", "no sprint in open air", "no sprint over 10 m/s", "no sprint")


def policy_of(net, temp: float, rng):
    if temp == 0:
        return net.act
    return lambda x, mask: net.sample(x, mask, rng, temp)


def describe(f, seeds) -> list[dict]:
    gx = f.goal()[0]
    rows = []
    for i, s in enumerate(seeds):
        c = f.courses[i]
        out = OUTCOMES[f.outcome[i]]
        row = {"seed": int(s), "outcome": out, "t": round(float(f.ticks[i] * f.ph.dt), 2),
               "togo": round(float(abs(gx[i] - f.x[i])), 1),
               "length": round(c["length"], 1),
               "kinds": [int(k) for _, k, _ in c["encounters"]],
               "frogs": [int(fr is not None) for fr in c["frogs"]],
               "air": [round(c[k], 2) for k in ("wind", "gust", "mass", "charge")]}  # fmt: skip
        if out == "frog":  # the mid-route frog (0) or the pad's (1): the nearer one
            d = np.where(f.frog_on[i], np.abs(f.frog_p[i, :, 0] - f.x[i]), np.inf)
            row["frog"] = int(np.argmin(d))
        if out in ("bird", "missile"):
            row["by"] = ENCOUNTERS[int(f.hit_enc[i])]
        rows.append(row)
    return rows


def _again(job):
    first, n, path, temp, rep, hold = job
    os.environ["OMP_NUM_THREADS"] = "1"
    net = load(HERE / path)
    seeds = list(range(first, first + n))
    env = se.SoloEnv(n, hold=hold, seed=7 + 1000 * rep)
    x = env.start(seeds)
    f = env.fleet
    policy = policy_of(net, temp, np.random.default_rng([first, rep]))
    t = 0
    while (f.outcome == FLYING).any() and t < CAP:
        x = env.step(policy(x, env.mask()))[0]
        t += hold
    return describe(f, seeds)


def _rewind(job):
    """(Snapshots are taken at decisions; with a move held 4 updates, every 0.6 s, not 0.5.)"""
    first, n, path, k, hold, own = job
    os.environ["OMP_NUM_THREADS"] = "1"
    net = load(HERE / path)
    seeds = list(range(first, first + n))
    env = se.SoloEnv(n, hold=hold, seed=7)
    x = env.start(seeds)
    f = env.fleet
    fly = policy_of(net, own, np.random.default_rng([first, 0]))
    snaps, at, t = [], [], 0
    end = np.zeros(n, np.int64)
    while (f.outcome == FLYING).any() and t < CAP:
        if not snaps or t - at[-1] >= SNAP:
            snaps.append(copy.deepcopy(env))
            at.append(t)
        x = env.step(fly(x, env.mask()))[0]
        t += hold
        end = np.where((f.outcome != FLYING) & (end == 0), f.ticks // 3, end)
    at = np.array(at)
    out = []
    for i in np.nonzero((f.outcome != LANDED) & (f.outcome != FLYING))[0]:
        row = {"seed": seeds[i], "outcome": OUTCOMES[f.outcome[i]], "end": int(end[i]), "alive": {}}
        for mode, temp in (("drawn", 1.0), ("top", 0.0)):
            alive = []
            for back in BACK:
                t0 = end[i] - round(back * 20)
                if t0 < 0:
                    alive.append(None)
                    continue
                j = int(np.searchsorted(at, t0, side="right") - 1)
                fork = snaps[j].fork(int(i), k, seed=int(1 + 7919 * seeds[i] + 31 * j))
                pol = policy_of(net, temp, np.random.default_rng([seeds[i], j, 1]))
                xf = fork.inputs()
                for _ in range(0, end[i] - at[j] + AFTER, hold):
                    if not (fork.fleet.outcome == FLYING).any():
                        break
                    xf = fork.step(pol(xf, fork.mask()))[0]
                o = fork.fleet.outcome
                alive.append(float(np.mean((o == FLYING) | (o == LANDED))))
            row["alive"][mode] = alive
        out.append(row)
    return out


def _moves(job):
    first, n, path, k, first_n, hold, own = job
    os.environ["OMP_NUM_THREADS"] = "1"
    net = load(HERE / path)
    seeds = list(range(first, first + n))
    env = se.SoloEnv(n, hold=hold, seed=7)
    x = env.start(seeds)
    f = env.fleet
    fly = policy_of(net, own, np.random.default_rng([first, 0]))
    snaps, at, t = [], [], 0
    end = np.zeros(n, np.int64)
    while (f.outcome == FLYING).any() and t < CAP:
        if not snaps or t - at[-1] >= SNAP:
            snaps.append(copy.deepcopy(env))
            at.append(t)
        x = env.step(fly(x, env.mask()))[0]
        t += hold
        end = np.where((f.outcome != FLYING) & (end == 0), f.ticks // 3, end)
    at = np.array(at)
    out = []
    for i in np.nonzero((f.outcome != LANDED) & (f.outcome != FLYING))[0]:
        for back in (1.0, 2.0, 3.0):
            t0 = end[i] - round(back * 20)
            if t0 < 0:
                continue
            j = int(np.searchsorted(at, t0, side="right") - 1)
            one = snaps[j].fork(int(i), 1, seed=1)
            offer = np.nonzero(one.mask()[0])[0]
            lg = net(one.inputs())[0, : se.NM][offer].astype(np.float64)
            p = np.exp(lg - lg.max())
            p /= p.sum()
            fork = snaps[j].fork(int(i), len(offer) * k, seed=int(3 + 7919 * seeds[i] + 31 * j))
            pol = policy_of(net, own, np.random.default_rng([seeds[i], j, 2]))
            xf = fork.inputs()
            for u in range((end[i] - at[j] + AFTER) // hold + 1):
                if not (fork.fleet.outcome == FLYING).any():
                    break
                m = np.repeat(offer, k) if u < first_n else pol(xf, fork.mask())
                xf = fork.step(m)[0]
            o = fork.fleet.outcome
            alive = ((o == FLYING) | (o == LANDED)).reshape(len(offer), k)
            out.append({"seed": seeds[i], "outcome": OUTCOMES[f.outcome[i]], "back": back,
                        "offer": offer.tolist(), "p": np.round(p, 4).tolist(),
                        "alive": alive.sum(1).tolist(),
                        "half": alive[:, : k // 2].sum(1).tolist()})  # fmt: skip
    return out


def _speed(job):
    first, n, path, rule, hold, own = job
    os.environ["OMP_NUM_THREADS"] = "1"
    net = load(HERE / path)
    seeds = list(range(first, first + n))
    env = se.SoloEnv(n, hold=hold, seed=7)
    x = env.start(seeds)
    f = env.fleet
    fly = policy_of(net, own, np.random.default_rng([first, 0]))
    sprint = np.arange(se.NM) == SPRINT
    speeds, moves, t = [], np.zeros(se.NM, np.int64), 0
    while (f.outcome == FLYING).any() and t < CAP:
        mask = env.mask()
        v = f.vx * f.dir
        if rule == "no sprint in open air":
            px, py, _, _, on = f.pursuers()
            seen = (on & (np.hypot(px - f.x[:, None], py - f.y[:, None]) < SIGHT)).any(1)
            mask = mask & ~(sprint[None] & ~seen[:, None])
        elif rule == "no sprint over 10 m/s":
            mask = mask & ~(sprint[None] & (v > 10.0)[:, None])
        elif rule == "no sprint":
            mask = mask & ~sprint[None]
        m = fly(x, mask)
        fl = f.outcome == FLYING
        speeds.append(v[fl])
        moves += np.bincount(m[fl], minlength=se.NM)
        x = env.step(m)[0]
        t += hold
    out = [OUTCOMES[o] for o in f.outcome]
    return out, np.concatenate(speeds), moves, (f.ticks * f.ph.dt).tolist()


def _planner(job):
    seeds, air = job
    os.environ["OMP_NUM_THREADS"] = "1"
    f, _ = se.fp.fly(se.FEAT, seeds, ph=se.PH45, air=air)
    gx = f.goal()[0]
    return [[int(s), OUTCOMES[f.outcome[i]], round(float(abs(gx[i] - f.x[i])), 1)]
            for i, s in enumerate(seeds)]  # fmt: skip


def again(pool, args, res) -> None:
    firsts = range(DIAG, DIAG + args.n, CHUNK)
    jobs = [(a, CHUNK, args.net, args.temp, rep, args.hold)
            for rep in range(args.reps) for a in firsts]  # fmt: skip
    rows = sum(pool.map(_again, jobs), [])
    per = {}
    for r in rows:
        per.setdefault(r["seed"], []).append(r)
    landed = np.array([sum(q["outcome"] == "landed" for q in v) for v in per.values()])
    p = landed.sum() / (args.n * args.reps)
    res["again"] = {
        "n": args.n,
        "reps": args.reps,
        "landed_share": float(p),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        # courses by how many of their flights landed (0..reps), and what one pilot with the
        # same odds on every course would give
        "courses_by_landings": np.bincount(landed, minlength=args.reps + 1).tolist(),
        "if_all_courses_alike": [
            round(float(args.n * _binom(args.reps, k, p)), 1) for k in range(args.reps + 1)
        ],
        "frog": dict(Counter(("route", "pad")[r["frog"]] for r in rows if "frog" in r)),
        "hit_by": dict(Counter(r["by"] for r in rows if "by" in r)),
    }
    feats = {}  # what makes a course hard: the landing share by what is on it
    for v in per.values():
        c = v[0]
        ok = np.mean([q["outcome"] == "landed" for q in v])
        keys = [f"route frog {c['frogs'][0]}"]
        for k in range(1, len(ENCOUNTERS)):
            n_k = c["kinds"].count(k)
            keys.append(f"{ENCOUNTERS[k]} x{min(n_k, 2)}{'+' * (n_k > 2)}")
        span = "60-80" if c["length"] < 80 else "80-100" if c["length"] < 100 else "100-120"
        keys.append(f"length {span}")
        for key in keys:
            feats.setdefault(key, []).append(ok)
    res["again"]["landed_by_course"] = {
        k: [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(feats.items())
    }
    print(json.dumps({k: v for k, v in res["again"].items() if k != "landed_by_course"}))
    res["courses"] = [
        {"seed": s, "landed": int(sum(q["outcome"] == "landed" for q in v)),
         "ends": [[q["outcome"], q["togo"], q.get("by", q.get("frog", ""))] for q in v],
         **{k: v[0][k] for k in ("length", "kinds", "frogs", "air")}}
        for s, v in sorted(per.items())
    ]  # fmt: skip


def temps(pool, args, res) -> None:
    res["temps"] = {}
    for temp in args.temps:
        jobs = [(a, CHUNK, args.net, temp, 0, args.hold)
                for a in range(DIAG, DIAG + args.n, CHUNK)]  # fmt: skip
        c = Counter(r["outcome"] for r in sum(pool.map(_again, jobs), []))
        res["temps"][str(temp)] = dict(c)
        print(f"temperature {temp}: landed {c['landed']}/{args.n}  {dict(c)}", flush=True)


def rewind(pool, args, res) -> None:
    jobs = [(a, CHUNK, args.net, args.k, args.hold, args.temp)
            for a in range(DIAG, DIAG + args.n, CHUNK)]  # fmt: skip
    rw = sum(pool.map(_rewind, jobs), [])
    res["rewind"] = {"back_s": list(BACK), "k": args.k, "flights": len(rw), "by_outcome": {}}
    groups = {"all": rw}
    for r in rw:
        groups.setdefault(r["outcome"], []).append(r)
    for g, v in groups.items():
        o = res["rewind"]["by_outcome"][g] = {"flights": len(v)}
        for mode in ("drawn", "top"):
            a = np.array([[np.nan if q is None else q for q in r["alive"][mode]] for r in v])
            o[mode] = np.round(np.nanmean(a, 0), 3).tolist()
            # flights with next to no chance left (under 10%), at each rewind
            doomed = np.nanmean(np.where(np.isnan(a), np.nan, a < 0.1), 0)
            o[mode + "_doomed"] = np.round(doomed, 3).tolist()
        print(f"rewind {g:8s} ({len(v):3d} flights) alive when wound back {list(BACK)} s: "
              f"drawn {o['drawn']}  top {o['top']}  doomed (drawn) {o['drawn_doomed']}",
              flush=True)  # fmt: skip
    res["rewind"]["rows"] = rw


def speed(pool, args, res) -> None:
    res["speed"] = {}
    for rule in RULES:
        jobs = [(a, CHUNK, args.net, rule, args.hold, args.temp)
                for a in range(DIAG, DIAG + args.n, CHUNK)]  # fmt: skip
        parts = pool.map(_speed, jobs)
        outs, secs = sum((p[0] for p in parts), []), sum((p[3] for p in parts), [])
        v = np.concatenate([p[1] for p in parts])
        moves = np.sum([p[2] for p in parts], 0)
        top = np.argsort(-moves)[:5]
        o = res["speed"][rule] = {
            "landed": outs.count("landed"),
            "n": len(outs),
            "outcomes": dict(Counter(outs)),
            "speed_mean": round(float(v.mean()), 2),
            "over_10": round(float(np.mean(v > 10.0)), 3),
            "over_12": round(float(np.mean(v > 12.0)), 3),
            "sprint_share": round(float(moves[SPRINT] / moves.sum()), 3),
            "seconds_landed": round(float(np.mean(
                [t for q, t in zip(outs, secs, strict=True) if q == "landed"])), 2),
            "top_moves": [[f"{KIND[expert.MK[m]]}({expert.MA[m]:g}, {expert.MB[m]:g})",
                           round(float(moves[m] / moves.sum()), 3)] for m in top],
        }  # fmt: skip
        print(f"{rule:22s}: {o}", flush=True)


def planner(pool, args, res) -> None:
    hard = [c["seed"] for c in res["courses"] if c["landed"] <= 1]
    res["planner"] = {"courses": len(hard), "landed_at_most": 1}
    for air in ("estimated", "nominal"):
        out = sum(pool.map(_planner, [(hard[i : i + 4], air) for i in range(0, len(hard), 4)]), [])
        res["planner"][air] = {
            "landed": sum(o == "landed" for _, o, _ in out),
            "failed": [q for q in out if q[1] != "landed"],
        }
        print(f"planner, air {air}, the {len(hard)} hardest courses: {res['planner'][air]}",
              flush=True)  # fmt: skip


def moves(pool, args, res) -> None:
    # the tried move lasts 0.3 s where a move lasts one update, else one decision
    k, first_n = 8, 6 if args.hold == 1 else 1
    jobs = [(a, CHUNK, args.net, k, first_n, args.hold, args.temp)
            for a in range(DIAG, DIAG + args.moves_n, CHUNK)]  # fmt: skip
    rows = sum(pool.map(_moves, jobs), [])
    res["moves"] = {"k": k, "tried_for_updates": first_n * args.hold, "courses": args.moves_n,
                    "by_back": {}}  # fmt: skip
    for back in (1.0, 2.0, 3.0):
        v = [r for r in rows if r["back"] == back]
        as_flown, top, best, good, spread = [], [], [], [], []
        for r in v:
            p, a, h = np.array(r["p"]), np.array(r["alive"]) / k, np.array(r["half"])
            rest = (np.array(r["alive"]) - h) / (k - k // 2)
            as_flown.append(float(p @ a))  # drawing the next move as the network does
            top.append(float(a[np.argmax(p)]))  # its most probable move
            # the move that looks best on half the copies (ties: the one it likes more),
            # scored on the other half
            pick = np.lexsort((p, h))[-1]
            best.append(float(rest[pick]))
            good.append(float(np.mean(a >= 0.75)))
            spread.append(float(a.max() - a.min()))
        res["moves"]["by_back"][str(back)] = o = {
            "states": len(v),
            "alive_as_flown": round(float(np.mean(as_flown)), 3),
            "alive_top_move": round(float(np.mean(top)), 3),
            "alive_best_move": round(float(np.mean(best)), 3),
            "moves_mostly_alive": round(float(np.mean(good)), 3),
            "moves_offered": round(float(np.mean([len(r["offer"]) for r in v])), 1),
            "best_minus_worst": round(float(np.mean(spread)), 3),
        }
        print(f"moves, wound back {back} s: {o}", flush=True)
    res["moves"]["rows"] = rows


def trace(pool, args, res) -> None:
    net = load(HERE / args.net)
    env = se.SoloEnv(1, hold=args.hold, seed=7)
    x = env.start([args.trace_seed])
    f = env.fleet
    fly = policy_of(net, args.temp, np.random.default_rng([args.trace_seed, 0]))
    rows, t = [], 0
    while f.outcome[0] == FLYING and t < CAP:
        mask = env.mask()
        lg = np.where(mask, net(x)[:, : se.NM], -np.inf)[0].astype(np.float64)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        m = fly(x, mask)
        px, py, _, _, on = f.pursuers()
        near = on[0] & (np.hypot(px[0] - f.x[0], py[0] - f.y[0]) < SIGHT)
        rows.append({
            "t": round(t * 0.05, 2),
            "to_go": round(float(abs(f.goal()[0][0] - f.x[0])), 1),
            "height": round(float(f.y[0]), 2),
            "climb": round(float(f.vy[0]), 2),
            "move": _name(int(m[0])),
            "top": [[_name(int(j)), round(float(p[j]), 2)] for j in np.argsort(-p)[:3]],
            # each bird (B) or missile (M) in sight: metres ahead of the drone, metres above it
            "threats": [["B" if k < 6 else "M", round(float((px[0, k] - f.x[0]) * f.dir[0]), 1),
                         round(float(py[0, k] - f.y[0]), 1)] for k in np.nonzero(near)[0]],
        })  # fmt: skip
        x = env.step(m)[0]
        t += args.hold
    res["trace"] = {"seed": args.trace_seed, "outcome": OUTCOMES[f.outcome[0]], "rows": rows}
    print(f"trace of course {args.trace_seed}: {OUTCOMES[f.outcome[0]]} after {t * 0.05:.2f} s")
    for r in rows[-44::4]:
        print("  ", r)


def _name(m: int) -> str:
    return f"{KIND[expert.MK[m]]}({expert.MA[m]:g}, {expert.MB[m]:g})"


def holds(pool, args, res) -> None:
    res["holds"] = {}
    for hold in (1, 2, 4):
        jobs = [(a, CHUNK, args.net, args.temp, 0, hold)
                for a in range(DIAG, DIAG + args.n, CHUNK)]  # fmt: skip
        c = Counter(r["outcome"] for r in sum(pool.map(_again, jobs), []))
        res["holds"][str(hold)] = dict(c)
        print(f"a move held {hold} updates: landed {c['landed']}/{args.n}  {dict(c)}", flush=True)


PARTS = {
    "trace": trace,
    "holds": holds,
    "again": again,
    "moves": moves,
    "temps": temps,
    "rewind": rewind,
    "speed": speed,
    "planner": planner,
}


def main() -> None:
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--parts", nargs="+", default=["again", "temps", "rewind", "speed"],
                    choices=tuple(PARTS))  # fmt: skip
    ap.add_argument("--n", type=int, default=1000, help="courses")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--temps", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75])
    ap.add_argument("--k", type=int, default=32, help="copies flown on from each rewind")
    ap.add_argument("--moves-n", type=int, default=400, help="courses for the moves part")
    ap.add_argument("--trace-seed", type=int, default=50_850)
    ap.add_argument("--hold", type=int, default=1, help="updates the pilot holds a move")
    ap.add_argument("--temp", type=float, default=1.0, help="its moves drawn (1) or top (0)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    path = HERE / "runs" / f"rewind-{args.name}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["net"], res["hold"], res["temp"] = args.net, args.hold, args.temp
    with Pool(args.workers) as pool:
        for part in args.parts:
            PARTS[part](pool, args, res)
            path.write_text(json.dumps(res))


def _binom(n: int, k: int, p: float) -> float:
    from math import comb

    return comb(n, k) * p**k * (1 - p) ** (n - k)


if __name__ == "__main__":
    main()
