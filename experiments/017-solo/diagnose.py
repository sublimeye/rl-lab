"""Why post 16's copy of the planner fails (post 17): from its saved data and networks.

Nothing is trained here. Post 16's DAgger runs saved, per round, every state the flights
visited with the planner's move and, for each of the 59 moves, the best score of the plans
starting with it (under -500: every such plan fails in the planner's copies). Round r's
flights were flown by the network fitted after round r-1, so that network never trained on
them: its picks there are what it does on its own, unseen states.

uv run experiments/017-solo/diagnose.py            # writes runs/diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import refit  # noqa: E402
from refit import ABSENT, FAIL, G1, pilot  # noqa: E402

expert = pilot.expert
D1 = HERE.parent / "016-flawless" / "runs" / "d1"  # post 16's CPU run
KIND = np.array(["cruise", "jink", "land", "wait", "level"])


def frog_near(X: np.ndarray) -> np.ndarray:
    fr = X[:, 58:74].reshape(-1, 2, 8)
    return ((fr[..., 4] > 0.5) & (np.abs(fr[..., 6]) * 10 < 8)).any(1)


def in_sight(X: np.ndarray) -> np.ndarray:
    return (X[:, 18:58].reshape(-1, 4, 10)[..., 4] > 0.5).any(1)


def on_policy(run: Path, r: int) -> dict:
    """The network fitted after round r-1, on round r's states."""
    d = refit.load_round(run, r)
    X, Y, S, D, fl, t = d["X"], d["Y"], d["S"], d["D"], d["fl"], d["t"]
    net = pilot.load_net(run, f"net_{r - 1:02d}")
    logits = net(X)[0]
    pick = logits.argmax(1)
    pm = refit.prev_move(X)
    safe, tried = S >= FAIL, S > ABSENT + 1
    dec = D & safe.any(1)
    fatal = dec & (S[np.arange(len(Y)), pick] < FAIL)
    urg, frog, seen = refit.urgent(X), frog_near(X), in_sight(X)
    calm = ~seen & ~frog
    order = np.lexsort((t, fl))
    same = fl[order][1:] == fl[order][:-1]
    change = np.zeros(len(Y), bool)
    change[order[1:]] = same & (Y[order[1:]] != Y[order[:-1]])
    rows = np.nonzero(D)[0]
    top = np.argsort(-logits[rows], 1)
    alive = safe[rows].any(1)
    out = {
        "round": r,
        "states": int(len(Y)),
        "accuracy": float(np.mean(pick == Y)),
        "accuracy_change": float(np.mean((pick == Y)[change])),
        "differs_from_last": float(np.mean(Y != pm)),
        "keeps_last": float(np.mean((pick == pm)[Y != pm])),
        "picks_label_then": float(np.mean((pick == Y)[Y != pm])),
        "fatal": float(fatal.sum() / dec.sum()),
        "fatal_urgent": float(fatal[urg].sum() / (dec & urg).sum()),
        "fatal_frog": float(fatal[frog].sum() / (dec & frog).sum()),
        "fatal_calm": float(fatal[calm].sum() / max((dec & calm).sum(), 1)),
        "top_k_safe": {
            k: float(np.take_along_axis(safe[rows], top[:, :k], 1).any(1)[alive].mean())
            for k in (1, 2, 3, 5)
        },  # fmt: skip
        "safe_moves_median": float(np.median(safe[rows].sum(1))),
        "tried_moves_mean": float(tried[rows].sum(1).mean()),
    }
    # each failed flight: the first decision whose executed move fails in the copies, and the
    # first decision with no safe move left, in seconds before the end
    nxt = np.full(len(Y), -1)
    nxt[order[:-1]] = np.where(same, pm[order[1:]], -1)  # the move flown at this update
    lead, lost, count, pairs = {}, [], [], Counter()
    for i, res in enumerate(d["res"]):
        if res["outcome"] in ("landed", "flying", "timeout"):
            continue
        fr = np.nonzero(fl == i)[0]
        fr = fr[np.argsort(t[fr])]
        dr = fr[D[fr]]
        ex = nxt[dr]
        ok = safe[dr].any(1)
        bad = (ex >= 0) & ok & (S[dr, np.maximum(ex, 0)] < FAIL)
        count.append(int(bad.sum()))
        if (~ok).any():
            lost.append((len(fr) - t[dr][~ok][0]) * 0.05)
        if bad.any():
            j = int(np.nonzero(bad)[0][0])
            lead.setdefault(res["outcome"], []).append((len(fr) - t[dr][j]) * 0.05)
            pairs[f"{KIND[expert.MK[Y[dr[j]]]]}/{KIND[expert.MK[ex[j]]]}"] += 1
    out["first_fatal_s"] = {k: [float(np.percentile(v, q)) for q in (25, 50, 75)] + [len(v)]
                            for k, v in lead.items() if len(v) >= 10}  # fmt: skip
    out["lost_s"] = [float(np.percentile(lost, q)) for q in (25, 50, 75)] + [len(lost)]
    out["fatal_per_failed_flight"] = float(np.median(count))
    out["first_fatal_planner_vs_flown"] = dict(pairs.most_common(6))
    return out


def planner_round(run: Path) -> dict:
    """Round 0: the planner's own flights."""
    d = refit.load_round(run, 0)
    X, Y, S, D = d["X"], d["Y"], d["S"], d["D"]
    safe, tried = (S >= FAIL)[D], (S > ABSENT + 1)[D]
    return {
        "states": int(len(Y)),
        "flights": len(d["res"]),
        "label_is_last_move": float(np.mean(refit.prev_move(X) == Y)),
        "kinds": {k: int(v) for k, v in Counter(KIND[expert.MK[Y]]).items()},
        "safe_moves_median": float(np.median(safe.sum(1))),
        "tried_moves_mean": float(tried.sum(1).mean()),
        "few_safe": float(np.mean(safe.sum(1) <= 5)),
    }


# --- post 16's network, flown as post 16 flew it (its top move; the planner object only keeps
# the air estimate and the frog memory) -------------------------------------------------------
def _fly16(job):
    seeds, tag, feat, veto = job
    os.environ["OMP_NUM_THREADS"] = "1"
    from gauntlet_env import FLYING, OUTCOMES, ROUGH, Config, FleetEnv

    net = pilot.load_net(G1, tag)
    env = FleetEnv(len(seeds), Config(pilot.PH45, ROUGH, feat), seed=7)
    obs = env.start(seeds)
    f = env.fleet
    pl = pilot.Teacher(f.n)
    prev = np.full(f.n, expert.DEFAULT)
    t = 0
    while (f.outcome == FLYING).any() and t < 1300:
        flying = f.outcome == FLYING
        pl.ph_real = f.ph
        pl.observe(f)
        pl.estimate(f)
        x = np.concatenate([obs, pilot.beliefs(pl, f, prev)], 1).astype(np.float32)
        logits, danger = net(x)
        if veto is not None:  # its danger head rules out moves it expects to fail
            ok = 1 / (1 + np.exp(-danger)) < veto
            m = np.where(ok, logits, -1e9).argmax(1)
            m = np.where(ok.any(1), m, danger.argmin(1))
        else:
            m = logits.argmax(1)
        a = expert.control(pl.assumed(f), m)
        pl.remember(f, a)
        prev = np.where(flying, m, prev)
        obs, _, _, _ = env.step(a)
        t += 1
    return [OUTCOMES[o] for o in f.outcome]


def fly16(pool, feat, veto=None, tag="net", first=20_000) -> dict:
    jobs = [(list(range(first + i, first + i + 20)), tag, feat, veto) for i in range(0, 200, 20)]
    c = Counter(sum(pool.map(_fly16, jobs), []))
    return {"landed": c["landed"], "outcomes": dict(c)}


def main() -> None:
    from multiprocessing import Pool

    from solo_eval import KINDS

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    out = {"planner_round": planner_round(G1), "on_policy": {}}
    print("round 0 (the planner's flights):", out["planner_round"])
    for name, run in (("gpu", G1), ("cpu", D1)):
        out["on_policy"][name] = [on_policy(run, r) for r in range(1, 6)]
        for o in out["on_policy"][name]:
            print(f"{name} round {o['round']}: accuracy {o['accuracy']:.3f} (planner changes "
                  f"move: {o['accuracy_change']:.3f}); fatal picks {o['fatal']:.3f} (urgent "
                  f"{o['fatal_urgent']:.3f}, frog near {o['fatal_frog']:.3f}, calm "
                  f"{o['fatal_calm']:.3f}); keeps its last move {o['keeps_last']:.3f}; top-k "
                  f"holds a safe move {o['top_k_safe']}")  # fmt: skip
        o = out["on_policy"][name][-1]
        print(f"   round 5 failed flights: first fatal move (s before the end, quartiles, n) "
              f"{o['first_fatal_s']}; lost {o['lost_s']}; fatal decisions per failed flight "
              f"{o['fatal_per_failed_flight']}; planner/flown kinds "
              f"{o['first_fatal_planner_vs_flown']}")  # fmt: skip
    with Pool(args.workers) as pool:
        full = replace(pilot.FEAT)
        out["post16_net"] = {"gauntlet": fly16(pool, full)}
        out["post16_net_fresh"] = fly16(pool, full, first=30_000)
        print(f"post 16's network alone, fresh courses: {out['post16_net_fresh']}")
        for kind, feat in KINDS.items():
            out["post16_net"][kind] = fly16(pool, feat)
        for k, v in out["post16_net"].items():
            print(f"post 16's network alone, {k}: landed {v['landed']}/200 {v['outcomes']}")
        out["veto"] = {str(tau): fly16(pool, full, veto=tau) for tau in (0.5, 0.2, 0.1)}
        print("post 16's network with its danger head as a veto:",
              {k: v["landed"] for k, v in out["veto"].items()})  # fmt: skip
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "diagnosis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
