"""Refit post 16's move classifier on its saved states, one change at a time (post 17, diagnosis).

Post 16 quoted 0.90-0.94 held-out accuracy, but held out *states* at random: every held-out
state had its neighbours (0.05 s earlier and later, same flight) in the training set. Here
whole flights are held out, and each fit is also flown alone on 200 dev courses (40000-40199,
used by nothing else), which takes seconds.

uv run experiments/017-solo/refit.py --name base
uv run experiments/017-solo/refit.py --name noprev --no-prev
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "016-flawless"))

import pilot  # noqa: E402
import solo_env as se  # noqa: E402
from solo_net import SoloNet, save  # noqa: E402

expert = pilot.expert
from gauntlet_env import FLYING, OUTCOMES, ROUGH, Config, FleetEnv  # noqa: E402

G1 = HERE.parent / "016-gpu" / "runs" / "g1"  # post 16's GPU run (local only: runs/ is ignored)
NM, FAIL, ABSENT = pilot.NM, pilot.FAIL, pilot.ABSENT
N_OBS = 318
AIR = slice(N_OBS, N_OBS + 9)  # the planner's estimate of the air and the mass
PREV = slice(N_OBS + 25, N_OBS + 32)  # the move made last, among the belief inputs
DEV, TEST, FRESH = 40_000, 20_000, 30_000


def load_round(run: Path, r: int) -> dict:
    """One round's states, with the flight and the update each belongs to (a worker flew 4
    courses at a time and stored, update by update, the drones still flying)."""
    z = np.load(run / f"data_{r:02d}.npz")
    res = json.loads(str(z["res"]))
    upd = [int(math.ceil(round(x["time"] * 60) / 3)) for x in res]
    n = len(z["Y"])
    fl, tt = np.zeros(n, np.int32), np.zeros(n, np.int32)
    k = 0
    for c in range(0, len(res), 4):
        us = upd[c : c + 4]
        for t in range(max(us)):
            for j, u in enumerate(us):
                if u > t:
                    fl[k], tt[k] = c + j, t
                    k += 1
    assert k == n, (k, n)
    return {"X": z["X"].astype(np.float32), "Y": z["Y"].astype(np.int64),
            "S": z["S"].astype(np.float32), "D": z["D"], "res": res, "fl": fl, "t": tt}  # fmt: skip


def load_all(run: Path = G1, rounds=range(6)) -> dict:
    parts, off = [], 0
    for r in rounds:
        d = load_round(run, r)
        d["fl"] = d["fl"] + off
        d["round"] = np.full(len(d["Y"]), r, np.int8)
        off += len(d["res"])
        parts.append(d)
    out = {
        k: np.concatenate([p[k] for p in parts]) for k in ("X", "Y", "S", "D", "fl", "t", "round")
    }
    out["flights"] = off
    return out


def prev_move(X: np.ndarray) -> np.ndarray:
    """The move made last, decoded from the belief inputs."""
    kind = X[:, N_OBS + 25 : N_OBS + 30].argmax(1)
    a, b = X[:, N_OBS + 30] * 10, X[:, N_OBS + 31] * 10
    d = (expert.MK[None] != kind[:, None]) * 100.0
    d = d + np.abs(expert.MA[None] - a[:, None]) + np.abs(expert.MB[None] - b[:, None])
    return d.argmin(1)


def urgent(X: np.ndarray) -> np.ndarray:
    """A bird or missile whose closest approach (flying straight) is within 2 s and 2 m."""
    sl = X[:, 18:58].reshape(-1, 4, 10)
    miss = np.hypot(sl[..., 8], sl[..., 9]) * 5
    return ((sl[..., 4] > 0.5) & (sl[..., 7] * 3 < 2) & (miss < 2)).any(1)


def score(pick: np.ndarray, d: dict, rows: np.ndarray) -> dict:
    """How a network's picks compare with the planner on states `rows`."""
    X, Y, S, D = d["X"][rows], d["Y"][rows], d["S"][rows], d["D"][rows]
    pm = prev_move(X)
    safe = S >= FAIL
    dec = D & safe.any(1)
    fatal = dec & (S[np.arange(len(Y)), pick] < FAIL)
    urg = urgent(X)
    return {
        "accuracy": float(np.mean(pick == Y)),
        "accuracy_change": float(np.mean((pick == Y)[Y != pm])),  # planner wants another move
        "keeps_last": float(np.mean((pick == pm)[Y != pm])),
        "fatal": float(fatal.sum() / dec.sum()),
        "fatal_urgent": float(fatal[urg].sum() / max((dec & urg).sum(), 1)),
        "fatal_calm": float(fatal[~urg].sum() / max((dec & ~urg).sum(), 1)),
    }


def fit(d: dict, tr: np.ndarray, args, log=print):
    import torch as th

    th.manual_seed(args.seed)
    dev = "mps" if th.backends.mps.is_available() and not args.cpu else "cpu"
    X = d["X"][tr].copy()
    hidden = [sl for sl, on in ((PREV, args.no_prev), (AIR, args.no_air)) if on]
    for sl in hidden:
        X[:, sl] = 0.0
    mean, std = X.mean(0), X.std(0) + 0.05
    for sl in hidden:
        mean[sl], std[sl] = 0.0, 1.0
    Xt = th.from_numpy((X - mean) / std).to(dev)
    Yt = th.from_numpy(d["Y"][tr]).to(dev)
    St = th.from_numpy(d["S"][tr]).to(dev)
    width = tuple(args.width)
    mods, k = [], Xt.shape[1]
    for w in width:
        mods += [th.nn.Linear(k, w), th.nn.ReLU()]
        if args.dropout:
            mods.append(th.nn.Dropout(args.dropout))
        k = w
    mods.append(th.nn.Linear(k, 2 * NM))
    model = th.nn.Sequential(*mods).to(dev)
    opt = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = th.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    w_state = None
    if args.urgent_w != 1.0:
        w_state = th.from_numpy(np.where(urgent(d["X"][tr]), args.urgent_w, 1.0)).float().to(dev)
    t0 = time.time()
    for k in range(args.steps):
        idx = th.randint(len(Xt), (args.batch,), device=dev)
        x = Xt[idx]
        if args.noise:
            x = x + args.noise * th.randn_like(x)
        out = model(x)
        lp = th.log_softmax(out[:, :NM], 1)
        s = St[idx]
        tried = s > ABSENT + 1
        safe = s >= FAIL
        ce = -lp.gather(1, Yt[idx][:, None])[:, 0]
        if args.loss == "safe":  # all the probability on moves the planner found safe
            has = safe.any(1)
            mass = th.logsumexp(lp.masked_fill(~safe, -1e9), 1)
            ce = th.where(has, -mass + args.label_w * ce, ce)
        elif args.loss == "soft":  # the planner's scores as a distribution over moves
            tgt = th.softmax((s - s.max(1, keepdim=True).values).masked_fill(~tried, -1e9)
                             / args.temp, 1)  # fmt: skip
            ce = -(tgt * lp).sum(1) + args.label_w * ce
        if w_state is not None:
            ce = ce * w_state[idx]
        bce = th.nn.functional.binary_cross_entropy_with_logits(
            out[:, NM:], (s < FAIL).float(), reduction="none"
        )
        bce = (bce * tried).sum() / tried.sum().clamp(min=1)
        loss = ce.mean() + 0.5 * bce
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (k + 1) % max(args.steps // 4, 1) == 0:
            log(f"    step {k + 1}/{args.steps}: loss {loss.item():.3f} ({time.time() - t0:.0f} s)")
    model.to("cpu").eval()
    lin = [m for m in model if hasattr(m, "weight")]
    layers = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()) for m in lin]
    for sl in hidden:
        layers[0][0][sl] = 0.0  # whatever the flight feeds there is ignored
    return pilot.NumpyNet(layers, mean, std)


def solo(net) -> SoloNet:
    """The same network for solo_env's inputs (no air estimate: those 9 inputs are dropped;
    it must have been fitted with --no-air)."""
    keep = np.r_[0:N_OBS, N_OBS + 9 : N_OBS + 32]
    w0, b0 = net.layers[0]
    assert not w0[AIR].any()
    return SoloNet([(w0[keep], b0), *net.layers[1:]], net.mean[keep], net.std[keep])


def _fly_solo(job):
    seeds, net, hold = job
    os.environ["OMP_NUM_THREADS"] = "1"
    return se.fly(net.act, seeds, hold=hold)[0]


def fly_solo(net: SoloNet, first: int, n: int = 1000, hold: int = 1, workers: int = 10) -> Counter:
    """A network alone in solo_env: still-air controller, no planner object at all."""
    from multiprocessing import Pool

    jobs = [(list(range(first + i, first + i + 25)), net, hold) for i in range(0, n, 25)]
    with Pool(workers) as p:
        return Counter(sum(p.map(_fly_solo, jobs), []))


def _fly(job):
    """A network alone on `seeds` (016's "net" pilot: its top move every update)."""
    seeds, net = job
    os.environ["OMP_NUM_THREADS"] = "1"
    env = FleetEnv(len(seeds), Config(pilot.PH45, ROUGH, pilot.FEAT), seed=7)
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
        m = net(x)[0].argmax(1)
        a = expert.control(pl.assumed(f), m)
        pl.remember(f, a)
        prev = np.where(flying, m, prev)
        obs, _, _, _ = env.step(a)
        t += 1
    return [OUTCOMES[o] for o in f.outcome]


def fly(net, first: int, n: int = 200, workers: int = 10) -> Counter:
    from multiprocessing import Pool

    jobs = [(list(range(first + i, first + i + 20)), net) for i in range(0, n, 20)]
    with Pool(workers) as p:
        return Counter(sum(p.map(_fly, jobs), []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-prev", action="store_true", help="hide the move made last")
    ap.add_argument("--no-air", action="store_true", help="hide the planner's air estimate")
    ap.add_argument("--all", action="store_true", help="fit on the held-out flights too")
    ap.add_argument("--loss", choices=("ce", "safe", "soft"), default="ce")
    ap.add_argument("--label-w", type=float, default=0.1)
    ap.add_argument("--temp", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0, help="input noise while fitting (sd)")
    ap.add_argument("--urgent-w", type=float, default=1.0)
    ap.add_argument("--width", type=int, nargs="+", default=[512, 512, 256])
    ap.add_argument("--share", type=float, default=1.0, help="share of the training flights used")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--test", action="store_true", help="also fly the test and fresh courses")
    args = ap.parse_args()
    d = load_all()
    rng = np.random.default_rng(0)
    held = rng.random(d["flights"]) < 0.1  # whole flights held out, the same for every fit
    use = (~held | args.all) & (rng.random(d["flights"]) < args.share)
    tr, va = np.nonzero(use[d["fl"]])[0], np.nonzero(held[d["fl"]])[0]
    print(f"{args.name}: {use.sum()} flights ({len(tr):,} states) to fit, {held.sum()} held out "
          f"({len(va):,} states)", flush=True)  # fmt: skip
    net = fit(d, tr, args)
    out = {"args": vars(args), "flights": int(use.sum()), "states": int(len(tr))}
    for tag, rows in (("fitted", tr[:: max(len(tr) // 40_000, 1)]), ("held_out", va)):
        out[tag] = score(net(d["X"][rows])[0].argmax(1), d, rows)
        print(f"  {tag:8s}: " + ", ".join(f"{k} {v:.3f}" for k, v in out[tag].items()), flush=True)
    for tag, first in (("dev", DEV),) + ((("test", TEST), ("fresh", FRESH)) if args.test else ()):
        c = fly(net, first, workers=args.workers)
        out[tag] = dict(c)
        print(f"  alone on the {tag} courses: landed {c['landed']}/200 {dict(c)}", flush=True)
    (HERE / "runs").mkdir(exist_ok=True)
    if args.no_air:  # the pilot post 17 trains on: no air estimate, still-air controller
        sn = solo(net)
        save(sn, HERE / "runs" / f"solo-{args.name}.npz")
        for hold in (1, 2, 4):
            c = fly_solo(sn, DEV, hold=hold, workers=args.workers)
            out[f"solo_dev_hold{hold}"] = dict(c)
            print(f"  alone in solo_env, a move held {hold} updates, 1000 dev courses: landed "
                  f"{c['landed']} {dict(c)}", flush=True)  # fmt: skip
    pilot.save_net(net, HERE / "runs" / f"refit-{args.name}.npz")
    (HERE / "runs" / f"refit-{args.name}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
