"""Second tries (post 17): the simulator shows the network which move would have saved it.

Practice (train.py) learns each decision from how one flight ended. Here the question is put
to the simulator directly. One pass:
1. the network flies fresh courses alone, a snapshot kept every half second;
2. every failed flight is wound back 1, 2 and 3 s, and from each of those moments every move
   on offer is tried as the next move (held 0.3 s, K copies each, the network flying on as
   usual): a move's score is the share of its copies still alive 4 s after the original end;
3. the network is fitted to put its probability, at those moments, on the moves that lived
   (the loss is minus the log of the survival it can expect from its own draw), while staying
   what it was on ordinary moments from the same flights (a KL term to the network as it was).
A tenth of the failed flights are held out of the fit; the survival the network can expect at
their rewound moments, before and after, says whether the lesson carries to flights it wasn't
shown. Copies of the simulator are used in training only: at flight time it is still a network
choosing a move.

uv run experiments/017-solo/retry.py --name g1 --init runs/f1-r350/last.npz --passes 6
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import solo_env as se  # noqa: E402
from solo_net import SoloNet, load, save  # noqa: E402

from gauntlet_env import FLYING, LANDED, OUTCOMES  # noqa: E402

RUNS = HERE / "runs"
DEV = 40_000
PRACTICE = 5_000_000  # courses for second tries: 5,000,000 up, a new block every pass
OPEN_PAD = replace(se.FEAT, guard=0.0)
SNAP = 10  # a snapshot every this many updates (0.5 s)
AFTER = 80  # a second try must outlive the original flight by this many updates (4 s)
CHUNK = 25
CAP = 1300
_NETS: dict[str, SoloNet] = {}


def _net(path: str) -> SoloNet:
    if path not in _NETS:
        _NETS.clear()
        _NETS[path] = load(path)
    return _NETS[path]


def _collect(job):
    """One chunk of courses: the flights, the second tries of those that failed, and a few
    ordinary moments. -> (outcomes, X, offer, score, flight seed, seconds back, XA, offerA)"""
    first, path, feat, k, hold0, backs, keep = job
    os.environ["OMP_NUM_THREADS"] = "1"
    net = _net(path)
    seeds = list(range(first, first + CHUNK))
    env = se.SoloEnv(CHUNK, feat, hold=1, seed=first % 100_003)
    x = env.start(seeds)
    f = env.fleet
    rng = np.random.default_rng([first, 0])
    snaps, t = [], 0
    end = np.zeros(CHUNK, np.int64)
    xa, ma = [], []
    while (f.outcome == FLYING).any() and t < CAP:
        mask = env.mask()
        if t % SNAP == 0:
            snaps.append(copy.deepcopy(env))
            pick = (f.outcome == FLYING) & (rng.random(CHUNK) < keep)
            xa.append(x[pick])
            ma.append(mask[pick])
        x = env.step(net.sample(x, mask, rng))[0]
        t += 1
        end = np.where((f.outcome != FLYING) & (end == 0), t, end)
    X, M, Q, who, when = [], [], [], [], []
    for i in np.nonzero((f.outcome != LANDED) & (f.outcome != FLYING))[0]:
        for back in backs:
            t0 = end[i] - round(back * 20)
            if t0 < 0:
                continue
            j = t0 // SNAP
            one = snaps[j].fork(int(i), 1, seed=1)
            offer = one.mask()[0]
            moves = np.nonzero(offer)[0]
            fork = snaps[j].fork(int(i), len(moves) * k, seed=int(3 + 7919 * seeds[i] + 31 * j))
            xf = fork.inputs()
            for u in range(end[i] - j * SNAP + AFTER):
                if not (fork.fleet.outcome == FLYING).any():
                    break
                m = np.repeat(moves, k) if u < hold0 else net.sample(xf, fork.mask(), rng)
                xf = fork.step(m)[0]
            o = fork.fleet.outcome
            alive = ((o == FLYING) | (o == LANDED)).reshape(len(moves), k).mean(1)
            q = np.zeros(se.NM, np.float32)
            q[moves] = alive
            X.append(one.inputs()[0])
            M.append(offer)
            Q.append(q)
            who.append(seeds[i])
            when.append(back)
    return ([OUTCOMES[o] for o in f.outcome],
            np.array(X, np.float32).reshape(-1, se.DIM), np.array(M, bool).reshape(-1, se.NM),
            np.array(Q, np.float32).reshape(-1, se.NM), np.array(who, np.int64),
            np.array(when, np.float32), np.concatenate(xa), np.concatenate(ma))  # fmt: skip


def _dev(job):
    first, path, feat, sampled = job
    os.environ["OMP_NUM_THREADS"] = "1"
    net = _net(path)
    seeds = list(range(first, first + CHUNK))
    if sampled:
        rng = np.random.default_rng(seeds[0])
        out, f = se.fly(lambda x, mask: net.sample(x, mask, rng), seeds, feat, hold=1)
    else:
        out, f = se.fly(net.act, seeds, feat, hold=1)
    return out, (f.ticks * f.ph.dt).tolist()


def fly_dev(pool, path: str, n: int, feat=se.FEAT, sampled: bool = True) -> dict:
    parts = pool.map(_dev, [(a, path, feat, sampled) for a in range(DEV, DEV + n, CHUNK)])
    outs, secs = sum((p[0] for p in parts), []), sum((p[1] for p in parts), [])
    c = dict(Counter(outs))
    t = [s for o, s in zip(outs, secs, strict=True) if o == "landed"]
    c["seconds"] = round(float(np.mean(t)), 2) if t else 0.0
    c.setdefault("landed", 0)
    return c


def expect(net: SoloNet, X, M, Q) -> np.ndarray:
    """The survival the network can expect from its own draw of the next move, per moment."""
    lg = np.where(M, net(X)[:, : se.NM].astype(np.float64), -np.inf)
    p = np.exp(lg - lg.max(1, keepdims=True))
    return (p / p.sum(1, keepdims=True) * Q).sum(1)


def fit(net: SoloNet, tr, va, anchors, args, log) -> SoloNet:
    import torch as th

    dev = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    th.manual_seed(args.seed)

    def mlp():
        mods = []
        for w, _ in net.layers[:-1]:
            mods += [th.nn.Linear(*w.shape), th.nn.ReLU()]
        mods.append(th.nn.Linear(*net.layers[-1][0].shape))
        model = th.nn.Sequential(*mods)
        lin = [m for m in model if hasattr(m, "weight")]
        with th.no_grad():
            for m, (w, b) in zip(lin, net.layers, strict=True):
                m.weight.copy_(th.from_numpy(w.T))
                m.bias.copy_(th.from_numpy(b))
        return model.to(dev)

    def to(a, kind=th.float32):
        return th.from_numpy(np.ascontiguousarray(a)).to(dev, kind)

    model, old = mlp(), mlp()
    mean, std = to(net.mean), to(net.std)
    X, M, Q = to((tr[0] - net.mean) / net.std), to(tr[1], th.bool), to(tr[2])
    XA, MA = to((anchors[0] - net.mean) / net.std), to(anchors[1], th.bool)
    del mean, std
    use = Q.max(1).values >= 0.5  # a moment with no move that mostly lives teaches nothing
    X, M, Q = X[use], M[use], Q[use]
    opt = th.optim.Adam(model.parameters(), lr=args.lr)

    def numpy_net() -> SoloNet:
        lin = [m for m in model if hasattr(m, "weight")]
        return SoloNet([(m.weight.detach().cpu().numpy().T.copy(),
                         m.bias.detach().cpu().numpy().copy()) for m in lin],
                       net.mean, net.std)  # fmt: skip

    best, best_net, since = float(expect(net, *va).mean()) if len(va[0]) else 0.0, net, 0
    log(
        f"    fitting {len(X)} moments ({int((~use).sum())} with no move that lives left out), "
        f"{len(XA)} ordinary ones; held-out moments {len(va[0])}: survival expected "
        f"{best:.3f}, with the best move {float(va[2].max(1).mean()) if len(va[0]) else 0:.3f}"
    )
    for step in range(1, args.steps + 1):
        idx = th.randint(len(X), (args.batch,), device=dev)
        x = X[idx] + args.noise * th.randn((args.batch, se.DIM), device=dev)
        lsm = th.log_softmax(model(x).masked_fill(~M[idx], -1e9), 1)
        lived = (lsm.exp() * Q[idx]).sum(1)
        loss_q = -th.log(lived + 1e-3).mean()
        ia = th.randint(len(XA), (args.batch,), device=dev)
        with th.no_grad():
            was = th.log_softmax(old(XA[ia]).masked_fill(~MA[ia], -1e9), 1)
        now = th.log_softmax(model(XA[ia]).masked_fill(~MA[ia], -1e9), 1)
        kl = (was.exp() * (was - now).masked_fill(~MA[ia], 0.0)).sum(1).mean()
        loss = loss_q + args.keep * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.check == 0:
            cand = numpy_net()
            held = float(expect(cand, *va).mean()) if len(va[0]) else 0.0
            log(
                f"    step {step}: fitted moments' survival {lived.mean().item():.3f}, drift on "
                f"ordinary moments (KL) {kl.item():.4f}, held-out survival {held:.3f}"
            )
            if held > best + 1e-4:
                best, best_net, since = held, cand, 0
            else:
                since += 1
                if since >= args.patience:
                    break
    return best_net


def main() -> None:
    from multiprocessing import get_context

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", required=True, help="the network to start from (.npz)")
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--courses", type=int, default=8000, help="flown per pass")
    ap.add_argument("--open", type=float, default=0.25, help="share with no frog on the pad")
    ap.add_argument("--k", type=int, default=4, help="copies per move")
    ap.add_argument("--first-hold", type=int, default=6, help="updates the tried move is held")
    ap.add_argument("--backs", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--memory", type=int, default=3, help="passes whose moments are fitted")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--check", type=int, default=50)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--keep", type=float, default=1.0, help="weight of staying as it was")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--dev-n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--first-pass", type=int, default=0, help="carry a run on from this pass")
    args = ap.parse_args()
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args)))
    logf = open(out / "log.txt", "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    hist = json.loads((out / "retry.json").read_text()) if args.first_pass else []
    path = str(HERE / args.init)
    with get_context("spawn").Pool(args.workers) as pool:
        if not args.first_pass:
            d = fly_dev(pool, path, args.dev_n)
            o = fly_dev(pool, path, 200, OPEN_PAD)
            log(
                f"start: {args.dev_n} dev courses landed {d['landed']} ({d['seconds']} s a "
                f"landing) {d}; open pads {o['landed']}/200"
            )
            hist.append({"pass": 0, "dev": d, "dev_open": o})
        memory: list[tuple] = []
        for it in range(args.first_pass + 1, args.passes + 1):
            t0 = time.monotonic()
            base = PRACTICE + it * 200_000
            jobs = []
            for c in range(args.courses // CHUNK):
                feat = OPEN_PAD if (c % 100) < 100 * args.open else se.FEAT
                jobs.append((base + c * CHUNK, path, feat, args.k, args.first_hold,
                             tuple(args.backs), 0.04))  # fmt: skip
            parts = pool.map(_collect, jobs, chunksize=1)
            outs = Counter(sum((p[0] for p in parts), []))
            X, M, Q, who, when, XA, MA = (np.concatenate([p[q] for p in parts])
                                          for q in range(1, 8))  # fmt: skip
            t_fly = time.monotonic() - t0
            net = load(path)
            asis = expect(net, X, M, Q)
            row = {
                "pass": it,
                "courses": args.courses,
                "landed": outs["landed"],
                "outcomes": dict(outs),
                "moments": int(len(X)),
                "failed_flights": int(len(np.unique(who))),
                "alive_as_flown": {str(b): round(float(asis[when == b].mean()), 3)
                                   for b in args.backs},
                "alive_best_move": {str(b): round(float(Q[when == b].max(1).mean()), 3)
                                    for b in args.backs},
            }  # fmt: skip
            log(
                f"pass {it}: {args.courses} courses flown, {outs['landed']} landed "
                f"{dict(outs)}; {row['failed_flights']} failed flights, {len(X)} rewound "
                f"moments ({t_fly:.0f} s). Survival expected there as it flies "
                f"{row['alive_as_flown']}; with the best move found (flattered by luck) "
                f"{row['alive_best_move']}"
            )
            held = np.isin(who, np.unique(who)[::10])  # a tenth of the failed flights
            memory = [*memory, (X[~held], M[~held], Q[~held])][-args.memory :]
            tr = tuple(np.concatenate([m[q] for m in memory]) for q in range(3))
            va = (X[held], M[held], Q[held])
            new = fit(net, tr, va, (XA, MA), args, log)
            path = str(out / f"pass_{it:02d}.npz")
            save(new, Path(path))
            save(new, out / "last.npz")
            row["held_before"] = round(float(expect(net, *va).mean()), 3)
            row["held_after"] = round(float(expect(new, *va).mean()), 3)
            row["dev"] = fly_dev(pool, path, args.dev_n)
            row["dev_top"] = fly_dev(pool, path, 1000, sampled=False)
            row["dev_open"] = fly_dev(pool, path, 200, OPEN_PAD)
            row["minutes"] = round((time.monotonic() - t0) / 60, 2)
            hist.append(row)
            (out / "retry.json").write_text(json.dumps(hist))
            d = row["dev"]
            log(
                f"pass {it} done in {row['minutes']} min: held-out failed flights' survival "
                f"{row['held_before']} -> {row['held_after']}; {args.dev_n} dev courses landed "
                f"{d['landed']} ({d['seconds']} s a landing) {d}; top move "
                f"{row['dev_top']['landed']}/1000; open pads {row['dev_open']['landed']}/200"
            )


if __name__ == "__main__":
    main()
