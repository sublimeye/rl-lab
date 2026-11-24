"""A network that picks the fair planner's moves, alone or checked by a short search (post 16).

Post 15 copied its planner's *sticks* into a network with DAgger and got 1-2 of 200. Two things
were against it: the teacher knew things the network couldn't (frog timers, the air), and the
teacher's choices are discrete (dodge up or down, wait or go) while a regression on sticks
averages them. Here:
- the teacher is fair_planner.py's, which knows only what a pilot could;
- the network gets what that planner believes as inputs (when each frog may strike next, the
  air's push and the mass as estimated from its own motion), and the move it made last;
- it picks a *move* (one of the planner's ~70: a cruise height and speed, a jink, a land run,
  a wait), as a classifier, and post 15's controller flies it. A second head says, for every
  move, whether the planner found that every plan starting with it fails (its danger sense);
- DAgger: round 0 the planner flies; then the network flies, the planner labelling.

Three pilots, from the same network:
- net: the network's top move, every update;
- shield: every 0.2 s the planner's search, but only over plans whose move now is one of the
  network's top k (and the rest of the last plan); the full search only when all of them fail;
- the planner itself (fair_planner.py).

uv run experiments/016-flawless/pilot.py dagger --name d1
uv run experiments/016-flawless/pilot.py test --name d1 --mode net
uv run experiments/016-flawless/pilot.py test --name d1 --mode shield --k 5
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(HERE.parent / "014-gauntlet"))
sys.path.insert(0, str(HERE.parent / "015-teacher"))

import fair_planner as fp  # noqa: E402  (patches expert's moves and the copies' step)

import expert  # noqa: E402
from gauntlet_env import (  # noqa: E402
    FLYING,
    FROG_JITTER,
    FULL,
    OUTCOMES,
    ROUGH,
    Config,
    FleetEnv,
    Physics,
)

RUNS = HERE / "runs"
FEAT = replace(FULL, predict=True)
PH45 = replace(Physics(), tilt_max=float(np.radians(45.0)))
NM = len(expert.MOVES)
TRAIN_SEED0 = 2_000_000  # training courses (the tests are 20000-20199, the fresh 30000-30199)
NB = 9 + 2 * 8 + 7
FAIL = -500.0  # a plan scoring under this fails in the copies
ABSENT = -3000.0


# --- what the network is given besides post 14's inputs -------------------------------------
def beliefs(pl: fp.FairPlanner, f, prev: np.ndarray) -> np.ndarray:
    """(n, NB): the planner's beliefs about the air and the frogs, and the move made last."""
    n = f.n
    t = f.ticks * f.ph.dt
    e = pl.est
    w = 2 * np.pi / e["period"]
    out = np.zeros((n, NB))
    out[:, 0] = e["wind"] / ROUGH.wind
    out[:, 1] = e["gust"] / ROUGH.gust
    out[:, 2] = np.sin(w * t + e["phase"])
    out[:, 3] = np.cos(w * t + e["phase"])
    for j, dt in enumerate((0.0, 0.5, 1.0)):  # the push now and ahead (m/s^2 / 2.5)
        out[:, 4 + j] = (e["wind"] + e["gust"] * np.sin(w * (t + dt) + e["phase"])) / 2.5
    out[:, 7] = (e["mass"] - fp.MASS0) / 0.2
    out[:, 8] = (e["period"] - 3.5) / 1.5
    jit = 2 * FROG_JITTER
    for k in range(2):
        c = 9 + 8 * k
        out[:, c + 1] = 5.0  # unseen: far away
        out[:, c + 2 : c + 6] = 2.0
        for i in range(n):
            if not (f.frog_on[i, k] and not f.frog_past[i, k] and np.isfinite(pl.first_seen[i, k])):
                continue
            lo, hi, _ = pl.interval(f, i, k)
            b_lo, b_hi = pl._beats
            now = t[i]
            dx = f.frog_p[i, k, 0] - f.x[i]
            win = (lo, hi + fp.LEAPING, lo + b_lo - jit, hi + b_hi + jit + fp.LEAPING)
            out[i, c] = 1.0
            out[i, c + 1] = np.clip(dx / 10, -5, 5)
            out[i, c + 2 : c + 6] = np.clip((np.array(win) - now) / 3, 0, 2)
            out[i, c + 6] = min(len(pl.starts[i][k]), 5) / 5
            out[i, c + 7] = float(abs(dx) < fp.ZONE)
    c = 9 + 16
    kind = expert.MK[prev]
    out[np.arange(n), c + kind] = 1.0
    out[:, c + 5] = expert.MA[prev] / 10
    out[:, c + 6] = expert.MB[prev] / 10
    return out


def move_now(plans: np.ndarray) -> np.ndarray:
    m1, t1, m2, t2, m3 = plans.T
    return np.where(t1 > 0, m1, np.where(t2 > 0, m2, m3))


class Teacher(fp.FairPlanner):
    """The fair planner, which also keeps, at each decision, the best score of the plans
    starting with each move (the network's danger head learns from it), and can be limited
    to plans whose move now is in `allowed` (the shield)."""

    def __init__(self, n: int, **kw):
        super().__init__(n, **kw)
        self.mscore = np.full((n, NM), ABSENT)
        self.decided = np.zeros(n, bool)
        self.allowed: np.ndarray | None = None  # (n, NM) bool
        self.full_searches = 0
        self._order = iter(())

    def believe(self, f, idx, h):
        self._order = iter(list(idx))
        return super().believe(f, idx, h)

    def candidates(self, near_pad: bool, k_h: int) -> np.ndarray:
        c = super().candidates(near_pad, k_h)
        if self.allowed is None:
            return c
        i = next(self._order, None)
        if i is None or self.allowed[i].all():
            return c
        keep = self.allowed[i][move_now(c)]
        return c[keep] if keep.any() else c

    def _decide(self, f, idx) -> None:
        if len(idx) == 0:
            return
        groups, sc, _ = self.evaluate(f, idx)
        if self.allowed is not None:  # the shield: all of the network's picks fail -> full
            k, bad = 0, []
            for grp, i in zip(groups, idx, strict=True):
                if sc[k : k + len(grp)].max() < FAIL and not self.allowed[i].all():
                    bad.append(i)
                k += len(grp)
            if bad:
                keep = self.allowed
                self.allowed = keep.copy()
                self.allowed[bad] = True
                self.full_searches += len(bad)
                groups, sc, _ = self.evaluate(f, idx)
                self.allowed = keep
        act0 = self.act0
        g = f.take(idx)
        gx, _ = g.goal()
        g.dir = np.where(gx >= g.x, 1.0, -1.0)
        canon = expert.control(g, np.full(len(idx), expert.CANON))
        k = 0
        for q, (grp, i) in enumerate(zip(groups, idx, strict=True)):
            s = sc[k : k + len(grp)].copy()
            ms = np.full(NM, ABSENT)
            np.maximum.at(ms, move_now(grp), s)
            self.mscore[i] = ms
            self.decided[i] = True
            if not self.fresh[i]:
                s[0] += self.loyal
            j = int(np.argmax(s))
            if self.band and s[j] > -500:
                near = np.nonzero(s >= s[j] - self.band)[0]
                d = ((act0[k + near] - canon[q]) ** 2).sum(1)
                j = int(near[np.lexsort((-s[near], d))[0]])
            self.plan[i] = grp[j]
            k += len(grp)
        self.age[idx] = 0
        self.fresh[idx] = False
        self.decisions += len(idx)

    def plan_now(self, f) -> np.ndarray:
        """fair_planner's act without the sticks: re-plan the drones due; the move now."""
        self.decided[:] = False
        flying = np.nonzero(f.outcome == FLYING)[0]
        due = flying[self.fresh[flying] | (self.age[flying] >= self.every)]
        live = f.frog_on[due] & ~f.frog_past[due]
        close = np.abs(f.frog_p[due, :, 0] - f.x[due][:, None]) < fp.FAR
        near = (live & close).any(1)
        self.decide(f, due[near])
        self.decide(f, due[~near])
        return self.current()


# --- the network ------------------------------------------------------------------------------
class NumpyNet:
    """ReLU layers, then two heads: the move's logits, and each move's chance of failing."""

    def __init__(self, layers, mean, std):
        self.layers = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in layers]
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)

    def __call__(self, x: np.ndarray):
        h = ((x - self.mean) / self.std).astype(np.float32)
        for w, b in self.layers[:-1]:
            h = np.maximum(h @ w + b, 0.0)
        w, b = self.layers[-1]
        o = h @ w + b
        return o[:, :NM], o[:, NM:]


def make_torch(dim: int, width=(512, 512, 256)):
    import torch as th

    mods, d = [], dim
    for w in width:
        mods += [th.nn.Linear(d, w), th.nn.ReLU()]
        d = w
    mods.append(th.nn.Linear(d, 2 * NM))
    return th.nn.Sequential(*mods)


def to_numpy(model, mean, std) -> NumpyNet:
    lin = [m for m in model if hasattr(m, "weight")]
    layers = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()) for m in lin]
    return NumpyNet(layers, mean, std)


def train(model, X, Y, S, D, steps, lr=1e-3, batch=4096, fail_w=0.5, log=print):
    """Cross-entropy on the planner's move (every update) + binary cross-entropy on whether
    each move fails (decision updates, moves the planner tried)."""
    import torch as th

    th.set_num_threads(8)
    opt = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = th.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    n = len(X)
    g = th.Generator().manual_seed(0)
    perm = th.randperm(n, generator=g)
    nva = max(1, n // 50)
    va, tr = perm[:nva], perm[nva:]
    Xt = th.from_numpy(X)
    Yt = th.from_numpy(Y.astype(np.int64))
    St = th.from_numpy(S)
    Dt = th.from_numpy(D)
    tot = np.zeros(2)
    for k in range(steps):
        idx = tr[th.randint(len(tr), (batch,), generator=g)]
        out = model(Xt[idx].float())
        ce = th.nn.functional.cross_entropy(out[:, :NM], Yt[idx])
        s, d = St[idx].float(), Dt[idx]
        tried = (s > ABSENT + 1) & d[:, None]
        bce = th.nn.functional.binary_cross_entropy_with_logits(
            out[:, NM:], (s < FAIL).float(), reduction="none"
        )
        bce = (bce * tried).sum() / tried.sum().clamp(min=1)
        loss = ce + fail_w * bce
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += (ce.item(), bce.item())
        if (k + 1) % (steps // 4) == 0:
            with th.no_grad():
                o = model(Xt[va].float())
                acc = (o[:, :NM].argmax(1) == Yt[va]).float().mean().item()
            ce, bce = tot / (steps // 4)
            log(f"    step {k + 1}/{steps}: move CE {ce:.3f}, fail BCE {bce:.3f}, "
                f"held-out move accuracy {acc:.3f}")  # fmt: skip
            tot[:] = 0
    return acc


# --- flying -----------------------------------------------------------------------------------
def _fly(job):
    """One worker: fly `seeds` (mode planner / net / shield / dagger), returns the states
    seen, the planner's labels (when it runs) and the results."""
    seeds, mode, net, opts = job
    os.environ["OMP_NUM_THREADS"] = "1"
    env = FleetEnv(len(seeds), Config(PH45, ROUGH, FEAT), seed=opts.get("noise_seed", 7))
    obs = env.start(seeds)
    f = env.fleet
    pl = Teacher(f.n)
    labels = mode in ("planner", "dagger")
    rng = np.random.default_rng(opts.get("noise_seed", 7))
    beta, k_top, cap = opts.get("beta", 0.0), opts.get("k", 5), opts.get("cap", 1300)
    prev = np.full(f.n, expert.DEFAULT)
    X, Y, S, D = [], [], [], []
    t = 0
    t0 = time.time()
    while (f.outcome == FLYING).any() and t < cap:
        flying = f.outcome == FLYING
        pl.ph_real = f.ph
        pl.observe(f)
        pl.estimate(f)
        x = np.concatenate([obs, beliefs(pl, f, prev)], 1).astype(np.float32)
        if net is not None:
            logits, _ = net(x)
        if mode == "shield":
            due = pl.fresh | (pl.age >= pl.every)
            if due.any():
                top = np.argsort(-logits, 1)[:, :k_top]
                allowed = np.zeros((f.n, NM), bool)
                allowed[np.arange(f.n)[:, None], top] = True
                pl.allowed = allowed
            m = pl.plan_now(f)
        elif labels:
            label = pl.plan_now(f)
            m = label
            if mode == "dagger":
                take = rng.random(f.n) < beta
                m = np.where(take, label, np.argmax(logits, 1))
        else:  # net
            m = np.argmax(logits, 1)
        if labels:
            X.append(x[flying].astype(np.float16))
            Y.append(label[flying].astype(np.int16))
            S.append(pl.mscore[flying].astype(np.float16))
            D.append(pl.decided[flying].copy())
        a = expert.control(pl.assumed(f), m)
        pl.remember(f, a)
        pl.age += 1
        prev = np.where(flying, m, prev)
        obs, _, _, _ = env.step(a)
        t += 1
    res = []
    for i, s in enumerate(seeds):
        res.append({"seed": int(s), "outcome": OUTCOMES[f.outcome[i]],
                    "time": round(float(f.ticks[i] * f.ph.dt), 2), "x": round(float(f.x[i]), 1),
                    "togo": round(float(abs(f.goal()[0][i] - f.x[i])), 1),
                    "hit_by": int(f.hit_enc[i])})  # fmt: skip
    stats = {"decisions": pl.decisions, "full": pl.full_searches, "cpu": time.time() - t0}
    if labels:
        return (
            (np.concatenate(X), np.concatenate(Y), np.concatenate(S), np.concatenate(D)),
            res,
            stats,
        )
    return None, res, stats


def fly_many(seeds, mode, net=None, workers=10, chunk=4, **opts):
    from multiprocessing import Pool

    seeds = list(seeds)
    jobs = []
    for j, i in enumerate(range(0, len(seeds), chunk)):
        o = dict(opts)
        o["noise_seed"] = opts.get("noise_seed", 7) + (j if mode in ("planner", "dagger") else 0)
        jobs.append((seeds[i : i + chunk], mode, net, o))
    with Pool(workers) as p:
        parts = p.map(_fly, jobs)
    res = sum((p[1] for p in parts), [])
    stats = {k: sum(p[2][k] for p in parts) for k in parts[0][2]}
    data = None
    if parts[0][0] is not None:
        data = tuple(np.concatenate([p[0][q] for p in parts]) for q in range(4))
    return data, res, stats


def tally(res) -> str:
    c = Counter(r["outcome"] for r in res)
    return f"landed {c['landed']}/{len(res)} {dict(c)}"


# --- DAgger -----------------------------------------------------------------------------------
def load_net(out: Path, tag: str = "net") -> NumpyNet:
    z = np.load(out / f"{tag}.npz", allow_pickle=True)
    n = (len(z.files) - 2) // 2
    layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(n)]
    return NumpyNet(layers, z["mean"], z["std"])


def save_net(net: NumpyNet, path: Path) -> None:
    arrs = {"mean": net.mean, "std": net.std}
    for i, (w, b) in enumerate(net.layers):
        arrs[f"w{i}"], arrs[f"b{i}"] = w, b
    np.savez(path, **arrs)


def dagger(args) -> None:
    import torch as th

    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args)))
    logf = open(out / "log.txt", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    hist = json.loads((out / "rounds.json").read_text()) if (out / "rounds.json").exists() else []
    data = []
    next_seed = TRAIN_SEED0
    net = None
    model, mean, std = None, None, None
    t_start = time.time()
    for r in range(args.rounds):
        n = args.first if r == 0 else args.flights
        seeds = range(next_seed, next_seed + n)
        next_seed += n
        path = out / f"data_{r:02d}.npz"
        t0 = time.time()
        if path.exists():
            z = np.load(path)
            part = (z["X"], z["Y"], z["S"], z["D"])
            res = json.loads(str(z["res"]))
            log(f"round {r}: loaded {len(part[0]):,} states")
        else:
            beta = 1.0 if r == 0 else args.beta**r
            mode = "planner" if r == 0 else "dagger"
            part, res, stats = fly_many(seeds, mode, net, args.workers, args.chunk,
                                        noise_seed=1000 * (r + 1), beta=beta,
                                        cap=1300 if r == 0 else args.cap)  # fmt: skip
            np.savez(path, X=part[0], Y=part[1], S=part[2], D=part[3], res=json.dumps(res))
        who = "planner" if r == 0 else "network"
        log(f"round {r}: {who} flew {n} training courses, {tally(res)}, {len(part[0]):,} states "
            f"({time.time() - t0:.0f} s)")  # fmt: skip
        data.append(part)
        X = np.concatenate([d[0] for d in data])
        Y = np.concatenate([d[1] for d in data])
        S = np.concatenate([d[2] for d in data])
        D = np.concatenate([d[3] for d in data])
        if mean is None:
            mean = X.astype(np.float32).mean(0)
            std = X.astype(np.float32).std(0) + 0.05
        t0 = time.time()
        if model is None or args.fresh:
            th.manual_seed(r)
            model = make_torch(X.shape[1])
            # the normalisation is folded into the inputs at fit time
        Xn = ((X.astype(np.float32) - mean) / std).astype(np.float16)
        acc = train(model, Xn, Y, S, D, args.steps, lr=args.lr, log=log)
        net = to_numpy(model, mean, std)
        save_net(net, out / f"net_{r:02d}.npz")
        save_net(net, out / "net.npz")
        _, test_res, _ = fly_many(range(20_000, 20_000 + args.test_n), "net", net,
                                  args.workers, 20)  # fmt: skip
        landed = sum(x["outcome"] == "landed" for x in test_res)
        log(f"  fit {time.time() - t0:.0f} s; network alone on the test courses: {tally(test_res)} "
            f"(total {(time.time() - t_start) / 60:.0f} min)")  # fmt: skip
        hist.append({"round": r, "flown_by": who, "train_flights": n,
                     "train_landed": sum(x["outcome"] == "landed" for x in res),
                     "states": int(len(X)), "accuracy": round(acc, 4),
                     "test_landed": landed, "test": Counter(x["outcome"] for x in test_res),
                     "minutes": round((time.time() - t_start) / 60, 1)})  # fmt: skip
        (out / "rounds.json").write_text(json.dumps(hist))


def test(args) -> None:
    out = RUNS / args.name
    net = load_net(out, args.net)
    seeds = range(args.first_seed, args.first_seed + args.n)
    t0 = time.time()
    _, res, stats = fly_many(seeds, args.mode, net, args.workers, args.chunk, k=args.k)
    msg = (f"{args.name}/{args.net} {args.mode} k={args.k} seeds {args.first_seed}+: "
           f"{tally(res)} ({time.time() - t0:.0f} s, {stats})")  # fmt: skip
    print(msg, flush=True)
    for r in res:
        if r["outcome"] != "landed":
            print("   ", r, flush=True)
    tag = f"test-{args.net}-{args.mode}" + (f"-k{args.k}" if args.mode == "shield" else "")
    tag += f"-{args.first_seed}"
    (out / f"{tag}.json").write_text(
        json.dumps({"flights": res, "stats": stats, "seconds": round(time.time() - t0)})
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dagger")
    d.add_argument("--name", required=True)
    d.add_argument("--rounds", type=int, default=6)
    d.add_argument("--first", type=int, default=400, help="courses the planner flies, round 0")
    d.add_argument("--flights", type=int, default=300)
    d.add_argument("--beta", type=float, default=0.5)
    d.add_argument("--cap", type=int, default=900)
    d.add_argument("--steps", type=int, default=8000)
    d.add_argument("--lr", type=float, default=1e-3)
    d.add_argument("--fresh", action="store_true", help="a new network every round")
    d.add_argument("--test-n", type=int, default=200)
    d.add_argument("--workers", type=int, default=10)
    d.add_argument("--chunk", type=int, default=4)
    t = sub.add_parser("test")
    t.add_argument("--name", required=True)
    t.add_argument("--net", default="net")
    t.add_argument("--mode", choices=("net", "shield", "planner"), default="net")
    t.add_argument("--k", type=int, default=5)
    t.add_argument("--n", type=int, default=200)
    t.add_argument("--first-seed", type=int, default=20_000)
    t.add_argument("--workers", type=int, default=10)
    t.add_argument("--chunk", type=int, default=5)
    args = ap.parse_args()
    dagger(args) if args.cmd == "dagger" else test(args)


if __name__ == "__main__":
    main()
