"""PPO over the planner's moves (post 17): the network picks a move, post 15's controller flies it,
and the only teacher is how the flight ends.

uv run experiments/017-solo/train.py --name m1 --minutes 45
uv run experiments/017-solo/train.py --name m2 --init runs/refit-x.npz --bc 0.5   # from a copy

Eight processes each step 1024 drones (solo_env.SoloEnv); the network runs on the Mac's GPU.
A course is, at random: a full test-like course (40%), a dodge drill (30%: 20-50 m, a waypoint
every 8-14 m, no frogs) or a frog drill (30%: 15-35 m, the pad's frog, half the time another).
Every so often the pilot flies the 200 dev courses (40000-40199) with its top move, no noise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, deque
from dataclasses import replace
from multiprocessing import Pipe, Process
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import solo_env as se  # noqa: E402
from solo_net import SoloNet, load, save  # noqa: E402

RUNS = HERE / "runs"
DEV, TEST, FRESH = 40_000, 20_000, 30_000
CALM = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
SCHOOL = replace(se.FEAT, route=(15.0, 130.0), first=(6.0, 12.0), spacing=(12.0, 20.0))
DODGE = replace(SCHOOL, route=(20.0, 50.0), first=(4.0, 8.0), spacing=(8.0, 14.0),
                route_frog=0.0, guard=0.0)  # fmt: skip
FROGS = replace(SCHOOL, route=(15.0, 35.0), mix=CALM, route_frog=0.5, guard=1.0)
MIXES = {
    "drills": [(0.4, se.FEAT), (0.3, DODGE), (0.3, FROGS)],
    "test": [(1.0, se.FEAT)],
}
DRILLS = ("course", "dodge", "frogs")


def worker(conn, n: int, seed: int, hold: int, mix: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    env = se.SoloEnv(n, se.FEAT, MIXES[mix], seed, hold)
    x = env.reset()

    def near():
        f = env.fleet
        return np.abs(f.goal()[0] - f.x) < se.PAD_NEAR

    conn.send((x, near()))
    while True:
        moves = conn.recv()
        if moves is None:
            break
        x, r, d, ended = env.step(moves)
        conn.send((x, r, d, near(), [(e["outcome"], e["drill"]) for e in ended]))


class Fleet:
    """The worker processes as one big vectorised env."""

    def __init__(self, workers: int, n: int, seed: int, hold: int, mix: str):
        self.conns, self.procs = [], []
        for w in range(workers):
            a, b = Pipe()
            p = Process(target=worker, args=(b, n, seed * 1000 + w, hold, mix), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.n, self.total = n, n * workers
        first = [c.recv() for c in self.conns]
        self.x = np.concatenate([f[0] for f in first])
        self.near = np.concatenate([f[1] for f in first])

    def step(self, moves: np.ndarray):
        for w, c in enumerate(self.conns):
            c.send(moves[w * self.n : (w + 1) * self.n])
        out = [c.recv() for c in self.conns]
        self.x = np.concatenate([o[0] for o in out])
        self.near = np.concatenate([o[3] for o in out])
        ended = sum((o[4] for o in out), [])
        return np.concatenate([o[1] for o in out]), np.concatenate([o[2] for o in out]), ended

    def close(self) -> None:
        for c in self.conns:
            c.send(None)


def _dev_job(job):
    seeds, net, hold, sampled = job
    os.environ["OMP_NUM_THREADS"] = "1"
    if not sampled:
        return se.fly(net.act, seeds, hold=hold)[0]
    rng = np.random.default_rng(seeds[0])
    return se.fly(lambda x, mask: net.sample(x, mask, rng), seeds, hold=hold)[0]


def fly_alone(net: SoloNet, first: int, hold: int, pool, n: int = 200, sampled=False) -> Counter:
    """The network alone on n fixed courses: its top move, or (sampled) a move drawn from its
    distribution, every decision."""
    jobs = [(list(range(first + i, first + i + 25)), net, hold, sampled) for i in range(0, n, 25)]
    return Counter(sum(pool.map(_dev_job, jobs), []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--envs", type=int, default=1024, help="drones per worker")
    ap.add_argument("--hold", type=int, default=2, help="updates a move is held")
    ap.add_argument("--steps", type=int, default=32, help="decisions per drone per round")
    ap.add_argument("--mix", choices=tuple(MIXES), default="drills")
    ap.add_argument("--width", type=int, nargs="+", default=[512, 512, 256])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=None, help="decay linearly to this")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.995, help="per update (0.05 s)")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--init", default=None, help="start the actor from this network (.npz)")
    ap.add_argument("--critic", default=None, help="and the value network from this one")
    ap.add_argument("--warm", type=int, default=0, help="rounds in which only the critic learns")
    ap.add_argument("--bc", type=float, default=0.0,
                    help="weight of the safe-set loss on post 16's saved states")  # fmt: skip
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--dev-n", type=int, default=400, help="dev courses flown at each check")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    from multiprocessing import Pool

    import torch as th

    th.manual_seed(args.seed)
    dev = "mps" if th.backends.mps.is_available() and not args.cpu else "cpu"
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args)))
    logf = open(out / "log.txt", "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    fleet = Fleet(args.workers, args.envs, args.seed, args.hold, args.mix)
    pool = Pool(8)
    N, T = fleet.total, args.steps
    gamma = args.gamma**args.hold
    pad = th.from_numpy(se.PAD_MOVES).to(dev)

    def mlp(sizes, last_gain):
        mods = []
        for a, b in zip(sizes[:-2], sizes[1:-1], strict=True):
            lin = th.nn.Linear(a, b)
            th.nn.init.orthogonal_(lin.weight, 2**0.5)
            th.nn.init.zeros_(lin.bias)
            mods += [lin, th.nn.ReLU()]
        lin = th.nn.Linear(sizes[-2], sizes[-1])
        th.nn.init.orthogonal_(lin.weight, last_gain)
        th.nn.init.zeros_(lin.bias)
        return th.nn.Sequential(*mods, lin)

    actor = mlp([se.DIM, *args.width, se.NM], 0.01)
    critic = mlp([se.DIM, *args.width, 1], 1.0)
    if args.init:
        src = load(HERE / args.init)
        mean, std = src.mean.copy(), src.std.copy()
        lin = [m for m in actor if hasattr(m, "weight")]
        with th.no_grad():
            for m, (w, b) in zip(lin, src.layers, strict=True):
                m.weight.copy_(th.from_numpy(w.T[: m.weight.shape[0]]))
                m.bias.copy_(th.from_numpy(b[: m.bias.shape[0]]))
    else:  # the inputs' scale, from a few rounds of random moves
        xs = []
        rng = np.random.default_rng(args.seed)
        for _ in range(40):
            xs.append(fleet.x[::4].copy())
            m = np.where(fleet.near[:, None] | ~se.PAD_MOVES[None], rng.random((N, se.NM)), -1)
            fleet.step(m.argmax(1))
        xs = np.concatenate(xs)
        mean, std = xs.mean(0), xs.std(0) + 0.05
    if args.critic:  # carrying on a run: its value network too (a fresh one sets practice back)
        lin = [m for m in critic if hasattr(m, "weight")]
        with th.no_grad():
            for m, (w, b) in zip(lin, load(HERE / args.critic).layers, strict=True):
                m.weight.copy_(th.from_numpy(w.T))
                m.bias.copy_(th.from_numpy(b))
    actor.to(dev)
    critic.to(dev)
    mean_t, std_t = th.from_numpy(mean).to(dev), th.from_numpy(std).to(dev)
    opt = th.optim.Adam([*actor.parameters(), *critic.parameters()], lr=args.lr, eps=1e-5)

    def numpy_net(model=None) -> SoloNet:
        lin = [m for m in (actor if model is None else model) if hasattr(m, "weight")]
        layers = [(m.weight.detach().cpu().numpy().T.copy(), m.bias.detach().cpu().numpy().copy())
                  for m in lin]  # fmt: skip
        return SoloNet(layers, mean, std)

    def logits_of(x, near):
        lg = actor((x - mean_t) / std_t)
        return lg.masked_fill(pad[None] & ~near[:, None], -1e9)

    aux = None
    if args.bc:  # post 16's states, labelled by the planner: stay on moves it found safe
        import refit

        d = refit.load_all()
        xa = d["X"][:, np.r_[0 : refit.N_OBS, refit.N_OBS + 9 : refit.N_OBS + 32]]
        xa[:, -se.N_PREV :] = 0.0
        aux = (th.from_numpy((xa - mean) / std).to(dev), th.from_numpy(d["S"]).to(dev),
               th.from_numpy(d["Y"]).to(dev))  # fmt: skip
        del d, xa

    def bc_loss():
        xa, sa, ya = aux
        idx = th.randint(len(xa), (4096,), device=dev)
        x = xa[idx] + 0.1 * th.randn((4096, se.DIM), device=dev)
        lsm = th.log_softmax(actor(x), 1)
        safe = sa[idx] >= refit.FAIL
        ce = -lsm.gather(1, ya[idx][:, None])[:, 0]
        mass = th.logsumexp(lsm.masked_fill(~safe, -1e9), 1)
        return th.where(safe.any(1), -mass + 0.1 * ce, ce).mean()

    X = th.zeros((T, N, se.DIM), device=dev)
    NEAR = th.zeros((T, N), dtype=th.bool, device=dev)
    A = th.zeros((T, N), dtype=th.long, device=dev)
    LP = th.zeros((T, N), device=dev)
    V = th.zeros((T + 1, N), device=dev)
    R = th.zeros((T, N), device=dev)
    DN = th.zeros((T, N), device=dev)
    recent = {k: deque(maxlen=4000) for k in range(len(DRILLS))}
    t_start, decisions, it, best = time.time(), 0, 0, -1
    hist = []
    while time.time() - t_start < args.minutes * 60:
        it += 1
        if args.lr_end is not None:
            left = 1.0 - (time.time() - t_start) / (args.minutes * 60)
            for group in opt.param_groups:
                group["lr"] = args.lr_end + (args.lr - args.lr_end) * left
        with th.no_grad():
            for t in range(T):
                x = th.from_numpy(fleet.x).to(dev)
                near = th.from_numpy(fleet.near).to(dev)
                lg = logits_of(x, near)
                u = th.rand_like(lg).clamp_(1e-9, 1 - 1e-9)
                a = (lg - th.log(-th.log(u))).argmax(1)  # a sample, by the Gumbel trick
                X[t], NEAR[t], A[t] = x, near, a
                LP[t] = th.log_softmax(lg, 1).gather(1, a[:, None])[:, 0]
                V[t] = critic((x - mean_t) / std_t)[:, 0]
                r, d, ended = fleet.step(a.cpu().numpy())
                R[t] = th.from_numpy(r).to(dev) * 0.1
                DN[t] = th.from_numpy(d).to(dev).float()
                for outcome, drill in ended:
                    recent[drill].append(outcome)
            V[T] = critic((th.from_numpy(fleet.x).to(dev) - mean_t) / std_t)[:, 0]
            adv = th.zeros((T, N), device=dev)
            last = th.zeros(N, device=dev)
            for t in reversed(range(T)):
                live = 1.0 - DN[t]
                delta = R[t] + gamma * V[t + 1] * live - V[t]
                last = delta + gamma * args.lam * live * last
                adv[t] = last
            ret = adv + V[:T]
        decisions += T * N
        fx, fn, fa = X.reshape(-1, se.DIM), NEAR.reshape(-1), A.reshape(-1)
        flp, fadv, fret = LP.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        stats = np.zeros(4)
        k = 0
        for _ in range(args.epochs):
            perm = th.randperm(T * N, device=dev)
            for s in range(0, T * N, args.batch):
                idx = perm[s : s + args.batch]
                lsm = th.log_softmax(logits_of(fx[idx], fn[idx]), 1)
                lp = lsm.gather(1, fa[idx][:, None])[:, 0]
                ent = -(lsm.exp() * lsm.clamp(min=-50)).sum(1).mean()
                ad = fadv[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                ratio = (lp - flp[idx]).exp()
                pg = -th.min(ratio * ad, ratio.clamp(1 - args.clip, 1 + args.clip) * ad).mean()
                v = critic((fx[idx] - mean_t) / std_t)[:, 0]
                vl = 0.5 * (v - fret[idx]).pow(2).mean()
                loss = vl if it <= args.warm else pg - args.ent * ent + vl
                if aux is not None and it > args.warm:
                    loss = loss + args.bc * bc_loss()
                opt.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], 0.5)
                opt.step()
                stats += (pg.item(), vl.item(), ent.item(), (flp[idx] - lp).mean().item())
                k += 1
        pg_m, vl_m, ent_m, kl_m = stats / k
        mins = (time.time() - t_start) / 60
        share = {DRILLS[k]: (np.mean([o == "landed" for o in v]) if v else 0.0)
                 for k, v in recent.items()}  # fmt: skip
        row = {"round": it, "minutes": round(mins, 2), "decisions": decisions,
               "landed": {k: round(float(v), 3) for k, v in share.items()},
               "entropy": round(float(ent_m), 3)}  # fmt: skip
        if it % args.eval_every == 0 or it == 1:
            net = numpy_net()
            top = fly_alone(net, DEV, args.hold, pool, args.dev_n)
            smp = fly_alone(net, DEV, args.hold, pool, args.dev_n, sampled=True)
            row["dev_top"], row["dev_sampled"] = dict(top), dict(smp)
            save(net, out / "last.npz")
            save(numpy_net(critic), out / "critic.npz")
            if max(top["landed"], smp["landed"]) > best:
                best = max(top["landed"], smp["landed"])
                save(net, out / "best.npz")
            log(f"{it:5d} {mins:6.1f} min {decisions / 1e6:7.1f}M decisions "
                f"({decisions / (mins * 60):,.0f}/s)  training flights landed "
                + " ".join(f"{k} {v:.2f}" for k, v in share.items() if recent[DRILLS.index(k)])
                + f"  entropy {ent_m:.2f} kl {kl_m:.4f} v {vl_m:.3f}"
                + f"  | {args.dev_n} dev courses: top move {top['landed']}, sampled "
                f"{smp['landed']}  {dict(smp)}")  # fmt: skip
        hist.append(row)
        (out / "train.json").write_text(json.dumps(hist))
    final = numpy_net()
    save(final, out / "last.npz")
    save(numpy_net(critic), out / "critic.npz")
    for tag, pilot in (("last", final), ("best", load(out / "best.npz"))):
        for sampled in (False, True):
            c = fly_alone(pilot, DEV, args.hold, pool, 1000, sampled)
            log(f"{tag} network alone, {'sampled' if sampled else 'top'} move, 1000 dev courses: "
                f"landed {c['landed']} {dict(c)}")  # fmt: skip
    fleet.close()


if __name__ == "__main__":
    main()
