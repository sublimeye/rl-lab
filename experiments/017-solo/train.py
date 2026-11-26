"""PPO over the planner's moves (post 17): the network picks a move, post 15's controller flies it,
and the only teacher is how the flight ends.

uv run experiments/017-solo/train.py --name m1 --minutes 45
uv run experiments/017-solo/train.py --name m2 --init runs/refit-x.npz --bc 0.5   # from a copy

Eight processes each step 1024 drones (solo_env.SoloEnv); the network runs on the GPU (the
Mac's, or a rented box's: more --workers there).
A course is, at random: a full test-like course (40%), a dodge drill (30%: 20-50 m, a waypoint
every 8-14 m, no frogs) or a frog drill (30%: 15-35 m, the pad's frog, half the time another).
Every so often the pilot flies dev courses (40000 up) alone: its top move, and its moves drawn.
The clock is the monotonic one, which stops while the laptop sleeps: --minutes are minutes awake.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, deque
from dataclasses import replace
from multiprocessing import get_context, shared_memory
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import solo_env as se  # noqa: E402
from solo_net import SoloNet, load, save  # noqa: E402

RUNS = HERE / "runs"
MP = get_context("spawn")  # (a forked child of a process that has touched the GPU is trouble)
DEV, TEST, FRESH = 40_000, 20_000, 30_000
CALM = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
SCHOOL = replace(se.FEAT, route=(15.0, 130.0), first=(6.0, 12.0), spacing=(12.0, 20.0))
DODGE = replace(SCHOOL, route=(20.0, 50.0), first=(4.0, 8.0), spacing=(8.0, 14.0),
                route_frog=0.0, guard=0.0)  # fmt: skip
FROGS = replace(SCHOOL, route=(15.0, 35.0), mix=CALM, route_frog=0.5, guard=1.0)
OPEN_PAD = replace(se.FEAT, guard=0.0)  # no frog on the pad
MIXES = {
    "drills": [(0.4, se.FEAT), (0.3, DODGE), (0.3, FROGS)],
    "test": [(1.0, se.FEAT)],
    # the test's courses, and a quarter of them with no frog guarding the pad
    "wide": [(0.75, se.FEAT), (0.25, OPEN_PAD)],
}
NAMES = {"drills": ("course", "dodge", "frogs"), "test": ("course",),
         "wide": ("course", "open pad")}  # fmt: skip


def _shared(names, total: int):
    """The arrays every process sees: each drone's inputs, move, reward, done, near the pad."""
    shms = [shared_memory.SharedMemory(name=nm) for nm in names]
    kinds = ((total, se.DIM), np.float32), (total, np.int64), (total, np.float32), \
        (total, np.bool_), (total, np.bool_)  # fmt: skip
    return shms, [np.ndarray(shape, dt, m.buf) for (shape, dt), m in zip(kinds, shms, strict=True)]


def worker(conn, names, total: int, lo: int, n: int, seed: int, hold: int, mix: str, reward,
           replay: float, timeout: float, again: float) -> None:  # fmt: skip
    os.environ["OMP_NUM_THREADS"] = "1"
    shms, (X, A, R, D, NEAR) = _shared(names, total)
    me = slice(lo, lo + n)
    env = se.SoloEnv(n, se.FEAT, MIXES[mix], seed, hold, reward=reward, replay=replay,
                     timeout=timeout, again=again)  # fmt: skip

    def near():
        f = env.fleet
        return np.abs(f.goal()[0] - f.x) < se.PAD_NEAR

    X[me], NEAR[me] = env.reset(), near()
    conn.send([])
    while conn.recv() is not None:
        x, r, d, ended = env.step(A[me].copy())
        X[me], R[me], D[me], NEAR[me] = x, r, d, near()
        conn.send([(e["outcome"], e["drill"]) for e in ended])
    for m in shms:
        m.close()


class Fleet:
    """The worker processes as one big vectorised env. They write into shared memory, so a
    step moves no arrays between processes: x, near (and step's rewards and dones) are views
    that the next step overwrites."""

    def __init__(self, workers: int, n: int, seed: int, hold: int, mix: str, reward=se.REWARD,
                 replay: float = 0.0, timeout: float = 0.0, again: float = 0.0):  # fmt: skip
        self.n, self.total = n, n * workers
        sizes = (self.total * se.DIM * 4, self.total * 8, self.total * 4, self.total, self.total)
        self.shms = [shared_memory.SharedMemory(create=True, size=k) for k in sizes]
        names = [m.name for m in self.shms]
        self._mine, arrays = _shared(names, self.total)
        self.x, self.a, self.r, self.d, self.near = arrays
        self.conns, self.procs = [], []
        for w in range(workers):
            a, b = MP.Pipe()
            args = (b, names, self.total, w * n, n, seed * 1000 + w, hold, mix, reward, replay,
                    timeout, again)  # fmt: skip
            p = MP.Process(target=worker, args=args, daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        for c in self.conns:
            c.recv()

    def step(self, moves: np.ndarray):
        self.a[:] = moves
        for c in self.conns:
            c.send(1)
        ended = sum((c.recv() for c in self.conns), [])
        return self.r, self.d, ended

    def close(self) -> None:
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join(5)
        for m in self._mine:
            m.close()
        for m in self.shms:
            m.close()
            m.unlink()


def _dev_job(job):
    seeds, net, hold, sampled, feat = job
    os.environ["OMP_NUM_THREADS"] = "1"
    if not sampled:
        out, f = se.fly(net.act, seeds, feat, hold=hold)
    else:
        rng = np.random.default_rng(seeds[0])
        out, f = se.fly(lambda x, mask: net.sample(x, mask, rng), seeds, feat, hold=hold)
    return out, (f.ticks * f.ph.dt).tolist()


def fly_alone(net: SoloNet, first: int, hold: int, pool, n: int = 200, sampled=False,
              feat=se.FEAT) -> Counter:  # fmt: skip
    """The network alone on n fixed courses: its top move, or (sampled) a move drawn from its
    distribution, every decision. The outcomes, and under "seconds" the landed flights' mean
    time."""
    jobs = [(list(range(first + i, first + i + 25)), net, hold, sampled, feat)
            for i in range(0, n, 25)]  # fmt: skip
    parts = pool.map(_dev_job, jobs)
    outs, secs = sum((p[0] for p in parts), []), sum((p[1] for p in parts), [])
    c = Counter(outs)
    t = [s for o, s in zip(outs, secs, strict=True) if o == "landed"]
    c["seconds"] = round(float(np.mean(t)), 2) if t else 0.0
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-workers", type=int, default=8)
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
    ap.add_argument("--ent-end", type=float, default=None, help="decay linearly to this")
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
    # the reward (the defaults are post 14's: paid by the metre, charged by the second)
    ap.add_argument("--landed", type=float, default=se.REWARD.landed)
    ap.add_argument("--failed", type=float, default=se.REWARD.failed)
    ap.add_argument("--time", type=float, default=se.REWARD.time, help="charged per update")
    ap.add_argument("--timeout", type=float, default=0.0, help="paid when the 60 s run out")
    ap.add_argument("--again", type=float, default=0.0,
                    help="share of new flights begun 2-6 s before a recent failure")  # fmt: skip
    ap.add_argument("--replay", type=float, default=0.0,
                    help="share of new courses that were failed the last time flown")  # fmt: skip
    args = ap.parse_args()

    import torch as th

    th.manual_seed(args.seed)
    dev = "cpu"
    if th.cuda.is_available() and not args.cpu:
        dev = "cuda"
        th.set_float32_matmul_precision("high")
    elif th.backends.mps.is_available() and not args.cpu:
        dev = "mps"
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args)))
    logf = open(out / "log.txt", "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    reward = replace(se.REWARD, landed=args.landed, failed=args.failed, time=args.time)
    fleet = Fleet(args.workers, args.envs, args.seed, args.hold, args.mix, reward, args.replay,
                  args.timeout, args.again)  # fmt: skip
    pool = MP.Pool(args.eval_workers)
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
    names = NAMES[args.mix]
    recent = {k: deque(maxlen=4000) for k in range(len(names))}
    t_start, decisions, it, best = time.monotonic(), 0, 0, -1
    hist = []
    while time.monotonic() - t_start < args.minutes * 60:
        it += 1
        left = 1.0 - (time.monotonic() - t_start) / (args.minutes * 60)
        if args.lr_end is not None:
            for group in opt.param_groups:
                group["lr"] = args.lr_end + (args.lr - args.lr_end) * left
        ent_w = args.ent
        if args.ent_end is not None:
            ent_w = args.ent_end + (args.ent - args.ent_end) * left
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
                loss = vl if it <= args.warm else pg - ent_w * ent + vl
                if aux is not None and it > args.warm:
                    loss = loss + args.bc * bc_loss()
                opt.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], 0.5)
                opt.step()
                stats += (pg.item(), vl.item(), ent.item(), (flp[idx] - lp).mean().item())
                k += 1
        pg_m, vl_m, ent_m, kl_m = stats / k
        mins = (time.monotonic() - t_start) / 60
        share = {names[k]: (np.mean([o == "landed" for o in v]) if v else 0.0)
                 for k, v in recent.items()}  # fmt: skip
        row = {"round": it, "minutes": round(mins, 2), "decisions": decisions,
               "landed": {k: round(float(v), 3) for k, v in share.items()},
               "entropy": round(float(ent_m), 3)}  # fmt: skip
        if it % args.eval_every == 0 or it == 1:
            net = numpy_net()
            top = fly_alone(net, DEV, args.hold, pool, args.dev_n)
            smp = fly_alone(net, DEV, args.hold, pool, args.dev_n, sampled=True)
            row["dev_top"], row["dev_sampled"] = dict(top), dict(smp)
            if args.mix == "wide":  # and the same pilot where no frog guards the pad
                row["dev_open"] = dict(fly_alone(net, DEV, args.hold, pool, 200, True, OPEN_PAD))
            save(net, out / "last.npz")
            save(numpy_net(critic), out / "critic.npz")
            if max(top["landed"], smp["landed"]) > best:
                best = max(top["landed"], smp["landed"])
                save(net, out / "best.npz")
            log(
                f"{it:5d} {mins:6.1f} min {decisions / 1e6:7.1f}M decisions "
                f"({decisions / (mins * 60):,.0f}/s)  training flights landed "
                + " ".join(f"{k} {v:.2f}" for k, v in share.items() if recent[names.index(k)])
                + f"  entropy {ent_m:.2f} kl {kl_m:.4f} v {vl_m:.3f}"
                + f"  | {args.dev_n} dev courses: top move {top['landed']}, sampled "
                f"{smp['landed']} ({smp['seconds']} s a landing)  {dict(smp)}"
                + (
                    f"  open pads: {row['dev_open'].get('landed', 0)}/200"
                    if "dev_open" in row
                    else ""
                )
            )
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
