"""Teach the post-14 network to fly like the planner, with DAgger (post 15).

DAgger (Ross, Gordon & Bagnell 2011): round 0, the planner flies and every state the network
would have seen is stored with the planner's sticks. Then, each round, the network flies; the
planner, following along on the same drones, says at every update what it would have done, and
those states and sticks join the pile. The network is refit on all of it. Copying the planner
only on the planner's own flights (behaviour cloning, round 0) teaches nothing about recovering
from the network's own mistakes; DAgger puts those states in the data.

The network is post 14's: two layers of 256, tanh, the same 318 inputs (three frames with
closest-approach inputs, 3 s of frog history), noise and a one-update delay. It is built as a
Stable-Baselines3 PPO policy, so finetune.py can carry on from it.

uv run experiments/015-teacher/dagger.py --name dagger-s1 --rounds 8 --flights 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "014-gauntlet"))

from gauntlet_env import FLYING, FULL, OUTCOMES, ROUGH, Config, FleetEnv, Physics  # noqa: E402

RUNS = HERE / "runs"
FEAT = replace(FULL, predict=True)
TRAIN_SEED0 = 1_000_000  # training courses: far from the test seeds (20000-20199)


class NumpyNet:
    """The actor as numpy: tanh layers, the mean action clipped to the sticks."""

    predict = True

    def __init__(self, layers):
        self.layers = [(np.asarray(w, np.float64), np.asarray(b, np.float64)) for w, b in layers]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        h = obs.astype(np.float64)
        for k, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if k < len(self.layers) - 1:
                h = np.tanh(h)
        return np.clip(h, -1.0, 1.0)


def obs_dim() -> int:
    return FleetEnv(1, Config(features=FEAT)).observation_space.shape[0]


# --- collecting ---------------------------------------------------------------------------
def _collect(job):
    """One worker: fly `seeds` with the network (or the planner when layers is None), the
    planner labelling every update. Returns (obs, sticks, flight results)."""
    seeds, layers, kw, noise_seed, beta, cap = job
    os.environ["OMP_NUM_THREADS"] = "1"
    from expert import Planner

    env = FleetEnv(len(seeds), Config(Physics(), ROUGH, FEAT), seed=noise_seed)
    obs = env.start(seeds)
    f = env.fleet
    pl = Planner(f.n, **kw)
    net = NumpyNet(layers) if layers is not None else None
    X, A = [], []
    rng = np.random.default_rng(noise_seed)
    t = 0
    while (f.outcome == FLYING).any() and t < cap:
        flying = f.outcome == FLYING
        label = pl.act(f)
        X.append(obs[flying].astype(np.float16))
        A.append(label[flying].astype(np.float32))
        if net is None:
            act = label
        else:  # the network flies; with probability beta, each update, the planner does
            act = np.where(rng.random(f.n)[:, None] < beta, label, net(obs))
        obs, _, _, _ = env.step(act)
        t += 1
    res = [
        {"seed": int(s), "outcome": OUTCOMES[f.outcome[i]], "time": float(f.ticks[i] * f.ph.dt)}
        for i, s in enumerate(seeds)
    ]
    return np.concatenate(X), np.concatenate(A), res


def collect(seeds, layers, workers=10, chunk=4, kw=None, noise_seed=0, beta=0.0, cap=1300):
    from multiprocessing import Pool

    seeds = list(seeds)
    jobs = [
        (seeds[i : i + chunk], layers, kw or {}, noise_seed + i, beta, cap)
        for i in range(0, len(seeds), chunk)
    ]
    with Pool(workers) as p:
        parts = p.map(_collect, jobs)
    X = np.concatenate([p[0] for p in parts])
    A = np.concatenate([p[1] for p in parts])
    res = sum((p[2] for p in parts), [])
    return X, A, res


# --- the network ----------------------------------------------------------------------------
def make_model(arch=(256, 256)):
    """A PPO model (and its policy) with post 14's network; the env only sets the shapes."""
    import torch as th
    from stable_baselines3 import PPO

    th.set_num_threads(4)
    env = FleetEnv(8, Config(Physics(), ROUGH, FEAT))
    kw = {"net_arch": {"pi": list(arch), "vf": list(arch)}}
    return PPO("MlpPolicy", env, policy_kwargs=kw, device="cpu", verbose=0, n_steps=32)


def actor_layers(model) -> list:
    lin = [m for m in model.policy.mlp_extractor.policy_net if hasattr(m, "weight")]
    lin.append(model.policy.action_net)
    return [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()) for m in lin]


def actor_params(model):
    pol = model.policy
    return list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())


def actor_forward(model, x):
    pol = model.policy
    return pol.action_net(pol.mlp_extractor.policy_net(x))


def stick_loss(out, target):
    """Squared error per state, except where the planner holds a stick at full deflection:
    the drone clips the network's output to the sticks' range, so going past +-1 the right
    way costs nothing."""
    import torch as th

    full = target.abs() > 0.99
    short = th.relu(1.0 - out * th.sign(target))  # how far short of full deflection
    return th.where(full, short**2, (out - target) ** 2).sum(1)


def fit(model, X, A, steps=6000, batch=4096, lr=1e-3, weight=None, log=print):
    """The network's mean sticks against the planner's (stick_loss): `steps` minibatches drawn
    from all the data, 2% held out to report the squared error of the clipped sticks."""
    import torch as th

    opt = th.optim.Adam(actor_params(model), lr=lr)
    n = len(X)
    Xt = th.from_numpy(X.astype(np.float32))
    At = th.from_numpy(A)
    Wt = th.from_numpy(weight.astype(np.float32)) if weight is not None else None
    g = th.Generator().manual_seed(0)
    perm = th.randperm(n, generator=g)
    va, tr = perm[: max(1, n // 50)], perm[max(1, n // 50) :]
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)
    tot = 0.0
    for k in range(steps):
        idx = tr[th.randint(len(tr), (min(batch, len(tr)),), generator=g)]
        err = stick_loss(actor_forward(model, Xt[idx]), At[idx])
        loss = (err * Wt[idx]).mean() if Wt is not None else err.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
        if (k + 1) % (steps // 4) == 0:
            with th.no_grad():
                out = actor_forward(model, Xt[va]).clamp(-1.0, 1.0)
                v = ((out - At[va]) ** 2).sum(1).mean().item()
            log(f"    step {k + 1}/{steps}: loss {tot / (steps // 4):.4f}, held out {v:.4f}")
            tot = 0.0
    return v


def weights_for(A: np.ndarray, boost: float) -> np.ndarray:
    """More weight on full-stick throttle (the planner's jinks): they are rare and decisive."""
    return 1.0 + boost * (np.abs(A[:, 0]) > 0.99)


# --- testing --------------------------------------------------------------------------------
def test(layers, n=200, tests=("gauntlet",)) -> dict:
    """The network on the 200 fixed test courses (post 14's fly(): noise seed 7)."""
    from evaluate import TESTS, fly, summary

    net = NumpyNet(layers)
    return {t: summary(fly(net, TESTS[t][0], hide=TESTS[t][1])) for t in tests}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--flights", type=int, default=400, help="courses flown per round")
    ap.add_argument("--first", type=int, default=None, help="courses in round 0 (planner)")
    ap.add_argument("--steps", type=int, default=6000, help="gradient steps per round")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--boost", type=float, default=0.0, help="extra weight on jinks")
    ap.add_argument("--beta", type=float, default=0.5, help="round r: planner flies beta^r")
    ap.add_argument("--cap", type=int, default=900, help="updates per network flight (45 s)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--init", default=None, help="start from this PPO zip (e.g. post 14's)")
    ap.add_argument("--resume", action="store_true", help="reuse saved rounds' data")
    args = ap.parse_args()

    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args)))
    logf = open(out / "log.txt", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    from stable_baselines3 import PPO

    model = make_model()
    if args.init:
        src = PPO.load(args.init, device="cpu")
        model.policy.load_state_dict(src.policy.state_dict())
        log(f"init from {args.init}")
    hist = json.loads((out / "rounds.json").read_text()) if args.resume and (
        out / "rounds.json").exists() else []  # fmt: skip
    data_X, data_A = [], []
    next_seed = TRAIN_SEED0
    t_start = time.time()
    for r in range(args.rounds):
        n = (args.first or args.flights) if r == 0 else args.flights
        seeds = range(next_seed, next_seed + n)
        next_seed += n
        path = out / f"data_{r:02d}.npz"
        t0 = time.time()
        if args.resume and path.exists():
            z = np.load(path)
            X, A = z["obs"], z["act"]
            res = json.loads(str(z["res"]))
            log(f"round {r}: loaded {len(X):,} states")
        else:
            layers = None if r == 0 else actor_layers(model)
            beta = args.beta**r if r else 1.0
            X, A, res = collect(
                seeds,
                layers,
                args.workers,
                args.chunk,
                noise_seed=1000 * r,
                beta=beta,
                cap=1300 if r == 0 else args.cap,
            )
            np.savez(path, obs=X, act=A, res=json.dumps(res))
        landed = sum(x["outcome"] == "landed" for x in res)
        who = "planner" if r == 0 else "network"
        log(
            f"round {r}: {who} flew {n} training courses, landed {landed}, "
            f"{len(X):,} states ({time.time() - t0:.0f} s)"
        )
        data_X.append(X)
        data_A.append(A)
        Xall, Aall = np.concatenate(data_X), np.concatenate(data_A)
        t0 = time.time()
        w = weights_for(Aall, args.boost) if args.boost else None
        held = fit(model, Xall, Aall, args.steps, lr=args.lr, weight=w, log=log)
        model.save(out / f"round_{r:02d}")
        ev = test(actor_layers(model))["gauntlet"]
        log(
            f"  fit {time.time() - t0:.0f} s; test: landed {ev['landed']}/200 {ev['outcomes']} "
            f"(total {(time.time() - t_start) / 60:.0f} min)"
        )
        hist.append(
            {
                "round": r,
                "flown_by": who,
                "train_landed": landed,
                "train_flights": n,
                "states": int(len(Xall)),
                "held_out_loss": round(held, 4),
                "test": ev,
                "minutes": round((time.time() - t_start) / 60, 1),
            }
        )
        (out / "rounds.json").write_text(json.dumps(hist))


if __name__ == "__main__":
    main()
