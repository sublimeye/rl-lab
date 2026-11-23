"""016-flawless/pilot.py's DAgger with the planner and the training on a GPU.

pilot.py runs as is, with three parts swapped:
- fly_many: instead of a pool of processes, each flying 4 courses with its own observation
  noise (and its own random draws for who flies, net or planner), one process flies every
  chunk in lockstep, with the same seeds and draws; each update, one GPU decision covers the
  drones due in all chunks (gpu_planner);
- the teacher: pilot.Teacher's bookkeeping (each move's best score, for the danger head) on
  the GPU planner;
- train: the same steps, batches (the same CPU generator) and model, on the GPU.
Results go to 016-gpu/runs/<name>/, not 016-flawless's.

uv run experiments/016-gpu/dagger_gpu.py dagger --name g1
uv run experiments/016-gpu/dagger_gpu.py test --name g1 --mode net
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gpu_planner as gp  # noqa: E402
import torch_rollout as tr  # noqa: E402

sys.path.insert(0, str(HERE.parent / "016-flawless"))
import pilot  # noqa: E402

expert = gp.expert
from gauntlet_env import FLYING, OUTCOMES, ROUGH, TIMEOUT, Config, FleetEnv  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BODY = None


class GPUTeacher(gp.GPUPlanner):
    """pilot.Teacher on the GPU planner (the shield's `allowed` is GPUPlanner's)."""

    def __init__(self, n, **kw):
        super().__init__(n, device=DEVICE, body=BODY, **kw)
        self.mscore = np.full((n, pilot.NM), pilot.ABSENT)
        self.decided = np.zeros(n, bool)

    def _record(self, due, best, drone_of, plans):
        m1, t1, m2, t2, m3 = plans.unbind(1)
        now = torch.where(t1 > 0, m1, torch.where(t2 > 0, m2, m3))
        nm = pilot.NM
        ms = torch.full((len(due) * nm,), pilot.ABSENT, dtype=best.dtype, device=best.device)
        ms = ms.scatter_reduce(0, drone_of * nm + now, best, "amax")
        self.mscore[due] = ms.reshape(len(due), nm).cpu().numpy()
        self.decided[due] = True

    def plan_now(self, f):
        self.decided[:] = False
        flying = np.nonzero(f.outcome == FLYING)[0]
        due = flying[self.fresh[flying] | (self.age[flying] >= self.every)]
        live = f.frog_on[due] & ~f.frog_past[due]
        close = np.abs(f.frog_p[due, :, 0] - f.x[due][:, None]) < gp.fp.FAR
        near = (live & close).any(1)
        if len(due):
            self.gpu_decide(f, due, near)
        return self.current()


def fly_many(seeds, mode, net=None, workers=10, chunk=4, **opts):
    """pilot.fly_many's flights, every chunk in one process, decided together."""
    t0 = time.time()
    seeds = list(seeds)
    parts = [seeds[i : i + chunk] for i in range(0, len(seeds), chunk)]
    envs, obs, rngs, rows = [], [], [], []
    k = 0
    for j, cs in enumerate(parts):
        ns = opts.get("noise_seed", 7) + (j if mode in ("planner", "dagger") else 0)
        env = FleetEnv(len(cs), Config(pilot.PH45, ROUGH, pilot.FEAT), seed=ns)
        obs.append(env.start(cs))
        envs.append(env)
        rngs.append(np.random.default_rng(ns))
        rows.append(slice(k, k + len(cs)))
        k += len(cs)
    n = k
    pl = GPUTeacher(n)
    labels = mode in ("planner", "dagger")
    beta, cap = opts.get("beta", 0.0), opts.get("cap", 1300)
    prev = np.full(n, expert.DEFAULT)
    data = [([], [], [], []) for _ in parts]
    active = [True] * len(parts)
    t = 0
    while any(active):
        f = expert.concat([e.fleet for e in envs])  # a copy: the planner only reads it
        for c, sl in enumerate(rows):
            if not active[c]:  # a chunk whose loop has ended: nothing more for its drones
                f.outcome[sl] = np.where(f.outcome[sl] == FLYING, TIMEOUT, f.outcome[sl])
        flying = f.outcome == FLYING
        pl.ph_real = f.ph
        pl.observe(f)
        pl.estimate(f)
        x = np.concatenate([np.concatenate(obs), pilot.beliefs(pl, f, prev)], 1)
        x = x.astype(np.float32)
        if net is not None:
            logits, _ = net(x)
        if mode == "shield":  # the planner's search over the network's top k moves
            due = pl.fresh | (pl.age >= pl.every)
            if due.any():
                top = np.argsort(-logits, 1)[:, : opts.get("k", 5)]
                allowed = np.zeros((n, pilot.NM), bool)
                allowed[np.arange(n)[:, None], top] = True
                pl.allowed = allowed
            m = pl.plan_now(f)
        elif labels:
            label = pl.plan_now(f)
            m = label
            if mode == "dagger":
                take = np.zeros(n, bool)
                for c, sl in enumerate(rows):
                    if active[c]:
                        take[sl] = rngs[c].random(sl.stop - sl.start) < beta
                m = np.where(take, label, np.argmax(logits, 1))
            for c, sl in enumerate(rows):
                if active[c]:
                    fl = flying[sl]
                    X, Y, S, D = data[c]
                    X.append(x[sl][fl].astype(np.float16))
                    Y.append(label[sl][fl].astype(np.int16))
                    S.append(pl.mscore[sl][fl].astype(np.float16))
                    D.append(pl.decided[sl][fl].copy())
        else:
            m = np.argmax(logits, 1)
        a = expert.control(pl.assumed(f), m)
        pl.remember(f, a)
        pl.age += 1
        prev = np.where(flying, m, prev)
        t += 1
        for c, sl in enumerate(rows):
            if active[c]:
                obs[c], _, _, _ = envs[c].step(a[sl])
                active[c] = bool((envs[c].fleet.outcome == FLYING).any()) and t < cap
    res = []
    for env, cs in zip(envs, parts, strict=True):
        f = env.fleet
        for i, s in enumerate(cs):
            res.append({"seed": int(s), "outcome": OUTCOMES[f.outcome[i]],
                        "time": round(float(f.ticks[i] * f.ph.dt), 2),
                        "x": round(float(f.x[i]), 1),
                        "togo": round(float(abs(f.goal()[0][i] - f.x[i])), 1),
                        "hit_by": int(f.hit_enc[i])})  # fmt: skip
    stats = {"decisions": pl.decisions, "full": pl.full_searches, "cpu": time.time() - t0}
    if not labels:
        return None, res, stats
    out = tuple(np.concatenate([np.concatenate(d[q]) for d in data if d[q]]) for q in range(4))
    return out, res, stats


def train(model, X, Y, S, D, steps, lr=1e-3, batch=4096, fail_w=0.5, log=print):
    """pilot.train on the GPU: the same model, steps and batches (drawn by the same CPU
    generator)."""
    th = torch
    dev = DEVICE
    model.to(dev)
    opt = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = th.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    nm, absent, fail = pilot.NM, pilot.ABSENT, pilot.FAIL
    n = len(X)
    g = th.Generator().manual_seed(0)
    perm = th.randperm(n, generator=g)
    nva = max(1, n // 50)
    va, trn = perm[:nva], perm[nva:]
    Xt = th.from_numpy(X).to(dev)
    Yt = th.from_numpy(Y.astype(np.int64)).to(dev)
    St = th.from_numpy(S).to(dev)
    Dt = th.from_numpy(D).to(dev)
    tot = th.zeros(2, device=dev)
    acc = 0.0
    for k in range(steps):
        idx = trn[th.randint(len(trn), (batch,), generator=g)].to(dev, non_blocking=True)
        out = model(Xt[idx].float())
        ce = th.nn.functional.cross_entropy(out[:, :nm], Yt[idx])
        s, d = St[idx].float(), Dt[idx]
        tried = (s > absent + 1) & d[:, None]
        bce = th.nn.functional.binary_cross_entropy_with_logits(
            out[:, nm:], (s < fail).float(), reduction="none"
        )
        bce = (bce * tried).sum() / tried.sum().clamp(min=1)
        loss = ce + fail_w * bce
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += th.stack([ce.detach(), bce.detach()])
        if (k + 1) % (steps // 4) == 0:
            with th.no_grad():
                o = model(Xt[va.to(dev)].float())
                acc = (o[:, :nm].argmax(1) == Yt[va.to(dev)]).float().mean().item()
            ce_m, bce_m = (tot / (steps // 4)).tolist()
            log(f"    step {k + 1}/{steps}: move CE {ce_m:.3f}, fail BCE {bce_m:.3f}, "
                f"held-out move accuracy {acc:.3f}")  # fmt: skip
            tot.zero_()
    model.to("cpu")
    return acc


def main() -> None:
    global BODY
    if DEVICE == "cuda":
        BODY = torch.compile(tr.rollout_body, dynamic=True, fullgraph=True)
    pilot.fly_many = fly_many
    pilot.train = train
    pilot.RUNS = HERE / "runs"
    pilot.main()


if __name__ == "__main__":
    main()
