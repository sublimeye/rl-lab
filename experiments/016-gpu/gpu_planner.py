"""The fair planner (016-flawless/fair_planner.py) with its decisions on a GPU.

What a pilot has seen (frog strikes, the air and mass estimated from its own motion) stays on
the CPU, in numpy, as in FairPlanner: it is per drone and cheap. The search moves to torch:
every drone due a decision, under each of its frog hypotheses (3 near a frog, 1 elsewhere), is
one *group*; all groups' candidate plans fly ahead in one batch (torch_rollout), then the three
deepenings, the scoring and the choice run as tensor ops over groups instead of Python loops:
- deepening against threats: per group, the best few copies (by progress) that end with a bird
  or missile still after them get a dodge search, up to 3 rounds of 4 (a segment rank);
- deepening at the pad: per group, the best 4 copies ending near it get a wait-then-land search;
- the hypotheses: a plan scores by its worst;
- the choice: the best plan, or within the band the one closest to a plain cruise (a segment
  lexicographic minimum).
Same rules as numpy, so the same choices up to round-off (float64: checked by check_planner.py).

One difference: numpy's look-ahead stops when nothing in the batch flies, so a copy that ran
out of time stops gaining progress then; here a batch holds every group at once, numpy's held
one hypothesis's. It only touches copies that time out (60 s) inside the look-ahead.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch_fleet as tf
import torch_rollout as tr

fp = tf.fp
expert = fp.expert
from gauntlet_env import FAILED, FLYING, FROG_REACH, GOLDEN, LANDED, MAX_ENC  # noqa: E402

LEAPING = fp.LEAPING
FROG_JITTER = tf.FROG_JITTER


def _seg_rank(gid, key, mask, n_groups):
    """For rows in mask: their rank within their group, ordered by key descending (ties: row
    order, as numpy's stable argsort of -key). Others: a large number."""
    n = len(gid)
    big = torch.full_like(gid, n_groups)
    g = torch.where(mask, gid, big)
    p1 = torch.argsort(-key, stable=True)
    p2 = torch.argsort(g[p1], stable=True)
    order = p1[p2]
    cnt = torch.bincount(g, minlength=n_groups + 1)
    start = torch.cumsum(cnt, 0) - cnt
    rank = torch.empty_like(gid)
    rank[order] = torch.arange(n, device=gid.device) - start[g[order]]
    return torch.where(mask, rank, torch.full_like(rank, 1 << 30))


def _seg_first(gid, keys, mask, n_groups):
    """Per group, the row (in mask) smallest by keys (a list, most significant first; ties: the
    first row). -1 for a group with none."""
    n = len(gid)
    order = torch.arange(n, device=gid.device)
    for k in reversed(keys):
        order = order[torch.argsort(k[order], stable=True)]
    g = torch.where(mask, gid, torch.full_like(gid, n_groups))
    order = order[torch.argsort(g[order], stable=True)]
    gs = g[order]
    first = torch.ones_like(gs, dtype=torch.bool)
    first[1:] = gs[1:] != gs[:-1]  # the first row of each group in that order
    out = torch.full((n_groups + 1,), -1, dtype=torch.long, device=gid.device)
    out[gs[first]] = order[first]
    return out[:n_groups]


class GPUPlanner(fp.FairPlanner):
    """FairPlanner whose decide runs on `device` in `dtype`."""

    def __init__(self, n, device="cuda", dtype=torch.float32, body=None, **kw):
        super().__init__(n, **kw)
        assert self.fair and self.frogs == "ghost" and self.air == "estimated"
        self.device, self.dtype = device, dtype
        self.body = body or tr.rollout_body
        self.T = tr.tables(device, dtype)
        self.dodge_t = torch.as_tensor(np.stack([self.dodges(k) for k in range(6)]),
                                       device=device)  # fmt: skip
        self.land_t = torch.as_tensor(self.landings(), device=device)
        self.heights = torch.as_tensor(expert.HEIGHTS, device=device, dtype=dtype)
        self.failed_set = torch.as_tensor(FAILED, device=device)
        self.hostile_kind = expert.HOSTILE_PER_KIND
        self.stats = Counter()  # how often each search ran (copies searched from)
        self.check = None  # a hook: (f, due, near, chosen) -> the plans to keep

    # --- the air: FairPlanner.fit, every gust period at once -------------------------------
    def fit(self, i: int) -> None:
        """FairPlanner.fit (steady wind + a sine, the best of the weather's periods), solved by
        normal equations for all periods together (lstsq per period: ~8x slower)."""
        tp = np.array(self.pushes[i][-200:])
        t, y = tp[:, 0], tp[:, 1]
        e = self.est
        if t[-1] - t[0] < 1.5:
            e["wind"][i], e["gust"][i] = float(np.mean(y[-20:])), 0.0
            return
        w = 2 * np.pi / fp.PERIODS
        X = np.stack([np.ones((len(w), len(t))), np.sin(w[:, None] * t), np.cos(w[:, None] * t)],
                     2)  # fmt: skip
        c = np.linalg.solve(X.transpose(0, 2, 1) @ X, (X.transpose(0, 2, 1) @ y)[..., None])
        err = (((X @ c)[..., 0] - y) ** 2).sum(1)
        b = int(np.argmin(err))
        c0, ca, cb = c[b, :, 0]
        e["wind"][i] = float(np.clip(c0, -fp.ROUGH.wind, fp.ROUGH.wind))
        e["gust"][i] = float(min(np.hypot(ca, cb), fp.ROUGH.gust))
        e["period"][i], e["phase"][i] = fp.PERIODS[b], float(np.arctan2(cb, ca))

    # --- one batch: every due drone x its hypotheses ----------------------------------------
    def act(self, f):
        self.ph_real = f.ph
        self.observe(f)
        self.estimate(f)
        flying = np.nonzero(f.outcome == FLYING)[0]
        due = flying[self.fresh[flying] | (self.age[flying] >= self.every)]
        live = f.frog_on[due] & ~f.frog_past[due]
        close = np.abs(f.frog_p[due, :, 0] - f.x[due][:, None]) < fp.FAR
        near = (live & close).any(1)
        if len(due):
            self.gpu_decide(f, due, near)
        a = expert.control(self.assumed(f), self.current())
        self.remember(f, a)
        self.age += 1
        return a

    def beliefs(self, f, drones, hs):
        """A numpy fleet with one row per (drone, hypothesis): the assumed air and mass, the
        frogs striking at hypothesis h (fair_planner.believe), ghost windows."""
        g = self.assumed(f)
        gb = g.take(drones)
        gb.ghost = np.full((gb.n, 2, 3, 2), -1e9)
        cache = {}
        for r, (i, h) in enumerate(zip(drones, hs, strict=True)):
            for k in range(2):
                if not g.frog_on[i, k] or g.frog_past[i, k]:
                    continue
                if (i, k) not in cache:
                    lo, hi, beat = self.interval(f, i, k)
                    cache[i, k] = (lo, hi, beat, self._beats)
                lo, hi, beat, (b_lo, b_hi) = cache[i, k]
                when = lo + (hi - lo) * (h / max(self.hyps - 1, 1))
                n = gb.frog_n[r, k]
                gb.frog_p[r, k, 2] = beat
                gb.frog_p[r, k, 3] = when - beat * n
                gb.frog_p[r, k, 4] = np.mod(0.5 - n * GOLDEN, 1.0)
                gb.sw_after[r] = 10**6
                jit = 2 * FROG_JITTER
                for j in range(3):
                    gb.ghost[r, k, j] = (lo + j * (b_lo - jit), hi + j * (b_hi + jit) + LEAPING)
        return gb

    def gpu_decide(self, f, due, near):
        """New plans for drones `due` of the real fleet f (near: their frog hypotheses)."""
        dev, dt = self.device, self.dtype
        af = self.assumed(f)  # what numpy's _decide decides on
        # groups: near drones x hyps, the rest x 1 (the middle hypothesis)
        mid = (self.hyps - 1) // 2
        drones, hs, slot = [], [], []
        for q, i in enumerate(due):
            for h in range(self.hyps) if near[q] else [mid]:
                drones.append(int(i))
                hs.append(h)
                slot.append(q)
        drones, hs, slot = np.array(drones), np.array(hs), np.array(slot)
        gb = self.beliefs(af, drones, hs)
        # candidates per group (what is left of the last plan first, for a drone not fresh)
        gx, _ = af.goal()
        cands = []
        for i in due:
            k_h = int(np.argmin(np.abs(np.array(expert.HEIGHTS) - af.y[i])))
            c = self.candidates(abs(gx[i] - af.x[i]) < expert.PAD_NEAR, k_h)
            if not self.fresh[i]:
                p = self.plan[i].copy()
                p[1] -= self.age[i]
                p[3] -= self.age[i]
                c = np.concatenate([p[None], c])
            cands.append(c)
        sizes = np.array([len(cands[q]) for q in slot])
        plans_np = np.concatenate([cands[q] for q in slot])
        gid_np = np.repeat(np.arange(len(slot)), sizes)
        # waypoints still ahead: constant in the copies (none is met in them)
        j = np.arange(MAX_ENC)[None]
        ahead = (j >= gb.enc_next[:, None]) & (j < gb.n_enc[:, None])
        hostile = (self.hostile_kind[gb.enc_kind] * ahead).sum(1)
        pend0 = self.open_cost * hostile + self.near_cost * self.safe * ahead.sum(1)
        gb.enc_at[:] = 1e9
        S0, C0 = tf.from_numpy(gb, device=dev, dtype=dt)
        P = tf.params(gb.ph, gb.feat.battery)
        gid = torch.as_tensor(gid_np, device=dev)
        plans = torch.as_tensor(plans_np, device=dev)
        S, C = tf.take(S0, C0, gid)
        sc = self._evaluate(S, C, P, plans, gid, len(slot), torch.as_tensor(pend0, dtype=dt,
                            device=dev)[gid])  # fmt: skip
        # the worst over each drone's hypotheses: groups of one drone hold the same plans
        first_group = np.zeros(len(slot), np.int64)
        for q in range(len(due)):
            gq = np.nonzero(slot == q)[0]
            first_group[gq] = gq[0]
        offs = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        canon_row = torch.as_tensor(
            np.repeat(offs[first_group], sizes) + (np.arange(len(gid_np)) - offs[gid_np]),
            device=dev,
        )
        best = torch.full_like(sc, float("inf")).scatter_reduce(0, canon_row, sc, "amin")
        keep = canon_row == torch.arange(len(gid_np), device=dev)  # one row per drone's plan
        drone_of = torch.as_tensor(slot, device=dev)[gid]
        # the choice (FairPlanner._decide)
        s = best
        lead = torch.zeros(len(gid_np), dtype=torch.bool, device=dev)
        lead[torch.as_tensor(offs[first_group], device=dev)] = True
        not_fresh = torch.as_tensor(~self.fresh[due], device=dev)[drone_of]
        s = torch.where(lead & not_fresh, s + self.loyal, s)
        n_d = len(due)
        neg = torch.full((n_d,), -float("inf"), dtype=dt, device=dev)
        smax = neg.scatter_reduce(0, drone_of[keep], s[keep], "amax")[drone_of]
        g2 = af.take(due)
        gxx, _ = g2.goal()
        g2.dir = np.where(gxx >= g2.x, 1.0, -1.0)
        canon = torch.as_tensor(expert.control(g2, np.full(n_d, expert.CANON)), device=dev,
                                dtype=dt)[drone_of]  # fmt: skip
        d = ((self._act0 - canon) ** 2).sum(1)
        banded = (smax > -500) if self.band else torch.zeros_like(keep)
        elig = keep & torch.where(banded, s >= smax - self.band, s == smax)
        dkey = torch.where(banded, d, torch.zeros_like(d))
        pick = _seg_first(drone_of, [dkey, -s], elig, n_d)
        chosen = plans[pick].cpu().numpy()
        self.last_gpu = {"best": best, "keep": keep, "drone_of": drone_of, "plans": plans}
        if self.check is not None:  # e.g. check_planner.py: numpy's decision, to compare
            chosen = self.check(f, due, near, chosen)
        self.plan[due] = chosen
        self.age[due] = 0
        self.fresh[due] = False
        self.decisions += len(due)

    # --- Planner.evaluate, as tensors ----------------------------------------------------------
    def _rollout(self, S, C, P, plans, H):
        return tr.rollout(S, C, P, plans, H, self.T, body=self.body)

    def _failed(self, S):
        return torch.isin(S["outcome"], self.failed_set)

    def _evaluate(self, S, C, P, plans, gid, n_groups, pend0):
        T, H, dev = self.T, self.H, self.device
        MK = T[0]
        m1, t1, m2, t2, m3 = plans.unbind(1)
        m0 = torch.where(0 < t1, m1, torch.where(0 < t2, m2, m3))
        self._act0 = tr.control(S, C, P, m0, T)
        d0 = tr.distance(S, C)
        S, alive, dmin, gain = self._rollout(S, C, P, plans, H)
        self.stats["groups"] += n_groups
        self.stats["copies landed"] += int((S["outcome"] == LANDED).sum())
        # progress_score
        core = 0.5 * (d0 - tr.distance(S, C)) + 0.5 * gain / H
        jinks = ((MK[m1] == expert.JINK) & (t1 > 0)).to(d0.dtype)
        jinks = jinks + ((MK[m2] == expert.JINK) & (t2 > 0)).to(d0.dtype)
        core = core - self.jink_cost * jinks
        # open_threats
        hostile = S["p_on"] & C["p_hostile"] & ~S["p_done"]
        r = torch.hypot(S["p_x"] - S["x"][:, None], S["p_y"] - S["y"][:, None])
        threat = (hostile * torch.clamp(25.0 / torch.clamp(r, min=1e-6), 0, 1)).sum(1)
        failed = self._failed(S)
        landed = S["outcome"] == LANDED
        lasted = alive / H
        flying = S["outcome"] == FLYING
        resolved = torch.zeros_like(failed)
        # deepening against threats: up to deep_rounds of deep_k per group
        deep = flying & (threat > 0)
        rank = _seg_rank(gid, core, deep, n_groups)
        kh_all = torch.argmin((self.heights[None] - S["y"][:, None]).abs(), 1)
        nd = self.dodge_t.shape[1]
        for rnd in range(self.deep_rounds):
            g_res = torch.zeros(n_groups + 1, dtype=torch.bool, device=dev)
            g_res[gid[resolved]] = True
            sel_m = (rank >= rnd * self.deep_k) & (rank < (rnd + 1) * self.deep_k)
            sel_m &= ~g_res[gid] & ~resolved
            sel = torch.nonzero(sel_m)[:, 0]
            if len(sel) == 0:
                break
            self.stats[f"dodge searches, round {rnd + 1}"] += len(sel)
            rows = sel.repeat_interleave(nd)
            sub, subC = tf.take(S, C, rows)
            sp = self.dodge_t[kh_all[sel]].reshape(-1, 5)
            sub, a2, dm2, _ = self._rollout(sub, subC, P, sp, self.H2)
            ok = (~self._failed(sub)).reshape(-1, nd)
            a2, dm2 = a2.reshape(-1, nd), dm2.reshape(-1, nd)
            any_ok = ok.any(1)
            best_dm = torch.where(ok, dm2, torch.full_like(dm2, -float("inf"))).amax(1)
            self.stats["dodged"] += int(any_ok.sum())
            resolved[sel] = any_ok
            dmin[sel] = torch.where(any_ok, torch.minimum(dmin[sel], best_dm), dmin[sel])
            failed[sel] = failed[sel] | ~any_ok
            lasted[sel] = torch.where(any_ok, lasted[sel],
                                      (alive[sel] + a2.amax(1)) / (H + self.H2))  # fmt: skip
        # deepening at the pad: wait, then land
        land_at = torch.full_like(d0, -1.0)
        gx, _ = tr.goal(S, C)
        near = flying & (threat == 0) & ((gx - S["x"]).abs() < expert.PAD_DEEP)
        rank = _seg_rank(gid, core, near, n_groups)
        sel = torch.nonzero(rank < self.deep_k)[:, 0]
        if len(sel):
            nl = self.land_t.shape[0]
            self.stats["landing searches"] += len(sel)
            sub, subC = tf.take(S, C, sel.repeat_interleave(nl))
            sp = self.land_t.repeat(len(sel), 1)
            sub, a2, dm2, _ = self._rollout(sub, subC, P, sp, self.H2)
            down = (sub["outcome"] == LANDED).reshape(-1, nl)
            a2, dm2 = a2.reshape(-1, nl), dm2.reshape(-1, nl)
            key = torch.where(down, a2, torch.full_like(a2, float("inf")))
            b = torch.argmin(key, 1)  # the first of the quickest
            any_down = down.any(1)
            self.stats["landing found"] += int(any_down.sum())
            a_b = a2.gather(1, b[:, None])[:, 0]
            d_b = dm2.gather(1, b[:, None])[:, 0]
            land_at[sel] = torch.where(any_down, alive[sel] + a_b, land_at[sel])
            dmin[sel] = torch.where(any_down, torch.minimum(dmin[sel], d_b), dmin[sel])
        # waypoints ahead (pend0) and the mid-route frog, each a close pass at worst
        worst = self.near_cost * self.safe
        pending = pend0 + worst * (C["frog_on"][:, 0] & ~S["frog_past"][:, 0]).to(d0.dtype)
        inside = (C["frog_p"][..., 0] - S["x"][:, None]).abs() < FROG_REACH + 1.0
        in_reach = (C["frog_on"] & ~S["frog_past"] & inside).sum(1).to(d0.dtype)
        zero = torch.zeros_like(threat)
        sc = core - self.open_cost * (torch.where(resolved, zero, threat) + in_reach) - pending
        sc = torch.where(landed, 1000.0 - alive, sc)
        sc = torch.where(land_at >= 0, 1000.0 - land_at, sc)
        sc = sc - self.near_cost * torch.clamp(self.safe - dmin, min=0.0)
        sc = torch.where(failed, -1000.0 + 10.0 * lasted, sc)
        return sc


def fly(feat, seeds, ph, weather=fp.ROUGH, device="cuda", dtype=torch.float32, log=None,
        max_updates=1300, **kw):  # fmt: skip
    """The GPU planner flies these courses, all at once."""
    f = fp.Fleet(len(seeds), ph, feat=feat)
    for i, s in enumerate(seeds):
        f.set_course(i, fp.make_course(int(s), weather, feat, f.ph))
    pl = GPUPlanner(f.n, device=device, dtype=dtype, **kw)
    t = 0
    while (f.outcome == FLYING).any() and t < max_updates:
        f.step(pl.act(f))
        t += 1
        if log and t % 100 == 0:
            log(t, f, pl)
    return f, pl
