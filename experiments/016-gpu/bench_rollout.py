"""The planner's main look-ahead on the GPU: parity with numpy, and the time per decision.

--parity: a few drones' candidate plans (expert.Planner.candidates), flown 90 updates in numpy
(Planner.rollout) and in torch; compares (updates flown, closest pass, progress) and the state.
Default: time one decision's look-ahead for D drones (every candidate plan, 90 updates), the
whole look-ahead in one CUDA graph.

uv run experiments/016-gpu/bench_rollout.py --parity
uv run experiments/016-gpu/bench_rollout.py --drones 250 --hyps 1 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import bench_step  # noqa: E402
import parity  # noqa: E402
import torch_fleet as tf  # noqa: E402
import torch_rollout as tr  # noqa: E402

expert = parity.expert


def plans_for(g, pl):
    """Every candidate plan of each drone of g (fresh), and the copy rows."""
    gx, _ = g.goal()
    groups, rows = [], []
    for i in range(g.n):
        k_h = int(np.argmin(np.abs(np.array(expert.HEIGHTS) - g.y[i])))
        c = pl.candidates(abs(gx[i] - g.x[i]) < expert.PAD_NEAR, k_h)
        groups.append(c)
        rows.append(np.full(len(c), i))
    return np.concatenate(groups), np.concatenate(rows)


def check_parity(dtype=torch.float64, device="cpu", H=90):
    g = bench_step.base_states(4, 150)
    pl = expert.Planner(g.n)
    plans, rows = plans_for(g, pl)
    sim = g.take(rows)
    S, C = tf.from_numpy(sim, device=device, dtype=dtype)
    P = tf.params(sim.ph, sim.feat.battery)
    T = tr.tables(device, dtype)
    t0 = time.time()
    alive, dmin, gain = pl.rollout(sim, plans, H)
    t_np = time.time() - t0
    pt = torch.as_tensor(plans, device=device)
    S, a2, d2, g2 = tr.rollout(S, C, P, pt, H, T)
    worst, bad = parity.compare(sim, S, 1e-9 if dtype == torch.float64 else 1e-2)
    da = float(np.max(np.abs(alive - a2.cpu().numpy())))
    dd = float(np.max(np.abs(np.minimum(dmin, 1e6) - np.minimum(d2.cpu().numpy(), 1e6))))
    dg = float(np.max(np.abs(gain - g2.cpu().numpy())))
    outs = dict(zip(parity.OUTCOMES, np.bincount(sim.outcome, minlength=9).tolist(), strict=True))
    print(f"{len(rows)} copies x {H} updates: state {'MATCH' if not bad else bad} "
          f"(largest {worst:.1e}); alive diff {da}, closest-pass diff {dd:.1e}, progress diff "
          f"{dg:.1e}; outcomes {outs}; numpy {t_np:.1f} s")  # fmt: skip
    return not bad and da == 0 and dd < 1e-6 and dg < 1e-6


FNS = {}


def time_decision(
    drones, hyps, H=90, dtype=torch.float32, device="cuda", reps=5, log=print, dynamic=False
):
    g = bench_step.base_states(16, 150)
    pl = expert.Planner(g.n)
    plans, rows = plans_for(g, pl)
    # D drones (repeating the 16) x hyps hypotheses, every candidate
    per = np.bincount(rows, minlength=g.n)
    pick = np.arange(drones) % g.n
    sel = np.concatenate([np.nonzero(rows == i)[0] for i in pick] * hyps)
    n = len(sel)
    S0, C0 = tf.from_numpy(g.take(rows[sel]), device=device, dtype=dtype)
    P = tf.params(g.ph, g.feat.battery)
    T = tr.tables(device, dtype)
    pt = torch.as_tensor(plans[sel], device=device)
    S = {k: v.clone() for k, v in S0.items()}
    d0 = tr.distance(S, C0)
    acc = [torch.zeros_like(d0), torch.full_like(d0, 1e9), torch.zeros_like(d0), d0]
    us = torch.arange(H, device=device)

    def body_(S, C, P, plans, T, u, acc):
        S2, acc2 = tr.rollout_body(S, C, P, plans, T, u, tuple(acc))
        for k, v in S.items():
            v.copy_(S2[k])
        for a, b in zip(acc, acc2, strict=True):
            a.copy_(b)

    fn = FNS.get(dynamic) or torch.compile(
        body_, fullgraph=True, dynamic=dynamic, mode="max-autotune-no-cudagraphs"
    )
    FNS[dynamic] = fn
    t0 = time.time()
    for u in range(3):
        fn(S, C0, P, pt, T, us[u], acc)
    torch.cuda.synchronize()
    warm = time.time() - t0

    def reset():
        for k, v in S.items():
            v.copy_(S0[k])
        acc[0].zero_()
        acc[1].fill_(1e9)
        acc[2].zero_()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for u in range(3):
            fn(S, C0, P, pt, T, us[u], acc)
    torch.cuda.current_stream().wait_stream(s)
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):
        for u in range(H):
            fn(S, C0, P, pt, T, us[u], acc)
    times = []
    for _ in range(reps):
        reset()
        torch.cuda.synchronize()
        t0 = time.time()
        cg.replay()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    ms = 1e3 * float(np.median(times))
    flying = int((S["outcome"] == tr.FLYING).sum())
    ns = ms * 1e6 / (n * H)
    log(f"{drones} drones x {per.mean():.0f} candidates x {hyps} hyps = {n} copies, {H} "
        f"updates: {ms:.1f} ms per look-ahead ({ms / H:.3f} ms/update, {ns:.2f} ns per "
        f"copy-step); compile {warm:.0f} s; {flying} flying at the end")  # fmt: skip
    return {"drones": drones, "hyps": hyps, "copies": n, "H": H, "ms": round(ms, 2),
            "compile_s": round(warm), "flying_end": flying}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--drones", type=int, nargs="+", default=[250])
    ap.add_argument("--hyps", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--out", default=None)
    ap.add_argument("--dynamic", action="store_true", help="one compile for every size")
    args = ap.parse_args()
    if args.parity:
        torch.set_num_threads(2)
        ok = check_parity(device=args.device)
        print("PARITY", "OK" if ok else "FAILED")
        return
    print(torch.cuda.get_device_name(), flush=True)
    res = [time_decision(d, h, dynamic=args.dynamic) for d in args.drones for h in args.hyps]
    if args.out:
        (HERE / "runs").mkdir(exist_ok=True)
        (HERE / "runs" / f"{args.out}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
