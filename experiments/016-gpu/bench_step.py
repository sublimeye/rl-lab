"""How fast is the torch step on a GPU? ms per step (one stick update, 3 physics steps) for N
copies, eager, compiled, and compiled inside a CUDA graph.

The copies are the planner's kind: drones flown a while in numpy (birds, missiles and frogs
about), copied with ghost windows, repeated up to N. Every copy is computed every step whether
it still flies or not (masks), so the time doesn't depend on how many are left.

uv run experiments/016-gpu/bench_step.py --n 65536 262144 1048576
uv run experiments/016-gpu/bench_step.py --explain   # graph breaks only (CPU, no codegen)
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
import parity  # noqa: E402  (puts 014-016's folders on the path)
import torch_fleet as tf  # noqa: E402

from gauntlet_env import FLYING, FULL, ROUGH, Fleet, make_course  # noqa: E402

expert = parity.expert


def base_states(n_drones=16, updates=200, seed=0):
    """Copies (numpy) of drones mid-flight, with ghost windows: the planner's kind."""
    rng = np.random.default_rng(seed)
    f = Fleet(n_drones, parity.PH45, feat=FULL)
    for i in range(n_drones):
        f.set_course(i, make_course(20_000 + i, ROUGH, FULL, f.ph))
    for _ in range(updates):
        f.step(expert.control(f, np.full(n_drones, expert.CANON)))
    rows = np.nonzero(f.outcome == FLYING)[0]
    return parity.copies(f, rows, rng, ghost=True)


def inplace(step):
    def step_(S, C, a, P):
        new = step(S, C, a, P)
        for k, v in S.items():
            v.copy_(new[k])

    return step_


def bench(n, device, dtype, modes, reps=50, graph_steps=10, log=print):
    g = base_states()
    S0, C0 = tf.from_numpy(g, device=device, dtype=dtype)
    P = tf.params(g.ph, g.feat.battery)
    rows = torch.arange(n, device=device) % g.n
    S, C = tf.take(S0, C0, rows)
    gen = torch.Generator(device=device).manual_seed(0)
    a = torch.rand((n, 2), device=device, dtype=dtype, generator=gen) * 2.4 - 1.2
    S_init = {k: v.clone() for k, v in S.items()}
    out = {}

    def reset():
        for k, v in S.items():
            v.copy_(S_init[k])

    def sync():
        if device != "cpu":
            torch.cuda.synchronize()

    for mode in modes:
        reset()
        if mode == "eager":
            fn = inplace(tf.step)
        else:
            fn = torch.compile(inplace(tf.step), fullgraph=True, dynamic=False,
                               mode="max-autotune-no-cudagraphs" if mode != "compile-default"
                               else None)  # fmt: skip
        t0 = time.time()
        fn(S, C, a, P)  # compile / warm up
        fn(S, C, a, P)
        sync()
        warm = time.time() - t0
        if mode == "graph":  # CUDA graph of graph_steps steps, replayed
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn(S, C, a, P)
            torch.cuda.current_stream().wait_stream(s)
            cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg):
                for _ in range(graph_steps):
                    fn(S, C, a, P)
            reset()
            sync()
            t0 = time.time()
            for _ in range(reps // graph_steps):
                cg.replay()
            sync()
            ms = (time.time() - t0) / (reps // graph_steps * graph_steps) * 1e3
        else:
            reset()
            sync()
            t0 = time.time()
            for _ in range(reps):
                fn(S, C, a, P)
            sync()
            ms = (time.time() - t0) / reps * 1e3
        alive = int((S["outcome"] == FLYING).sum())
        out[mode] = {"ms_per_step": round(ms, 4), "warmup_s": round(warm, 1), "flying_after": alive}
        per = ms * 1e6 / n
        log(f"n={n:>8} {str(dtype):>14} {mode:>16}: {ms:8.3f} ms/step  ({per:.2f} ns per "
            f"copy-step; warm-up {warm:.0f} s; {alive} still flying)")  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[65_536, 262_144, 1_048_576])
    ap.add_argument("--modes", nargs="+", default=["eager", "compile", "graph"])
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--explain", action="store_true", help="list graph breaks (CPU) and stop")
    ap.add_argument("--out", default=None, help="save results to runs/<out>.json")
    args = ap.parse_args()
    dtype = torch.float64 if args.float64 else torch.float32
    if args.explain:
        torch.set_num_threads(2)
        g = base_states(4, 100)
        S, C = tf.from_numpy(g, dtype=dtype)
        a = torch.zeros((g.n, 2), dtype=dtype)
        ex = torch._dynamo.explain(tf.step)(S, C, a, tf.params(g.ph))
        print(f"graphs {ex.graph_count}, breaks {ex.graph_break_count}, ops {ex.op_count}")
        for b in ex.break_reasons:
            print("  ", b)
        return
    device = "cuda"
    print(torch.cuda.get_device_name(), torch.__version__, torch.version.cuda, flush=True)
    res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "runs": {}}
    for n in args.n:
        res["runs"][n] = bench(n, device, dtype, args.modes, args.reps)
        print(f"   memory peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    if args.out:
        (HERE / "runs").mkdir(exist_ok=True)
        (HERE / "runs" / f"{args.out}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
