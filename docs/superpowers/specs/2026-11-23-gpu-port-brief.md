# GPU port of the 016 planner and DAgger: brief for the GPU thread

Written 2026-11-23 by the CPU thread (main checkout). Read
`docs/superpowers/specs/2026-11-23-flawless-plan.md` first: it has the findings so far.

## Goal

A full DAgger run of `experiments/016-flawless/pilot.py dagger` (the planner flies 400 training
courses, then 5 rounds of 250 network flights, labelled by the planner; a network refit and a
200-course test after each round) in **10 minutes or less on one rented GPU**. Same results as the
CPU code, within float tolerance.

## Where the time goes (measured on the M1 Pro, numpy)

- The fair planner (`fair_planner.py`, on top of `015-teacher/expert.py`) costs **7.7 core-s per
  flight-second** over a full test run (fair-t45-v6: 2370 s wall, 10 processes, 3096 flight-s).
- About 90% is `Fleet.step` (`014-gauntlet/gauntlet_env.py`) on the planner's copies:
  **~2 us per copy-step** on one core. The planner keeps ~900 candidate plans per drone
  (x3 frog-timing hypotheses near a frog), flies each 90 updates ahead (4.5 s), plus deepening
  searches. It re-plans every 4 updates (0.2 s).
- A DAgger run is about 31,000 flight-s, so **~66 core-hours**: 6-8 h on 10 cores.
- Estimate for a GPU, unverified: about 1e11 copy-steps per run, at a few KB of memory traffic
  each, is ~3 min on an H100 and ~6 min on an RTX 4090. Kernel-launch overhead adds ~4 min
  (6 rounds x ~225 sequential decisions x ~250 sequential steps). Batching all drones and
  hypotheses into one tensor, and using CUDA graphs or torch.compile, is what makes this work.

## Plan, with a go/no-go before the big part

1. **Port `Fleet.step` to PyTorch** (batched over copies, float32 on the GPU, float64 on the
   CPU for tests). Include the ghost check in `fair_planner._ghost_step`.
   - **Parity:** random courses and actions, many steps, against the numpy Fleet; float64 should
     match to ~1e-9, and outcomes (hits, landings) must match exactly.
   - **Benchmark on the rented GPU:** ms per step at 1M copies.
   - **Go/no-go:** if it is above ~0.5 ms/step, stop and report to the user (the 10-minute
     target would not hold) with the measured number and cost.
2. **Port the planner** (expert.Planner + FairPlanner: candidates, rollout, scoring, the three
   deepenings, beliefs about frogs, the air/mass estimate, the band). Rewrite the per-drone
   Python selection in the deepening as tensor ops (top-k per group, segment max).
   - **Parity:** the same chosen plan as numpy on a sample of states.
   - **The real test:** the GPU planner lands about 200/200 on the test courses 20000-20199 at
     45 deg (`fair_planner.py --tilt 45` defaults; reference: `runs/fair-t45-v6.json`, 200/200).
3. **Port pilot.py's DAgger** (move classifier + fail head; the "net" and "shield" pilots) and
   time a full run. The CPU thread is running the same thing (`d1`). Compare results with it
   later, through the user.

Report the measured times and dollars after each stage.

## Workspace rules (two threads share the repo)

- Work only in `/Users/rom/projects/rl-lab-gpu` (git worktree, branch `gpu-port`). Never touch
  `/Users/rom/projects/rl-lab` (the CPU thread's checkout) or its processes.
- New code goes in **`experiments/016-gpu/`**, new files only. `014-gauntlet/`, `015-teacher/`
  and `016-flawless/` are a read-only reference; import from them (sys.path), don't edit them.
  - `experiments/016-flawless/` and `docs/superpowers/specs/2026-11-23-flawless-plan.md` are
    uncommitted snapshots owned by the CPU thread: **never commit or edit them** on this branch
    (the CPU thread commits them on main; a copy here would conflict at merge).
- Don't edit `pyproject.toml` or `uv.lock`. torch comes in through stable-baselines3, and on
  Linux the default wheel has CUDA. Use PyTorch; **ask the user before adding JAX** or any
  other dependency.
- **Local CPU is busy:** the CPU thread's DAgger uses all 10 cores for the next ~8 h. Locally,
  run only small parity tests (1-4 drones, short flights, no process pools,
  `torch.set_num_threads(2)`). Heavy runs go on the rented GPU.
- Commit to `gpu-port` as you go; don't merge into main or push without the user's say.
- CLAUDE.md applies: env modules import only numpy + gymnasium (the torch port is not an env
  module, and CI must not import it); `ruff check`/`ruff format` only your own directory.

## The rented machine

- The user is choosing the provider (Vast.ai, RunPod, etc.) and does sign-up, payment, SSH keys
  and passwords themselves. Never enter credentials or payment details.
- Ask the user for the SSH host and port once it's up.
- Getting the code there: `rsync` the worktree, including the untracked `016-flawless/` snapshot
  (a git clone won't have it); then `uv sync` and `nvidia-smi`.
- Keep the box idle as little as possible. Tell the user when it can be stopped, and what it
  cost (hours x the rate they report).

## About the user

- They want experiments driven end to end without check-ins, except the go/no-go above and the
  machine setup.
- Every claim must be measured, not assumed; label estimates as estimates.
- Treat their email and personal details as private.
