# GPU port of experiment 016: what was done, measured and learned

Written 2026-11-23 by the GPU thread (worktree `/Users/rom/projects/rl-lab-gpu`, branch
`gpu-port`), for the CPU thread and any later session. Brief: `2026-11-23-gpu-port-brief.md`.
Findings before this: `2026-11-23-flawless-plan.md` (owned by the CPU thread, uncommitted in the
main checkout). Code: `experiments/016-gpu/` (new files only; 014/015/016-flawless untouched).

## In one paragraph

The fair planner and pilot.py's DAgger now run on a GPU, with the same decisions as the numpy
code. On a rented H200 the planner flies the 200 test courses in 85 s (MacBook, 10 cores:
2370 s) and a full DAgger run takes 14 min (MacBook: ~4.5 h for d1). The network alone stays at
1-6/200 on both CPU and GPU. New: the *shield* (the planner's search limited to the network's
top 5 moves, a full search only when all fail) lands 200/200 on the test courses with the
GPU-trained network, needing the full search in 2% of decisions, and on the fresh courses too (200/200, 1.7%).

## What was built (experiments/016-gpu/)

| File | What it is |
|---|---|
| `torch_fleet.py` | `Fleet.step` (014) in PyTorch for the planner's copies, plus fair_planner's ghost checks; masks instead of index lists; no waypoint spawns (the fair planner's copies never meet one) |
| `torch_rollout.py` | expert's controller (with 016's gentle touchdown) and `Planner.rollout` |
| `gpu_planner.py` | `GPUPlanner(FairPlanner)`: every due drone x frog hypothesis is a group; candidates, dodge rounds, pad search, scoring and the band's choice as segment tensor ops; air fit over all gust periods at once |
| `dagger_gpu.py` | pilot.py's DAgger: all 4-course chunks flown in lockstep in one process (same seeds and random draws as the pool), one GPU decision per update; training on the GPU |
| `fly_gpu.py` | the GPU planner on N courses at once |
| `cpu_pilots.py` | pilot.py's `test` (net / shield / planner, numpy, process pool) for networks in `016-gpu/runs/` |
| `parity.py`, `check_planner.py`, `bench_rollout.py`, `test_torch_fleet.py` | accuracy checks against numpy (the test skips without torch, so CI is unaffected) |
| `bench_step.py` | step benchmark |
| `remote.sh` | rsync this worktree to a rented box, `uv sync`, GPU check |

Run outputs are in `experiments/016-gpu/runs/` (gitignored, like every experiment's runs; on
the MacBook only): `bench-h200.json`, `rollout-h200-dynamic.json`, `gpu-t45*.json`, `g1/` (the
DAgger run: log, rounds.json, data_00-05.npz = 410k labelled states, net_00-05.npz, the shield
tests).

## Accuracy (measured)

- Step, float64 vs numpy: 12 seeds x 320 copies x 90 updates, every field to <= 6e-14, every
  outcome identical (landed, crashed, water, lost, timeout, bird, missile, frog, ghost catches).
  Same on the GPU in float64. Float32: outcomes identical on 5 of 6 seeds (one bird hit came one
  physics tick later).
- Look-ahead (controller + rollout), 1782 copies x 90 updates: state to 3e-14, updates flown
  exact.
- Whole decisions, float64 on CPU, numpy FairPlanner vs GPUPlanner, from the same states:
  45 + 75 + 30 + 135 = 285 decisions, every chosen plan identical, every candidate's score
  within 1.5e-13. Covered: hypotheses, ghost windows, pad search, landings. **Not yet covered:
  the dodge rounds** (no checked state ended a look-ahead with a bird or missile still after
  it); they ran ~27,000 times in the GPU's 200/200.
- Known difference, by design: numpy's look-ahead stops when nothing in its batch flies; a GPU
  batch holds all groups. It only affects copies that time out (60 s) inside a look-ahead.

## Speed (measured; H200 NVL, float32, torch 2.15 + CUDA 13.2)

| | MacBook M1 Pro (10 cores, numpy) | H200 | |
|---|---|---|---|
| Planner, test courses 20000-20199 | 2370 s, 200/200 (fair-t45-v6) | 85 s, 200/200 | 28x |
| Planner, fresh courses 30000-30199 | 2361 s, 200/200 | 77 s, 200/200 | 31x |
| DAgger, round 0 (planner, 400 courses) | 6504 s (d1) | 180 s (g1) | 36x |
| DAgger, whole run (6 rounds) | ~4.5 h (d1, excluding pauses; estimate until it ends) | 14 min (g1) | ~19x |

- Step: 1.2 ns per copy-step at 1M copies in a CUDA graph (1.26 ms/step; the brief's 0.5 ms
  go/no-go line was not met, but the end-to-end target nearly was).
- Compile: one 120-140 s compile per run with dynamic shapes (33 s from the disk cache).
- Cost: H200 NVL at $3.889/h, ~72 min used (~$4.60). An H100 SXM ($1.49-1.79/h) should do a
  DAgger run for ~$0.40-0.60 (estimate, not measured on an H100).

## Learning (measured)

| Round | g1 (GPU, 300 flights/round) network alone | d1 (CPU, 250 flights/round) network alone |
|---|---|---|
| 0 (planner flies 400) | 1/200 | 4/200 |
| 1 | 4 | 4 |
| 2 | 1 | 6 |
| 3 | 3 | 3 |
| 4 | 6 | 4 |
| 5 | 3 | (running) |

Same conclusion on both: the network alone doesn't learn to fly the gauntlet (held-out move
accuracy 0.90-0.94; one wrong move in ten is fatal on a long course; training courses flown by
the mixed pilot fall from ~130 landed to 3-11 as the network takes over).

Shield, g1's final network, k = 5, numpy on the MacBook's 10 cores:
- test courses 20000-20199: **200/200**, 1300 s, full searches 310 of 16,029 decisions (2%);
- fresh courses 30000-30199: **200/200**, 1367 s, full searches 267 of 15,755 decisions (1.7%).

So the network's top 5 holds a safe move 98% of the time: it learned a shortlist, not a choice.
The shield costs ~half the planner's CPU here (the smoke test had suggested ~1/4.7).

## The MacBook's CPU was shared (d1 paused twice, at the user's request)

d1 (`pilot.py dagger --name d1 --rounds 6 --first 400 --flights 250`, in the main checkout) was
paused with SIGSTOP and resumed with SIGCONT, losing nothing:
- 14:38:52 - 15:01:08 (22 min 16 s), during round 4;
- 15:21:55 - 15:44:54 (22 min 59 s), during round 5.
In all 45 min 15 s.
Subtract these from d1's logged times.

## Open items

1. Shield with k = 3 and 2 (how much of the pilot can be the network).
2. The shield is ported to the GPU (gpu_planner `allowed`, dagger_gpu shield mode). Checked
   against pilot.Teacher's numpy shield (check_shield.py, float64, random allowed moves so the
   full search runs): 30/30 decisions identical, full searches 6 = 6, each move's best score
   within 1.4e-14. Not yet run on a GPU.
3. A parity check that covers the dodge rounds.
4. Network experiments from the saved 410k states (on the Mac's GPU, MPS): size, training
   length, weighting of dangerous moments.
5. Recorded shield flights as data for the 016 post's page.

## Merging the two threads (not done yet)

- Nothing is merged or pushed. `gpu-port` holds only `experiments/016-gpu/` and this file.
- The CPU thread's `experiments/016-flawless/` and `2026-11-23-flawless-plan.md` exist only
  uncommitted in the main checkout: they must be committed on main (by the CPU thread, or with
  the user's go-ahead) before anything else; deleting that checkout or its changes would lose
  them. The brief's copy of them here is uncommitted on purpose.
- Then: merge `gpu-port` into main (no file overlaps); one 016 post tells the whole story in
  order (fair planner -> GPU port -> network vs shield), so the site reads as one journey.
