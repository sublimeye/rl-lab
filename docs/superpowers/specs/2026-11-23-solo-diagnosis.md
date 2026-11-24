# Experiment 017: why the copy fails, and what to try (written before any training run)

Written 2026-11-23, from post 16's saved data only (`experiments/016-gpu/runs/g1`, copied from
the GPU worktree; local only: 410k planner-labelled states from 1,900 flights, six saved
networks) plus cheap test flights of
those networks. Code: `experiments/017-solo/` (`refit.py`, `sensitivity.py`). Drone: 45° tilt,
as in post 16.

## What was measured

1. **The 0.90-0.94 "held-out accuracy" was a leak.** Post 16 held out states at random, so every
   held-out state had its neighbours (0.05 s earlier and later, same flight) in the training
   set. On states from its own next round of flights, which it never trained on, each round's
   network picks the planner's move 59-67% of the time (21-28% where the planner changes move),
   in both the CPU and the GPU run, with no improvement from round 1 to round 5.
2. **It memorises.** Post 16's recipe refitted with whole flights held out: 98.0% on fitted
   states, 63.1% on unseen flights.
3. **Its errors are systematic, not rare slips.** In unseen decisions where a safe move exists,
   its top move is one the planner found fatal (every plan starting with it fails) 13.2% of the
   time: 23.5% with a bird or missile about to pass (closest approach within 2 s and 2 m), 8.5%
   otherwise, 1.7% with nothing in sight and no frog near. In failed flights the first such
   move comes a median 1.0-1.2 s before a bird hit, 2.2-3.9 s before a missile hit, 3.6-3.7 s
   before a frog hit; the state is lost (no safe move left) only 0.8-1.4 s before the end. So
   there is usually a stretch where safe moves exist and the network keeps picking fatal ones
   (median 3-4 fatal decisions per failed flight).
4. **It fails every part of the task separately.** Final network, 200 test courses each: no
   threats at all 0 landed (193 fly off the course: every training course had a frog on the
   pad); one bird per waypoint 0 (160 bird hits); frogs only 6 (184 caught); the gauntlet 3.
   A scripted pilot that ignores every threat (cruise, then land) lands 3-4 of 200 gauntlet
   courses: post 16's network is no better than not looking.
5. **It repeats itself.** When the planner's move differs from the move the network made last
   (22-41% of its own states), the network keeps its last move 76% of the time and picks the
   planner's 7%. But hiding the last-move input does not help by itself (fatal picks 13.2% ->
   14.1%, 12 -> 7 dev courses landed).
6. **The planner's pick is partly arbitrary.** Typically 37 of the ~47 moves it tries are safe,
   and it picks one by a tie-break. Nudging the world by the sensors' noise (5 cm, 0.1 m/s; 40
   flights, 1,050 decisions made twice) changes its move in 9.8% of decisions (11.7% with an
   urgent threat, 15.1% near a frog, 0.7% in calm air), and in 2.7% (4.9% urgent) the nudged
   planner picks a move the real one finds fatal. Near-identical inputs do carry conflicting
   labels, and the planner's margins are thinner than the sensor noise.
7. **More of the same data helps slowly.** Fatal picks on unseen flights with 215 / 438 / 881 /
   1,703 flights fitted: 16.8% / 15.6% / 14.8% / 14.1%.
8. **What does carry over is which moves are safe.** Same data, same network size, target
   changed to "all your probability on moves the planner found safe", with dropout 0.3, input
   noise 0.1, weight decay 0.01 and no last-move input: fatal picks 7.5% (14.0% urgent), and
   **42 of 200** dev courses landed alone (the argmax copy: 12). Either change alone does
   little (safe-set target only: 11.6%, 10-14 landed; regularisation only: 11.8%, 13 landed).
   The unused danger head as a veto on post 16's network: 3 -> 17 of 200.

## Hypothesis

Copying the planner's single pick is the wrong target. That pick is the argmax of a 4.5 s search
plus a tie-break, it flips under sensor-sized nudges, and 1,900 flights are far too few to pin
it down, so the network memorises flights and learns to repeat itself. What a pilot needs, and
what generalises from the same data, is *which moves get it killed*. The planner's per-move
verdicts give that directly (item 8), and so do the outcomes of the network's own flights.

Predictions that would refute it: a network trained on outcomes (RL over the same 59 moves),
starting from the safe-set network, fails to improve on it; or more on-policy safe-set labels
fail to lower the fatal-pick rate.

## Routes, in order, each with a kill criterion

Selection on dev courses 40000-40999 (not the test or fresh courses, which are flown once at
the end).

A. **RL fine-tune over moves from the safe-set network** (laptop, no planner, no GPU): PPO where
   the action is one of the 59 moves and post 15's controller flies it, the safe-set loss on the
   saved states as an auxiliary term so it can't forget. *Kill:* not at least 5 points above its
   starting network on the dev courses after 45 minutes.
B. **Safe-set DAgger on the GPU** (the planner labelling safe moves on the new network's own
   flights; ~2 min a round on a rented H100/H200): *Kill:* under +5 points after 3 rounds.
C. **Better offline fits** of the safe-set target (minutes each) run alongside, as the starting
   point for A and B.

Not chosen: PPO from scratch on sticks (post 14's route, 33/200 after 280 min); a bigger or
recurrent copy of the argmax (items 2, 6, 7 say the target is the problem, not the capacity); a
curriculum by threat (item 4: every part fails, none dominates).

## Two facts about the pilot being compared

- Post 16's "network alone" is a network picking a move 20 times a second; a hand-written
  controller turns the move into sticks from the true state, and the network's inputs include
  the planner's air estimate, which replays each update on a simulator copy. Experiment 17's
  pilot keeps the moves and the controller but drops the air estimate (the controller assumes
  still air and the average mass), so nothing at flight time copies the simulator.
- Post 14's 33/200 is a network on the sticks themselves, 30° drone. Any comparison with it
  has to say so.

---

# Results (added the same evening, after the runs)

Selection on the dev courses; the test (20000-20199) and fresh (30000-30199) courses were flown
once per network, and every network flown on them is listed. "Drawn": each move drawn from the
network's probabilities (seeded); "top": always its most probable move.

## C. The safe-set fit without the air estimate (`refit.py --name sr`, the starting network)

Fatal picks on unseen flights 7.3% (urgent 13.2%). Alone in `solo_env` (still-air controller, no
planner object): 163 of 1,000 dev courses (top). Test 48 drawn / 28 top; fresh 43 / 37.

## A. PPO over moves from that network (`train.py`, laptop, no GPU rented)

- **a1** (the time-boxed attempt: 45 min, side task weight 0.3, 152.6M decisions, ~56k/s):
  dev (400) from 64 top / 101 drawn to 147 / 167 after 2.7 min (the kill line was +5 points in
  45 min) and 291 / 298 at 43.6 min. 1,000 dev courses at the end: 716 top, 749 drawn.
  **Test 145 drawn / 129 top; fresh 152 / 140.** One kind of encounter at every waypoint (top
  move, hits per encounter): bird 5 of 661, two birds 14 of 651, missile 17 of 638, bird +
  missile 30 of 637, flock 45 of 598; frogs only 167 of 200 landed (188 drawn). On any course
  without a frog on the pad it lands 0 of 200 (flies off the end of the course): it never saw
  one.
- **s1** (ablation, 15 min: from nothing, lr 3e-4): 0 or 1 of 400 dev courses at every check,
  0 of 1,000 at the end. One seed, one setting.
- **n1** (ablation, 15 min: from the copy, no side task): 242 top / 260 drawn of 400 at 49.2M
  decisions, against a1's 206 / 221 at the same count; 638 / 671 of 1,000 at the end. The side
  task slows practice. (A stuck script of mine shared the CPU with n1 for about 5 of its 15
  minutes, so compare by decisions, not minutes.)
- **a2** (carrying a1 on, no side task, a fresh value network, lr decaying): dev 294 / 301 at its
  start fell to 249 / 255 by 7.5 min and was 266 / 284 at 12.7 min. Stopped at 13.7 min: the
  fresh value network cost more than the time left would win back. `train.py` now saves the
  value network (`critic.npz`, `--critic`). Not flown on test or fresh.

- **b1** (the recipe n1 favoured, as one clean run: 45 min from the copy, no side task, 170.7M
  decisions, ~63k/s; its first six checks repeat n1's exactly): dev (400) 316 top / 339 drawn at
  44 min; 1,000 dev courses at the end: 772 top, **843 drawn** (a1: 716 / 749), so b1 is the
  pilot the post shows. **Test 162 drawn / 156 top; fresh 163 / 157.** One kind of encounter at
  every waypoint (top move): bird 0 of 665, two birds 9 of 661, missile 13 of 652, bird + missile
  27 of 641, flock 32 of 623; frogs only 171 of 200 landed (188 drawn). No frog on the pad: 0 of
  200 landed (139 off the far end past the pad, 61 out through the ceiling).

## B. Safe-set DAgger on a GPU: not run

Route A passed its line first. No GPU was rented; the session cost nothing.

## Post 16's network, for reference

Test 3, fresh 5 (its own flight loop). A scripted pilot that ignores every threat: test 4,
fresh 4, clear courses 198.
