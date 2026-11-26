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

---

# Second diagnosis, 2026-11-24: what still kills the pilot practice made (written before more training)

The user's reply to 162/163: not another post where the drone gets hit; can it land all 200, and
land on a pad with no frog? So b1 was taken apart the same way post 16's copy was, before any
new run. Code: `rewind.py` (courses 50000-50999, used for nothing else). Nothing below is
trained.

## What was measured (b1, moves drawn, unless said)

1. **The same courses fail, at the same place.** 1,000 courses flown 8 times each, with
   different draws and sensor noise: 80.7% landed. 450 courses landed all 8 times (a pilot with
   the same odds on every course: 180) and 63 landed at most twice (the same-odds pilot: 1).
   On those the flights end at the same spot by the same threat, 6-8 times of 8.
2. **A failed flight is not lost early, and not saved by a second try either.** 178 failed
   flights wound back and flown on 32 times each: still alive 4 s past the original end when
   wound back 0.5 / 1 / 2 / 3 / 5 / 8 s: 23 / 37 / 51 / 56 / 59 / 60%. Flights with under a 10%
   chance left: 52% at 0.5 s, 11% at 2 s, 7% at 3 s. So 3 s before the end nearly every flight
   still had a way out, and the pilot finds it about half the time; from 8 s back it still dies
   40% of the time (a whole flight: 19%).
3. **What kills it** (8,000 flights): birds 570, missiles 413, frogs 376 (mid-route 175, the
   pad's 201), crashes 145, left the course 21, water 20. By encounter: bird + missile 409,
   flock 226, two birds 156, missile 127, one bird 65.
4. **Always taking its top move is worse**, at every temperature between: 750 / 763 / 775 / 813
   / 822 of 1,000 at temperature 0 / 0.25 / 0.5 / 0.75 / 1 (frogs catch the top-move pilot 94
   times, the drawing one 47).
5. **It sprints.** It holds the full-throttle, full-tilt jink in 34% of its updates and is over
   10 m/s (the move table's fastest cruise) 36% of the time, over 12 m/s 22%. Its landings on
   the test courses take 12.6 s; the planner's (200 of 200) 15.5 s; the copy it started from
   23.7 s. Where nothing is in sight it picks the sprint with probability 1.00.
6. **Taking the sprint away at flight time doesn't help**: 774-783 of 1,000 against 822 (it
   never practised without it).
7. **The courses can be flown**: the planner lands 34 of the 35 courses b1 landed at most once
   in 8 (one missile hit 2 m from the pad). Told to assume still air and the average mass, as
   the solo pilot's controller does, it lands 31 (two crashes on the pad, a frog, a missile).

## Hypothesis

It was paid to hurry. Post 14's reward (0.3 a metre, +20 landed, -10 failed, 0.2 a second) is
discounted by 0.995 an update, which makes a reward one second away worth 10% less. Estimate,
from those numbers: mid-course the future is worth about 25, so a second costs about 2.5 and a
death about 35; a 7% chance of dying is a fair price for one second saved. Over four encounters
and a frog or two that is the 16-19% it loses. PPO did what it was paid for: it is faster than
the planner. And it cannot find out that slower is safer, because in open air it never draws
anything but the sprint.

Prediction: the same start (the safe-move copy), the same PPO, and a reward that pays for
landing and nothing for speed (discount 0.9998 an update, no charge for time, -50 for a failure
or for running out the 60 s) gives slower flights and clearly more landings at the same number
of decisions.

## Attempts, in order, each with a kill line (1,000 dev courses at each check)

D. **Paid to land** (c1): as b1 but for the reward above. *Kill:* at 20 minutes not at least
   3 points above b1 at the same number of decisions (b1: 76% of 400 at 20.6 min), or no slower.
E. **Practise what kills it** (c2): D, and half of each drone's new courses are ones it failed
   the last time it flew them (kept until it lands them). *Kill:* at 20 minutes not at least
   3 points above c1 at the same number of decisions.
F. **The long run**: what D and E leave standing, a quarter of the courses with no frog on the
   pad (b1 lands 0 of 200 of those), the learning rate decaying. Target before the test and
   fresh courses are flown (once): 99% of 1,000 dev courses. Landing exactly 200 of 200 needs
   under one failure in a thousand flights; that is the aim, not a promise.

Not chosen first: labels from the planner on b1's own flights (route B, a rented GPU): it would
teach dodges, but item 5 says the trouble starts before the dodge; kept for after F if specific
encounters still kill it. A bigger network: no evidence yet that size is the limit.

## Results of the second round (added as the runs finished)

Dev checks are 1,000 courses, moves drawn; b1's were 400 courses, so its shares carry about
+-2 points and these about +-1.3.

- **D, paid to land (c1): passed.** The same start and PPO as b1, 20 minutes (70.8M decisions).
  At the same number of decisions, c1 against b1: 44.1 / 45.8% at 9.8M, 60.9 / 58.3% at 19.7M,
  66.7 / 63.0% at 29.5M, 70.2 / 63.3% at 39.3M, 74.1 / 65.0% at 49.2M, 77.4 / 70.5% at 59.0M,
  79.0 / 73.0% at 68.8M; 810 of 1,000 at the end (b1 at 78.6M: 75.8%). Taking the top move it
  is further ahead: 72-74% against b1's 62-65% at 59-69M. Its landings take 22 s (b1 at the
  end: 12.6 s on the test courses). New failure: 1% of its flights run out the 60 s.
  A first try was cut to about 3 minutes by the laptop going to sleep on battery
  (`runs/c1-cut-short`); the trainer's clock now stops during sleep and the run was repeated.
- **E, practising what it failed (c2): no gain, dropped.** c1 plus replay of failed courses
  (half of the new courses), 20 minutes: 44.0, 59.3, 63.9, 70.5, 76.2, 78.2, 79.0% at c1's
  checks and 806 of 1,000 at the end against c1's 810. Under its kill line. (At a 20% failure
  rate half of any batch of courses is being failed anyway; the lever could matter when
  failures are rare. Not tested.)
- **Where the trainer's time goes** (measured): one process steps 1,024 drones in 10.6 ms
  (10 us a drone), and eight of them with the network on the Mac's GPU gather 178,000 decisions
  a second; the PPO update manages 260-370,000 sample-passes a second, so three passes over
  each round's 393,000 decisions take about 60% of the time. The simulator is not the limit on
  this laptop; the network update is.
- **F, the long run (f1), stopped at 36 minutes.** From c1, a quarter of the courses with no
  frog on the pad. Dev (1,000, drawn) 798 at its start, then 802, 835, 806, 853, 869, 863, 878
  every 19.7M decisions (137.6M in all, 208M with c1's). Open pads (200 dev courses): 0, 0, 0,
  140, 170, 183, 177, 185. So it learned the unguarded pad by itself, between 39M and 59M
  decisions; before that it hovered where the pad comes into sight (40 m out) until the 60 s ran
  out. Stopped because of what follows.
- **A larger learning rate (h1): worse.** From f1's network at 98M decisions, 3e-4 instead of
  about 8e-5: 825 and 832 at the two checks where f1 itself had 863 and 878.
- **Extrapolation (an estimate, not a measurement).** In the c1-f1 line the failure share fell
  from 29.8% at 39M decisions to 12.2% at 208M: about (decisions)^-0.5. On that line 1% needs
  roughly 100 times more flying (around 20 billion decisions: days on this laptop, hours on a
  rented box). More of the same practice is not the way to "never hit".

## Third look: does the next move decide a failed flight? (rewind.py, part "moves")

f1's network, 400 courses, the 50 flights that failed, each wound back 1, 2 and 3 s. From each
point every move on offer (about 48) was tried as the next move, held 0.3 s, on 8 copies each,
with the network flying on as usual. Share still alive 4 s after the original end:

| Wound back | Next move drawn as it does | Its top move | The best next move* |
|---|---|---|---|
| 1 s | 53% | 54% | 78% |
| 2 s | 65% | 63% | 85% |
| 3 s | 67% | 69% | 84% |

*picked on 4 of the copies, scored on the other 4, so luck doesn't flatter it. The best and
worst next move differ by 64-83 points of survival, and about half to two thirds of the moves
on offer mostly survive.

**Hypothesis.** Practice (PPO) learns each decision from one flight's outcome, and nine
flights in ten teach it nothing about dying. The simulator can answer the question directly:
rewind a failed flight, try every next move, and see which live. Fitting the network to put
its probability on the moves that lived, at those moments, should remove a large share of the
failures per pass, far faster than practice. It is still only a network at flight time; the
copies are used in training only, as the planner's were in post 16.

G. **Second tries** (retry.py): fly a few thousand courses, rewind the failures 1, 2 and 3 s,
   try every move (4 copies), fit, repeat. *Kill:* after the first pass, on failed flights it
   was not fitted to, survival at the rewound moments not up by 5 points, or 2,000 dev courses
   not up by 1.5 points.
- **G, second tries (g1, one pass): failed its kill line.** 8,000 courses flown by f1's
  network, 854 failures, 2,562 rewound moments, every move tried on 4 copies (3.4 minutes).
  Fitted moments: expected survival 62% -> 79%. Failed flights held out of the fit: 61.7% ->
  63.0% (the line was +5). 2,000 dev courses: 1,778 -> 1,750. It memorised the moments it was
  shown, as post 16's copy memorised its flights.

## Fourth look: why each decision teaches so little

f1's network (rewind.py on `runs/f1-r350`): 88.6% of 8,000 flights landed; 569 of 1,000 courses
landed 8 times of 8 (a same-odds pilot: 381) and 34 at most 3 times. Its moves drawn at
temperature 0 / 0.25 / 0.5 / 0.75: 827 / 864 / 883 / 883 of 1,000. One course it fails 8 times of
8 (50850), looked at update by update: a flock of three comes head-on, one bird level with the
drone and two about 2 m below. For the two seconds before the hit the network's probabilities are
split between holding its height (about 50%) and a cruise 5 m lower (12-26%); it draws a new move
every 0.05 s, so it does some of each and sinks slowly into the lower birds. Held longer without
retraining its moves are useless (2,000 dev courses: 1,778 with a move per update, 1,623 held 2
updates, 1,132 held 4): it has learned to steer by mixing moves.

**Hypothesis.** Twenty decisions a second, each drawn afresh, means no single decision matters:
the outcome hangs on what it does for the next second, not the next 0.05 s. That makes each
decision's share of the credit tiny (slow practice), makes "try something else" last 0.05 s
(no exploring a dive), and lets two good answers average into a bad one. The planner decides
every 0.2 s and its plans hold their first move 0.6 s. Deciding every 0.2 s (a move held 4
updates) should make practice both faster per decision and better at these moments.

H. **Decide every 0.2 s** (k1): c1's recipe (from the safe-move copy, paid to land), the move
   held 4 updates, a quarter of the courses with open pads. *Kill:* at 20 minutes not above
   c1's 81% of dev courses at 20 minutes.
- **H, decide every 0.2 s (k1): passed.** c1's recipe with the move held 4 updates (and a
  quarter of the courses with open pads), 20 minutes, 59.0M decisions (49,500 a second, each
  now 0.2 s of flight). Dev (1,000), top move / drawn: 568 / 508 at 3.3 min, 725 / 689 at 6.7,
  807 / 779 at 10.0, 856 / 816 at 13.2, 873 / 793 at 16.5, **892 / 831 at 19.8**. c1 at 20
  minutes: 737 / 810; the c1-f1 line needed 56 minutes (208M decisions) for 848 / 878. Always
  taking the top move is now the better pilot. Open pads: 0 of 200 until 16.5 min, 90 at 19.8.
  Continued as k2 (60 minutes, the learning rate decaying to 3e-5).
- **k2 (k1 carried on, 60 minutes awake; the laptop slept through most of a night in the
  middle).** Dev (1,000), top move / drawn: 914 / 882, 947 / 907, 958 / 936, 957 / 943, 957 /
  940, 965 / 954 at 6.4, 12.8, 19.5, 30.2, 42.9, 56.0 minutes, **971 / 963 at the end**
  (125.8M decisions; 184.8M with k1's, 428 days of flight at 0.2 s a decision). Open pads
  (200 dev courses, drawn): 94 at its start, then 174, 172, 183, 192, 189, 192.
- **k2 on the test and fresh courses, flown once (top move, a move held 0.2 s): 193 and 195 of
  200** (moves drawn: 190 and 191). The gauntlet with no frog on the pad, test seeds: 189 (188
  drawn). One kind of encounter at every waypoint, top move, hits per encounter: bird 0 of
  667, two birds 5 of 666, missile 0 of 667, bird + missile 8 of 665, flock 4 of 665; frogs
  only 197 of 200 landed; no threats at all 195 (5 run out the 60 s; 200 of 200 with moves
  drawn). Test failures: 3 birds, 2 missiles, 1 frog, 1 water; fresh: 4 birds, 1 frog.

I. **Practice restarts** (written before the run): a share of new practice flights begin 2 to
   6 s before one of the fleet's own recent failures (the same drone, course and moment), the
   rest as usual. k1's top-move pilot fails the same courses every time (of 500 courses flown
   4 times, 381 always land and 22 never), so its failures are places, not dice. Two runs
   side by side from k2, 30 minutes each at half the laptop: k3a without, k3b with half of the
   new flights restarted. *Kill:* k3b not at least 1 point above k3a on 2,000 dev courses.
- **I, practice restarts (k3b against k3a): no gain, dropped.** 30 minutes each from k2, four
  processes each (59M decisions each). 2,000 dev courses, top move: k2 1,939; k3a (plain) 1,958;
  k3b (half of the new flights restarted) 1,943. Under its kill line.
- **The final pilot is k3a**, chosen on the dev courses: k1 (20 min) + k2 (60 min) + k3a (30
  min on half the laptop) = 110 minutes, 241.4M decisions, 559 days of flight. Flown once (top
  move, a move held 0.2 s): **test 194 of 200, fresh 190 of 200**; the gauntlet with no frog
  on the pad 193; no threats at all 200; frogs only 199; hits per encounter: bird 1 of 667,
  two birds 7 of 662, missile 2 of 667, bird + missile 6 of 664, flock 6 of 662. Its 16
  failures on test and fresh: 6 frogs, 6 birds, 1 missile, 2 left the course, 1 water. (k2,
  the same line 30 minutes earlier, also flown once: 193 and 195. Paid to hurry, b1, on the
  gauntlet with no frog on the pad: 0 of 200.)
- **Not reached:** 200 of 200. On the 0.2 s line the dev failures halved from 10.8% to 5.3% in
  13 minutes and again to about 2.4% in 77 more.

Every lever tried, in one list: the reward (kept), deciding every 0.2 s (kept), open-pad
courses in the mix (kept), replaying failed courses (no gain), a larger learning rate (worse),
second tries fitted to rewound moments (memorised), practice restarts from rewound failures
(no gain). No GPU was rented; the cost was nothing but the laptop's time.
