# Experiment 015: a pilot taught by a planner

Asked by the user on 2026-11-21: build on 014 (same courses, threats, test seeds 20000-20199,
inputs). `experiments/014-gauntlet/planner.py` (a look-ahead planner on copies of the simulator)
landed 150/200 with 2 threat hits and 48 timeouts from stalling. Steps:
1. make the planner land >= 199/200: look further ahead, sharp dodge moves, fix the landing
   under the pad's frog;
2. teach post 14's network (MLP 256x256, same inputs) to copy it with DAgger;
3. fine-tune with PPO from the taught network, kept close to the teacher; optionally a reward
   for clean dodges.
Target 198/200 on the full test courses. CPU only (ask before JAX/GPU). Drive end to end, then
post 15.

## Why the 014 planner stalls (diagnosed before changing anything)

- 18 of its 48 timeouts hover beside the pad under the frog; the rest hover before a waypoint.
- Its moves are slow (vertical speed capped at 2.5 m/s), so the missile can't be dodged; every
  plan that crosses a missile waypoint dies inside the 3.5 s look-ahead, while one that crosses
  at the very end of it doesn't see the missile arrive. So it puts the crossing off forever (the
  horizon effect).

## The new planner (expert.py)

- Moves: cruise (height, speed), jink (full-deflection sticks), land, wait beside the pad.
- Candidates: two moves; a jink after a delay (0-2.6 s); wait then land (near the pad); and what
  is left of the last plan (deterministic world: the choice never gets worse).
- Score: landing 1000 - time; failing -1000; otherwise progress (half final, half mean over the
  horizon), a cost for passing within 1.5 m of a threat, a cost per jink.
- The horizon effect, fixed three ways: a charge for every waypoint (and the mid-route frog)
  still ahead, so putting one off saves nothing; deepening: the best few copies that end with a
  bird or missile still after them get a 3 s dodge search, and those that end near the pad a
  3 s wait-then-land search; ending inside a frog's reach unlanded costs as an open threat.
- Controllers know the wind, mass and charge (feed-forward): privileged, as is the look-ahead.
- Event deepening: crossings of a waypoint are replayed to the moment the threat appears and a
  dodge search starts there, so an early crossing gets as deep a search as a late one (a late
  crossing used to look safer, and the drone sat on the start pillar).
- Planner: 150/200 (014's) -> 192 -> 194 -> 198 -> 200/200 on the test courses.

## Teaching (dagger.py)

- Behaviour cloning on 400 planner flights (120k states) lands 1-2/200; DAgger rounds 1-3 (beta
  0.5^r mixing) only reach 5/200, held-out loss rising as the network's own states come in.
- Diagnosis: a third of the labels are full throttle and the planner flips between plans that
  score alike, so the labels are partly a coin toss. Fix, measured on 60 courses + 20 unseen:
  among plans within `band` of the best, take the one whose sticks are closest to a plain
  cruise's. Unseen-course error 1.17 (band 0) -> 0.87 (4) -> 0.67 (8) -> 0.53 (16) -> 0.57 (30),
  planner still 80/80. Band 16 is the teacher.
- Loss: squared error, but a full-deflection label only penalises falling short of it.
- DAgger with the band-16 teacher: rounds 0-4 land 3, 0, 1, 0, 1 of 200; the network's own
  flights end at the first threat. Cause: the planner foresees waypoints (where, which threat)
  and its best missile escapes start before the missile exists; nothing in the network's
  inputs says a waypoint is coming. Fair teacher: the copies' waypoints never fire (only threats
  already flying are foreseen). It still lands 200/200 (512 s vs 722 s), unseen-course error
  0.49 vs 0.53. DAgger restarted with it.
- DAgger with the fair teacher (dagger-s1): rounds 0-3 land 2, 2, 2, 1 of 200. The copies are
  hit by 29-39% of single birds (post 14: 8%); the planner's dodges are late and exact, and an
  aimed bird's commit moment isn't in the inputs. Weighting dangerous states (x10, x30) doesn't
  help; fitting post 14's pilot to the planner wrecks it (33 -> 0-3).

## Fine-tuning (finetune.py)

- Anchored (stay loss to the frozen copy + DAPG-style BC on the DAgger data), full courses:
  0-3/200 to 75M steps, with or without a dodge bonus. Unanchored on full courses: 0 from 30M.
- Anchored on post 14's school (drills + dials): dials never move, 0-4/200 to 70M.
- Unanchored on post 14's school (ft-school-free, ft-school-free-dodge), 284 min of compute
  (the laptop slept 27 min; resumed with --resume to make it up): 1 -> 0 (10-30M) -> 23 by
  110 min -> 12-26 -> 22 at the end; dodge bonus ends at 17. Post 14 from scratch: 33 and 17
  at 280 min. The taught start gives no head start and the same ceiling.
- Target 198/200 not met: the best network lands 22.
