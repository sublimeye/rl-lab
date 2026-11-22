# Experiment 014: the gauntlet (one pilot, every threat, a long flight)

Asked by the user on 2026-11-21: combine posts 12 and 13 into one drone that deals with every
enemy; a larger world with more sight and time before threats arrive; 5-10x the travel time;
the pilot knows the pad's direction but not its distance, and keeps going until it finds it;
dodges that handle two threats at once (two birds, a bird and a missile) by predicting both;
reliable, ideally 180+ of 200 landings. Run end to end without check-ins.

## Design

- Course: 60-120 m from start pillar to pad (post 11: 8-20 m). Rough weather, battery, sensor
  noise and delay as before. The pad doesn't slide (post 13's frog guards a fixed pad).
- Sight 40 m: threats, frogs and the pad are only in the inputs within 40 m. The pad's offset
  reads as "40 m that way" until it's in sight, plus a direction sign and an in-sight bit.
- Encounters at waypoints every 18-28 m (test): calm, an aimed bird, two aimed birds timed to
  arrive together, the missile (ahead or behind), a bird + the missile, a crossing flock.
  Odds 0.1/0.2/0.2/0.15/0.2/0.15.
- Aimed birds home with proportional navigation (slow, agile: 4-7 m/s, 2 rad/s) on a collision
  course, then commit to a straight line 1 s before contact. The missile is post 12's
  (11 m/s, 0.45 rad/s, 4 s fuel).
- Frogs: post 13's guard frog over the pad (every test course), a second frog mid-route on
  half of them. Memory: 3 s of each frog's offset from its perch.
- Pilot: an MLP (256x256), 3 stacked frames + frog history, PPO 512 drones x 4 epochs.

## Known flaws, checked before training

- Aimed birds have to be aimed: a fixed straight line set when the bird appears is beaten
  blind by any change of speed. They home, then commit late. Checked with a scripted pilot that
  ignores threats: every one-bird and two-bird course ends in a bird hit, and the escape test
  says 100% of one-bird hits were avoidable with 0.5 s notice.
- The missile is the hardest: fixed full-stick moves avoid ~60% of its hits (post 12: same
  bound, and a trained pilot did much better). Kept at post 12's strength.
- Everything at once from the start: tried first (no dials) and it never landed (0% in 40M
  steps; the pilots hovered until time ran out, as post 13's did).
- Dials can be gamed: a flailing beginner makes early-committing birds miss by accident; the
  bird and missile dials only move once the pilot lands 25% of its flights.
- A bug caught early: every landing counted as getting past the pad's frog, frog or not.

## Recipe (training), as it ended up

- Mix: 40% long courses (15-130 m, waypoints every 12-20 m), 30% dodge drills (20-50 m, an
  encounter every 8-14 m, no frogs), 30% frog drills (15-35 m, guard always, route frog half).
- Dials: presence (long courses only; +0.1 when 60% of 1000 land), missile speed 6 -> 11 m/s,
  bird commit 2 s -> 1 s (both gated on landing 25%), frog beat 6-7 s -> 3-4 s.
- Reward: 0.3/m progress, +20 landed, -10 failed, -0.01 per update in the air.
- Closest-approach inputs (time to closest approach, miss offset) per threat slot.
- 512 drones, n_steps 32, 4 epochs, batch 4096, gamma 0.995, lr 3e-4 -> 3e-5; 280 min each.

## Measurements

- Landed of 200 test courses; per encounter kind, hits per encounter met.
- One kind at every waypoint (bird, two birds, missile, bird + missile, flock), frogs only,
  clear.
- Blindfolds: pursuers hidden, frogs hidden.

## Outcome (2026-11-21)

- Target missed: the best pilot (final-s1) lands 33 of 200 full test courses, seed 2 lands 17;
  clear 192. Failures: frog 67, bird 37, missile 26, water 20, crashed 15, lost 2.
- Frogs only (both): 127 of 200 caught. Blindfolded to threats it still lands 33 (more hits,
  fewer crashes and splashes).
- Closest-approach inputs halved single-bird and missile hit rates on drills; a near-miss
  cost and potential-based danger shaping didn't help.
- A hand-written rule on the same inputs hits 2 of 666 birds and 2 of 663 flocks (PPO: 46 of
  601, 114 of 508), so the inputs carry it; it can't handle the missile.
- A reflex override lifts flock courses 70 -> 110 but not the full gauntlet (30 vs 33).
- Next: imitation from a scripted dodger then PPO; the reflex inside training; more compute
  (GPU/JAX, with approval).
