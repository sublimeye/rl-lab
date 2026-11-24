# Experiment 016: what it takes to land 200 of 200

Asked by the user on 2026-11-22, after post 15 (best network 22/200): reason out what is missing
and iterate until 200/200 on the gauntlet's test courses with the current threats (or 5-10
homing missiles at once); see whether the drone needs more agility. Sharp moves, turning
around, anything goes, as long as it lands.

## What the network sees vs what the planner used

- Birds and missiles: deterministic rules (proportional navigation; a bird commits to a line
  1 s before contact) from positions and velocities the network sees, plus closest-approach
  inputs. Enough in principle; a rule using the same inputs hit 25-75x fewer birds (post 14).
- Frogs: strike on a 3-4 s beat, each strike up to 0.1 s off it. The network's frog memory is
  3.05 s (16 samples, 0.2 s apart): the last strike, not the beat.
- Wind, gusts (a deterministic sine per course, period 2-5 s, up to 2.5 m/s^2 sideways) and
  mass: not given; estimable from the drone's own motion.
- Post 15's planner knew all of these exactly (frog timer, wind, mass, future gusts).

## A planner that knows only what a pilot could (fair_planner.py)

- Frogs from strikes seen (within 40 m), with the beat's range and the jitter: an interval for
  the next strike. Air and mass estimated by replaying each update in still air (mass to 0.004
  at once; the air's push to ~0.03 after a few seconds).
- 30 deg tilt (today's drone): 168/200. 15 timeouts at the world's edge past the pad, 12 frog
  hits, 3 birds, 2 lost. Air knowledge was the big loss (true frogs + nominal air: 6/10; seen
  frogs + true air: 9/10 on the first 10).
- Landing under the pad's frog: with the strike timing known only to +-0.35 s, the quiet
  window after a strike is ~0.8-1.1 s; from a safe spot 2.7 m off, a 30-deg drone needs ~1.5 s
  to dash and stop. Agility is the limit here: at 45 deg it's ~1.05 s.
- 45 deg: the 32 failures at 30 deg -> 29 land. Full 200: 180 (17 frog hits: the faster drone
  arrives at the mid-route frog having seen few strikes, and 3 sampled timings miss).
- Frog guarantee ("ghost"): in the copies a frog owns its column (2.8 m either side) while a
  leap could be under way: from the earliest possible strike start to the latest + 0.35 s, for
  the next three strikes. Plus far-side waits by the pad and a 7 m/s landing run.
- 45 deg + ghost (whole strike) + far-side waits: 193/200; leap-only ghost + 3 hypotheses for
  hold/return: same failures fixed, pad timeouts remained.
- Margins in the copies (their own flight is only as good as the air estimate): +0.25 m birds,
  +0.4 m missiles, +0.3 m frogs; copies land only within 90% of the landing limits (80% made
  tall pads unlandable). Waits 1.2/2.5 m above the pad (the frog's hold then sits above the
  way in). 198 -> 199.
- Landing controller: touchdown tilt capped at 80% of the 20-deg limit (post 15's cap was 0.6
  of the tilt limit: 27 deg at 45).
- The copyability band (post 15) compared with a cruise *along the route*; past the pad that
  points away from it and the drone hovered at the world's edge. Band off: 196 (stalls
  mid-route: coin tosses again). Band toward the pad: **200/200** (2370 s, 15,558 decisions).
- Tuned on the test seeds' failures, so validated on 200 fresh courses (30000-30199).
- Fresh courses (30000-30199): **200/200** (2361 s).
- The final planner at 30 deg: 198/200 (2882 s). One pad timeout 1.5 m short (the landing dash
  under the pad frog), one missile. The extra agility is worth 2 courses, not the result.

## Can a network do it? (pilot.py)

Post 15's DAgger copy (sticks, regression, privileged teacher) landed 1-2/200, and none of the
threat-free courses: averaging the teacher's discrete choices breaks even plain flying.
- The network picks a *move* (one of 59, flown by post 15's controller), as a classifier.
- Inputs: post 14's 318 + 32: the fair planner's beliefs (air push now/+0.5/+1 s, wind, gust,
  mass, period; per frog: seen, offset, the next two strike windows, strikes seen, in its
  zone) + the move it made last.
- A second head: for each move, does every plan starting with it fail (the planner's scores).
- Pilots: net (top move every update); shield (the planner's search limited to the network's
  top k moves, full search only when all fail); the planner.
- Shield k=5 on the smoke test: ~1/4.7 of the planner's CPU per flight-second.
