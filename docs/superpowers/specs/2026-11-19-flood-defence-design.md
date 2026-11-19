# Experiment 008: flood defence (design)

## Intent

A new experiment and post in rl-lab: a small 2D side-view world with sand and water physics.
The tide rises from one side; a house stands on the other. A digger can dig sand, carry it and
drop it. It's rewarded only for keeping the house dry, so building a dam (or anything else that
works) has to emerge. The post shows what it builds and how that changes over training, and lets
the reader watch the trained network live in the browser, or play the same game themselves.

Why: the user wants an experiment where an agent uses physics in its world to survive, i.e.
construction / tool use rather than movement alone. It should be fun to watch.

## Goals

- A numpy-only Gymnasium env (`experiments/008-flood/flood_env.py`) with sand and water
  physics, tests in CI, like 001 and 003.
- PPO (SB3, CPU) learns to keep the house dry clearly more often than doing nothing, on random
  terrain it hasn't seen.
- A post `site/src/content/posts/008-flood.mdx` in the house style: short answer first, every
  claim checked against exported JSON, numbers from 200-episode evals on held-out maps.
- In the browser: the network runs live on random maps (JS port of the env + network, parity
  test against Python, as in 003-tag), and a "play it yourself" mode with keyboard controls.

## Non-goals

- 3D, or a general falling-sand engine (no fire, no other materials).
- Evolving the body or the brain (no neuroevolution); "evolution" here means behaviour that
  emerges in training.
- Learned opponents or self-play.

## Anti-goals (would make it a failure even if it runs)

- Rewarding dam-building directly (e.g. reward for wall height). The reward is about the house
  staying dry. Any shaping must be about the outcome, and the post must say what it is.
- A world where you can win without building: the tide must top the natural terrain in front
  of the house on every map, so doing nothing always floods (checked by a test).
- Claims in the post that the data doesn't show.
- Training that needs more than a few hours per run on the M1 Pro.

## Constraints (from the repo)

- Env module imports numpy + gymnasium only; unique module name; tests in `test_flood_env.py`.
- Training on CPU (`device="cpu"`), runs in git-ignored `runs/`.
- Exported JSON in `site/public/data/008-flood/`, each file well under 1 MB.
- Links via `url()`; chart colours from the validated trio; check build, 390 px mobile, dark mode.
- JS port must match Python; parity fixture + `npm test`.
- After deploying: measure page-load numbers on the live page (as posts 004–007 did).

## Approaches considered

1. **Flood defence: dig, carry and drop sand against a rising tide** (chosen by the user).
   Medium risk: it may first find a lazy strategy, which is part of the story.
2. Jelly walker: soft-body creature learns to walk. Lowest risk, but no construction.
3. River crossing with blocks: most "tool use", but the reward only comes once the bridge is
   finished, which is hard for PPO without a curriculum. Highest risk.

## The world (my calls; correct any)

- **Grid:** 48 columns × 24 rows, side view. Cells are rendered as pixels-like blocks.
- **Sand as a height field:** each column has a sand height. Sand only ever sits on sand, so the
  world has no holes or overhangs, and a column's height describes it fully. Angle of repose:
  if a column is more than 1 cell higher than a neighbour, one grain slides down to it each
  physics tick (deterministic left-then-right order, so the JS port can match).
- **Bedrock:** a fixed bottom row the digger can't dig, so it can't dig to nothing.
- **Water as a height field too:** each column has a water depth on top of its sand. Each tick
  water flows from a column to a neighbour whose surface (sand + water) is lower, moving part of
  the difference (integer cells, deterministic order). The left edge is the sea: its level rises
  on a schedule (the tide) and it feeds water into column 0.
- **Tide:** rises from a start level to a peak over the episode, then the episode ends. The peak
  is above the house's floor, so without a barrier the house floods. Tide speed and peak are
  randomised per episode within ranges.
- **Terrain:** random per episode: a low beach on the sea side, a rise or dip in the middle, the
  house on higher ground on the right (house position also randomised a little). A test checks
  that doing nothing floods the house on every generated map.
- **House:** occupies a few columns; it floods when water reaches its floor in any of its
  columns. That ends the episode (failure). Surviving to the end of the tide is success.
- **Digger:** stands on the surface of a column (on top of sand, or on the water's floor; it can
  wade). Actions (discrete, 6): walk left, walk right, dig (take one grain from the column in
  front, if not carrying), drop (put one grain on the column in front, if carrying), turn, wait.
  It carries at most one grain (maybe 2–3; tune). It can climb a step of 1 cell, not more.
  Whether it can drown (water over its head ends the episode) is a tuning call: start without.
- **Speed:** several physics ticks per agent step if needed so the tide feels like a real
  threat. Everything is integer cells: fast in numpy and exact in JS.

## Learning setup (my calls)

- **Observation:** sand heights (48), water depths (48), digger x, y, facing, carrying, tide
  level, time left, house position: about 100 numbers. MLP policy (tanh, as in 003 so the JS
  forward pass is reused).
- **Reward:** +1 for each step the house is dry, scaled so a full survival is ~1; a penalty on
  flooding. If PPO can't find building at all, add the smallest shaping about the outcome (e.g.
  how far the water's edge is from the house), and say so in the post. No reward for sand moved.
- **Algorithm:** PPO, vectorised envs across the 10 cores, a few million steps per run, 3
  seeds for the final variant. Checkpoints at fixed fractions of training for the "how it
  changes over training" part.
- **Eval:** 200 held-out maps per checkpoint (fixed seeds), outcome = dry / flooded, plus how
  much sand it moved and where (to describe what it builds). Baselines: do-nothing, and a
  scripted builder (digs behind the house's side, builds a wall at the narrowest point) as a
  reference for "what a decent strategy scores".

## The post (shape)

- Title along the lines of "Flood defence: a digger learns to build a dam".
- Live game at the top: the network plays a random map; buttons for new map, speed, and switch
  to "you play" (arrow keys / on-screen buttons on mobile).
- Short answer bullets, then: the world and its rules, what it learned over training (replays of
  checkpoints on the same map), the numbers vs do-nothing and the scripted builder, the lazy or
  surprising strategies it found, what didn't work.

## Calls settled after the builder check (driven end-to-end by Claude, at the user's request)

- **Success / what if it never builds:** publish whatever the data shows, measured against
  do-nothing and the scripted builder. Escalation allowed, in this order, each one named in the
  post: longer training, a tide curriculum (low tides first), smallest outcome shaping (distance
  of the water's edge from the house), a smaller grid. Never a reward for sand moved or wall height.
- **House rules:** the house's columns are fixed: no digging or dropping on them, the digger
  can't walk into them, and the house sits at the right edge, so it can't be walked past. The
  floor doesn't move. A pit that soaks up the tide is allowed (a legitimate trick if it works).
  The scripted builder digs on the house side of its wall.
- **Deploying:** yes. The user asked for commit, test, post and verify, so the job ends with a
  push to `main`, a check of the live page, and the page-load numbers.

## Left to the builder

- Exact grid size, physics constants, tide ranges and episode length (tune so do-nothing
  always floods and a scripted builder usually survives).
- Carry capacity, drowning, physics ticks per step.
- PPO hyperparameters, step budget, the number of re-runs.
- Component design for the game and replays (new `FloodGame.astro`, `site/src/lib/flood-sim.js`).

## What was built (2026-11-19)

Changes from the design, all made while building:

- Water uses "virtual pipes" (flow with momentum); plain diffusion lagged ~3 cells behind the
  tide and nothing ever flooded.
- Sand slopes up to 2 cells (not 1) and the digger carries 3 grains: at 1-cell slopes even the
  scripted builder saved 19 of 200 houses.
- Maps come from a seeded mulberry32 PRNG so the browser rebuilds the same map from a seed.
- Round 1 runs found the house itself worked as a wall (sand can't slide onto it). Round 2 adds
  `house_wall`: sand against the house above its 2-cell wall spills over the roof. The post
  covers both rounds, the transfer between them, tides outside training, and sampled vs top
  action.
