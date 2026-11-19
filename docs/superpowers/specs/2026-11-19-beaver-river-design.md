# Experiment 009: beaver vs river (design)

## Intent

Keep post 008's story (water against a defender) but make it a small turn-based game the
reader plays: they control the water, a trained beaver protects its den. It should feel like
checkers or a simple board game, not a reflex game: every move is one choice, each game takes
a minute or two. Both sides have limited resources: the river has time (one dig per round) and
strength (how far water reaches); the beaver has a few logs and walks one tile per round.

## Rules (as built)

- 9 x 9 board, seen from above in isometric. Tiles: soil, clay, low ground (dry streambed or
  dug channel), rock, tree, log, the spring, the den.
- Round order: the river digs, the beaver acts, the water moves.
- River: dig one soil or clay tile next to water (clay takes two digs), or pass. 24 rounds.
- Water: spreads one tile per round into low ground; never more than `reach` = 16 tiles from
  the spring along open ground; water cut off from the spring drains at once.
- Beaver: step to a neighbouring tile, or drop a log on a neighbouring low tile. 4 logs. No
  dams closer than 3 tiles to the spring (fast water) or on the tiles touching the den.
- River wins when water enters the den; the beaver wins if the den is dry after round 24.

## Calls made during the build (no check-ins, per the user's standing preference)

- **No dams next to the den.** The first pilot learned to stand on its den and plug every tile
  touching it as soon as it was dug: with 4 logs and at most 4 sides it can't lose. That makes
  the game unwinnable for a person, so those tiles became off-limits; the beaver has to read
  where the river is going and cut it off further out.
- **No dams near the spring.** Otherwise 3 logs seal the spring for good.
- **The player is the river, not a creature.** Digs happen anywhere along the water's edge, so
  the choice is *where*, and spreading effort over several fronts is slower than pushing one.
- **Turn order gives the beaver the last word** each round (it sees the dig before acting);
  water moves after both, so a fresh dig can be plugged only if the beaver is next to it.
- **Plain MLP (128, 128) with masked PPO** (sb3-contrib). A small conv net was ~10x slower per
  step on the laptop CPU, so it was dropped.
- **Network weights shipped as float16** (base64) to keep the model files under 400 KB; Python
  evaluates with the same rounded weights, so the scores are the browser network's.

## Experiment plan

1. Round 1: beaver vs a mix of scripted rivers (route-following, beaver-dodging, wandering),
   3 seeds.
2. A river network trained against a frozen round-1 beaver: does a learner find holes?
3. Round 2: beaver trained against the scripted rivers plus the learned rivers.
4. Tests on 200 held-out boards: scripted rivers, the learned rivers, and a lookahead river
   that simulates the real beaver for each candidate dig (a stand-in for a careful person).

## Left to the builder

Board generator details, colours, animation timing, which network ships as the default.

## Outcome

- Round 1 (3 seeds, 20M steps): 522–530 of 600 test games dry vs the scripted rivers (scripted
  beaver 245). Logs go on flowing water 4.5–4.8 tiles from the spring, around round 13.
- A planning river floods round 1 on about 60% of boards; learned rivers up to about a third.
- Round 2 (trained with the learned river in the mix): better against that river (155–170 of
  200 vs 129–134), slightly worse vs scripted rivers (482–501 of 600), no change vs the planner.
  A fresh river finds less to exploit in it (49 floods vs 67 against round 1).
- Post: site/src/content/posts/009-beaver.mdx; numbers from experiments/009-beaver/summarize.py.
