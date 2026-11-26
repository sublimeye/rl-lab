# rl-lab

Small reinforcement-learning experiments, trained on a laptop and replayed in the browser.

**Site:** https://sublimeye.github.io/rl-lab/

| Experiment | What it is |
| --- | --- |
| [001 Cube to flag](https://sublimeye.github.io/rl-lab/posts/001-cube-to-flag/) | A cube crosses a lava arena with PPO; sparse rewards vs a walking-distance hint |
| [002 Does more training help?](https://sublimeye.github.io/rl-lab/posts/002-does-more-training-help/) | Same cube, 10M steps × 3 seeds, 200-episode evals per checkpoint; random starts vs more steps |
| [003 Tag](https://sublimeye.github.io/rl-lab/posts/003-tag/) | A square learns to run away from scripted chasers; play against it in the browser |
| [004 Tag, part 2](https://sublimeye.github.io/rl-lab/posts/004-tag-selfplay/) | Learned chasers expose the 003 runner; self-play, a league and PSRO; randomness + pillar orbit wins |
| [005 Tag, part 3](https://sublimeye.github.io/rl-lab/posts/005-tag-maps/) | Random maps (pillars, blocks, walls, mazes) with 16 distance sensors; 40 rounds of self-play vs 5M-step exploiters |
| [006 Tag, part 4](https://sublimeye.github.io/rl-lab/posts/006-tag-seek/) | Hide and seek: a chaser that only sees along lines of sight, with a memory and a map of where you could be |
| [007 Tag, part 5](https://sublimeye.github.io/rl-lab/posts/007-tag-cone/) | A field of view: the seeker sees only in front, turns its head at a limited speed, reacts with a delay; one network for any settings |
| [008 Flood defence](https://sublimeye.github.io/rl-lab/posts/008-flood/) | Sand and water physics: a digger rewarded only for keeping a house dry learns to build a dam against the rising tide; play it in the browser |
| [009 Beaver vs river](https://sublimeye.github.io/rl-lab/posts/009-beaver/) | A turn-based board game: you dig channels for the water, a PPO beaver drops logs to keep its den dry; learned and planning rivers test it |
| [010 Drone](https://sublimeye.github.io/rl-lab/posts/010-drone/) | A 2D drone flown with two analog sticks hops between pillars: PPO vs SAC vs a hand-written autopilot, bang-bang sticks, wind and payload; fly it yourself |
| [011 Drone, part 2](https://sublimeye.github.io/rl-lab/posts/011-drone2/) | A vectorised simulator (hundreds of drones per numpy step), a 1D depth camera, birds, a moving pad, battery, sensor noise and delay; CAPS for smooth sticks |
| [012 Dodge school](https://sublimeye.github.io/rl-lab/posts/012-dodge/) | Birds, falling rocks and a homing missile; an escape test for avoidable hits; curricula (a missile that speeds up, birds aimed at the drone) that teach dodging where random threats teach nothing |
| [013 Watch, then go](https://sublimeye.github.io/rl-lab/posts/013-frogs/) | Air-frogs that strike on a rhythm; 3 s of memory lets a pilot learn to wait and land between strikes, with a slow-to-fast ramp; a crouch "tell" partly substitutes; a countdown head |
| [014 The gauntlet](https://sublimeye.github.io/rl-lab/posts/014-gauntlet/) | One pilot, a 60-120 m flight to a pad it can't see, homing birds, the missile, flocks and air-frogs; difficulty dials, drills, closest-approach inputs, and a hand-written rule that dodges better than the trained pilot |
| [015 Taught by a planner](https://sublimeye.github.io/rl-lab/posts/015-teacher/) | A look-ahead planner on copies of the simulator lands all 200 gauntlet courses; copying it into post 14's network (DAgger) and fine-tuning with PPO; what makes a teacher copyable, and what doesn't |
| [016 All 200, fairly](https://sublimeye.github.io/rl-lab/posts/016-flawless/) | A planner that knows only what a pilot could lands all 200 (and 200 new courses) with a 45° drone; a network copying it lands 1-6 alone, 400 of 400 as a shortlist for a short search; the planner and the training ported to a rented GPU |
| [017 Flying solo](https://sublimeye.github.io/rl-lab/posts/017-solo/) | Three looks before training, each finding a different fault: post 16's copy had memorised its flights; practice (PPO over the moves) was paid to hurry; and a pilot deciding 20 times a second can't commit. Fitted to "any safe move", paid only to land and deciding every 0.2 s, the network alone lands 194 of 200 test courses and 190 of 200 fresh ones (193 of 200 with no frog on the pad) after 110 minutes of practice on a laptop |

## Layout

```
experiments/NNN-slug/     one folder per experiment
  *_env.py                Gymnasium environment (numpy only, tested in CI)
  test_*.py               environment tests
  train.py                training (Stable-Baselines3), writes runs/ (git-ignored)
  export.py               rolls out checkpoints → site/public/data/NNN-slug/*.json
  evaluate.py             200-episode evals per checkpoint, report replays + start maps
  summarize.py            prints the numbers quoted in the report
  selfplay.py, psro.py    003-tag: runner vs learned chasers (posts 004)
  exploit.py              003-tag: fresh chaser against a frozen runner (the robustness test)
  maps.py                 003-tag: random maps, walk-through check, ray sensors, bot path planning (005)
  evaluate_maps.py        003-tag: scores on 200 held-out maps, exports post 005 data + parity fixture
  sight.py, seek.py       003-tag: line of sight, the possibility map, hider/search bots; seeker training (006)
  evaluate_seek.py        003-tag: scores the seekers, exports post 006 data + parity fixture
  cone.py, seek_cone.py   003-tag: field of view, head turning, reaction time; training (007)
  evaluate_cone.py        003-tag: scores the field-of-view seekers, exports post 007 data + parity fixture
experiments/008-flood/
  flood_env.py            sand + water height fields, the tide, the digger; scripted builder baseline
  train.py                PPO, 16 envs in subprocesses
  evaluate.py             200 held-out maps per checkpoint, tide tests, exports post 008 data + parity fixture
  summarize.py            prints the numbers quoted in the post
experiments/009-beaver/
  beaver_env.py           the board game: tiles, water, digs, logs, random boards, scripted river + beaver
  cnn.py                  a conv front end (tried; too slow on CPU, not used for the posted runs)
  train.py                masked PPO (sb3-contrib) for either side, against scripted or trained opponents
  evaluate.py             200 held-out boards vs scripted, learned and lookahead rivers; site data + parity fixture
  summarize.py            prints the numbers quoted in the post
experiments/010-drone/
  drone_env.py            side-view drone physics, angle-mode sticks, random courses, weather, the autopilot
  train.py                PPO (16 envs) or SAC (1 env), optional stick penalty / stick speed limit / weather
  evaluate.py             200 held-out courses per checkpoint, every weather; site data + parity fixture
  summarize.py            prints the numbers quoted in the post
experiments/011-drone2/
  fleet_env.py            the drone vectorised (N drones per numpy step) + moving pad, birds, battery, noise, delay, memory, depth camera
  train.py                PPO or PPO + CAPS on the fleet (or post 10's env, for the speed comparison)
  evaluate.py, export.py  200 held-out courses flown at once; blindfold test; site data + parity fixture
  bench.py                raw simulator speed, one-at-a-time vs vectorised
  summarize.py            prints the numbers quoted in the post
experiments/012-dodge/
  dodge_env.py            post 11's fleet + rocks, a proportional-navigation missile, arenas, urgency-sorted threat inputs, shadows on the camera
  escape.py               the escape test (was a hit avoidable?) and a scripted hover pilot
  train.py                PPO with optional warm start, gSDE, missile-speed ramp, arena difficulty dial, course mixes
  evaluate.py, export.py  test courses and arenas, blindfold and escape tests; site data + parity fixture
  summarize.py            prints the numbers quoted in the post
experiments/013-frogs/
  frog_env.py             post 12's fleet + air-frogs (rhythm, lunge, crouch, guard), frog history (memory), strike-time labels
  train.py                PPO with labels dropped from the input, optional countdown head, LSTM, frog-rhythm ramp
  evaluate.py, export.py  test courses, a rhythm change, entry timing, probes; site data + parity fixture
  summarize.py            prints the numbers quoted in the post
experiments/014-gauntlet/
  gauntlet_env.py         post 13's fleet on long courses: 40 m sight, encounters at waypoints (aimed birds, two birds, the missile, a flock), frogs, closest-approach inputs
  scripted.py             a cruise pilot that ignores threats and the escape test for its hits
  train.py                PPO with difficulty dials (presence, missile speed, bird commit, frog beat), drill mixes, a cost for time
  evaluate.py, export.py  test courses per encounter kind, frogs, blindfolds; site data + parity fixture
  diagnostics.py          drills, frog drills, a flock-only pilot, a hand-written dodge rule, a reflex
  summarize.py            prints the numbers quoted in the post
experiments/015-teacher/
  expert.py               the planner: moves (cruise, jink, land, wait, level) flown ahead on copies of post 14's simulator, deepening, a copyable band, no foreknowledge of waypoints
  dagger.py               DAgger: post 14's network copies the planner's sticks on the planner's and its own flights
  finetune.py             PPO from the taught network, kept close to it and to the planner's labels; optional dodge bonus
  publish.py              tests the pilots, records planner flights for the site, exports site data, prints the post's numbers
experiments/016-flawless/
  fair_planner.py         post 15's planner without what a pilot can't know: frog timing from strikes seen, the air and mass estimated from the drone's own motion
  pilot.py                a network that picks one of the planner's 59 moves (DAgger); alone, or as a shortlist for the planner's search (the shield)
experiments/016-gpu/
  torch_fleet.py, torch_rollout.py, gpu_planner.py   the simulator's step, the look-ahead and the fair planner in PyTorch, batched
  dagger_gpu.py, fly_gpu.py, cpu_pilots.py           post 16's DAgger and flights on a GPU; numpy tests of the pilots
  parity.py, check_planner.py, check_shield.py       checks against the numpy code
  export_016.py           recorded flights and the numbers post 16 quotes
  remote.sh               copy the repo to a rented GPU box and install
experiments/017-solo/
  diagnose.py             why post 16's copy fails, from its saved flights and networks (nothing trained)
  refit.py                refits on those flights with whole flights held out: the planner's pick, any safe move, regularisation
  sensitivity.py          the planner deciding twice, the second time from a world nudged by the sensors' noise
  solo_env.py, solo_net.py   the gauntlet flown in moves with no planner (frog-strike memory, still-air controller); the network as numpy
  train.py                PPO over the 59 moves, from a safe-move copy or from nothing (8 processes x 1024 drones, the network on the Mac's GPU)
  solo_eval.py, publish.py, summarize.py   tests, recorded flights + site data, the numbers the post quotes
site/                     Astro blog, deployed to GitHub Pages
  src/content/posts/      one .mdx post per experiment
  src/components/         Replay (canvas player), OutcomeChart, EvalChart, StartMaps, TagGame, MapGallery, SeekGame, SeekFrames, ConeGame, FloodGame, FloodReplay, FloodChart, BeaverGame, BeaverChart, DroneGame, DroneChart, StickTrace, Drone2Game, DodgeGame, FrogGame, GauntletGame
  src/lib/tag-sim.js      browser port of experiments/003-tag/tag_env.py, maps.py + the policy network
  src/lib/tag-sight.js    browser port of experiments/003-tag/sight.py (006)
  src/lib/tag-cone.js     browser port of experiments/003-tag/cone.py (007)
  src/lib/flood-sim.js    browser port of experiments/008-flood/flood_env.py; flood-draw.js draws it
  src/lib/beaver-sim.js   browser port of experiments/009-beaver/beaver_env.py + the networks; beaver-draw.js draws it
  src/lib/drone-sim.js    browser port of experiments/010-drone/drone_env.py + the pilot network; drone-draw.js draws it
  src/lib/drone2-sim.js   browser port of experiments/011-drone2/fleet_env.py (one drone) + the camera and the pilots
  src/lib/dodge-sim.js    browser port of experiments/012-dodge/dodge_env.py (one drone, its threats, camera and pilots)
  src/lib/frog-sim.js     browser port of experiments/013-frogs/frog_env.py (+ frogs, memory and the countdown head)
  src/lib/gauntlet-sim.js browser port of experiments/014-gauntlet/gauntlet_env.py (one drone, its pursuers, frogs and the pilot)
  test/                   node tests: the JS port must match trajectories and maps exported from Python
  public/data/            exported replays (committed, small JSON)
```

## Run an experiment

```sh
uv sync                                    # Python 3.12, numpy, gymnasium, SB3, torch
uv run experiments/001-cube-to-flag/train.py --variant shaped --steps 1000000
uv run experiments/001-cube-to-flag/export.py --variant shaped
cd site && npm install && npm run dev      # http://localhost:4321/rl-lab/
```

Training runs on the CPU: for small MLP policies it's faster than the Apple GPU.

## Add an experiment

1. Copy the shape of `experiments/001-cube-to-flag/` into `experiments/NNN-slug/` (give the
   env module a unique name, e.g. `rover_env.py`, so pytest doesn't mix them up).
2. Export replays in the same JSON shape (`arena`, `checkpoints[].runs[]`, `training.bins`), or
   write a new viewer component if the world isn't a 2D box.
3. Add `site/src/content/posts/NNN-slug.mdx` and embed `<Replay src="NNN-slug/x.json" />`.
4. Push to `main`. CI lints and tests the environments, builds the site and deploys it.

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request: ruff + pytest for the
environments, then `npm test` (JS/Python parity) and `astro build`. Pushes to `main` also deploy to GitHub Pages. Training never
runs in CI.
