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
site/                     Astro blog, deployed to GitHub Pages
  src/content/posts/      one .mdx post per experiment
  src/components/         Replay (canvas player), OutcomeChart, EvalChart, StartMaps, TagGame, MapGallery, SeekGame, SeekFrames, ConeGame, FloodGame, FloodReplay, FloodChart
  src/lib/tag-sim.js      browser port of experiments/003-tag/tag_env.py, maps.py + the policy network
  src/lib/tag-sight.js    browser port of experiments/003-tag/sight.py (006)
  src/lib/tag-cone.js     browser port of experiments/003-tag/cone.py (007)
  src/lib/flood-sim.js    browser port of experiments/008-flood/flood_env.py; flood-draw.js draws it
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
