# rl-lab

Small reinforcement-learning experiments, trained on a laptop and replayed in the browser.

**Site:** https://sublimeye.github.io/rl-lab/

| Experiment | What it is |
| --- | --- |
| [001 Cube to flag](https://sublimeye.github.io/rl-lab/posts/001-cube-to-flag/) | A cube crosses a lava arena with PPO; sparse rewards vs a walking-distance hint |

## Layout

```
experiments/NNN-slug/     one folder per experiment
  *_env.py                Gymnasium environment (numpy only, tested in CI)
  test_*.py               environment tests
  train.py                training (Stable-Baselines3), writes runs/ (git-ignored)
  export.py               rolls out checkpoints → site/public/data/NNN-slug/*.json
site/                     Astro blog, deployed to GitHub Pages
  src/content/posts/      one .mdx post per experiment
  src/components/         Replay (canvas player), OutcomeChart (training chart)
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
environments, then `astro build`. Pushes to `main` also deploy to GitHub Pages. Training never
runs in CI.
