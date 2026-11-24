# rl-lab: notes for Claude Code

- Read README.md for layout and commands. Python via `uv`, site via `npm` in `site/`.
- Training happens locally (CPU, `device="cpu"`); CI only lints/tests envs and builds the site.
- Env modules must import only numpy + gymnasium (CI installs no torch). Give each a unique
  module name (`cube_env.py`, `rover_env.py`, ...).
- Exported replay JSON goes to `site/public/data/<experiment>/`; keep each file well under 1 MB.
- Posts: `site/src/content/posts/NNN-slug.mdx`. Every claim about agent behaviour must match the
  exported data; check the JSON before writing it.
- Site design tokens live in `site/src/styles/global.css` (light + dark). Chart colours use the
  validated flag/lava/timeout trio; don't add series colours without re-validating.
- Before pushing a site change: `npm run build` in `site/`, preview it, and check desktop, 390px
  mobile and dark mode.
- Base path is `/rl-lab`: build links with `url()` from `site/src/lib.ts`, never hard-code `/`.
- Playable experiments (003-tag, 008-flood, 009-beaver, 010-drone, 011-drone2, 012-dodge, 013-frogs, 014-gauntlet, 015-teacher) run the trained network in the browser: the JS port in
  `site/src/lib/` must match the Python env. After changing either side, re-export the parity
  fixture (`export_parity` in `experiments/003-tag/evaluate.py`, and in `evaluate_maps.py` for
  random maps, in `evaluate_seek.py` for line of sight and the possibility map, in `evaluate_cone.py` for
  the field of view; for 008 it's `export_parity` in `experiments/008-flood/evaluate.py` and
  `site/src/lib/flood-sim.js`; for 009 `evaluate.py --parity <beaver run> --parity-river <ckpt>`
  in `experiments/009-beaver/` and `site/src/lib/beaver-sim.js`; for 010 `evaluate.py --parity
  <runs...>` in `experiments/010-drone/` and `site/src/lib/drone-sim.js`; for 011 `evaluate.py --parity
  <runs...>` in `experiments/011-drone2/` and `site/src/lib/drone2-sim.js`; for 012 `evaluate.py
  --parity <state run> <camera run>` in `experiments/012-dodge/` and `site/src/lib/dodge-sim.js`; for 013 `evaluate.py --parity <memory run>
  <countdown run> <tell run>` in `experiments/013-frogs/` and `site/src/lib/frog-sim.js`; for 014 `evaluate.py --parity <run>` in `experiments/014-gauntlet/`
  and `site/src/lib/gauntlet-sim.js`; 015 reuses 014's env and port: after changing either, re-record the planner's
  flights with `publish.py --record 24` in `experiments/015-teacher/`, checked by `test/teacher-replay.test.mjs`; 016 replays
  recorded flights through the same port with a 45-degree tilt: re-record with `export_016.py --record 24` in
  `experiments/016-gpu/`, checked by `test/flawless-replay.test.mjs`) and run `npm test` in `site/`.
