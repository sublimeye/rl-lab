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
- Playable experiments (003-tag) run the trained network in the browser: the JS port in
  `site/src/lib/` must match the Python env. After changing either side, re-export the parity
  fixture (`export_parity` in `experiments/003-tag/evaluate.py`, and in `evaluate_maps.py` for
  random maps, in `evaluate_seek.py` for line of sight and the possibility map) and run
  `npm test` in `site/`.
