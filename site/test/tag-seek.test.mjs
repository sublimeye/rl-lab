// Hide and seek (post 006): checks the browser port of sight.py against episodes exported
// from Python (experiments/003-tag/evaluate_seek.py export_parity -> tag-seek-parity.json).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { ACTIONS, body, caught, logits, move, NavGrid, rays } from '../src/lib/tag-sim.js';
import {
  Grid,
  hiderDirection,
  obsSize,
  seekbotDirection,
  seekObserve,
  sees,
  Tracker,
} from '../src/lib/tag-sight.js';

const fx = JSON.parse(readFileSync(new URL('./tag-seek-parity.json', import.meta.url), 'utf8'));
const close = (a, b, tol, what) => {
  assert.equal(a.length, b.length, `${what}: length`);
  a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) <= tol, `${what}[${i}]: ${v} vs ${b[i]}`));
};

test('line of sight is blocked by obstacles only', () => {
  const w = { width: 12, height: 8, pillars: [[6, 4, 1]], blocks: [[2, 1, 2.3, 7]] };
  assert.ok(!sees(4, 4, 8, 4, w)); // through the pillar
  assert.ok(sees(4, 6, 8, 6, w)); // above it
  assert.ok(!sees(1, 4, 3, 4, w)); // through the wall
  assert.ok(sees(1, 7.5, 3, 7.5, w)); // past its end
});

test('input sizes match Python', () => {
  assert.deepEqual(
    ['xray', 'none', 'last', 'map'].map((s) => obsSize(s)),
    [26, 26, 32, 49],
  );
});

for (const ep of fx.episodes) {
  const name = `${ep.chaser} chaser vs ${ep.runner} runner on ${ep.map.family} #${ep.map.seed}`;
  test(`hide and seek matches Python: ${name}`, () => {
    const w = { ...fx.world, ...fx.maps[`${ep.map.seed}-${ep.map.family}`] };
    const grid = new Grid(new NavGrid(w));
    const tr = new Tracker(grid, w.maxSpeed, w.dt);
    const e = body(...ep.start.evader);
    const c = body(...ep.start.chaser);
    const hide = { target: null, arrived: false };
    tr.reset(c, e);
    const check = (t) => {
      const [es, cs, seen, nPossible, wSum, obs] = ep.states[t];
      close([e.x, e.y, e.vx, e.vy], es, 1e-6, `step ${t} runner`);
      close([c.x, c.y, c.vx, c.vy], cs, 1e-6, `step ${t} chaser`);
      assert.equal(Number(tr.seen), seen, `step ${t} seen`);
      assert.equal(
        tr.possible.reduce((a, b) => a + b, 0),
        nPossible,
        `step ${t} possible cells`,
      );
      assert.ok(Math.abs(tr.weightSum() - wSum) < 1e-6, `step ${t} weight`);
      close(seekObserve(c, e, tr, 'map', w, rays), obs, 1e-5, `step ${t} obs`);
    };
    check(0);
    for (let t = 1; t < ep.states.length; t++) {
      let ed;
      if (ep.runner === 'hider') ed = hiderDirection(hide, t - 1, grid, tr, e, c);
      else if (ep.runner === 'still') ed = [0, 0];
      else ed = ACTIONS[ep.runnerActions[t - 1]];
      const cd =
        ep.chaser === 'net'
          ? ACTIONS[ep.chaserActions[t - 1]]
          : seekbotDirection(ep.chaser, ep.ratio, grid, tr, c, e, w);
      move(e, ed[0], ed[1], w.maxSpeed, w);
      move(c, cd[0], cd[1], w.maxSpeed * ep.ratio, w);
      tr.update(c, e);
      check(t);
    }
    assert.equal(caught(e, c, w), ep.caught);
  });
}

test('the seeker network matches PyTorch', () => {
  for (const { obs, logits: want } of fx.policy.samples) {
    close(logits(fx.policy.net, obs), want, 1e-4, 'logits');
  }
});
