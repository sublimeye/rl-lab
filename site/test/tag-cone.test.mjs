// Field of view (post 007): checks the browser port of cone.py against episodes exported
// from Python (experiments/003-tag/evaluate_cone.py export_parity -> tag-cone-parity.json).
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { ACTIONS, body, caught, logits, NavGrid, rays } from '../src/lib/tag-sim.js';
import { Grid, hiderDirection } from '../src/lib/tag-sight.js';
import {
  ConeTracker,
  chaserStep,
  coneObserve,
  obsSize,
  speedFactor,
  turnToward,
  wrap,
} from '../src/lib/tag-cone.js';

const close = (a, b, tol, what) => {
  assert.equal(a.length, b.length, `${what}: length`);
  a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) <= tol, `${what}[${i}]: ${v} vs ${b[i]}`));
};

test('angles and speed match Python', () => {
  assert.equal(wrap(3 * Math.PI), Math.PI);
  assert.ok(Math.abs(turnToward(3, -3, 0.5) + 3) < 1e-12);
  assert.equal(speedFactor(-1, 0, 0, 0.5), 0.5);
  assert.ok(Math.abs(speedFactor(0, 1, 0, 0.5) - 0.75) < 1e-12);
  assert.equal(obsSize(), 63);
});

const FIXTURE = new URL('./tag-cone-parity.json', import.meta.url);
const fx = existsSync(FIXTURE) ? JSON.parse(readFileSync(FIXTURE, 'utf8')) : { episodes: [] };

for (const ep of fx.episodes) {
  const s = ep.senses;
  const name = `${ep.chaser} chaser vs ${ep.runner} runner, ${s.fov} degrees, on ${ep.map.family} #${ep.map.seed}`;
  test(`field of view matches Python: ${name}`, () => {
    const w = { ...fx.world, ...fx.maps[`${ep.map.seed}-${ep.map.family}`] };
    const grid = new Grid(new NavGrid(w));
    const tr = new ConeTracker(grid, w.maxSpeed, w.dt, s);
    const e = body(...ep.start.evader);
    const c = body(...ep.start.chaser);
    const hide = { target: null, arrived: false };
    tr.reset(c, e, ep.facing);
    const chaser = ep.chaser === 'net' ? { kind: 'net', net: fx.policy.net } : { kind: ep.chaser };
    const check = (t) => {
      const [es, cs, facing, seen, nPossible, wSum, obs] = ep.states[t];
      close([e.x, e.y, e.vx, e.vy], es, 1e-6, `step ${t} runner`);
      close([c.x, c.y, c.vx, c.vy], cs, 1e-6, `step ${t} chaser`);
      assert.ok(Math.abs(tr.facing - facing) < 1e-9, `step ${t} facing ${tr.facing} vs ${facing}`);
      assert.equal(Number(tr.seen), seen, `step ${t} seen`);
      assert.equal(
        tr.possible.reduce((a, b) => a + b, 0),
        nPossible,
        `step ${t} possible cells`,
      );
      assert.ok(Math.abs(tr.weightSum() - wSum) < 1e-6, `step ${t} weight`);
      close(coneObserve(c, e, tr, w, rays), obs, 1e-5, `step ${t} obs`);
    };
    check(0);
    for (let t = 1; t < ep.states.length; t++) {
      let ed;
      if (ep.runner === 'hider') ed = hiderDirection(hide, t - 1, grid, tr, e, c);
      else if (ep.runner === 'still') ed = [0, 0];
      else ed = ACTIONS[ep.runnerActions[t - 1]];
      const picked = chaserStep(chaser, ep.ratio, grid, tr, c, e, ed, w, rays);
      if (ep.chaser === 'net')
        assert.deepEqual(picked, ep.chaserActions[t - 1], `step ${t} actions`);
      check(t);
    }
    assert.equal(caught(e, c, w), ep.caught);
  });
}

test('the seeker network matches PyTorch', { skip: !fx.policy }, () => {
  for (const { obs, logits: want } of fx.policy.samples) {
    close(logits(fx.policy.net, obs), want, 1e-4, 'logits');
  }
});
