// Random maps (post 005): checks the browser port against data exported from Python
// (experiments/003-tag/evaluate_maps.py export_parity writes test/tag-maps-parity.json).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import {
  ACTIONS,
  body,
  caught,
  chaserDirection,
  generateMap,
  logits,
  move,
  mulberry32,
  NavGrid,
  navigable,
  observe,
  rays,
} from '../src/lib/tag-sim.js';

const fx = JSON.parse(readFileSync(new URL('./tag-maps-parity.json', import.meta.url), 'utf8'));
const close = (a, b, tol, what) => {
  assert.equal(a.length, b.length, `${what}: length`);
  a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) <= tol, `${what}[${i}]: ${v} vs ${b[i]}`));
};
const world = (m) => ({ ...fx.world, ...generateMap(m.seed, m.family) });

test('random numbers match Python', () => {
  for (const [seed, want] of Object.entries(fx.prng)) {
    const r = mulberry32(Number(seed));
    close(
      want.map(() => r()),
      want,
      0,
      `seed ${seed}`,
    );
  }
});

test('the same seed builds the same map', () => {
  for (const m of fx.maps) {
    const got = generateMap(m.seed, m.family);
    assert.equal(got.pillars.length, m.pillars.length, `${m.family} ${m.seed} pillars`);
    assert.equal(got.blocks.length, m.blocks.length, `${m.family} ${m.seed} blocks`);
    got.pillars.forEach((p, i) => close(p, m.pillars[i], 1e-12, `${m.family} ${m.seed} pillar`));
    got.blocks.forEach((b, i) => close(b, m.blocks[i], 1e-12, `${m.family} ${m.seed} block`));
  }
});

test('every generated map can be walked through', () => {
  for (const fam of ['pillars', 'blocks', 'walls', 'maze', 'mixed']) {
    for (let seed = 1000; seed < 1060; seed++) assert.ok(navigable(generateMap(seed, fam)));
  }
  // and the check does catch a closed room
  const room = {
    width: 12,
    height: 8,
    pillars: [],
    blocks: [
      [4, 2, 8, 2.3],
      [4, 5.7, 8, 6],
      [4, 2, 4.3, 6],
      [7.7, 2, 8, 6],
    ],
  };
  assert.ok(!navigable(room));
});

test('ray sensors match Python', () => {
  const w = world(fx.rayProbe.map);
  for (const c of fx.rayProbe.cases) close(rays(c.x, c.y, w, fx.rays), c.rays, 1e-9, 'rays');
});

test('bot path planning matches Python', () => {
  const w = world(fx.rayProbe.map);
  const nav = new NavGrid(w);
  for (const c of fx.nav) close(nav.direction(...c.from, ...c.to, 0.36), c.dir, 1e-9, 'dir');
});

for (const ep of fx.episodes) {
  test(`simulation on a random map matches Python: ${ep.name}`, () => {
    const w = world(ep.map);
    const nav = new NavGrid(w);
    const e = body(...ep.start.evader);
    const c = body(...ep.start.chaser);
    let dir = [0, 0];
    ep.actions.forEach((a, t) => {
      if (t % ep.chaser.reaction === 0) dir = chaserDirection(ep.chaser, c, e, w, nav);
      move(e, ...ACTIONS[a], w.maxSpeed, w);
      move(c, dir[0], dir[1], w.maxSpeed * ep.chaser.ratio, w);
      const [es, cs, obs] = ep.states[t];
      close([e.x, e.y, e.vx, e.vy], es, 1e-6, `step ${t} evader`);
      close([c.x, c.y, c.vx, c.vy], cs, 1e-6, `step ${t} chaser`);
      close(observe(e, c, w, fx.rays), obs, 1e-5, `step ${t} obs`);
    });
    assert.equal(caught(e, c, w), ep.caught);
  });
}

test('the map runner network matches PyTorch', () => {
  for (const { obs, logits: want } of fx.policy.samples) {
    close(logits(fx.policy.net, obs), want, 1e-4, 'logits');
  }
});
