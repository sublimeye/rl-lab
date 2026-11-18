// Checks the browser port against a trajectory and network outputs exported from Python
// (experiments/003-tag/evaluate.py writes test/tag-parity.json).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { ACTIONS, body, caught, chaserDirection, logits, move, observe } from '../src/lib/tag-sim.js';

const fx = JSON.parse(readFileSync(new URL('./tag-parity.json', import.meta.url), 'utf8'));
const close = (a, b, tol, what) =>
  a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) <= tol, `${what}[${i}]: ${v} vs ${b[i]}`));

for (const ep of fx.episodes) {
  test(`simulation matches Python: ${ep.name}`, () => {
    const w = fx.world;
    const e = body(...ep.start.evader);
    const c = body(...ep.start.chaser);
    let dir = [0, 0];
    ep.actions.forEach((a, t) => {
      if (t % ep.chaser.reaction === 0) dir = chaserDirection(ep.chaser, c, e, w);
      move(e, ...ACTIONS[a], w.maxSpeed, w);
      move(c, dir[0], dir[1], w.maxSpeed * ep.chaser.ratio, w);
      const [es, cs, obs] = ep.states[t];
      close([e.x, e.y, e.vx, e.vy], es, 1e-6, `step ${t} evader`);
      close([c.x, c.y, c.vx, c.vy], cs, 1e-6, `step ${t} chaser`);
      close(observe(e, c, w), obs, 1e-6, `step ${t} obs`);
    });
    assert.equal(caught(e, c, w), ep.caught);
  });
}

test('policy network matches PyTorch', () => {
  for (const { obs, logits: want } of fx.policy.samples) {
    close(logits(fx.policy.net, obs), want, 1e-4, 'logits');
  }
});
