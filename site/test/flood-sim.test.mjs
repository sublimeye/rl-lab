// Checks the browser port against maps, trajectories and network outputs exported from Python
// (experiments/008-flood/evaluate.py writes test/flood-parity.json).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { logits, makeMap, newState, observe, step } from '../src/lib/flood-sim.js';

const fx = JSON.parse(readFileSync(new URL('./flood-parity.json', import.meta.url), 'utf8'));
const w = fx.world;
const close = (a, b, tol, what) => {
  assert.equal(a.length, b.length, `${what}: length`);
  a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) <= tol, `${what}[${i}]: ${v} vs ${b[i]}`));
};

test('random maps match Python', () => {
  for (const m of fx.maps) {
    const js = makeMap(m.seed, w);
    assert.deepEqual(js.sand, m.sand, `seed ${m.seed} sand`);
    assert.equal(js.houseFloor, m.houseFloor, `seed ${m.seed} floor`);
    assert.equal(js.peak, m.peak, `seed ${m.seed} peak`);
    assert.equal(js.startX, m.startX, `seed ${m.seed} start`);
  }
});

for (const ep of fx.episodes) {
  test(`simulation matches Python: ${ep.name}`, () => {
    const s = newState(makeMap(ep.seed, w), w);
    const byT = new Map(ep.states.map((st) => [st.t, st]));
    ep.actions.forEach((a) => {
      const r = step(s, a);
      const t = s.t;
      const py = byT.get(t);
      if (!py) return;
      assert.deepEqual(Array.from(s.sand), py.sand, `step ${t} sand`);
      close(Array.from(s.water), py.water, 1e-9, `step ${t} water`);
      assert.equal(s.x, py.x, `step ${t} x`);
      assert.equal(s.carrying, py.carrying, `step ${t} carrying`);
      assert.equal(s.spilled, py.spilled, `step ${t} spilled`);
      assert.ok(Math.abs(r - py.reward) < 1e-12, `step ${t} reward`);
      close(observe(s), py.obs, 1e-9, `step ${t} obs`);
    });
    assert.ok(s.done, 'episode ended');
    assert.ok(byT.has(s.t), 'last state checked');
    assert.equal(s.outcome, ep.outcome);
  });
}

test('network forward pass matches PyTorch', () => {
  for (const { obs, logits: z } of fx.policy.samples) close(logits(fx.policy.net, obs), z, 1e-4, 'logits');
});
