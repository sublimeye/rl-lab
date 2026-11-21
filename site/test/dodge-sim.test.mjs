// Checks the browser port against courses, flights and network outputs exported from Python
// (experiments/012-dodge/evaluate.py --parity <runs> writes test/dodge-parity.json).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  FULL,
  THREATS,
  WEATHER,
  look,
  makeCourse,
  netForward,
  newDrone,
  newEyes,
  push,
  step,
  unpackNet,
} from "../src/lib/dodge-sim.js";

const fx = JSON.parse(
  readFileSync(new URL("./dodge-parity.json", import.meta.url), "utf8"),
);
const close = (a, b, tol, what) => {
  assert.equal(a.length, b.length, `${what}: length`);
  a.forEach((v, i) =>
    assert.ok(
      Math.abs(v - b[i]) <= tol * Math.max(1, Math.abs(b[i])),
      `${what}[${i}]: ${v} vs ${b[i]}`,
    ),
  );
};
// export.school(obs): FULL or FULL_VISION with the dodge-school threats, sensor noise off
const featOf = (obs) => ({ ...FULL, noise: 0, obs, threats: THREATS.school });
const STATE_KEYS = "x y vx vy th om motor charge padX0 mX mY mPsi".split(" ");

test("the fixture's state columns are the ones checked here", () => {
  assert.deepEqual(
    fx.state_keys,
    "x y vx vy th om motor charge pad_x0 m_x m_y m_psi".split(" "),
  );
});

test("courses match Python", () => {
  for (const c of fx.courses) {
    const what = `${c.obs} ${c.seed}`;
    const js = makeCourse(c.seed, WEATHER.rough, { ...featOf(c.obs), threats: THREATS[c.threats ?? "school"] });
    close(js.start, c.start, 1e-12, `${what} start`);
    close(js.target, c.target, 1e-12, `${what} target`);
    close(
      [
        js.wind,
        js.gust,
        js.period,
        js.phase,
        js.mass,
        js.amp,
        js.padPeriod,
        js.padPhase,
        js.charge,
      ],
      [
        c.wind,
        c.gust,
        c.period,
        c.phase,
        c.mass,
        c.amp,
        c.pad_period,
        c.pad_phase,
        c.charge,
      ],
      1e-12,
      `${what} weather, pad, battery`,
    );
    assert.equal(js.arena, c.arena, `${what} arena`);
    close(js.goal, c.goal, 1e-12, `${what} goal`);
    assert.equal(js.birds.length, c.birds.length, `${what} birds`);
    js.birds.forEach((b, k) =>
      close(b, c.birds[k], 1e-12, `${what} bird ${k}`),
    );
    assert.equal(js.rocks.length, c.rocks.length, `${what} rocks`);
    js.rocks.forEach((r, k) =>
      close(r, c.rocks[k], 1e-12, `${what} rock ${k}`),
    );
    assert.equal(js.missile === null, c.missile === null, `${what} missile`);
    if (c.missile) close(js.missile, c.missile, 1e-12, `${what} missile`);
  }
});

for (const f of fx.flights) {
  test(`flight matches Python: ${f.obs} network, course ${f.seed}`, () => {
    const net = unpackNet(fx.nets[f.net]);
    const feat = featOf(f.obs);
    const d = newDrone(makeCourse(f.seed, WEATHER.rough, feat), feat);
    const eyes = newEyes(d, 0);
    const looks = new Map(f.looks.map(([t, o]) => [t, o]));
    let obs = look(d, eyes);
    f.states.forEach((py, t) => {
      if (looks.has(t))
        close(obs, looks.get(t), 1e-9, `update ${t} what it saw`);
      close(netForward(net, obs), f.wants[t], 1e-9, `update ${t} network`);
      step(d, f.wants[t][0], f.wants[t][1]);
      obs = push(d, eyes);
      close(d.stick, f.sticks[t], 1e-12, `update ${t} sticks`);
      close(
        [...STATE_KEYS.map((k) => d[k]), ...d.rockX],
        py.slice(0, 14),
        1e-9,
        `update ${t} state`,
      );
      assert.deepEqual(d.rockN, py.slice(14, 16), `update ${t} rocks dropped`);
      assert.equal(d.mState, py[16], `update ${t} missile state`);
      assert.equal(d.ticks, py[17], `update ${t} ticks`);
      assert.equal(d.outcome, py[18], `update ${t} outcome`);
    });
  });
}

test("the site ships the networks tested here", () => {
  const shipped = ["model-school.json", "model-camera.json"].map((f) =>
    JSON.parse(
      readFileSync(new URL(`../public/data/012-dodge/${f}`, import.meta.url), "utf8"),
    ),
  );
  shipped.forEach((m, i) => {
    assert.equal(m.obs, fx.nets[i].obs);
    assert.deepEqual(
      m.layers.map((L) => [L.shape, L.w, L.b]),
      fx.nets[i].layers.map((L) => [L.shape, L.w, L.b]),
    );
  });
});
