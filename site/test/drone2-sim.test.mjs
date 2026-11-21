// Checks the browser port against courses, flights and network outputs exported from Python
// (experiments/011-drone2/evaluate.py --parity <runs> writes test/drone2-parity.json).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  FULL,
  WEATHER,
  look,
  makeCourse,
  netForward,
  newDrone,
  newEyes,
  push,
  step,
  unpackNet,
} from "../src/lib/drone2-sim.js";

const fx = JSON.parse(
  readFileSync(new URL("./drone2-parity.json", import.meta.url), "utf8"),
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
const featOf = (task) => ({ ...FULL, noise: 0, obs: task });

test("courses match Python", () => {
  for (const c of fx.courses) {
    const js = makeCourse(c.seed, WEATHER.rough, featOf(c.task));
    close(js.start, c.start, 1e-12, `${c.seed} start`);
    close(js.target, c.target, 1e-12, `${c.seed} target`);
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
      `${c.seed} weather, pad, battery`,
    );
    assert.equal(js.birds.length, c.birds.length, `${c.seed} birds`);
    js.birds.forEach((b, k) =>
      close(b, c.birds[k], 1e-12, `${c.seed} bird ${k}`),
    );
  }
});

for (const f of fx.flights) {
  test(`flight matches Python: ${f.task} network, course ${f.seed}`, () => {
    const net = unpackNet(fx.nets[f.net]);
    const feat = featOf(f.task);
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
        [d.x, d.y, d.vx, d.vy, d.th, d.om, d.motor, d.charge, d.padX0],
        py.slice(0, 9),
        1e-9,
        `update ${t} state`,
      );
      assert.equal(d.ticks, py[9], `update ${t} ticks`);
      assert.equal(d.outcome, py[10], `update ${t} outcome`);
    });
  });
}

test("the site ships the networks tested here", () => {
  const shipped = ["model-state.json", "model-vcaps.json"].map((f) =>
    JSON.parse(
      readFileSync(
        new URL(`../public/data/011-drone2/${f}`, import.meta.url),
        "utf8",
      ),
    ),
  );
  shipped.forEach((m, i) => {
    assert.equal(m.task, fx.nets[i].task);
    assert.deepEqual(
      m.layers.map((L) => [L.shape, L.w, L.b]),
      fx.nets[i].layers.map((L) => [L.shape, L.w, L.b]),
    );
  });
});
