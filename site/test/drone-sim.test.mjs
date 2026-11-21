// Checks the browser port against courses, flights and network outputs exported from Python
// (experiments/010-drone/evaluate.py --parity <runs> writes test/drone-parity.json).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  PHYSICS,
  WEATHER,
  autopilot,
  makeCourse,
  netForward,
  newDrone,
  observe,
  physicsOf,
  step,
} from "../src/lib/drone-sim.js";

const fx = JSON.parse(
  readFileSync(new URL("./drone-parity.json", import.meta.url), "utf8"),
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

test("courses match Python", () => {
  for (const c of fx.courses) {
    const js = makeCourse(c.seed, WEATHER[c.weather]);
    close(
      [js.start.x0, js.start.x1, js.start.top],
      c.start,
      1e-12,
      `seed ${c.seed} start`,
    );
    close(
      [js.target.x0, js.target.x1, js.target.top],
      c.target,
      1e-12,
      `seed ${c.seed} target`,
    );
    close(
      [js.wind, js.gust, js.period, js.phase, js.mass],
      [c.wind, c.gust, c.period, c.phase, c.mass],
      1e-12,
      `seed ${c.seed} weather`,
    );
  }
});

for (const f of fx.flights) {
  const name = f.pilot === "autopilot" ? "autopilot" : `network ${f.pilot}`;
  test(`flight matches Python: ${name}, course ${f.seed}, ${f.weather}`, () => {
    const net = f.pilot === "autopilot" ? null : fx.nets[f.pilot];
    const ph = net ? physicsOf(net) : PHYSICS;
    assert.equal(ph.stickRate, f.physics.stick_rate);
    const d = newDrone(makeCourse(f.seed, WEATHER[f.weather], ph), ph);
    f.states.forEach((py, t) => {
      // the pilot's own choice must match Python's; then Python's exact choice is applied
      const want = net ? netForward(net, observe(d)) : autopilot(d);
      close(want, f.wants[t], 1e-9, `update ${t} pilot`);
      step(d, f.wants[t][0], f.wants[t][1]);
      close(d.stick, f.sticks[t], 1e-12, `update ${t} sticks`);
      close(
        [d.x, d.y, d.vx, d.vy, d.th, d.om, d.motor, d.ticks],
        py.slice(0, 8),
        1e-9,
        `update ${t} state`,
      );
      close(observe(d), py.slice(8), 1e-9, `update ${t} obs`);
    });
    assert.equal(d.outcome, f.outcome); // null: cut short in Python too
  });
}
