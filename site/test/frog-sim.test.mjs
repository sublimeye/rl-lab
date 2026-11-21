// Checks the browser port against courses, flights and network outputs exported from Python
// (experiments/013-frogs/evaluate.py --parity <runs> writes test/frog-parity.json).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  SWITCH,
  TASKS,
  WEATHER,
  countdown,
  crouching,
  frogs,
  look,
  makeCourse,
  netForward,
  newDrone,
  newEyes,
  push,
  step,
  unpackNet,
} from "../src/lib/frog-sim.js";

const fx = JSON.parse(
  readFileSync(new URL("./frog-parity.json", import.meta.url), "utf8"),
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
// evaluate.condition(task, switch) with sensor noise off
const featOf = (task, sw, more = {}) => ({
  ...TASKS[task],
  noise: 0,
  switch: sw,
  ...more,
});
const STATE_KEYS = "x y vx vy th om motor charge".split(" ");

test("the fixture's state columns are the ones checked here", () => {
  assert.deepEqual(fx.state_keys, STATE_KEYS);
  assert.deepEqual(SWITCH, fx.flights[0].switch);
});

test("courses match Python", () => {
  for (const c of fx.courses) {
    const what = `${c.task} ${JSON.stringify(c.switch)} ${c.seed}`;
    const js = makeCourse(c.seed, WEATHER.rough, featOf(c.task, c.switch));
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
    assert.deepEqual(
      [js.birds.length, js.rocks.length, js.missile],
      [c.birds.length, c.rocks.length, c.missile],
      `${what} no post-12 threats`,
    );
    assert.deepEqual(js.switch, c.switch, `${what} switch`);
    assert.equal(js.frogs.length, c.frogs.length, `${what} frogs`);
    js.frogs.forEach((f, k) =>
      close(f, c.frogs[k], 1e-12, `${what} frog ${k}`),
    );
  }
});

for (const f of fx.flights) {
  test(`flight matches Python: ${f.task} network, course ${f.seed}`, () => {
    const net = unpackNet(fx.nets[f.net]);
    assert.equal(net.aux !== null, f.counts.length > 0, "countdown head");
    const feat = featOf(f.task, f.switch);
    const d = newDrone(makeCourse(f.seed, WEATHER.rough, feat), feat);
    const eyes = newEyes(d, 0);
    const looks = new Map(f.looks.map(([t, o]) => [t, o]));
    let obs = look(d, eyes);
    f.states.forEach((py, t) => {
      if (looks.has(t))
        close(obs, looks.get(t), 1e-9, `update ${t} what it saw`);
      close(netForward(net, obs), f.wants[t], 1e-9, `update ${t} network`);
      if (net.aux)
        close(countdown(net), f.counts[t], 1e-9, `update ${t} countdown`);
      step(d, f.wants[t][0], f.wants[t][1]);
      obs = push(d, eyes);
      close(d.stick, f.sticks[t], 1e-12, `update ${t} sticks`);
      const fr = frogs(d);
      close(
        [
          ...STATE_KEYS.map((k) => d[k]),
          ...fr.map((q) => q.x),
          ...fr.map((q) => q.y),
        ],
        py.slice(0, 12),
        1e-9,
        `update ${t} state`,
      );
      assert.deepEqual(d.frogN, py.slice(12, 14), `update ${t} strikes`);
      assert.equal(d.ticks, py[14], `update ${t} ticks`);
      assert.equal(d.outcome, py[15], `update ${t} outcome`);
    });
  });
}

test("frogs match Python past the rhythm switch, crouching, with the drone sitting", () => {
  const feat = featOf("tell", SWITCH, { frogs: [2, 2] });
  const drones = fx.sit.seeds.map((s) =>
    newDrone(makeCourse(s, WEATHER.rough, feat), feat),
  );
  fx.sit.frames.forEach((frame, t) =>
    drones.forEach((d, i) => {
      step(d, -1, 0);
      const py = frame[i];
      const fr = frogs(d);
      const what = `course ${fx.sit.seeds[i]} update ${t}`;
      close(
        [...fr.map((q) => q.x), ...fr.map((q) => q.y)],
        py.slice(0, 4),
        1e-9,
        `${what} frogs`,
      );
      assert.deepEqual(d.frogN, py.slice(4, 6), `${what} strikes`);
      assert.deepEqual(
        crouching(d).map(Number),
        py.slice(6, 8),
        `${what} crouching`,
      );
      assert.equal(d.outcome, py[8], `${what} outcome`);
    }),
  );
  // the fixture has to reach the switch (strike 3) and show a crouch, or this checks little
  const last = fx.sit.frames.at(-1);
  assert.ok(last.some((p) => Math.max(p[4], p[5]) > SWITCH[0] + 1));
  assert.ok(fx.sit.frames.some((fr) => fr.some((p) => p[6] || p[7])));
});

test("the site ships the networks tested here", () => {
  const shipped = ["model-memory.json", "model-countdown.json", "model-tell.json"].map((f) =>
    JSON.parse(readFileSync(new URL(`../public/data/013-frogs/${f}`, import.meta.url), "utf8")),
  );
  shipped.forEach((m, i) => {
    assert.equal(m.task, fx.nets[i].task);
    assert.deepEqual(
      m.layers.map((L) => [L.shape, L.w, L.b]),
      fx.nets[i].layers.map((L) => [L.shape, L.w, L.b]),
    );
    assert.deepEqual(m.aux ?? null, fx.nets[i].aux ?? null);
  });
});
