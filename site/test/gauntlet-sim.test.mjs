// Checks the browser port against courses, flights and network outputs exported from Python
// (cd experiments/014-gauntlet && uv run evaluate.py --parity <run> writes test/gauntlet-parity.json).
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";
import {
  CLEAR,
  COMBO_TEST,
  FULL,
  K_PURSUERS,
  P_KIND,
  WEATHER,
  frogs,
  inputWidth,
  look,
  makeCourse,
  netForward,
  newDrone,
  newEyes,
  push,
  step,
  unpackNet,
} from "../src/lib/gauntlet-sim.js";

const fx = JSON.parse(readFileSync(new URL("./gauntlet-parity.json", import.meta.url), "utf8"));
let worst = {};
const close = (a, b, tol, what, tag = what) => {
  assert.equal(a.length, b.length, `${what}: length`);
  a.forEach((v, i) => {
    const err = Math.abs(v - b[i]) / Math.max(1, Math.abs(b[i]));
    worst[tag] = Math.max(worst[tag] ?? 0, err);
    assert.ok(err <= tol, `${what}[${i}]: ${v} vs ${b[i]}`);
  });
};
const TESTS = { gauntlet: FULL, clear: CLEAR, "bird + missile": COMBO_TEST };
const STATE_KEYS = "x y vx vy th om motor charge".split(" ");
// a state row: STATE_KEYS, p_x, p_y, p_on (K_PURSUERS each), frog x, y (2 each), enc_next, ticks, outcome
const K = K_PURSUERS;
const ON = STATE_KEYS.length + 2 * K;

test("the fixture's state columns are the ones checked here", () => {
  assert.deepEqual(fx.state_keys, STATE_KEYS);
});

test("courses match Python", () => {
  let encounters = 0;
  let routeFrogs = 0;
  for (const c of fx.courses) {
    const what = `${c.test} ${c.seed}`;
    const js = makeCourse(c.seed, WEATHER.rough, TESTS[c.test]);
    close(js.start, c.start, 1e-12, `${what} start`, "courses");
    close(js.target, c.target, 1e-12, `${what} target`, "courses");
    close(js.world, c.world, 1e-12, `${what} world`, "courses");
    close(
      [js.dir, js.length, js.wind, js.gust, js.period, js.phase, js.mass, js.charge, js.mSpeed, js.birdLock],
      [c.dir, c.length, c.wind, c.gust, c.period, c.phase, c.mass, c.charge, c.m_speed, c.bird_lock],
      1e-12,
      `${what} scalars`,
      "courses",
    );
    assert.deepEqual(js.switch, c.switch, `${what} switch`);
    assert.equal(js.frogs.length, c.frogs.length, `${what} frogs`);
    js.frogs.forEach((f, k) => {
      assert.equal(f === null, c.frogs[k] === null, `${what} frog ${k} present`);
      if (f) close(f, c.frogs[k], 1e-12, `${what} frog ${k}`, "courses");
    });
    routeFrogs += c.frogs[0] ? 1 : 0;
    assert.equal(js.encounters.length, c.encounters.length, `${what} waypoints`);
    js.encounters.forEach(([at, kind, p], k) => {
      const [pat, pkind, pp] = c.encounters[k];
      close([at], [pat], 1e-12, `${what} waypoint ${k} at`, "courses");
      assert.equal(kind, pkind, `${what} waypoint ${k} kind`);
      close(p, pp, 0, `${what} waypoint ${k} draws`, "courses");
      encounters += pkind ? 1 : 0;
    });
  }
  assert.ok(encounters > 20 && routeFrogs > 0, "the fixture's courses have encounters and frogs");
});

for (const f of fx.flights) {
  test(`flight matches Python: course ${f.seed}`, () => {
    const net = unpackNet(fx.net);
    assert.equal(net.predict, fx.predict);
    const feat = { ...FULL, noise: 0, predict: fx.predict };
    assert.equal(net.width, inputWidth(feat), "input width");
    const d = newDrone(makeCourse(f.seed, WEATHER.rough, feat), feat);
    const eyes = newEyes(d, 0);
    const looks = new Map(f.looks.map(([t, o]) => [t, o]));
    let obs = look(d, eyes);
    f.states.forEach((py, t) => {
      if (looks.has(t)) close(obs, looks.get(t), 1e-9, `update ${t} what it saw`, "looks");
      close(netForward(net, obs), f.wants[t], 1e-9, `update ${t} network`, "wants");
      step(d, f.wants[t][0], f.wants[t][1]);
      obs = push(d, eyes);
      const fr = frogs(d);
      close(
        [
          ...STATE_KEYS.map((k) => d[k]),
          ...d.pX,
          ...d.pY,
          ...fr.map((q) => q.x),
          ...fr.map((q) => q.y),
        ],
        [...py.slice(0, ON), ...py.slice(ON + K, ON + K + 4)],
        1e-9,
        `update ${t} state`,
        "states",
      );
      assert.deepEqual(d.pOn.map(Number), py.slice(ON, ON + K), `update ${t} pursuers on`);
      assert.equal(py.length, ON + K + 7, "state row width");
      assert.deepEqual([d.encNext, d.ticks, d.outcome], py.slice(ON + K + 4), `update ${t} waypoint, ticks, outcome`);
    });
    assert.notEqual(d.outcome, 0, "the flight ended");
  });
}

test("the fixture's flights meet pursuers", () => {
  // otherwise the encounter and pursuer code goes unchecked
  const seen = fx.flights.flatMap((f) => f.states.map((s) => s.slice(ON, ON + K)));
  assert.ok(seen.some((on) => on.some((v, k) => v && P_KIND[k] === 0)), "a bird");
  assert.ok(seen.some((on) => on.some((v, k) => v && P_KIND[k] === 1)), "a missile");
});

test("the site's gauntlet pilots decode with the right input width", (t) => {
  const dir = new URL("../public/data/014-gauntlet/", import.meta.url);
  const files = existsSync(dir) ? readdirSync(dir).filter((n) => /^model-.*\.json$/.test(n)) : [];
  if (!files.length) return t.skip("no model files yet");
  for (const name of files) {
    const m = JSON.parse(readFileSync(new URL(name, dir), "utf8"));
    assert.equal(typeof m.predict, "boolean", `${name}: predict`);
    const net = unpackNet(m);
    assert.equal(net.memory, "hist", `${name}: memory`);
    assert.equal(net.width, inputWidth({ ...FULL, predict: m.predict }), `${name}: input width`);
    net.layers.forEach((L, i) => {
      assert.equal(L.w.length, m.layers[i].shape[0], `${name} layer ${i} rows`);
      assert.ok(L.w.every((r) => r.length === m.layers[i].shape[1] && r.every(Number.isFinite)), `${name} layer ${i}`);
    });
    assert.equal(net.layers.at(-1).b.length, 2, `${name}: two sticks`);
    const out = netForward(net, new Array(net.width).fill(0));
    assert.ok(out.every((v) => Math.abs(v) <= 1), `${name}: output`);
  }
});

test.after(() => console.log("gauntlet parity, max relative error:", JSON.stringify(worst)));
