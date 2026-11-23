// Post 15's planner flights replayed in the browser port: the recorded sticks (exported by
// experiments/015-teacher/publish.py --record) must fly each test course to the same end.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { FULL, OUTCOMES, WEATHER, makeCourse, newDrone, step } from "../src/lib/gauntlet-sim.js";

const data = JSON.parse(
  readFileSync(new URL("../public/data/015-teacher/planner-flights.json", import.meta.url), "utf8"),
);

test("recorded planner flights land in the browser port as in Python", () => {
  assert.ok(data.flights.length > 0);
  for (const fl of data.flights) {
    const buf = Buffer.from(fl.sticks, "base64");
    const st = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    const d = newDrone(makeCourse(fl.seed, WEATHER.rough, FULL), FULL);
    let t = 0;
    while (d.outcome === 0 && t < fl.updates) {
      step(d, st[2 * t], st[2 * t + 1]);
      t++;
    }
    assert.equal(OUTCOMES[d.outcome], fl.outcome, `seed ${fl.seed}`);
    assert.equal(t, fl.updates, `seed ${fl.seed}: updates`);
    assert.ok(Math.abs(d.x - fl.end[0]) < 1e-6 && Math.abs(d.y - fl.end[1]) < 1e-6, `seed ${fl.seed}: end`);
  }
});
