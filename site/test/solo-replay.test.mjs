// Post 17's recorded flights (the final pilot, the one that was paid to hurry, and the copy both
// started from; a 45-degree drone) replayed in the browser port: the sticks exported by
// experiments/017-solo/publish.py --record must fly each test course to the same end as in Python.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { FULL, OUTCOMES, PHYSICS, WEATHER, makeCourse, newDrone, step } from "../src/lib/gauntlet-sim.js";

const PH45 = { ...PHYSICS, tiltMax: (45 * Math.PI) / 180 };

for (const name of ["solo", "hurry", "copy"]) {
  const data = JSON.parse(
    readFileSync(new URL(`../public/data/017-solo/${name}-flights.json`, import.meta.url), "utf8"),
  );
  test(`recorded ${name} flights end in the browser port as in Python`, () => {
    assert.ok(data.flights.length > 0);
    for (const fl of data.flights) {
      const buf = Buffer.from(fl.sticks, "base64");
      const st = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
      const d = newDrone(makeCourse(fl.seed, WEATHER.rough, FULL, PH45), FULL, PH45);
      let t = 0;
      while (d.outcome === 0 && t < fl.updates) {
        step(d, st[2 * t], st[2 * t + 1]);
        t++;
      }
      assert.equal(OUTCOMES[d.outcome], fl.outcome, `seed ${fl.seed}`);
      assert.equal(t, fl.updates, `seed ${fl.seed}: updates`);
      assert.ok(
        Math.abs(d.x - fl.end[0]) < 1e-6 && Math.abs(d.y - fl.end[1]) < 1e-6,
        `seed ${fl.seed}: end ${d.x},${d.y} vs ${fl.end}`,
      );
    }
  });
}
