// Checks the browser port against boards, games, bot moves and network outputs exported from
// Python (experiments/009-beaver/evaluate.py --parity writes test/beaver-parity.json).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  beaver,
  beaverLegal,
  dig,
  diggable,
  flow,
  netLogits,
  netRiver,
  makeBoard,
  maskedLogits,
  bestOf,
  newGame,
  observe,
  planRoute,
  riverBot,
  unpackNet,
  wardenAction,
} from "../src/lib/beaver-sim.js";

const fx = JSON.parse(
  readFileSync(new URL("./beaver-parity.json", import.meta.url), "utf8"),
);
const r = fx.rules;
const net = unpackNet(fx.net);

test("random boards match Python", () => {
  for (const b of fx.boards) {
    const js = makeBoard(b.seed, r);
    assert.deepEqual(js.tiles, b.tiles, `seed ${b.seed} tiles`);
    assert.equal(js.spring, b.spring, `seed ${b.seed} spring`);
    assert.equal(js.den, b.den, `seed ${b.seed} den`);
    assert.equal(js.start, b.start, `seed ${b.seed} start`);
  }
});

test("river options match Python", () => {
  const g = newGame(makeBoard(fx.games[0].seed, r), r);
  const legal = new Array(g.n * g.n + 1).fill(false);
  for (const i of diggable(g)) legal[i] = true;
  legal[g.n * g.n] = true;
  assert.deepEqual(legal, fx.riverLegal);
});

for (const [k, game] of fx.games.entries()) {
  test(`game ${k} matches Python (board ${game.seed}, ${k < 3 ? "network" : "scripted"} beaver)`, () => {
    const g = newGame(makeBoard(game.seed, r), r);
    game.steps.forEach((st, t) => {
      assert.equal(riverBot(g, game.dodge), st.dig, `round ${t} river bot`);
      dig(g, st.dig);
      const obs = observe(g);
      const sum = obs.reduce((a, v, i) => a + v * ((i % 7) + 1), 0);
      assert.ok(Math.abs(sum - st.obsSum) < 1e-6, `round ${t} observation`);
      assert.equal(wardenAction(g), st.warden, `round ${t} scripted beaver`);
      if (st.logits) {
        const z = netLogits(net, obs);
        z.forEach((v, i) =>
          assert.ok(
            Math.abs(v - st.logits[i]) < 1e-6,
            `round ${t} logit ${i}: ${v} vs ${st.logits[i]}`,
          ),
        );
      }
      if (k < 3)
        assert.equal(
          bestOf(maskedLogits(net, obs, beaverLegal(g))),
          st.move,
          `round ${t} network move`,
        );
      beaver(g, st.move);
      flow(g);
      assert.equal(g.tiles.join(""), st.tiles, `round ${t} tiles`);
      assert.equal(g.wet.map(Number).join(""), st.wet, `round ${t} water`);
      assert.equal(g.beaverAt, st.beaver, `round ${t} beaver`);
      assert.equal(g.logs, st.logs, `round ${t} logs`);
      assert.equal(g.done, st.done, `round ${t} done`);
      assert.equal(g.flooded, st.flooded, `round ${t} flooded`);
      assert.deepEqual(planRoute(g), st.route, `round ${t} route`);
    });
  });
}

for (const [k, game] of (fx.riverGames ?? []).entries()) {
  test(`river network game ${k} matches Python (board ${game.seed})`, () => {
    const river = unpackNet(fx.river);
    const g = newGame(makeBoard(game.seed, r), r);
    game.steps.forEach((st, t) => {
      if (st.logits) {
        const z = netLogits(river, observe(g));
        z.forEach((v, i) =>
          assert.ok(
            Math.abs(v - st.logits[i]) < 1e-6,
            `round ${t} river logit ${i}`,
          ),
        );
      }
      assert.equal(netRiver(river, g, false), st.dig, `round ${t} river move`);
      dig(g, st.dig);
      beaver(g, st.move);
      flow(g);
      assert.equal(g.tiles.join(""), st.tiles, `round ${t} tiles`);
      assert.equal(g.wet.map(Number).join(""), st.wet, `round ${t} water`);
      assert.equal(g.beaverAt, st.beaver, `round ${t} beaver`);
      assert.equal(g.done, st.done, `round ${t} done`);
    });
  });
}
