// Browser port of experiments/009-beaver/beaver_env.py. Keep them in step:
// site/test/beaver-sim.test.mjs replays games exported from Python and checks every board,
// every observation, the bots' moves and the network's outputs.

import { logits } from "./tag-sim.js";

export const [ROCK, TREE, SOIL, CLAY, LOW, LOG, SPRING, DEN] = [
  0, 1, 2, 3, 4, 5, 6, 7,
];
export const B_ACTIONS = [
  "stay",
  "up",
  "right",
  "down",
  "left",
  "log-up",
  "log-right",
  "log-down",
  "log-left",
];
const DIRS = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];
const INF = 1e6;
const WATER_OPEN = new Set([LOW, SPRING, DEN]);
const DIGGABLE = new Set([SOIL, CLAY]);
const BLOCKED = new Set([ROCK, TREE]);

/** Defaults of beaver_env.Rules; the exported JSON carries the real values. */
export const RULES = { size: 9, rounds: 24, logs: 4, reach: 16, calm: 3 };

/** Same sequence as beaver_env.mulberry32: floats in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const NB = new Map();
export function nbrs(n) {
  if (!NB.has(n)) {
    const out = [];
    for (let i = 0; i < n * n; i++) {
      const x = i % n;
      const y = (i / n) | 0;
      const row = [];
      for (const [dx, dy] of DIRS)
        if (x + dx >= 0 && x + dx < n && y + dy >= 0 && y + dy < n)
          row.push((y + dy) * n + x + dx);
      out.push(row);
    }
    NB.set(n, out);
  }
  return NB.get(n);
}

export function stepTo(i, d, n) {
  const x = (i % n) + DIRS[d][0];
  const y = ((i / n) | 0) + DIRS[d][1];
  return x >= 0 && x < n && y >= 0 && y < n ? y * n + x : -1;
}

export function distances(tiles, n, src, open) {
  const d = new Array(n * n).fill(INF);
  d[src] = 0;
  const q = [src];
  const nb = nbrs(n);
  for (let h = 0; h < q.length; h++) {
    const i = q[h];
    for (const j of nb[i]) {
      if (d[j] === INF && open.has(tiles[j])) {
        d[j] = d[i] + 1;
        q.push(j);
      }
    }
  }
  return d;
}

/** Random map from a seed (beaver_env.make_board). */
export function makeBoard(seed, r = RULES) {
  const n = r.size;
  const rand = mulberry32(seed);
  const ri = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  const nb = nbrs(n);
  for (let attempt = 0; attempt < 100; attempt++) {
    const t = new Array(n * n).fill(SOIL);
    const spring = ri(0, 3);
    const den = ri(n - 4, n - 2) + ri(n - 4, n - 2) * n;
    for (let k = ri(2, 4); k > 0; k--) {
      const cx = ri(0, n - 1);
      const cy = ri(0, n - 1);
      const rad = ri(1, 2);
      for (let i = 0; i < n * n; i++)
        if (
          Math.abs((i % n) - cx) + Math.abs(((i / n) | 0) - cy) <= rad &&
          rand() < 0.75
        )
          t[i] = CLAY;
    }
    for (let k = ri(5, 8); k > 0; k--) {
      let i = ri(0, n * n - 1);
      const kind = rand() < 0.5 ? ROCK : TREE;
      for (let m = ri(1, 3); m > 0; m--) {
        t[i] = kind;
        const j = stepTo(i, ri(0, 3), n);
        i = j >= 0 ? j : i;
      }
    }
    let i = spring;
    for (let k = ri(3, 6); k > 0; k--) {
      const x = i % n;
      const y = (i / n) | 0;
      const dx = (den % n) - x;
      const dy = ((den / n) | 0) - y;
      let d;
      if (rand() < 0.5 && dy !== 0) d = dy > 0 ? 2 : 0;
      else if (dx !== 0) d = dx > 0 ? 1 : 3;
      else d = ri(0, 3);
      if (rand() < 0.25) d = ri(0, 3);
      const j = stepTo(i, d, n);
      if (j >= 0 && j !== den) {
        i = j;
        t[i] = LOW;
      }
    }
    for (let k = ri(1, 2); k > 0; k--) {
      let i = ri(n, n * n - n - 1);
      let d = ri(0, 3);
      for (let m = ri(3, 5); m > 0; m--) {
        if (i !== den && i !== spring) t[i] = LOW;
        if (rand() < 0.3) d = ri(0, 3);
        const j = stepTo(i, d, n);
        i = j >= 0 ? j : i;
      }
    }
    t[spring] = SPRING;
    t[den] = DEN;
    for (const j of nb[spring]) if (BLOCKED.has(t[j])) t[j] = SOIL;
    const openDen = nb[den].filter((j) => !BLOCKED.has(t[j]));
    if (openDen.length < 2) continue;
    const starts = nb[den].filter((j) => !BLOCKED.has(t[j]) && t[j] !== SPRING);
    const start = starts[ri(0, starts.length - 1)];
    const reachable = distances(
      t,
      n,
      spring,
      new Set([LOW, SPRING, DEN, SOIL, CLAY]),
    );
    if (reachable[den] > r.reach - 2) continue;
    const b = { tiles: t, spring, den, start, seed };
    if (digsNeeded(newGame(b, r))[den] < 3) continue;
    return transform(b, ri(0, 7), n);
  }
  throw new Error(`no board for seed ${seed}`);
}

function transform(b, k, n) {
  const f = (i) => {
    let x = i % n;
    let y = (i / n) | 0;
    if (k & 1) x = n - 1 - x;
    if (k & 2) y = n - 1 - y;
    if (k & 4) [x, y] = [y, x];
    return y * n + x;
  };
  const tiles = new Array(n * n);
  b.tiles.forEach((v, i) => (tiles[f(i)] = v));
  return {
    tiles,
    spring: f(b.spring),
    den: f(b.den),
    start: f(b.start),
    seed: b.seed,
  };
}

// ------------------------------------------------------------------------------ the game
export function newGame(board, r = RULES) {
  const n = r.size;
  const g = {
    r,
    n,
    nb: nbrs(n),
    board,
    tiles: board.tiles.slice(),
    beaverAt: board.start,
    logs: r.logs,
    round: 0,
    flooded: false,
    done: false,
    lastDig: -1,
    lastLog: -1,
    dug: 0,
    dams: 0,
    wet: null,
  };
  g.springDist = distances(
    g.tiles,
    n,
    board.spring,
    new Set([LOW, SPRING, DEN, SOIL, CLAY, LOG]),
  );
  // No dams in the fast water near the spring, or right next to the den (its way in).
  g.noDam = g.springDist.map((d) => d < r.calm);
  for (const j of g.nb[board.den]) g.noDam[j] = true;
  const d = strength(g);
  g.wet = d.map((v) => v <= r.reach);
  return g;
}

/** Steps from the spring along open water ground right now. */
export function strength(g) {
  return distances(g.tiles, g.n, g.board.spring, WATER_OPEN);
}

/** End of round: cut-off water drains, then water spreads one tile within reach. */
export function flow(g) {
  const n2 = g.n * g.n;
  const t = g.tiles;
  const sp = g.board.spring;
  const keep = new Array(n2).fill(false);
  keep[sp] = true;
  const q = [sp];
  for (let h = 0; h < q.length; h++) {
    const i = q[h];
    for (const j of g.nb[i]) {
      if (!keep[j] && g.wet[j] && WATER_OPEN.has(t[j])) {
        keep[j] = true;
        q.push(j);
      }
    }
  }
  const d = strength(g);
  const wet = keep.slice();
  for (let i = 0; i < n2; i++) {
    if (!keep[i]) continue;
    for (const j of g.nb[i])
      if (!keep[j] && WATER_OPEN.has(t[j]) && d[j] <= g.r.reach) wet[j] = true;
  }
  g.wet = wet;
  g.round += 1;
  if (g.wet[g.board.den]) {
    g.flooded = true;
    g.done = true;
  } else if (g.round >= g.r.rounds) g.done = true;
}

export function canDig(g, i, d = null) {
  if (i < 0 || i >= g.n * g.n || !DIGGABLE.has(g.tiles[i])) return false;
  const s = d ?? strength(g);
  return g.nb[i].some((j) => g.wet[j] && s[j] < g.r.reach);
}

export function diggable(g) {
  const d = strength(g);
  const out = [];
  for (let i = 0; i < g.n * g.n; i++) if (canDig(g, i, d)) out.push(i);
  return out;
}

/** The river's move: dig tile i (or pass with -1). Returns whether it dug. */
export function dig(g, i) {
  g.lastDig = -1;
  if (i < 0 || !canDig(g, i)) return false;
  g.tiles[i] = g.tiles[i] === CLAY ? SOIL : LOW;
  g.lastDig = i;
  g.dug += 1;
  return true;
}

export const walkable = (g, i) => !BLOCKED.has(g.tiles[i]);
export const canLog = (g, i) =>
  i >= 0 && g.logs > 0 && g.tiles[i] === LOW && !g.noDam[i];

export function beaverLegal(g) {
  const out = [true, false, false, false, false, false, false, false, false];
  for (let d = 0; d < 4; d++) {
    const j = stepTo(g.beaverAt, d, g.n);
    out[1 + d] = j >= 0 && walkable(g, j);
    out[5 + d] = canLog(g, j);
  }
  return out;
}

export function beaver(g, a) {
  g.lastLog = -1;
  if (a >= 1 && a <= 4) {
    const j = stepTo(g.beaverAt, a - 1, g.n);
    if (j >= 0 && walkable(g, j)) g.beaverAt = j;
  } else if (a >= 5 && a <= 8) {
    const j = stepTo(g.beaverAt, a - 5, g.n);
    if (canLog(g, j)) {
      g.tiles[j] = LOG;
      g.wet[j] = false;
      g.logs -= 1;
      g.dams += 1;
      g.lastLog = j;
    }
  }
}

/** Fewest digs before water could reach each tile (beaver_env.Game.digs_needed). */
export function digsNeeded(g) {
  const cost = { [LOW]: 0, [DEN]: 0, [SPRING]: 0, [SOIL]: 1, [CLAY]: 2 };
  const n2 = g.n * g.n;
  const best = new Array(n2).fill(INF);
  const first = [];
  for (let i = 0; i < n2; i++) {
    if (g.wet[i]) {
      best[i] = 0;
      first.push(i);
    }
  }
  const buckets = [first];
  for (let k = 0; k < buckets.length; k++) {
    for (const i of buckets[k]) {
      if (best[i] !== k) continue;
      for (const j of g.nb[i]) {
        const c = cost[g.tiles[j]];
        if (c === undefined) continue;
        const v = k + c;
        if (v < best[j]) {
          best[j] = v;
          while (buckets.length <= v) buckets.push([]);
          buckets[v].push(j);
        }
      }
    }
  }
  return best;
}

// ------------------------------------------------------------------------------ bots
/** Binary-heap priority queue on [cost, tile], same pop order as Python's heapq on tuples. */
function heapPush(h, item) {
  h.push(item);
  let i = h.length - 1;
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (less(h[i], h[p])) {
      [h[i], h[p]] = [h[p], h[i]];
      i = p;
    } else break;
  }
}
function heapPop(h) {
  const top = h[0];
  const last = h.pop();
  if (h.length) {
    h[0] = last;
    let i = 0;
    for (;;) {
      const l = 2 * i + 1;
      const r = l + 1;
      let m = i;
      if (l < h.length && less(h[l], h[m])) m = l;
      if (r < h.length && less(h[r], h[m])) m = r;
      if (m === i) break;
      [h[i], h[m]] = [h[m], h[i]];
      i = m;
    }
  }
  return top;
}
const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);

/** The river bot's route (beaver_env.plan_route). */
export function planRoute(g, extra = null, digW = 3.0) {
  const n2 = g.n * g.n;
  const cost = { [LOW]: 0, [DEN]: 0, [SOIL]: 1, [CLAY]: 2 };
  const sd = strength(g);
  const dist = new Array(n2).fill(Infinity);
  const steps = new Array(n2).fill(INF);
  const prev = new Array(n2).fill(-1);
  const pq = [];
  for (let i = 0; i < n2; i++) {
    if (g.wet[i]) {
      dist[i] = 0;
      steps[i] = sd[i];
      heapPush(pq, [0, i]);
    }
  }
  while (pq.length) {
    const [c, i] = heapPop(pq);
    if (c > dist[i]) continue;
    if (i === g.board.den) break;
    for (const j of g.nb[i]) {
      const k = cost[g.tiles[j]];
      if (k === undefined || g.wet[j]) continue;
      const s = steps[i] + 1;
      if (s > g.r.reach) continue;
      const v = c + digW * k + 1 + (extra ? extra[j] : 0);
      if (v < dist[j]) {
        dist[j] = v;
        steps[j] = s;
        prev[j] = i;
        heapPush(pq, [v, j]);
      }
    }
  }
  if (prev[g.board.den] < 0 && !g.wet[g.board.den]) return [];
  const route = [];
  let i = g.board.den;
  while (i >= 0 && !g.wet[i]) {
    route.push(i);
    i = prev[i];
  }
  return route.reverse();
}

/** The scripted river without noise (beaver_env.RiverBot with defaults). */
export function riverBot(g, dodge = 0) {
  const options = diggable(g);
  if (!options.length) return -1;
  let extra = null;
  if (dodge) {
    extra = new Array(g.n * g.n).fill(0);
    const b = g.beaverAt;
    for (let i = 0; i < g.n * g.n; i++) {
      const m =
        Math.abs((i % g.n) - (b % g.n)) +
        Math.abs(((i / g.n) | 0) - ((b / g.n) | 0));
      extra[i] += dodge * Math.max(0, 3 - m);
    }
  }
  const route = planRoute(g, extra);
  const ok = new Set(options);
  for (const i of route) if (ok.has(i)) return i;
  if (route.length) return -1;
  const den = g.board.den;
  const md = (i) =>
    Math.abs((i % g.n) - (den % g.n)) +
    Math.abs(((i / g.n) | 0) - ((den / g.n) | 0));
  let best = options[0];
  for (const i of options) if (md(i) < md(best)) best = i;
  return best;
}

function beaverPath(g, goals) {
  if (goals.has(g.beaverAt)) return 0;
  const n2 = g.n * g.n;
  const first = new Array(n2).fill(-1);
  const seen = new Array(n2).fill(false);
  seen[g.beaverAt] = true;
  const q = [];
  for (let d = 0; d < 4; d++) {
    const j = stepTo(g.beaverAt, d, g.n);
    if (j >= 0 && walkable(g, j) && !seen[j]) {
      seen[j] = true;
      first[j] = d;
      q.push(j);
    }
  }
  for (let h = 0; h < q.length; h++) {
    const i = q[h];
    if (goals.has(i)) return 1 + first[i];
    for (const j of g.nb[i]) {
      if (!seen[j] && walkable(g, j)) {
        seen[j] = true;
        first[j] = first[i];
        q.push(j);
      }
    }
  }
  return -1;
}

/** The scripted beaver (beaver_env.warden_action). */
export function wardenAction(g) {
  const route = planRoute(g);
  let damable = g.logs
    ? route.filter((i) => g.tiles[i] === LOW && canLog(g, i))
    : [];
  if (!damable.length && g.logs && route.length)
    damable = g.nb[route[0]].filter((j) => g.wet[j] && canLog(g, j));
  for (let d = 0; d < 4; d++)
    if (damable.includes(stepTo(g.beaverAt, d, g.n))) return 5 + d;
  if (damable.length) {
    const stands = new Set();
    for (const i of damable)
      for (const j of g.nb[i]) if (walkable(g, j)) stands.add(j);
    const a = beaverPath(g, stands);
    if (a > 0) return a;
  }
  if (route.length) {
    const stands = new Set(g.nb[route[0]].filter((j) => walkable(g, j)));
    const a = beaverPath(g, stands);
    if (a > 0) return a;
  }
  return 0;
}

// ------------------------------------------------------------------------------ the network
export const N_CH = 12;

/** What both sides see (beaver_env.observe). */
export function observe(g) {
  const n2 = g.n * g.n;
  const t = g.tiles;
  const need = digsNeeded(g);
  const out = new Array(N_CH * n2 + 2).fill(0);
  for (let i = 0; i < n2; i++) {
    const v = t[i];
    out[i] = BLOCKED.has(v) ? 1 : 0;
    out[n2 + i] = v === SOIL ? 1 : 0;
    out[2 * n2 + i] = v === CLAY ? 1 : 0;
    out[3 * n2 + i] = v === LOW && !g.wet[i] ? 1 : 0;
    out[4 * n2 + i] = v === LOG ? 1 : 0;
    out[5 * n2 + i] = g.wet[i] ? 1 : 0;
    out[6 * n2 + i] = v === DEN ? 1 : 0;
    out[9 * n2 + i] = need[i] < INF ? 1 / (1 + need[i]) : 0;
    out[10 * n2 + i] = g.noDam[i] ? 1 : 0;
  }
  out[7 * n2 + g.beaverAt] = 1;
  if (g.lastDig >= 0) out[8 * n2 + g.lastDig] = 1;
  if (g.lastLog >= 0) out[11 * n2 + g.lastLog] = 1;
  out[N_CH * n2] = g.logs / g.r.logs;
  out[N_CH * n2 + 1] = (g.r.rounds - g.round) / g.r.rounds;
  return out;
}

/** The conv front end (beaver_env.Net.features): 3x3 padded convolutions with ReLU on the
 * planes, flattened channel-major, then the scalars. No convs: the input unchanged. */
export function features(net, input) {
  if (!net.convs) return input;
  const n = Math.round(Math.sqrt((input.length - 2) / N_CH));
  const n2 = n * n;
  let h = input.slice(0, N_CH * n2);
  let cin = N_CH;
  for (const c of net.convs) {
    const cout = c.b.length;
    const out = new Array(cout * n2);
    for (let o = 0; o < cout; o++) {
      const wo = c.w[o];
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          let s = c.b[o];
          for (let ci = 0; ci < cin; ci++) {
            const wk = wo[ci];
            const base = ci * n2;
            for (let dy = 0; dy < 3; dy++) {
              const yy = y + dy - 1;
              if (yy < 0 || yy >= n) continue;
              for (let dx = 0; dx < 3; dx++) {
                const xx = x + dx - 1;
                if (xx < 0 || xx >= n) continue;
                s += wk[dy * 3 + dx] * h[base + yy * n + xx];
              }
            }
          }
          out[o * n2 + y * n + x] = s > 0 ? s : 0;
        }
      }
    }
    h = out;
    cin = cout;
  }
  return h.concat(input.slice(N_CH * n2));
}

/** The network's logits for an observation. */
export function netLogits(net, input) {
  return logits(net, features(net, input));
}

/** Network logits with illegal actions set to -Infinity. */
export function maskedLogits(net, input, legal) {
  const z = netLogits(net, input);
  return z.map((v, i) => (legal[i] ? v : -Infinity));
}

export function bestOf(z) {
  let best = 0;
  for (let i = 1; i < z.length; i++) if (z[i] > z[best]) best = i;
  return best;
}

export function sampleOf(z, rand = Math.random) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  let r = rand() * s;
  for (let i = 0; i < e.length; i++) {
    r -= e[i];
    if (r < 0 && e[i] > 0) return i;
  }
  return bestOf(z);
}

/** The beaver network's move. sample: draw from its probabilities (as in training). */
export function netBeaver(net, g, sample = true, rand = Math.random) {
  const z = maskedLogits(net, observe(g), beaverLegal(g));
  return sample ? sampleOf(z, rand) : bestOf(z);
}

/** float16 (IEEE half) bits to a number, exactly. */
function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

function decodeHalves(b64) {
  const bin =
    typeof atob === "function"
      ? atob(b64)
      : Buffer.from(b64, "base64").toString("binary");
  const out = new Array(bin.length / 2);
  for (let i = 0; i < out.length; i++)
    out[i] = halfToFloat(
      bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8),
    );
  return out;
}

/** The exported network ({convs?, layers: [{shape, w, b}]}, float16 base64) in plain arrays:
 * layers {w: rows, b}; convs {w: [out][in][9], b}. */
export function unpackNet(packed) {
  const net = {
    layers: packed.layers.map((L) => {
      const [rows, cols] = L.shape;
      const flat = decodeHalves(L.w);
      const w = [];
      for (let r = 0; r < rows; r++)
        w.push(flat.slice(r * cols, (r + 1) * cols));
      return { w, b: decodeHalves(L.b) };
    }),
  };
  if (packed.convs) {
    net.convs = packed.convs.map((L) => {
      const [o, i, k1, k2] = L.shape;
      const flat = decodeHalves(L.w);
      const k = k1 * k2;
      const w = [];
      for (let a = 0; a < o; a++) {
        const row = [];
        for (let b = 0; b < i; b++)
          row.push(flat.slice((a * i + b) * k, (a * i + b + 1) * k));
        w.push(row);
      }
      return { w, b: decodeHalves(L.b) };
    });
  }
  return net;
}

/** Which tiles the river can dig, plus 'pass' last (beaver_env.river_legal). */
export function riverLegal(g) {
  const out = new Array(g.n * g.n + 1).fill(false);
  for (const i of diggable(g)) out[i] = true;
  out[g.n * g.n] = true;
  return out;
}

/** A trained river network's dig: a tile, or -1 to pass. */
export function netRiver(net, g, sample = true, rand = Math.random) {
  const z = maskedLogits(net, observe(g), riverLegal(g));
  const a = sample ? sampleOf(z, rand) : bestOf(z);
  return a === g.n * g.n ? -1 : a;
}
