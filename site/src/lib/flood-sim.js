// Browser port of experiments/008-flood/flood_env.py. Keep them in step:
// site/test/flood-sim.test.mjs replays trajectories exported from Python and checks every
// state, every observation, the network's outputs and the random maps.

export { act, logits, probs, sampleAct } from './tag-sim.js';

/** Same order as Python: wait, walk left/right, dig left/right, drop left/right. */
export const ACTIONS = ['wait', 'left', 'right', 'dig-left', 'dig-right', 'drop-left', 'drop-right'];
export const [WAIT, LEFT, RIGHT, DIG_L, DIG_R, DROP_L, DROP_R] = [0, 1, 2, 3, 4, 5, 6];

/** Defaults of flood_env.World and Config; the exported JSON carries the real values. */
export const WORLD = {
  width: 40,
  height: 20,
  houseWidth: 5,
  houseFloor: [6, 9],
  tideOver: [1.0, 3.0],
  tideStart: 2.5,
  rise: 0.7,
  beach: [3, 6],
  flow: 0.2,
  damping: 0.95,
  substeps: 4,
  settlePasses: 2,
  floodDepth: 0.05,
  carry: 3,
  maxStep: 2,
  houseWall: null, // round 2: sand against the house can't rise above floor + houseWall
  maxSteps: 500,
  rewardDry: 1.0,
  rewardFlood: -1.0,
};

/** Same sequence as flood_env.mulberry32 in Python: floats in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Random coast from a seed (flood_env.make_map). Returns { sand, houseFloor, peak, startX, seed }. */
export function makeMap(seed, w = WORLD, tideScale = 1.0) {
  const rand = mulberry32(seed);
  const u = (lo, hi) => lo + (hi - lo) * rand();
  const int = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  const n = w.width - w.houseWidth;
  const hf = int(w.houseFloor[0], w.houseFloor[1]);
  let peak = hf + u(w.tideOver[0], w.tideOver[1]) * tideScale;
  peak = Math.max(peak, hf + 0.3);
  const cap = Math.max(2, Math.floor(peak - 0.5));
  const beach = int(w.beach[0], w.beach[1]);

  const span = Math.max(1, n - 1 - beach);
  const prof = [];
  for (let x = 0; x < n; x++) prof.push(2.0 + (hf - 2.0) * Math.min(1.0, Math.max(0.0, (x - beach) / span)));
  const bumps = int(1, 3);
  for (let k = 0; k < bumps; k++) {
    const c = u(beach + 2, n - 2);
    const width = u(1.5, 5.0);
    const amp = u(-4.0, 5.0);
    for (let x = 0; x < n; x++) {
      const z = (x - c) / width;
      prof[x] += amp * Math.exp(-0.5 * z * z);
    }
  }
  const sand = [];
  for (let x = 0; x < n; x++) sand.push(x < beach ? 2 : Math.min(cap, Math.max(2, Math.floor(prof[x] + 0.5))));
  let changed = true;
  while (changed) {
    changed = false;
    for (let x = 0; x < n; x++) {
      let lo = sand[x];
      if (x > 0) lo = Math.min(lo, sand[x - 1] + 1);
      if (x < n - 1) lo = Math.min(lo, sand[x + 1] + 1);
      if (lo < sand[x]) {
        sand[x] = lo;
        changed = true;
      }
    }
  }
  const land = [];
  for (let x = beach + 1; x < n; x++) if (sand[x] > w.tideStart) land.push(x);
  const startX = land.length ? land[Math.floor(rand() * land.length)] : n - 1;
  for (let k = 0; k < w.houseWidth; k++) sand.push(hf);
  return { sand, houseFloor: hf, peak, startX, seed };
}

/** A fresh episode on a map. */
export function newState(map, w = WORLD) {
  const sand = Int32Array.from(map.sand);
  const water = new Float64Array(w.width);
  for (let i = 0; i < w.width; i++) {
    if (sand[i] >= w.tideStart) break;
    water[i] = w.tideStart - sand[i];
  }
  return {
    w,
    map,
    n: w.width - w.houseWidth,
    sand,
    water,
    flux: new Float64Array(w.width),
    x: map.startX,
    carrying: 0,
    t: 0,
    dug: 0,
    dropped: 0,
    spilled: 0,
    done: false,
    outcome: null,
  };
}

export function seaLevel(s) {
  const { w, map } = s;
  const frac = Math.min(1.0, s.t / (w.rise * w.maxSteps));
  return w.tideStart + (map.peak - w.tideStart) * frac;
}

/** One settling pass over columns [0, n): a grain slides off any step steeper than maxStep. */
export function settle(sand, n, maxStep) {
  const moves = new Int32Array(n);
  for (let i = 0; i < n - 1; i++) {
    const d = sand[i + 1] - sand[i];
    if (d < -maxStep) {
      moves[i] -= 1;
      moves[i + 1] += 1;
    } else if (d > maxStep) {
      moves[i] += 1;
      moves[i + 1] -= 1;
    }
  }
  for (let i = 0; i < n; i++) sand[i] += moves[i];
}

/** One water sub-step ("virtual pipes"), in the same order of operations as Python. */
export function flowWater(sand, water, flux, sea, w) {
  const W = water.length;
  const q = new Float64Array(W);
  for (let p = 0; p < W; p++) {
    const left = p === 0 ? sea : sand[p - 1] + water[p - 1];
    q[p] = flux[p] * w.damping + w.flow * (left - (sand[p] + water[p]));
  }
  const scale = new Float64Array(W);
  for (let i = 0; i < W; i++) {
    let out = Math.max(-q[i], 0.0);
    if (i < W - 1) out += Math.max(q[i + 1], 0.0);
    scale[i] = out > water[i] ? water[i] / (out > 0 ? out : 1.0) : 1.0;
  }
  for (let p = 0; p < W; p++) {
    const src = p === 0 ? 1.0 : scale[p - 1];
    q[p] = q[p] > 0 ? q[p] * src : q[p] * scale[p];
  }
  for (let i = 0; i < W; i++) {
    let v = water[i] + q[i];
    if (i < W - 1) v -= q[i + 1];
    water[i] = Math.max(v, 0.0);
    flux[i] = q[i];
  }
}

export function applyAction(s, a) {
  const { w, n, sand } = s;
  const x = s.x;
  if (a === LEFT || a === RIGHT) {
    const nx = a === LEFT ? x - 1 : x + 1;
    if (nx >= 0 && nx < n && sand[nx] - sand[x] <= w.maxStep) s.x = nx;
  } else if (a === DIG_L || a === DIG_R) {
    const tx = a === DIG_L ? x - 1 : x + 1;
    if (tx >= 0 && tx < n && s.carrying < w.carry && sand[tx] > 1) {
      sand[tx] -= 1;
      s.carrying += 1;
      s.dug += 1;
    }
  } else if (a === DROP_L || a === DROP_R) {
    const tx = a === DROP_L ? x - 1 : x + 1;
    if (tx >= 0 && tx < n && s.carrying > 0 && sand[tx] < w.height - 1) {
      sand[tx] += 1;
      s.carrying -= 1;
      s.dropped += 1;
    }
  }
}

export function flooded(s) {
  for (let i = s.n; i < s.w.width; i++) if (s.water[i] > s.w.floodDepth) return true;
  return false;
}

/** One agent step. Returns the reward; sets s.done and s.outcome at the end. */
export function step(s, a) {
  const { w } = s;
  applyAction(s, a);
  for (let k = 0; k < w.settlePasses; k++) {
    settle(s.sand, s.n, w.maxStep);
    if (w.houseWall != null && s.sand[s.n - 1] > s.map.houseFloor + w.houseWall) {
      s.sand[s.n - 1] -= 1; // spills over the roof
      s.spilled += 1;
    }
  }
  s.t += 1;
  const sea = seaLevel(s);
  for (let k = 0; k < w.substeps; k++) flowWater(s.sand, s.water, s.flux, sea, w);
  if (flooded(s)) {
    s.done = true;
    s.outcome = 'flooded';
    return w.rewardFlood;
  }
  if (s.t >= w.maxSteps) {
    s.done = true;
    s.outcome = 'dry';
  }
  return w.rewardDry / w.maxSteps;
}

/** The network's input: sand heights, water depths, a one-hot position, then 4 numbers. */
export function observe(s) {
  const { w } = s;
  const H = w.height;
  const out = [];
  for (let i = 0; i < w.width; i++) out.push(s.sand[i] / H);
  for (let i = 0; i < w.width; i++) out.push(Math.min(s.water[i], H) / H);
  for (let i = 0; i < w.width; i++) out.push(i === s.x ? 1 : 0);
  out.push(s.carrying / w.carry, seaLevel(s) / H, s.map.peak / H, s.t / w.maxSteps);
  return out;
}

/** The scripted builder's wall column (flood_env.wall_site). */
export function wallSite(s) {
  const need = Math.floor(s.map.peak) + 1;
  let best = 0;
  let bestCost = Infinity;
  for (let x = 0; x < s.n - 2; x++) {
    const cost = Math.max(need - s.sand[x], 0) ** 2 + 0.01 * (s.n - x);
    if (cost < bestCost) {
      bestCost = cost;
      best = x;
    }
  }
  return best;
}

/** The scripted builder (flood_env.builder_action). */
export function builderAction(s, c) {
  const { w, n, sand, water } = s;
  const need = Math.floor(s.map.peak) + 1;
  const x = s.x;
  const go = (target) => (target === x ? WAIT : target > x ? RIGHT : LEFT);
  if (sand[c] >= need && s.carrying === 0) return WAIT;
  if (s.carrying === w.carry || (s.carrying > 0 && sand[c] >= need)) {
    const stand = c + 1 < n ? c + 1 : c - 1;
    if (x === stand) return stand > c ? DROP_L : DROP_R;
    return go(stand);
  }
  let lo = Infinity;
  for (let i = Math.max(0, c - 4); i < Math.min(s.map.sand.length, c + 5); i++) lo = Math.min(lo, s.map.sand[i]);
  const k = Math.max(3, Math.floor((need - lo + 1) / w.maxStep) + 2);
  const spots = [];
  for (let d = c + k; d < n; d++) if (sand[d] > 1 && water[d] < 0.5) spots.push(d);
  for (let d = c - k; d >= 0; d--) if (sand[d] > 1) spots.push(d);
  if (!spots.length) return WAIT;
  let d = spots[0];
  for (const e of spots) if (Math.abs(e - x) < Math.abs(d - x)) d = e;
  const stand = d > c ? d - 1 : d + 1;
  if (x === stand) return d > stand ? DIG_R : DIG_L;
  return go(stand);
}
