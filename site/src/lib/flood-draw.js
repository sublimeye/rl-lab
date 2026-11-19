// Draws a flood-world state (site/src/lib/flood-sim.js) on a canvas whose size is
// width x height cells (2:1 for the default 40 x 20 world). Shared by FloodGame and FloodReplay.

/** Scene colours from the CSS tokens, read from an element (follows light / dark mode). */
export function readColors(el) {
  const cs = getComputedStyle(el);
  const v = (name) => cs.getPropertyValue(name).trim();
  return {
    sky: v('--sky'),
    sand: v('--sand'),
    built: v('--sand-built'),
    bedrock: v('--bedrock'),
    water: v('--water'),
    ink: v('--ink'),
    muted: v('--muted'),
    wall: v('--wall'),
    roof: v('--lava'),
    floor: v('--floor'),
    font: cs.fontFamily,
  };
}

/**
 * s: a flood-sim state. opts.labels: draw the high-tide label; opts.digger: draw the digger
 * (default true).
 */
export function drawWorld(ctx, s, C, opts = {}) {
  const { labels = true, digger = true } = opts;
  const { w } = s;
  const W = w.width;
  const H = w.height;
  const cw = ctx.canvas.width / W;
  const ch = ctx.canvas.height / H;
  const X = (c) => c * cw;
  const Y = (v) => ctx.canvas.height - v * ch;
  // Whole pixels, so neighbouring columns meet without anti-aliased seams.
  const rect = (x0, v0, x1, v1) => {
    const l = Math.round(X(x0));
    const t = Math.round(Y(v1));
    ctx.fillRect(l, t, Math.round(X(x1)) - l, Math.round(Y(v0)) - t);
  };

  ctx.fillStyle = C.sky;
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  for (let x = 0; x < W; x++) {
    const s0 = s.map.sand[x];
    const sx = s.sand[x];
    ctx.fillStyle = C.bedrock;
    rect(x, 0, x + 1, 1);
    ctx.fillStyle = C.sand;
    rect(x, 1, x + 1, Math.min(sx, s0));
    if (sx > s0) {
      ctx.fillStyle = C.built;
      rect(x, s0, x + 1, sx);
    }
  }
  // Grain lines on built-up sand, so a dam reads as stacked grains.
  if (cw >= 6) {
    ctx.strokeStyle = C.sand;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = Math.max(1, cw / 14);
    ctx.beginPath();
    for (let x = 0; x < W; x++) {
      for (let v = s.map.sand[x] + 1; v < s.sand[x]; v++) {
        ctx.moveTo(X(x), Y(v));
        ctx.lineTo(X(x + 1), Y(v));
      }
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  drawHouse(ctx, s, C, X, Y);

  ctx.fillStyle = C.water;
  ctx.globalAlpha = 0.82;
  for (let x = 0; x < W; x++) {
    const d = s.water[x];
    if (d > 0.02) rect(x, s.sand[x], x + 1, s.sand[x] + d);
  }
  ctx.globalAlpha = 1;

  // The tide's high-water mark.
  const peak = s.map.peak;
  ctx.strokeStyle = C.muted;
  ctx.lineWidth = Math.max(1, cw / 12);
  ctx.setLineDash([cw * 0.5, cw * 0.4]);
  ctx.beginPath();
  ctx.moveTo(0, Y(peak));
  ctx.lineTo(X(s.n), Y(peak));
  ctx.stroke();
  ctx.setLineDash([]);
  if (labels) {
    ctx.fillStyle = C.muted;
    ctx.font = `600 ${Math.round(Math.max(11, cw * 0.62))}px ${C.font}`;
    ctx.textBaseline = 'bottom';
    ctx.fillText('high tide', X(0.4), Y(peak) - ch * 0.15);
  }

  if (digger) drawDigger(ctx, s, C, X, Y, cw, ch);
}

function drawHouse(ctx, s, C, X, Y) {
  const hf = s.map.houseFloor;
  const x0 = s.n + 0.45;
  const x1 = s.w.width - 0.45;
  const top = hf + (s.w.houseWall ?? 2.4); // round 2: the wall sand can lean on
  ctx.fillStyle = C.wall;
  ctx.fillRect(X(x0), Y(top), X(x1) - X(x0), Y(hf) - Y(top));
  ctx.fillStyle = C.roof;
  ctx.beginPath();
  ctx.moveTo(X(s.n + 0.05), Y(top));
  ctx.lineTo(X((s.n + s.w.width) / 2), Y(top + 1.7));
  ctx.lineTo(X(s.w.width - 0.05), Y(top));
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = C.ink;
  const mid = (s.n + s.w.width) / 2;
  ctx.fillRect(X(mid - 0.45), Y(hf + 1.45), X(mid + 0.45) - X(mid - 0.45), Y(hf) - Y(hf + 1.45));
  ctx.fillStyle = C.sky;
  ctx.fillRect(X(s.n + 1.1), Y(hf + 1.9), X(s.n + 1.9) - X(s.n + 1.1), Y(hf + 1.1) - Y(hf + 1.9));
  ctx.fillRect(X(mid + 1.0), Y(hf + 1.9), X(mid + 1.8) - X(mid + 1.0), Y(hf + 1.1) - Y(hf + 1.9));
}

function drawDigger(ctx, s, C, X, Y, cw, ch) {
  const x = s.x;
  const base = s.sand[x];
  const r = Math.min(cw, ch) * 0.18;
  ctx.fillStyle = C.ink;
  roundRect(ctx, X(x + 0.14), Y(base + 1.0), X(x + 0.86) - X(x + 0.14), Y(base) - Y(base + 1.0), r);
  ctx.fill();
  ctx.fillStyle = C.sky; // eyes
  const e = Math.max(1.5, cw * 0.09);
  ctx.fillRect(X(x + 0.32), Y(base + 0.78), e, e);
  ctx.fillRect(X(x + 0.6), Y(base + 0.78), e, e);
  // Carried grains, stacked above its head.
  ctx.fillStyle = C.built;
  for (let k = 0; k < s.carrying; k++) {
    const v = base + 1.12 + k * 0.42;
    ctx.fillRect(X(x + 0.3), Y(v + 0.36), X(x + 0.7) - X(x + 0.3), Y(v) - Y(v + 0.36));
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Sizes a canvas to its CSS box at the device pixel ratio. Returns true if it changed. */
export function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const wpx = Math.max(1, Math.round(r.width * dpr));
  const hpx = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width === wpx && canvas.height === hpx) return false;
  canvas.width = wpx;
  canvas.height = hpx;
  return true;
}
