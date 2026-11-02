// All eight attention maps. One canvas, drawn either by the GPU straight from a storage buffer (no readback) or
// by a 2D fallback from recorded frames. Both lerp from the previous reading to the latest over one eval interval
// so the picture moves continuously. Row is the position reading, column is the position being read.

const clamp01 = (x) => Math.max(0, Math.min(1, x));
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

function cssColor(name) {
  const probe = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
  probe.fillStyle = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  probe.fillRect(0, 0, 1, 1);
  const [r, g, b] = probe.getImageData(0, 0, 1, 1).data;
  return [r / 255, g / 255, b / 255];
}

const SHADER = /* wgsl */ `
struct U {
  color: vec4f,
  mix: f32, size: f32, stepX: f32, stepY: f32,
  side: u32, heads: u32, layers: u32, pad: u32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> prev: array<f32>;
@group(0) @binding(2) var<storage, read> curr: array<f32>;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array(vec2f(-1, -3), vec2f(3, 1), vec2f(-1, 1));
  return vec4f(p[i], 0, 1);
}

fn at(base: u32, q: u32, k: u32) -> f32 {
  let i = base + min(q, u.side - 1) * u.side + min(k, u.side - 1);
  return mix(prev[i], curr[i], u.mix);
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let cell = floor(pos.xy / vec2f(u.stepX, u.stepY));
  let local = pos.xy - cell * vec2f(u.stepX, u.stepY);
  if (local.x >= u.size || local.y >= u.size || cell.x >= f32(u.heads) || cell.y >= f32(u.layers)) { discard; }
  let scale = u.size / f32(u.side);
  let tex = local / scale;
  let ij = floor(tex);
  // Nearest neighbor at any scale: a device pixel that straddles two texels takes each by its coverage,
  // which is exactly nearest when the scale is a whole number.
  let w = clamp((tex - ij + 0.5 / scale - 1.0) * scale, vec2f(0), vec2f(1));
  let base = (u32(cell.y) * u.heads + u32(cell.x)) * u.side * u.side;
  let q = u32(ij.y);
  let k = u32(ij.x);
  let v = mix(mix(at(base, q, k), at(base, q, k + 1), w.x), mix(at(base, q + 1, k), at(base, q + 1, k + 1), w.x), w.y);
  let a = pow(clamp(v, 0.0, 1.0), 0.65);
  return vec4f(u.color.rgb * a, a);
}
`;

export function createGpuMaps(device, canvas, view, { T, H, L }) {
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'premultiplied' });
  const uniform = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const copies = [0, 1].map(() => device.createBuffer({ size: view.size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
  const module = device.createShaderModule({ code: SHADER, label: 'attention' });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  // groups[i] reads copies[i] as the newest and the other as the one before it.
  const groups = [[1, 0], [0, 1]].map(([p, c]) => device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: copies[p] } },
      { binding: 2, resource: { buffer: copies[c] } },
    ],
  }));

  let latest = 0; // copies[latest] holds the newest reading
  let from = 0;
  let span = 280;
  let dirty = true;
  let geo = { size: 1, stepX: 1, stepY: 1 };
  let color = cssColor('--glow');
  const data = new ArrayBuffer(48);
  const f32 = new Float32Array(data);
  const u32 = new Uint32Array(data);

  return {
    layout(g) { geo = g; dirty = true; },
    recolor() { color = cssColor('--glow'); dirty = true; },
    // Copies the view into the older copy, which becomes the newest. `ms` is how long to take getting there.
    push(ms) {
      const enc = device.createCommandEncoder();
      const next = 1 - latest;
      enc.copyBufferToBuffer(view, 0, copies[next], 0, view.size);
      device.queue.submit([enc.finish()]);
      latest = next;
      from = performance.now();
      span = reduced() ? 1 : ms;
      dirty = true;
    },
    // Returns true while there is motion left to draw.
    draw(now) {
      const mixT = clamp01((now - from) / span);
      if (!dirty && mixT >= 1) return false;
      f32.set(color, 0);
      f32[4] = mixT; f32[5] = geo.size; f32[6] = geo.stepX; f32[7] = geo.stepY;
      u32[8] = T; u32[9] = H; u32[10] = L;
      device.queue.writeBuffer(uniform, 0, data);
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, groups[latest]);
      pass.draw(3);
      pass.end();
      device.queue.submit([enc.finish()]);
      dirty = mixT < 1;
      return dirty;
    },
    clear() {
      const enc = device.createCommandEncoder();
      copies.forEach((b) => enc.clearBuffer(b));
      device.queue.submit([enc.finish()]);
      dirty = true;
    },
  };
}

// The recording's renderer. Frames are bytes [layer, head, q, k]: attention quantized to 8 bits.
export function createCpuMaps(canvas, { T, H, L }) {
  const ctx = canvas.getContext('2d');
  const tile = document.createElement('canvas');
  tile.width = T;
  tile.height = T;
  const tileCtx = tile.getContext('2d');
  const image = tileCtx.createImageData(T, T);
  const n = L * H * T * T;
  let prev = new Uint8Array(n);
  let curr = new Uint8Array(n);
  let from = 0;
  let span = 280;
  let dirty = true;
  let geo = { size: 1, stepX: 1, stepY: 1 };
  let color = cssColor('--glow').map((x) => Math.round(x * 255));
  // The same alpha ramp as the shader, v^0.65.
  const ramp = new Uint8Array(256).map((_, i) => Math.round(Math.pow(i / 255, 0.65) * 255));

  return {
    layout(g) { geo = g; dirty = true; },
    recolor() { color = cssColor('--glow').map((x) => Math.round(x * 255)); dirty = true; },
    push(bytes, ms) {
      prev = curr;
      curr = bytes;
      from = performance.now();
      span = reduced() ? 1 : ms;
      dirty = true;
    },
    draw(now) {
      const t = clamp01((now - from) / span);
      if (!dirty && t >= 1) return false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = false;
      const px = image.data;
      for (let l = 0; l < L; l++) {
        for (let h = 0; h < H; h++) {
          const base = (l * H + h) * T * T;
          for (let i = 0; i < T * T; i++) {
            const o = i * 4;
            px[o] = color[0]; px[o + 1] = color[1]; px[o + 2] = color[2];
            px[o + 3] = ramp[Math.round(prev[base + i] + (curr[base + i] - prev[base + i]) * t)];
          }
          tileCtx.putImageData(image, 0, 0);
          ctx.drawImage(tile, h * geo.stepX, l * geo.stepY, geo.size, geo.size);
        }
      }
      dirty = t < 1;
      return dirty;
    },
    clear() {
      prev = new Uint8Array(n);
      curr = new Uint8Array(n);
      dirty = true;
    },
  };
}

const MAPS_SHARE = 0.62;
const FOOT_SPACE = 28;
const PREV_EARNED = 0.2;
const PREFIX_EARNED = 0.2;
const JOBS = {
  prev: { short: 'reads the previous character', why: 'of its attention lands on the character just before', metric: 'prev' },
  prefix: { short: 'finds where this character came before', why: 'of its attention in the repeat lands right after the earlier copy of this character', metric: 'prefix' },
};

// The DOM over the canvas: one button per head, a caption per layer, the labels each head has earned.
export function createMaps(root, { canvas: firstCanvas, explain, T, H, L, onSilence, getEnabled, onLayout }) {
  let canvas = firstCanvas;
  const heads = [];
  const captions = [];
  const scores = [];
  const masks = new Array(L).fill(0);
  let renderer = null;
  for (let l = 0; l < L; l++) {
    const cap = document.createElement('p');
    cap.className = 'layer';
    cap.textContent = `Layer ${l}`;
    cap.style.gridColumn = '1 / -1';
    root.append(cap);
    captions.push(cap);
    const row = [];
    for (let h = 0; h < H; h++) {
      const b = document.createElement('button');
      b.className = 'head';
      b.type = 'button';
      b.setAttribute('aria-pressed', 'false');
      const label = document.createElement('span');
      label.className = 'job';
      b.append(label);
      root.append(b);
      row.push({ el: b, label, job: null });
      const info = () => headText(l, h);
      // On a phone the words sit in the flow, so changing them on blur would move the buttons under a finger.
      const floating = (text) => { if (getComputedStyle(explain).position === 'absolute') explain.textContent = text; };
      b.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') floating(info()); });
      b.addEventListener('focus', () => floating(info()));
      b.addEventListener('pointerleave', () => floating(''));
      b.addEventListener('blur', () => floating(''));
      b.addEventListener('click', () => {
        if (getEnabled()) toggle(l, 1 << h);
        explain.textContent = info();
      });
    }
    heads.push(row);
  }

  const pct = (x) => `${Math.round(x * 100)}%`;
  function headText(l, h) {
    const where = `Layer ${l}, head ${h + 1}.`;
    const job = heads[l][h].job;
    const score = scores[l]?.[h];
    const tail = getEnabled() ? (masks[l] >> h & 1 ? ' Press to bring it back.' : ' Press to silence it.') : '';
    if (masks[l] >> h & 1) return `${where} Silenced for the probe and the text box. Training never sees this.${tail}`;
    if (!score) return `${where} No readings yet.`;
    if (job) return `${where} It ${JOBS[job].short}: ${pct(score[JOBS[job].metric])} ${JOBS[job].why}.${tail}`;
    return `${where} No clear job yet. Previous-character score ${score.prev.toFixed(2)}, prefix-matching score ${score.prefix.toFixed(2)}.${tail}`;
  }

  function toggle(l, bits) {
    masks[l] = (masks[l] & bits) === bits ? masks[l] & ~bits : masks[l] | bits;
    paint();
    onSilence(masks.slice());
  }

  function paint() {
    for (let l = 0; l < L; l++) {
      let sayPrev = false;
      let sayPrefix = false;
      for (let h = 0; h < H; h++) {
        const off = Boolean(masks[l] >> h & 1);
        const { el, label, job } = heads[l][h];
        el.setAttribute('aria-pressed', String(off));
        const text = off ? 'silenced' : job ? JOBS[job].short : '';
        if (label.textContent !== text) label.textContent = text;
        el.classList.toggle('earned', Boolean(job) && !off);
        el.setAttribute('aria-label', `Layer ${l} head ${h + 1}${text ? `, ${text}` : ''}`);
        if (job === 'prev' && !off) sayPrev = true;
        if (job === 'prefix' && !off) sayPrefix = true;
      }
      const role = sayPrefix ? ' finds where this character came before' : sayPrev ? ' reads the previous character' : '';
      const text = `Layer ${l}${role}`;
      if (captions[l].textContent !== text) captions[l].textContent = text;
    }
  }

  // A device-pixel layout: whole-number texel scale when the screen allows it, so nearest neighbor stays crisp.
  // Beside the clock the maps take what the window can show without scrolling, down to one CSS pixel per cell.
  function resize() {
    if (!renderer) return;
    const dpr = window.devicePixelRatio || 1;
    const column = root.parentElement;
    const below = column.parentElement;
    const beside = matchMedia('(min-width: 1181px)').matches;
    const width = beside ? below.clientWidth * MAPS_SHARE : column.clientWidth;
    const narrow = width < 620;
    const gapCss = narrow ? 6 : 16;
    const capCss = narrow ? 26 : 30;
    const room = (width - gapCss * (H - 1)) / H;
    let target = narrow ? room : Math.min(room, 192);
    if (beside) {
      const foot = column.querySelector('.mapfoot').offsetHeight + FOOT_SPACE;
      target = Math.max(T, Math.min(room, (innerHeight - below.offsetTop - L * capCss - foot) / L));
    }
    const exact = (target * dpr) / T;
    const scale = exact < 1.9 ? exact : Math.max(2, Math.floor(exact));
    const size = Math.round(scale * T);
    const free = Math.floor((width * dpr - H * size) / (H - 1));
    const gap = narrow ? Math.max(1, free) : Math.round(gapCss * dpr);
    const cap = Math.round(capCss * dpr);
    canvas.width = H * size + (H - 1) * gap;
    canvas.height = L * size + (L - 1) * cap;
    canvas.style.width = `${canvas.width / dpr}px`;
    canvas.style.height = `${canvas.height / dpr}px`;
    if (beside) document.documentElement.style.setProperty('--maps-width', canvas.style.width);
    root.style.setProperty('--map', `${size / dpr}px`);
    root.style.setProperty('--map-gap', `${gap / dpr}px`);
    root.style.setProperty('--cap', `${cap / dpr}px`);
    renderer.layout({ size, stepX: size + gap, stepY: size + cap });
    onLayout?.();
  }

  new ResizeObserver(resize).observe(root.parentElement);
  addEventListener('resize', resize);

  return {
    use(r, c) { renderer = r; canvas = c; resize(); },
    get renderer() { return renderer; },
    get masks() { return masks.slice(); },
    resize,
    toggleLayer(l) { toggle(l, (1 << H) - 1); },
    restore() {
      masks.fill(0);
      paint();
      onSilence(masks.slice());
    },
    update(frameScores) {
      for (let l = 0; l < L; l++) {
        scores[l] = frameScores[l];
        for (let h = 0; h < H; h++) {
          if (masks[l] >> h & 1) continue;
          const s = frameScores[l][h];
          const row = heads[l][h];
          if (s.prefix >= PREFIX_EARNED) row.job = 'prefix';
          else if (s.prev >= PREV_EARNED && row.job !== 'prefix') row.job = 'prev';
          else if (s.prev < PREV_EARNED / 2 && s.prefix < PREFIX_EARNED / 2) row.job = null;
        }
      }
      paint();
    },
    reset() {
      masks.fill(0);
      scores.length = 0;
      heads.flat().forEach((x) => { x.job = null; });
      explain.textContent = '';
      paint();
    },
  };
}
