// Parameter layout shared by the WGSL trainer and the PyTorch reference (ref/induction_ref/layout.py
// mirrors it; tools/parity.mjs checks the two agree). One flat float32 buffer. Matrices are stored
// [in, out] so a layer is x @ W. Defaults follow the 2609.19000 fleet: pre-LN blocks with a 4x GELU
// MLP, no attention biases, PyTorch default init, AdamW (0.9, 0.95) with decay on every parameter.

export const DEFAULTS = {
  V: 512, T: 128, d: 128, H: 4, L: 2, unique: 2, mlp: 4, ln: true,
  B: 32, lr: 1e-3, b1: 0.9, b2: 0.95, eps: 1e-8, wd: 0.01, warmup: 100,
  pRep: 0.75, langSeed: 777, seed: 1, steps: 16000, evalEvery: 25, probeB: 64, probeT: 128,
};

export function resolve(cfg) {
  const c = { ...DEFAULTS, ...cfg };
  if (c.d % c.H) throw new Error(`d=${c.d} not divisible by H=${c.H}`);
  if (c.L % c.unique) throw new Error(`L=${c.L} not a multiple of unique=${c.unique}`);
  c.dh = c.d / c.H;
  c.f = c.mlp * c.d;
  c.rMin ??= Math.floor(c.T / 16);
  c.rMax ??= c.T - Math.floor(c.T / 16);
  if (c.probeT > c.T) c.probeT = c.T;
  return c;
}

// [{name, shape, fanIn, init}] where init is 'normal' (N(0,1), torch nn.Embedding), 'kaiming'
// (U(-1/sqrt(fanIn), 1/sqrt(fanIn)), torch nn.Linear weight and bias), 'one' or 'zero'.
export function paramSpecs(cfg) {
  const c = resolve(cfg);
  const s = [];
  const add = (name, shape, init, fanIn = 0) => s.push({ name, shape, init, fanIn });
  add('wte', [c.V, c.d], 'normal');
  add('wpe', [c.T, c.d], 'normal');
  for (let u = 0; u < c.unique; u++) {
    const p = `b${u}.`;
    if (c.ln) { add(p + 'ln1g', [c.d], 'one'); add(p + 'ln1b', [c.d], 'zero'); }
    add(p + 'wqkv', [c.d, 3 * c.d], 'kaiming', c.d);
    add(p + 'wo', [c.d, c.d], 'kaiming', c.d);
    if (c.mlp) {
      if (c.ln) { add(p + 'ln2g', [c.d], 'one'); add(p + 'ln2b', [c.d], 'zero'); }
      add(p + 'w1', [c.d, c.f], 'kaiming', c.d);
      add(p + 'b1', [c.f], 'kaiming', c.d);
      add(p + 'w2', [c.f, c.d], 'kaiming', c.f);
      add(p + 'b2', [c.d], 'kaiming', c.f);
    }
  }
  if (c.ln) { add('lnfg', [c.d], 'one'); add('lnfb', [c.d], 'zero'); }
  add('wu', [c.d, c.V], 'kaiming', c.d);
  let off = 0;
  return s.map((x) => {
    const size = x.shape.reduce((a, b) => a * b, 1);
    const out = { ...x, off, size };
    off += size;
    return out;
  });
}

export function layoutOf(cfg) {
  const specs = paramSpecs(cfg);
  const by = Object.fromEntries(specs.map((x) => [x.name, x]));
  const total = specs.reduce((a, x) => a + x.size, 0);
  return { specs, by, total };
}
