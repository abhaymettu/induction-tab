// Every WGSL kernel the trainer runs. Forward, backward and AdamW are written by hand here;
// there is no autodiff. Shapes are row-major float32. `P` and `G` are the flat parameter and
// gradient buffers; activations live in their own buffers so no dispatch aliases a writable binding.

import { PRNG_WGSL } from './prng.js';

export const WG = 64;

// C[z] = alpha * opA(A[z]) @ opB(B[z]) (+ bias from B) (+ beta * D[z]) (+ C[z] when accumulating).
// z is split as (zo, zi) so one dispatch walks (batch, head) with independent strides.
export const GEMM = /* wgsl */ `
struct G {
  M: u32, N: u32, K: u32, zInner: u32,
  aOff: u32, aLd: u32, aSo: u32, aSi: u32,
  bOff: u32, bLd: u32, bSo: u32, bSi: u32,
  cOff: u32, cLd: u32, cSo: u32, cSi: u32,
  dOff: u32, dLd: u32, dSo: u32, dSi: u32,
  transA: u32, transB: u32, flags: u32, biasOff: u32,
  alpha: f32, beta: f32, pad0: u32, pad1: u32,
};
@group(0) @binding(0) var<uniform> g: G;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> D: array<f32>;

const TM = 32u;
const TN = 32u;
const TK = 16u;
var<workgroup> As: array<f32, 512>;
var<workgroup> Bs: array<f32, 512>;

fn loadA(base: u32, m: u32, k: u32) -> f32 {
  if (m >= g.M || k >= g.K) { return 0.0; }
  if (g.transA == 1u) { return A[base + k * g.aLd + m]; }
  return A[base + m * g.aLd + k];
}
fn loadB(base: u32, k: u32, n: u32) -> f32 {
  if (k >= g.K || n >= g.N) { return 0.0; }
  if (g.transB == 1u) { return B[base + n * g.bLd + k]; }
  return B[base + k * g.bLd + n];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let zo = wg.z / g.zInner;
  let zi = wg.z % g.zInner;
  let aBase = g.aOff + zo * g.aSo + zi * g.aSi;
  let bBase = g.bOff + zo * g.bSo + zi * g.bSi;
  let cBase = g.cOff + zo * g.cSo + zi * g.cSi;
  let dBase = g.dOff + zo * g.dSo + zi * g.dSi;
  let m0 = wg.y * TM;
  let n0 = wg.x * TN;
  var acc: array<array<f32, 4>, 4>;
  for (var k0 = 0u; k0 < g.K; k0 += TK) {
    for (var i = 0u; i < 8u; i++) {
      let e = li + i * 64u;
      if (g.transA == 1u) {
        As[e] = loadA(aBase, m0 + e % TM, k0 + e / TM);
      } else {
        As[(e % TK) * TM + e / TK] = loadA(aBase, m0 + e / TK, k0 + e % TK);
      }
      if (g.transB == 1u) {
        Bs[(e % TK) * TN + e / TK] = loadB(bBase, k0 + e % TK, n0 + e / TK);
      } else {
        Bs[e] = loadB(bBase, k0 + e / TN, n0 + e % TN);
      }
    }
    workgroupBarrier();
    for (var kk = 0u; kk < TK; kk++) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) {
        a[i] = As[kk * TM + lid.y * 4u + i];
        b[i] = Bs[kk * TN + lid.x * 4u + i];
      }
      for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
          acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let m = m0 + lid.y * 4u + i;
    if (m >= g.M) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let n = n0 + lid.x * 4u + j;
      if (n >= g.N) { continue; }
      var v = g.alpha * acc[i][j];
      if ((g.flags & 2u) != 0u) { v += B[g.biasOff + n]; }
      if ((g.flags & 1u) != 0u) { v += g.beta * D[dBase + m * g.dLd + n]; }
      let idx = cBase + m * g.cLd + n;
      if ((g.flags & 4u) != 0u) { v += C[idx]; }
      C[idx] = v;
    }
  }
}
`;

const U32_UNIFORM = (fields) => `struct U { ${fields.map((f) => `${f}: u32`).join(', ')} };
@group(0) @binding(0) var<uniform> u: U;`;

// x[n, c] = wte[tok[n], c] + wpe[n % T, c]
export const EMBED_FWD = /* wgsl */ `
${U32_UNIFORM(['rows', 'T', 'd', 'wte', 'wpe', 'p0', 'p1', 'p2'])}
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> tok: array<u32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = id.x;
  if (n >= u.rows) { return; }
  let te = u.wte + tok[n] * u.d;
  let pe = u.wpe + (n % u.T) * u.d;
  for (var c = 0u; c < u.d; c++) { X[n * u.d + c] = P[te + c] + P[pe + c]; }
}
`;

// One invocation per table row: dwte[v] += sum_{n: tok[n] = v} dX[n];  dwpe[t] += sum_b dX[b*T + t]
export const EMBED_BWD = /* wgsl */ `
${U32_UNIFORM(['rows', 'T', 'd', 'wte', 'wpe', 'V', 'p0', 'p1'])}
@group(0) @binding(1) var<storage, read_write> Gr: array<f32>;
@group(0) @binding(2) var<storage, read> tok: array<u32>;
@group(0) @binding(3) var<storage, read> dX: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < u.V) {
    let out = u.wte + i * u.d;
    for (var n = 0u; n < u.rows; n++) {
      if (tok[n] == i) {
        for (var c = 0u; c < u.d; c++) { Gr[out + c] += dX[n * u.d + c]; }
      }
    }
  } else if (i < u.V + u.T) {
    let t = i - u.V;
    let out = u.wpe + t * u.d;
    for (var n = t; n < u.rows; n += u.T) {
      for (var c = 0u; c < u.d; c++) { Gr[out + c] += dX[n * u.d + c]; }
    }
  }
}
`;

// In place over [rows, T, T] scores: causal mask then softmax along the last axis.
// Rows of ablated heads are zeroed (mask bit h of u.ablate), which removes the head's output.
export const SOFTMAX_FWD = /* wgsl */ `
${U32_UNIFORM(['rows', 'T', 'H', 'ablate'])}
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let r = id.x;
  if (r >= u.rows) { return; }
  let i = r % u.T;
  let h = (r / u.T) % u.H;
  let base = r * u.T;
  if (((u.ablate >> h) & 1u) != 0u) {
    for (var j = 0u; j < u.T; j++) { S[base + j] = 0.0; }
    return;
  }
  var mx = -3.0e38;
  for (var j = 0u; j <= i; j++) { mx = max(mx, S[base + j]); }
  var sum = 0.0;
  for (var j = 0u; j <= i; j++) {
    let e = exp(S[base + j] - mx);
    S[base + j] = e;
    sum += e;
  }
  let inv = 1.0 / sum;
  for (var j = 0u; j <= i; j++) { S[base + j] *= inv; }
  for (var j = i + 1u; j < u.T; j++) { S[base + j] = 0.0; }
}
`;

// In place over dP: dS = P * (dP - sum_j P * dP)
export const SOFTMAX_BWD = /* wgsl */ `
${U32_UNIFORM(['rows', 'T', 'p0', 'p1'])}
@group(0) @binding(1) var<storage, read> Pr: array<f32>;
@group(0) @binding(2) var<storage, read_write> dS: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let r = id.x;
  if (r >= u.rows) { return; }
  let i = r % u.T;
  let base = r * u.T;
  var dot = 0.0;
  for (var j = 0u; j <= i; j++) { dot += Pr[base + j] * dS[base + j]; }
  for (var j = 0u; j < u.T; j++) {
    dS[base + j] = Pr[base + j] * (dS[base + j] - dot);
  }
}
`;

// Row-wise LayerNorm. Saves mean and rstd for the backward pass.
export const LN_FWD = /* wgsl */ `
${U32_UNIFORM(['rows', 'd', 'gOff', 'bOff'])}
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let r = id.x;
  if (r >= u.rows) { return; }
  let base = r * u.d;
  var mean = 0.0;
  for (var c = 0u; c < u.d; c++) { mean += X[base + c]; }
  mean /= f32(u.d);
  var vr = 0.0;
  for (var c = 0u; c < u.d; c++) { let t = X[base + c] - mean; vr += t * t; }
  vr /= f32(u.d);
  let rstd = inverseSqrt(vr + 1e-5);
  for (var c = 0u; c < u.d; c++) {
    Y[base + c] = (X[base + c] - mean) * rstd * P[u.gOff + c] + P[u.bOff + c];
  }
  stats[2u * r] = mean;
  stats[2u * r + 1u] = rstd;
}
`;

// dX[r] (+)= rstd * (dy*g - mean(dy*g) - xhat * mean(dy*g*xhat)). Accumulates after the first call
// of a backward pass because the residual stream gradient arrives from two branches.
export const LN_BWD_X = /* wgsl */ `
${U32_UNIFORM(['rows', 'd', 'gOff', 'acc'])}
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> dY: array<f32>;
@group(0) @binding(4) var<storage, read> stats: array<f32>;
@group(0) @binding(5) var<storage, read_write> dX: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let r = id.x;
  if (r >= u.rows) { return; }
  let base = r * u.d;
  let mean = stats[2u * r];
  let rstd = stats[2u * r + 1u];
  var m1 = 0.0;
  var m2 = 0.0;
  for (var c = 0u; c < u.d; c++) {
    let dyg = dY[base + c] * P[u.gOff + c];
    m1 += dyg;
    m2 += dyg * (X[base + c] - mean) * rstd;
  }
  m1 /= f32(u.d);
  m2 /= f32(u.d);
  for (var c = 0u; c < u.d; c++) {
    let xh = (X[base + c] - mean) * rstd;
    var v = rstd * (dY[base + c] * P[u.gOff + c] - m1 - xh * m2);
    if (u.acc == 1u) { v += dX[base + c]; }
    dX[base + c] = v;
  }
}
`;

// dg[c] += sum_r dy * xhat;  db[c] += sum_r dy
export const LN_BWD_P = /* wgsl */ `
${U32_UNIFORM(['rows', 'd', 'gOff', 'bOff'])}
@group(0) @binding(1) var<storage, read_write> Gr: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> dY: array<f32>;
@group(0) @binding(4) var<storage, read> stats: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = id.x;
  if (c >= u.d) { return; }
  var sg = 0.0;
  var sb = 0.0;
  for (var r = 0u; r < u.rows; r++) {
    let dy = dY[r * u.d + c];
    sg += dy * (X[r * u.d + c] - stats[2u * r]) * stats[2u * r + 1u];
    sb += dy;
  }
  Gr[u.gOff + c] += sg;
  Gr[u.bOff + c] += sb;
}
`;

// Exact GELU, x * Phi(x), matching torch.nn.GELU(). WGSL has no erf, so Phi uses Abramowitz and
// Stegun 7.1.26, whose absolute error is under 1.5e-7.
const PHI = /* wgsl */ `
fn phi(x: f32) -> vec2<f32> {
  let z = abs(x) * 0.70710678118654752;
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let e = exp(-z * z);
  let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  let erf = 1.0 - poly * e;
  let s = select(-1.0, 1.0, x >= 0.0);
  return vec2<f32>(0.5 * (1.0 + s * erf), 0.3989422804014327 * e);
}
`;

export const GELU_FWD = /* wgsl */ `
${U32_UNIFORM(['rows', 'w', 'p1', 'p2'])}
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
${PHI}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.rows) { return; }
  let base = id.x * u.w;
  for (var c = 0u; c < u.w; c++) {
    let x = X[base + c];
    Y[base + c] = x * phi(x).x;
  }
}
`;

// In place: dY becomes dX = dY * gelu'(x) = dY * (Phi(x) + x * phi(x)).
export const GELU_BWD = /* wgsl */ `
${U32_UNIFORM(['rows', 'w', 'p1', 'p2'])}
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> dY: array<f32>;
${PHI}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.rows) { return; }
  let base = id.x * u.w;
  for (var c = 0u; c < u.w; c++) {
    let x = X[base + c];
    let f = phi(x);
    dY[base + c] *= f.x + x * f.y;
  }
}
`;

// Softmax cross-entropy fused with its gradient. The target of row r is tok[r + 1]; the last position
// of each sequence has none, so it contributes nothing, and L becomes dlogits = (p - onehot) / count.
export const XENT = /* wgsl */ `
${U32_UNIFORM(['rows', 'V', 'T', 'withGrad'])}
@group(0) @binding(1) var<storage, read_write> L: array<f32>;
@group(0) @binding(2) var<storage, read> tok: array<u32>;
@group(0) @binding(3) var<storage, read_write> loss: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let r = id.x;
  if (r >= u.rows) { return; }
  let base = r * u.V;
  if (r % u.T == u.T - 1u) {
    loss[r] = 0.0;
    if (u.withGrad == 1u) { for (var v = 0u; v < u.V; v++) { L[base + v] = 0.0; } }
    return;
  }
  var mx = -3.0e38;
  for (var v = 0u; v < u.V; v++) { mx = max(mx, L[base + v]); }
  var sum = 0.0;
  for (var v = 0u; v < u.V; v++) { sum += exp(L[base + v] - mx); }
  let y = tok[r + 1u];
  loss[r] = log(sum) + mx - L[base + y];
  if (u.withGrad == 0u) { return; }
  let scale = 1.0 / f32(u.rows - u.rows / u.T);
  let inv = 1.0 / sum;
  for (var v = 0u; v < u.V; v++) {
    var p = exp(L[base + v] - mx) * inv;
    if (v == y) { p -= 1.0; }
    L[base + v] = p * scale;
  }
}
`;

// Mean training loss into the history ring, indexed by the step counter.
export const LOSS_REDUCE = /* wgsl */ `
${U32_UNIFORM(['rows', 'T', 'cap', 'p1'])}
@group(0) @binding(1) var<storage, read> loss: array<f32>;
@group(0) @binding(2) var<storage, read_write> hist: array<f32>;
@group(0) @binding(3) var<storage, read> state: array<u32>;
@compute @workgroup_size(1)
fn main() {
  var s = 0.0;
  for (var r = 0u; r < u.rows; r++) { s += loss[r]; }
  hist[(state[0] - 1u) % u.cap] = s / f32(u.rows - u.rows / u.T);
}
`;

// G[off + c] += sum_r dY[r, c]: the gradient of a bias added to every row.
export const BIAS_BWD = /* wgsl */ `
${U32_UNIFORM(['rows', 'w', 'off', 'p0'])}
@group(0) @binding(1) var<storage, read_write> Gr: array<f32>;
@group(0) @binding(2) var<storage, read> dY: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = id.x;
  if (c >= u.w) { return; }
  var s = 0.0;
  for (var r = 0u; r < u.rows; r++) { s += dY[r * u.w + c]; }
  Gr[u.off + c] += s;
}
`;

// Y[n] (+)= X[n], used where a layer has no LayerNorm in its backward path.
export const ADD = /* wgsl */ `
${U32_UNIFORM(['n', 'acc', 'p1', 'p2'])}
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.n) { return; }
  Y[id.x] = select(X[id.x], Y[id.x] + X[id.x], u.acc == 1u);
}
`;

// A training batch, drawn on the GPU from the step counter so a whole chunk of steps needs no CPU.
// Sequences follow the bigram table. With probability pRep the sequence repeats itself from a random
// offset r: positions r..T-1 take the original tokens 0..T-1-r (2609.19000's clone rule, in place
// walking down so sources are read before they are overwritten).
export const DATA = /* wgsl */ `
struct U { B: u32, T: u32, V: u32, K: u32, seed: u32, rMin: u32, rMax: u32, stream: u32, pRep: f32, p0: f32, p1: f32, p2: f32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> succ: array<u32>;
@group(0) @binding(2) var<storage, read> cum: array<f32>;
@group(0) @binding(3) var<storage, read_write> tok: array<u32>;
@group(0) @binding(4) var<storage, read> state: array<u32>;
${PRNG_WGSL}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let b = id.x;
  if (b >= u.B) { return; }
  let base = (state[0] * u.B + b) * (u.T + 3u);
  let seq = b * u.T;
  var cur = hash3(u.seed, u.stream, base) % u.V;
  for (var t = 0u; t < u.T; t++) {
    tok[seq + t] = cur;
    let x = unit(hash3(u.seed, u.stream, base + 1u + t));
    var slot = 0u;
    loop {
      if (slot == u.K - 1u || x < cum[cur * u.K + slot]) { break; }
      slot += 1u;
    }
    cur = succ[cur * u.K + slot];
  }
  if (unit(hash3(u.seed, u.stream, base + u.T + 1u)) < u.pRep) {
    let r = u.rMin + hash3(u.seed, u.stream, base + u.T + 2u) % (u.rMax - u.rMin);
    var t = u.T;
    loop {
      t -= 1u;
      if (t < r) { break; }
      tok[seq + t] = tok[seq + t - r];
    }
  }
}
`;

// Probe instruments on a [x ; x] batch of half length L (definitions from 2609.19000):
// out[l*H*2 + h*2 + 0] previous-token score, + 1 prefix-matching score, of head h in the attention S.
export const PROBE_SCORE = /* wgsl */ `
${U32_UNIFORM(['B', 'T', 'H', 'outOff'])}
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let h = id.x;
  let L = u.T / 2u;
  var prev = 0.0;
  var pre = 0.0;
  for (var b = 0u; b < u.B; b++) {
    let base = (b * u.H + h) * u.T * u.T;
    for (var q = 1u; q < u.T; q++) { prev += S[base + q * u.T + q - 1u]; }
    for (var q = L; q < u.T - 1u; q++) { pre += S[base + q * u.T + q - L + 1u]; }
  }
  out[u.outOff + 2u * h] = prev / f32(u.B * (u.T - 1u));
  out[u.outOff + 2u * h + 1u] = pre / f32(u.B * (L - 1u));
}
`;

// out[off + q] = mean over the batch of the loss at position q.
export const PROBE_CE = /* wgsl */ `
${U32_UNIFORM(['B', 'T', 'outOff', 'p0'])}
@group(0) @binding(1) var<storage, read> loss: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let q = id.x;
  if (q >= u.T) { return; }
  var s = 0.0;
  for (var b = 0u; b < u.B; b++) { s += loss[b * u.T + q]; }
  out[u.outOff + q] = s / f32(u.B);
}
`;

// AdamW over the flat parameter buffer (torch.optim.AdamW semantics, decay on every parameter),
// with linear warmup. Clears the gradient as it goes so the next backward pass can accumulate.
export const ADAMW = /* wgsl */ `
struct U { n: u32, warmup: u32, p0: u32, p1: u32, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, p2: f32, p3: f32, p4: f32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> P: array<f32>;
@group(0) @binding(2) var<storage, read_write> Gr: array<f32>;
@group(0) @binding(3) var<storage, read_write> M: array<f32>;
@group(0) @binding(4) var<storage, read_write> Vv: array<f32>;
@group(0) @binding(5) var<storage, read> state: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u.n) { return; }
  let t = f32(state[0]);
  var lr = u.lr;
  if (u.warmup > 0u) { lr = u.lr * min(1.0, t / f32(u.warmup)); }
  let gi = Gr[i];
  Gr[i] = 0.0;
  let m = u.b1 * M[i] + (1.0 - u.b1) * gi;
  let v = u.b2 * Vv[i] + (1.0 - u.b2) * gi * gi;
  M[i] = m;
  Vv[i] = v;
  let mh = m / (1.0 - pow(u.b1, t));
  let vh = v / (1.0 - pow(u.b2, t));
  let p = P[i] * (1.0 - lr * u.wd);
  P[i] = p - lr * mh / (sqrt(vh) + u.eps);
}
`;

export const TICK = /* wgsl */ `
${U32_UNIFORM(['inc', 'p0', 'p1', 'p2'])}
@group(0) @binding(1) var<storage, read_write> state: array<u32>;
@compute @workgroup_size(1)
fn main() { state[0] += u.inc; }
`;


