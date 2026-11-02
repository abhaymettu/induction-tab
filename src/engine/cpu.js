// The same forward pass as kernels.js, in plain JS over the same flat parameter buffer, for browsers without
// WebGPU. Slower by orders of magnitude, fine for one reader typing. tools/record.mjs checks it agrees with
// t.predict on a prompt.
import { layoutOf, resolve } from './layout.js';

const LN_EPS = 1e-5;

// The polynomial erf the GELU kernel uses, not Math.erf-grade accuracy, so the two passes agree.
function gelu(x) {
  const z = Math.abs(x) * 0.70710678118654752;
  const t = 1 / (1 + 0.3275911 * z);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erf = 1 - poly * Math.exp(-z * z);
  return x * 0.5 * (1 + (x >= 0 ? erf : -erf));
}

function layerNorm(X, n, d) {
  const Y = new Float32Array(n * d);
  for (let r = 0; r < n; r++) {
    const base = r * d;
    let mean = 0;
    for (let c = 0; c < d; c++) mean += X[base + c];
    mean /= d;
    let vr = 0;
    for (let c = 0; c < d; c++) { const t = X[base + c] - mean; vr += t * t; }
    const rstd = 1 / Math.sqrt(vr / d + LN_EPS);
    for (let c = 0; c < d; c++) Y[base + c] = (X[base + c] - mean) * rstd;
  }
  return Y;
}

// Y[r, o] = sum over k of A[r, k] * P[off + k * out + o], rows first r0 to n.
function matmul(A, n, inDim, P, off, out, r0 = 0) {
  const Y = new Float32Array(n * out);
  for (let r = r0; r < n; r++) {
    const ar = r * inDim;
    const yr = r * out;
    for (let k = 0; k < inDim; k++) {
      const a = A[ar + k];
      const pk = off + k * out;
      for (let o = 0; o < out; o++) Y[yr + o] += a * P[pk + o];
    }
  }
  return Y;
}

export function createCpuModel(cfg, P) {
  const c = resolve(cfg);
  if (c.lnAffine) throw new Error('cpu pass handles LayerNorm without learned gains only');
  const { by } = layoutOf(c);
  const { d, H, dh, f, V, L, T } = c;
  const scale = 1 / Math.sqrt(dh);

  // attention, when kept, is [layer, head, q, k] with side n, the same layout as t.view.
  const forward = (tokens, { masks = [], keep = false, lastOnly = false } = {}) => {
    const n = Math.min(tokens.length, T);
    const toks = tokens.subarray ? tokens.subarray(tokens.length - n) : tokens.slice(-n);
    let X = new Float32Array(n * d);
    for (let i = 0; i < n; i++) {
      const te = by.wte.off + toks[i] * d;
      const pe = by.wpe.off + i * d;
      for (let j = 0; j < d; j++) X[i * d + j] = P[te + j] + P[pe + j];
    }
    const attention = keep ? new Float32Array(L * H * n * n) : null;
    const row = new Float64Array(n);
    for (let l = 0; l < L; l++) {
      const p = `b${l % c.unique}.`;
      const q3 = matmul(layerNorm(X, n, d), n, d, P, by[p + 'wqkv'].off, 3 * d);
      const O = new Float32Array(n * d);
      for (let h = 0; h < H; h++) {
        if ((masks[l] ?? 0) >> h & 1) continue;
        const qOff = h * dh;
        const kOff = d + h * dh;
        const vOff = 2 * d + h * dh;
        for (let i = 0; i < n; i++) {
          const qi = i * 3 * d + qOff;
          let mx = -Infinity;
          for (let j = 0; j <= i; j++) {
            const kj = j * 3 * d + kOff;
            let s = 0;
            for (let m = 0; m < dh; m++) s += q3[qi + m] * q3[kj + m];
            s *= scale;
            row[j] = s;
            if (s > mx) mx = s;
          }
          let sum = 0;
          for (let j = 0; j <= i; j++) { row[j] = Math.exp(row[j] - mx); sum += row[j]; }
          const oi = i * d + h * dh;
          for (let j = 0; j <= i; j++) {
            const w = row[j] / sum;
            if (attention) attention[(l * H + h) * n * n + i * n + j] = w;
            const vj = j * 3 * d + vOff;
            for (let m = 0; m < dh; m++) O[oi + m] += w * q3[vj + m];
          }
        }
      }
      const out = matmul(O, n, d, P, by[p + 'wo'].off, d);
      for (let i = 0; i < n * d; i++) out[i] += X[i];
      const H1 = matmul(layerNorm(out, n, d), n, d, P, by[p + 'w1'].off, f);
      for (let r = 0; r < n; r++) {
        for (let o = 0; o < f; o++) H1[r * f + o] = gelu(H1[r * f + o] + P[by[p + 'b1'].off + o]);
      }
      const next = matmul(H1, n, f, P, by[p + 'w2'].off, d);
      for (let r = 0; r < n; r++) {
        for (let o = 0; o < d; o++) next[r * d + o] += out[r * d + o] + P[by[p + 'b2'].off + o];
      }
      X = next;
    }
    const logits = matmul(layerNorm(X, n, d), n, d, P, by.wu.off, V, lastOnly ? n - 1 : 0);
    return { logits, attention, n };
  };

  // Logits for the token after `tokens`, as t.predict returns them.
  const predict = (tokens, masks) => {
    const { logits, n } = forward(tokens, { masks, lastOnly: true });
    return logits.slice((n - 1) * V, n * V);
  };

  // The probe's instruments for B sequences of length T, as t.parseProbe reads them from the GPU.
  const probe = (batch, B, masks) => {
    const half = T >> 1;
    const ce = new Float64Array(T);
    const scores = [];
    for (let l = 0; l < L; l++) scores.push(Array.from({ length: H }, () => ({ prev: 0, prefix: 0 })));
    const view = new Float32Array(L * H * T * T);
    let seqLoss = null;
    for (let b = 0; b < B; b++) {
      const toks = batch.subarray(b * T, (b + 1) * T);
      const { logits, attention } = forward(toks, { masks, keep: true });
      const loss = new Float32Array(T);
      for (let q = 0; q < T - 1; q++) {
        let mx = -Infinity;
        for (let v = 0; v < V; v++) mx = Math.max(mx, logits[q * V + v]);
        let sum = 0;
        for (let v = 0; v < V; v++) sum += Math.exp(logits[q * V + v] - mx);
        loss[q] = Math.log(sum) + mx - logits[q * V + toks[q + 1]];
        ce[q] += loss[q] / B;
      }
      if (b === 0) seqLoss = loss;
      for (let i = 0; i < view.length; i++) view[i] += attention[i] / B;
      for (let l = 0; l < L; l++) {
        for (let h = 0; h < H; h++) {
          const base = (l * H + h) * T * T;
          for (let q = 1; q < T; q++) scores[l][h].prev += attention[base + q * T + q - 1] / (B * (T - 1));
          for (let q = half; q < T - 1; q++) scores[l][h].prefix += attention[base + q * T + q - half + 1] / (B * (half - 1));
        }
      }
    }
    let first = 0;
    let second = 0;
    for (let q = 0; q < half - 1; q++) first += ce[q];
    for (let q = half - 1; q < T - 1; q++) second += ce[q];
    return { scores, copyAdv: first / (half - 1) - second / half, seqLoss, view };
  };

  return { c, forward, predict, probe };
}
