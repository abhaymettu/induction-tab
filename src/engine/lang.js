// The synthetic language of 2609.19000: a fixed bigram table with 20 successor slots per token, weighted
// by softmax(1.5 * N(0, 1)). Built from the shared PRNG so JS, WGSL and numpy agree bit for bit
// (the upstream table came from PyTorch's generator and cannot be rebuilt elsewhere).

import { hash3, unit, STREAM } from './prng.js';

export const SLOTS = 20;

export function buildLanguage(V, langSeed) {
  const succ = new Uint32Array(V * SLOTS);
  const cum = new Float32Array(V * SLOTS);
  const w = new Float64Array(SLOTS);
  for (let i = 0; i < V; i++) {
    let sum = 0;
    for (let k = 0; k < SLOTS; k++) {
      const j = (i * SLOTS + k) * 3;
      succ[i * SLOTS + k] = hash3(langSeed, STREAM.bigram, j) % V;
      const u1 = ((hash3(langSeed, STREAM.bigram, j + 1) >>> 8) + 1) * 5.9604644775390625e-8;
      const u2 = unit(hash3(langSeed, STREAM.bigram, j + 2));
      w[k] = Math.exp(1.5 * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
      sum += w[k];
    }
    let acc = 0;
    for (let k = 0; k < SLOTS; k++) {
      acc += w[k] / sum;
      cum[i * SLOTS + k] = acc;
    }
  }
  return { succ, cum };
}

// [x ; x] with x uniform on [lo, V - lo): the out-of-distribution copy probe of 2609.19000.
export function copyProbe(V, B, T, seed = 1234) {
  const L = T / 2;
  const lo = 10;
  const tok = new Uint32Array(B * T);
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < L; t++) {
      const x = lo + hash3(seed, STREAM.eval, b * L + t) % (V - 2 * lo);
      tok[b * T + t] = x;
      tok[b * T + L + t] = x;
    }
  }
  return tok;
}
