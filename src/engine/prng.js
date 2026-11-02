// Counter-based PRNG shared bit-for-bit by JS, WGSL and numpy (ref/induction_ref/prng.py).
// Every random draw is a pure function of (seed, stream, index), so the browser and the
// native reference see the same init and the same data stream without sharing state.

export function pcg(v) {
  const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

export function hash3(seed, stream, i) {
  return pcg((seed ^ pcg((stream ^ pcg(i >>> 0)) >>> 0)) >>> 0);
}

// Uniform in [0, 1) with 24 bits, exact in float32.
export function unit(h) {
  return (h >>> 8) * 5.9604644775390625e-8;
}

export const STREAM = {
  init: 0x1000,
  data: 0x2000,
  bigram: 0x3000,
  eval: 0x4000,
};

export const PRNG_WGSL = /* wgsl */ `
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn hash3(seed: u32, stream: u32, i: u32) -> u32 {
  return pcg(seed ^ pcg(stream ^ pcg(i)));
}
fn unit(h: u32) -> f32 {
  return f32(h >> 8u) * 5.9604644775390625e-8;
}
`;
