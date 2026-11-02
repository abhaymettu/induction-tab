import { hash3, unit, STREAM } from './prng.js';
import { layoutOf } from './layout.js';

// Deterministic init from (seed, tensor index, element index). Box-Muller in float64, rounded to
// float32; ref/induction_ref/init.py computes the same numbers.
export function initParams(cfg, seed) {
  const { specs, total } = layoutOf(cfg);
  const out = new Float32Array(total);
  specs.forEach((s, ti) => {
    const stream = STREAM.init + ti;
    for (let i = 0; i < s.size; i++) {
      let v = 0;
      if (s.init === 'one') v = 1;
      else if (s.init === 'kaiming') v = (2 * unit(hash3(seed, stream, i)) - 1) / Math.sqrt(s.fanIn);
      else if (s.init === 'normal') {
        const u1 = ((hash3(seed, stream, 2 * i) >>> 8) + 1) * 5.9604644775390625e-8;
        const u2 = unit(hash3(seed, stream, 2 * i + 1));
        v = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      out[s.off + i] = v;
    }
  });
  return out;
}
