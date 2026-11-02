// The probe's first sequence as two aligned rows. A character in the lower row shows how sure the model is of
// it twice over: brightness on a log scale from chance to certain, so the moment copying starts is visible even
// at low probability, and a bar whose height is the probability itself.
import { decode } from '../engine/text.js';

const CHANCE = Math.log(96);
const EPS = 0.004;

export const lightOf = (p) => Math.max(0, Math.min(1, 1 + Math.log(Math.max(p, 1e-6)) / CHANCE));

const cell = (cls, text) => {
  const el = document.createElement('span');
  el.className = cls;
  el.textContent = text;
  return el;
};

export function createStream(root, tokens, { onHover }) {
  const half = tokens.length >> 1;
  const hi = cell('row hi', '');
  const lo = cell('row lo', '');
  const lit = [];
  const bars = [];
  const hiCells = [];
  const loCells = [];
  const lastLit = new Float32Array(half).fill(-1);
  const lastBar = new Float32Array(half).fill(-1);
  const probs = new Float32Array(half);
  for (let i = 0; i < half; i++) {
    const ch = decode(tokens[half + i]);
    const top = cell('c', decode(tokens[i]));
    const bottom = cell('c', ch);
    const light = cell('lit', ch);
    const bar = cell('bar', '');
    bottom.append(light, bar);
    hi.append(top);
    lo.append(bottom);
    hiCells.push(top);
    loCells.push(bottom);
    lit.push(light);
    bars.push(bar);
  }
  root.replaceChildren(hi, lo);

  let hovered = -1;
  const setHover = (i) => {
    if (i === hovered) return;
    if (hovered >= 0) { hiCells[hovered].classList.remove('on'); loCells[hovered].classList.remove('on'); }
    hovered = i;
    if (i >= 0) { hiCells[i].classList.add('on'); loCells[i].classList.add('on'); }
    onHover(i < 0 ? null : { column: i + 1, char: decode(tokens[half + i]), p: probs[i] });
  };
  const columnAt = (e) => {
    const box = root.getBoundingClientRect();
    return Math.min(half - 1, Math.max(0, Math.floor(((e.clientX - box.left) / box.width) * half)));
  };
  root.addEventListener('pointermove', (e) => setHover(columnAt(e)));
  root.addEventListener('pointerleave', () => setHover(-1));

  return {
    // seqLoss[t - 1] is the loss on the character at t, so the first character of the repeat reads position half - 1.
    update(seqLoss) {
      for (let i = 0; i < half; i++) {
        const p = Math.exp(-seqLoss[half + i - 1]);
        probs[i] = p;
        const light = lightOf(p);
        if (Math.abs(light - lastLit[i]) > EPS) { lit[i].style.opacity = light.toFixed(3); lastLit[i] = light; }
        if (Math.abs(p - lastBar[i]) > EPS) { bars[i].style.transform = `scaleY(${p.toFixed(3)})`; lastBar[i] = p; }
      }
      if (hovered >= 0) onHover({ column: hovered + 1, char: decode(tokens[half + hovered]), p: probs[hovered] });
    },
    clear() {
      for (let i = 0; i < half; i++) {
        lit[i].style.opacity = '0';
        bars[i].style.transform = 'scaleY(0)';
        lastLit[i] = 0;
        lastBar[i] = 0;
        probs[i] = 0;
      }
    },
  };
}
