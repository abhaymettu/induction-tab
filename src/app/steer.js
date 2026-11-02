// The third row: the reader types, the model continues as ghost text, greedy, one character at a time, drawn in
// the same language as the stream (brightness and a bar). Chips are measured, never assumed: a prompt is offered
// only if the model's greedy continuation at this moment really is the right one.
import { decode, encode } from '../engine/text.js';
import { lightOf } from './stream.js';

const GHOST = 24;
const DEBOUNCE_MS = 40;
const DEL = 95;
const CHECK = 10;

const CANDIDATES = [
  { text: 'abcabcabcabcab', period: 3 },
  { text: '0123456789 0123456', period: 11 },
  { text: 'qwertyuiop[] qwertyu', period: 13 },
  { text: 'hello hello hello he', period: 6 },
  { text: 'xyz xyz xyz xy', period: 4 },
  { text: 'ab12 ab12 ab12 ab', period: 5 },
  { text: '<div><div><div><d', period: 5 },
  { text: 'red blue red blue r', period: 9 },
];
const STUMBLE = { text: 'the quick brown fox. the quick br', period: 21 };

export const softmax = (logits) => {
  let mx = -Infinity;
  for (let v = 0; v < DEL; v++) if (logits[v] > mx) mx = logits[v];
  let sum = 0;
  const p = new Float64Array(DEL);
  for (let v = 0; v < DEL; v++) { p[v] = Math.exp(logits[v] - mx); sum += p[v]; }
  for (let v = 0; v < DEL; v++) p[v] /= sum;
  return p;
};
const argmax = (p) => {
  let best = 0;
  for (let v = 1; v < p.length; v++) if (p[v] > p[best]) best = v;
  return best;
};
const glyph = (t) => (t === 0 ? 'space' : decode(t));

export function createSteer({ box, typed, ghost, next, chips, help, predict }) {
  let gen = 0;
  let timer = 0;
  let enabled = true;
  let chipGen = 0;
  let accepted = '';

  const bars = Array.from({ length: 5 }, () => {
    const row = document.createElement('span');
    row.className = 'n';
    const ch = document.createElement('b');
    const track = document.createElement('i');
    const fill = document.createElement('u');
    track.append(fill);
    const pct = document.createElement('small');
    row.append(ch, track, pct);
    next.append(row);
    return { row, ch, fill, pct };
  });

  function paintNext(p) {
    const order = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, 5);
    order.forEach((t, i) => {
      const b = bars[i];
      b.ch.textContent = glyph(t);
      b.fill.style.transform = `scaleX(${p[t].toFixed(3)})`;
      b.pct.textContent = `${(p[t] * 100).toFixed(p[t] < 0.1 ? 1 : 0)}%`;
    });
    next.dataset.on = '1';
  }

  function clearNext() {
    delete next.dataset.on;
    bars.forEach((b) => { b.fill.style.transform = 'scaleX(0)'; });
  }

  function addGhost(t, p) {
    const s = document.createElement('span');
    s.className = 'g';
    s.textContent = decode(t);
    s.style.setProperty('--light', lightOf(p).toFixed(3));
    s.style.setProperty('--p', p.toFixed(3));
    ghost.append(s);
    accepted += decode(t);
  }

  async function run() {
    const mine = ++gen;
    const text = box.value;
    typed.textContent = text;
    if (!text.length) {
      ghost.replaceChildren();
      accepted = '';
      clearNext();
      return;
    }
    const ids = Array.from(encode(text));
    let fresh = true;
    for (let i = 0; i < GHOST; i++) {
      const logits = await predict(Uint32Array.from(ids));
      if (mine !== gen) return;
      const p = softmax(logits);
      const t = argmax(p);
      if (fresh) {
        // Replaced only once the new first character is ready, so the ghost never flickers empty while typing.
        ghost.replaceChildren();
        accepted = '';
        paintNext(p);
        fresh = false;
      }
      addGhost(t, p[t]);
      ids.push(t);
    }
  }

  function schedule(ms = DEBOUNCE_MS) {
    clearTimeout(timer);
    typed.textContent = box.value;
    timer = setTimeout(run, ms);
  }

  box.addEventListener('input', () => schedule());
  box.addEventListener('keydown', (e) => {
    const atEnd = box.selectionStart === box.value.length && box.selectionEnd === box.value.length;
    if (e.key === 'Enter') e.preventDefault();
    if (e.key === 'ArrowRight' && atEnd && accepted) {
      e.preventDefault();
      box.value = (box.value + accepted).slice(0, Number(box.maxLength));
      schedule(0);
    }
  });

  async function continuation(text, n) {
    const ids = Array.from(encode(text));
    let out = '';
    for (let i = 0; i < n; i++) {
      const t = argmax(softmax(await predict(Uint32Array.from(ids))));
      out += decode(t);
      ids.push(t);
    }
    return out;
  }
  const expected = ({ text, period }, n) => {
    let s = text;
    for (let i = 0; i < n; i++) s += s[s.length - period];
    return s.slice(text.length);
  };

  function put(text) {
    box.value = text;
    box.focus({ preventScroll: true });
    box.setSelectionRange(text.length, text.length);
    schedule(0);
  }

  // The chips the current weights have earned. Stale runs are dropped if the model changed underneath them.
  async function measureChips() {
    const mine = ++chipGen;
    const good = [];
    for (const c of CANDIDATES) {
      const got = await continuation(c.text, CHECK);
      if (mine !== chipGen) return;
      if (got === expected(c, CHECK)) good.push(c);
      await new Promise((r) => setTimeout(r));
    }
    const got = await continuation(STUMBLE.text, CHECK);
    if (mine !== chipGen) return;
    const stumbles = got !== expected(STUMBLE, CHECK);
    const picks = good.length > 3 ? [good[1], good[2], good[good.length - 1]] : good;
    const buttons = picks.map((c) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.textContent = c.text;
      b.addEventListener('click', () => put(c.text));
      return b;
    });
    if (stumbles && good.length) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip wrong';
      b.textContent = 'one it gets wrong';
      b.addEventListener('click', () => put(STUMBLE.text));
      buttons.push(b);
    }
    chips.replaceChildren(...buttons);
    chips.dataset.count = String(good.length);
    help(good.length);
  }

  return {
    refresh() { if (box.value) schedule(0); },
    measureChips,
    setEnabled(on, placeholder) {
      enabled = on;
      box.disabled = !on;
      box.placeholder = placeholder;
    },
    get enabled() { return enabled; },
    put,
    reset() {
      gen++;
      chipGen++;
      box.value = '';
      typed.textContent = '';
      ghost.replaceChildren();
      accepted = '';
      chips.replaceChildren();
      delete chips.dataset.count;
      clearNext();
    },
  };
}
