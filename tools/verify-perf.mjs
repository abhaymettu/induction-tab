// Throughput and frame times, a tight loop and the real page taking turns so both see the same machine.
// `node tools/verify-perf.mjs [rounds]`
import { chromium } from 'playwright';
import path from 'node:path';
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { serve } from './serve.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const cfg = JSON.parse(fs.readFileSync(path.join(ROOT, 'configs/tab.json'), 'utf8'));
const rounds = Number(process.argv[2] ?? 5);
const SEED = 1;
const AFTER_EVENT_MS = 3000;

const srv = await serve(ROOT);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });

const sorted = (xs) => [...xs].sort((a, b) => a - b);
const at = (xs, q) => sorted(xs)[Math.min(xs.length - 1, Math.floor(q * xs.length))];
const summary = (xs) => `median ${at(xs, 0.5).toFixed(2)}, range ${Math.min(...xs).toFixed(2)} to ${Math.max(...xs).toFixed(2)}`;
const frames = (xs) => `n=${xs.length} p50 ${at(xs, 0.5).toFixed(1)} p95 ${at(xs, 0.95).toFixed(1)} p99 ${at(xs, 0.99).toFixed(1)} max ${Math.max(...xs).toFixed(1)}`;

async function tightLoop() {
  const page = await browser.newPage();
  await page.goto(`${srv.url}/tools/harness.html`);
  await page.waitForFunction('window.ready === true');
  const r = await page.evaluate(async ({ cfg, seed }) => {
    const { trainLoop } = await import('/src/engine/run.js');
    const { device } = await window.engine.getDevice();
    const t = await window.engine.createTrainer(device, { ...cfg, seed });
    t.reset(seed);
    const res = await trainLoop(device, t, {
      budget: 3001,
      chunkSteps: () => 25,
      shouldStop: (s) => s.tEvent !== null,
    });
    t.destroy();
    device.destroy();
    return res;
  }, { cfg, seed: SEED });
  await page.close();
  return { tEvent: r.tEvent, ms: r.wallEvent };
}

async function realPage() {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: 'dark' });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${srv.url}/`);
  await page.waitForFunction(() => document.documentElement.dataset.phase === 'ready');
  await page.evaluate(() => {
    const w = (window.perf = { t0: 0, tEvent: 0, before: [], after: [], typed: [] });
    document.getElementById('start').addEventListener('click', () => { w.t0 = performance.now(); }, { capture: true });
    new MutationObserver(() => {
      if (!w.tEvent && 'arrived' in document.documentElement.dataset) w.tEvent = performance.now();
    }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-arrived'] });
    let last = performance.now();
    const f = (now) => {
      if (w.t0) (w.tEvent ? w.after : w.before).push(now - last);
      last = now;
      requestAnimationFrame(f);
    };
    requestAnimationFrame(f);
  });
  await page.click('#start');
  await page.waitForFunction(() => window.perf.tEvent > 0, null, { timeout: 60000 });
  const atEvent = await page.textContent('#sub');
  await page.waitForTimeout(AFTER_EVENT_MS);
  const r = await page.evaluate(() => ({ ms: window.perf.tEvent - window.perf.t0, before: window.perf.before.slice(2), after: window.perf.after }));
  assert.deepEqual(errors, []);
  await context.close();
  const m = atEvent.match(/Step (\d+) of 3000\. Copy advantage ([\d.]+) nats/);
  assert.ok(m && Number(m[2]) >= 2, `the event is real: ${atEvent}`);
  return { tEvent: Number(m[1]), ...r };
}

try {
  const tight = [];
  const live = [];
  for (let i = 0; i < rounds; i++) {
    const a = await tightLoop();
    const b = await realPage();
    assert.equal(a.tEvent, b.tEvent, 'both sides reach the event at the same step');
    tight.push(a);
    live.push(b);
    const perStep = (x) => x.ms / (x.tEvent + 1);
    console.log(`round ${i + 1}: step ${a.tEvent}. tight ${(a.ms / 1000).toFixed(2)} s, ${perStep(a).toFixed(2)} ms/step. page ${(b.ms / 1000).toFixed(2)} s, ${perStep(b).toFixed(2)} ms/step. frames ${frames(b.before)}`);
  }
  const perStep = (xs) => xs.map((x) => x.ms / (x.tEvent + 1));
  const gaps = live.map((b, i) => (b.ms / tight[i].ms - 1) * 100);
  console.log(`tight loop ms/step: ${summary(perStep(tight))}. seconds to event: ${summary(tight.map((x) => x.ms / 1000))}`);
  console.log(`page ms/step: ${summary(perStep(live))}. seconds to event: ${summary(live.map((x) => x.ms / 1000))}`);
  console.log(`page over tight loop, same round, percent: ${summary(gaps)}`);
  console.log(`frames while training, all rounds: ${frames(live.flatMap((x) => x.before))}`);
  console.log(`frames in the ${AFTER_EVENT_MS / 1000} s after the event, chips being measured: ${frames(live.flatMap((x) => x.after))}`);
} finally {
  await browser.close();
  srv.close();
}
