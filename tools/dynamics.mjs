// Trains seeds of one config in the WGSL trainer and writes the same JSONL as ref/train.py, resumable, so browser and
// native runs compare row for row. `node tools/dynamics.mjs '<cfg json>' 1-64 6000 results/out.jsonl [maxMinutes]`
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { serve } from './serve.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const cfg = JSON.parse(process.argv[2]);
const [a, b] = process.argv[3].split('-').map(Number);
const budget = Number(process.argv[4]);
const out = process.argv[5];
const maxMs = Number(process.argv[6] ?? 1e9) * 60000;
const stopAfter = Number(process.env.STOP_AFTER ?? 0);
const done = new Set(fs.existsSync(out) ? fs.readFileSync(out, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l).seed) : []);
const srv = await serve(ROOT);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage();
page.on('pageerror', (e) => console.error(e.message));
await page.goto(`${srv.url}/tools/harness.html`);
await page.waitForFunction('window.ready === true');
await page.evaluate(async (cfg) => {
  const { adapter, device } = await window.engine.getDevice();
  window.ctx = { adapter, device, t: await window.engine.createTrainer(device, cfg) };
}, cfg);
const t0 = Date.now();
for (let seed = a; seed <= b; seed++) {
  if (done.has(seed)) continue;
  if (Date.now() - t0 > maxMs) break;
  const r = await page.evaluate(async ({ seed, budget, stopAfter }) => {
    const { trainLoop } = await import('/src/engine/run.js');
    const { device, t, adapter } = window.ctx;
    t.reset(seed);
    const curve = [];
    const res = await trainLoop(device, t, {
      budget,
      chunkSteps: () => 25,
      shouldStop: (s, done) => s.tEvent !== null && done > s.tEvent + stopAfter,
      onEval: (step, p) => curve.push({ step, loss: p.trainLoss, copyAdv: p.copyAdv, firstPass: p.firstPass, secondPass: p.secondPass, prev: p.scores.map((l) => Math.max(...l.map((h) => h.prev))), prefix: p.scores.map((l) => Math.max(...l.map((h) => h.prefix))) }),
    });
    return { t_event: res.tEvent, t_pv: res.tPrev, wall_s: res.wallMs / 1000, wall_event_s: res.wallEvent === null ? null : res.wallEvent / 1000, steps_run: res.steps, curve, adapter: `${adapter.info.vendor} ${adapter.info.architecture}` };
  }, { seed, budget, stopAfter });
  fs.appendFileSync(out, `${JSON.stringify({ cfg, seed, budget, ...r })}\n`);
  console.log(seed, r.t_event, r.t_pv, r.wall_s.toFixed(1), (1000 * r.wall_s / r.steps_run).toFixed(1) + ' ms/step');
}
await browser.close();
srv.close();
