// The clock a person sees: the real page with drawing on, seeds 1 to 5, each beside a tight loop on the same seed so
// both sides see the same machine. Writes results/watched-m4pro.json. `node tools/watched.mjs [out.json] [rounds]`
import { chromium } from 'playwright';
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { serve } from './serve.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const cfg = JSON.parse(fs.readFileSync(path.join(ROOT, 'configs/tab.json'), 'utf8'));
const out = process.argv[2] ?? path.join(ROOT, 'results/watched-m4pro.json');
const SEEDS = [1, 2, 3, 4, 5];
const ROUNDS = Number(process.argv[3] ?? 3);
const clean = JSON.parse(fs.readFileSync(path.join(ROOT, 'results/bench-m4pro.txt'), 'utf8').trim().split('\n').at(-1)).out;
const VIEWPORT = { width: 1440, height: 900 };

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const at = (xs, q) => [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(q * xs.length))];
const gpuPercent = () => {
  const m = spawnSync('ioreg', ['-r', '-c', 'IOAccelerator', '-d', '1'], { encoding: 'utf8' }).stdout.match(/"Device Utilization %"=(\d+)/);
  return m ? Number(m[1]) : null;
};

const srv = await serve(ROOT);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });

async function tightLoop(seed) {
  const page = await browser.newPage();
  await page.goto(`${srv.url}/tools/harness.html`);
  await page.waitForFunction('window.ready === true');
  const r = await page.evaluate(async ({ cfg, seed }) => {
    const { trainLoop } = await import('/src/engine/run.js');
    const { device, adapter } = await window.engine.getDevice();
    const t = await window.engine.createTrainer(device, { ...cfg, seed });
    t.reset(seed);
    const res = await trainLoop(device, t, { budget: 3001, chunkSteps: () => 25, shouldStop: (s) => s.tEvent !== null });
    t.destroy();
    device.destroy();
    return { tEvent: res.tEvent, seconds: res.wallEvent / 1000, adapter: `${adapter.info.vendor} ${adapter.info.architecture}` };
  }, { cfg, seed });
  await page.close();
  return r;
}

// The first press of Start runs seed 1. Any other seed starts from the New seed button, which draws from Math.random.
async function watchedPage(seed) {
  const context = await browser.newContext({ viewport: VIEWPORT, colorScheme: 'dark' });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${srv.url}/`);
  await page.waitForFunction(() => document.documentElement.dataset.phase === 'ready');
  assert.equal(await page.evaluate(() => document.documentElement.dataset.source), 'live');
  await page.evaluate(() => {
    const w = (window.watch = { t0: 0, arrived: 0, frames: [] });
    new MutationObserver(() => {
      if (!w.arrived && 'arrived' in document.documentElement.dataset) w.arrived = performance.now();
    }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-arrived'] });
    let last = performance.now();
    const tick = (now) => {
      if (w.t0 && !w.arrived) w.frames.push(now - last);
      last = now;
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
  await page.click('#start');
  if (seed !== 1) {
    await page.evaluate((seed) => {
      const real = Math.random;
      Math.random = () => { Math.random = real; return (seed - 0.5) / 1e6; };
    }, seed);
    await page.click('#reseed');
  }
  await page.evaluate(() => { window.watch.t0 = performance.now(); });
  await page.waitForFunction(() => window.watch.arrived > 0, null, { timeout: 120000 });
  const sub = await page.textContent('#sub');
  const head = await page.textContent('#resultHead');
  const frames = await page.evaluate(() => window.watch.frames.slice(2));
  const cells = await page.evaluate(() => document.getElementById('stream').childElementCount);
  assert.deepEqual(errors, []);
  await context.close();
  const m = sub.match(/Step (\d+) of 3000\. Copy advantage ([\d.]+) nats/);
  assert.ok(m && Number(m[2]) >= 2, `the event is real: ${sub}`);
  return { tEvent: Number(m[1]), clock: Number(head.match(/([\d.]+) seconds/)[1]), frames, streamChildren: cells };
}

try {
  const runs = [];
  const gpu = [gpuPercent()];
  let adapter = null;
  for (let round = 1; round <= ROUNDS; round++) for (const seed of SEEDS) {
    const tight = await tightLoop(seed);
    gpu.push(gpuPercent());
    const live = await watchedPage(seed);
    gpu.push(gpuPercent());
    assert.equal(tight.tEvent, live.tEvent, `seed ${seed}: both sides reach the event at the same step`);
    assert.ok(live.frames.length > 50, `seed ${seed}: the page drew while it trained, ${live.frames.length} frames`);
    assert.ok(live.streamChildren > 0, `seed ${seed}: the stream drew`);
    adapter = tight.adapter;
    runs.push({ round, seed, tEvent: live.tEvent, clockSeconds: live.clock, drawingOffSeconds: Number(tight.seconds.toFixed(2)), cleanBenchmarkSeconds: Number((clean.find((c) => c.seed === seed).wallEvent / 1000).toFixed(2)), framesWhileTraining: live.frames.length, frameP50Ms: Number(at(live.frames, 0.5).toFixed(1)), frameP95Ms: Number(at(live.frames, 0.95).toFixed(1)) });
    console.log(JSON.stringify(runs.at(-1)));
  }
  const result = {
    what: 'Clock at the event on the real page with drawing on, beside a tight loop with drawing off on the same seed, alternating.',
    command: 'node tools/watched.mjs',
    browser: await browser.version(),
    headless: true,
    adapter,
    viewport: VIEWPORT,
    measuredAt: new Date().toISOString(),
    rounds: ROUNDS,
    gpuDeviceUtilizationPercent: gpu,
    runs,
    medianClockSeconds: median(runs.map((r) => r.clockSeconds)),
    // Another job's bursts only ever add time, so each seed's fastest round is its least-loaded reading.
    medianLeastLoadedClockSeconds: median(SEEDS.map((seed) => Math.min(...runs.filter((r) => r.seed === seed).map((r) => r.clockSeconds)))),
    medianDrawingOffSeconds: median(runs.map((r) => r.drawingOffSeconds)),
    medianCleanBenchmarkSeconds: median(runs.map((r) => r.cleanBenchmarkSeconds)),
    // The drawing-off side against the stored clean benchmark says whether something else was on the GPU.
    otherLoadFactor: Number(median(runs.map((r) => r.drawingOffSeconds / r.cleanBenchmarkSeconds)).toFixed(2)),
  };
  fs.writeFileSync(out, `${JSON.stringify(result, null, 1)}\n`);
  console.log(`median clock ${result.medianClockSeconds} s, least loaded ${result.medianLeastLoadedClockSeconds} s, drawing off beside it ${result.medianDrawingOffSeconds} s, ${result.otherLoadFactor} times the clean benchmark, GPU busy ${gpu.join(' ')}`);
} finally {
  await browser.close();
  srv.close();
}
