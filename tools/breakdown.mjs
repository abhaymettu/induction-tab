// Where one training step spends its time, using the sequential-dispatch method of arXiv 2608.08730: replay a
// subset of the step's dispatches R times per submit, wait for completion, and divide. Chrome quantizes
// timestamp queries to 100 us, coarser than one dispatch, so raw timestamps are not used.
// `node tools/breakdown.mjs ['{"d":64}']`
import { chromium } from 'playwright';
import path from 'node:path';
import fs from 'node:fs';
import { serve } from './serve.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const cfg = JSON.parse(process.argv[2] ?? fs.readFileSync(path.join(ROOT, 'configs/tab.json'), 'utf8'));
const srv = await serve(ROOT);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage();
page.on('pageerror', (e) => console.error(e.message));
await page.goto(`${srv.url}/tools/harness.html`);
await page.waitForFunction('window.ready === true');
const res = await page.evaluate(async (cfg) => {
  const { adapter, device } = await window.engine.getDevice();
  const t = await window.engine.createTrainer(device, cfg);
  t.reset(1);
  const ops = t.train.ops;
  const replay = async (subset, reps, rounds = 6) => {
    const samples = [];
    for (let r = 0; r < rounds; r++) {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let i = 0; i < reps; i++) t.train.encode(pass, subset);
      pass.end();
      const cb = enc.finish();
      const s = performance.now();
      device.queue.submit([cb]);
      await device.queue.onSubmittedWorkDone();
      samples.push((performance.now() - s) / reps);
    }
    samples.sort((a, b) => a - b);
    return samples[0];
  };
  for (let i = 0; i < 3; i++) { const enc = device.createCommandEncoder(); t.encodeSteps(enc, 2); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone(); }
  const tickOp = ops.find((o) => o.kernel === 'TICK');
  const overheadMs = (await replay(new Array(200).fill(tickOp), 5)) / 200;
  const full = await replay(ops, 5);
  const phases = {};
  for (const ph of ['data', 'fwd', 'loss', 'bwd', 'tick', 'opt']) {
    const sub = ops.filter((o) => o.phase === ph);
    phases[ph] = { dispatches: sub.length, ms: await replay(sub, 5) };
  }
  const kernels = {};
  for (const name of new Set(ops.map((o) => o.kernel))) {
    const sub = ops.filter((o) => o.kernel === name);
    kernels[name] = { dispatches: sub.length, ms: await replay(sub, 5) };
  }
  // Each GEMM on its own, largest first, to see which matmuls dominate.
  const gemms = [];
  for (const o of ops.filter((x) => x.kernel === 'GEMM')) {
    gemms.push({ phase: o.phase, wg: o.x * o.y * o.z, ms: await replay([o], 5, 4) });
  }
  gemms.sort((a, b) => b.ms - a.ms);
  const info = adapter.info;
  const out = { adapter: `${info.vendor} ${info.architecture}`, params: t.lay.total, dispatches: ops.length, stepMs: full, overheadMsPerDispatch: overheadMs, phases, kernels, gemms: gemms.slice(0, 8) };
  t.destroy(); device.destroy();
  return out;
}, cfg);
await browser.close();
srv.close();
const f = (x) => x.toFixed(3).padStart(8);
console.log(`${res.adapter}, chrome ${browser.version()}, ${res.params} params, ${res.dispatches} dispatches`);
console.log(`step ${res.stepMs.toFixed(2)} ms; empty-dispatch overhead ${(res.overheadMsPerDispatch * 1000).toFixed(1)} us each, ${(res.overheadMsPerDispatch * res.dispatches).toFixed(2)} ms per step`);
for (const [k, v] of Object.entries(res.phases)) console.log(`  ${k.padEnd(12)} ${String(v.dispatches).padStart(3)} ${f(v.ms)} ms`);
for (const [k, v] of Object.entries(res.kernels).sort((a, b) => b[1].ms - a[1].ms)) console.log(`  ${k.padEnd(12)} ${String(v.dispatches).padStart(3)} ${f(v.ms)} ms`);
console.log('  top GEMM: ' + res.gemms.map((g) => `${g.phase} ${g.ms.toFixed(2)}`).join(', '));
if (process.env.OUT) fs.writeFileSync(process.env.OUT, JSON.stringify({ cfg, browser: browser.version(), ...res }, null, 1));
