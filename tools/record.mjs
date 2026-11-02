// Saves a real run of the tab config for browsers without WebGPU: eval curves, 8-bit attention snapshots around
// the transition, the probe tokens and the final weights, under assets/recording/. Runs the seed twice, once for
// honest timings and once with snapshots, then checks the JS forward pass against t.predict on those weights.
// `node tools/record.mjs [seed]`
import { chromium } from 'playwright';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import zlib from 'node:zlib';
import assert from 'node:assert/strict';
import { serve } from './serve.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const OUT = path.join(ROOT, 'assets/recording');
const cfg = JSON.parse(fs.readFileSync(path.join(ROOT, 'configs/tab.json'), 'utf8'));
const seed = Number(process.argv[2] ?? 1);
const LIMIT_BYTES = 3 * 1024 * 1024;

const srv = await serve(ROOT);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error(e.message));
  await page.goto(`${srv.url}/tools/harness.html`);
  await page.waitForFunction('window.ready === true');

  const run = await page.evaluate(async ({ cfg, seed }) => {
    const { trainLoop } = await import('/src/engine/run.js');
    const { adapterInfo } = await import('/src/engine/bench.js');
    const { createCpuModel } = await import('/src/engine/cpu.js');
    const { copyProbe } = await import('/src/engine/lang.js');
    const { encode } = await import('/src/engine/text.js');
    const { adapter, device } = await window.engine.getDevice();
    const t = await window.engine.createTrainer(device, { ...cfg, seed });
    const round = (x) => +x.toFixed(3);
    const half = t.c.probeT / 2;
    const budget = 3001;

    t.reset(seed);
    const evals = [];
    const first = await trainLoop(device, t, {
      budget,
      chunkSteps: () => 1,
      onEval: (step, p, _s, wall) => evals.push({
        step,
        ms: Math.round(wall),
        trainLoss: round(p.trainLoss),
        copyAdv: round(p.copyAdv),
        scores: p.scores.map((l) => l.map((h) => ({ prev: round(h.prev), prefix: round(h.prefix) }))),
        seqLoss: Array.from(p.seqLoss.subarray(half - 1, 2 * half - 1), round),
      }),
    });

    const E = t.c.evalEvery;
    const ev = first.tEvent;
    const frameSteps = evals.map((e) => e.step).filter((s) => {
      if (s <= ev - 150) return s % 100 === 0;
      if (s <= ev + 150) return true;
      if (s <= ev + 650) return s % 100 === 0;
      return s % 400 === 0 || s === budget - 1;
    });

    t.reset(seed);
    const frames = [];
    const n = t.view.size / 4;
    const encodeFrame = async () => {
      const v = new Float32Array(await t.readback(t.view));
      const bytes = new Uint8Array(n);
      for (let i = 0; i < n; i++) bytes[i] = Math.round(Math.min(1, v[i]) * 255);
      let s = '';
      for (let i = 0; i < n; i += 8192) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
      return btoa(s);
    };
    const pending = new Set(frameSteps);
    const second = await trainLoop(device, t, {
      budget,
      chunkSteps: () => E,
      onChunk: async (done) => {
        const step = done - 1;
        if (pending.has(step)) { pending.delete(step); frames.push(await encodeFrame()); }
      },
    });

    const P = new Float32Array(await t.readback(t.P));
    const model = createCpuModel({ ...cfg }, P);
    const prompts = ['abcabcabcabcab', '0123456789 0123456', 'qwertyuiop[] qwertyu', 'the quick brown fox. the quick br'];
    const checks = [];
    for (const masks of [[0, 0], [0, 15], [15, 0]]) {
      t.setAblation(masks);
      for (const text of prompts) {
        const ids = encode(text);
        const gpu = await t.predict(ids);
        const cpu = model.predict(ids, masks);
        let worst = 0;
        let bestG = 0;
        let bestC = 0;
        for (let v = 0; v < t.c.V; v++) {
          worst = Math.max(worst, Math.abs(gpu[v] - cpu[v]));
          if (gpu[v] > gpu[bestG]) bestG = v;
          if (cpu[v] > cpu[bestC]) bestC = v;
        }
        checks.push({ text, masks, worst, sameTop: bestG === bestC });
      }
    }
    t.setAblation([0, 0]);

    const staging = device.createBuffer({ size: t.probeOut.size, usage: 0x01 | 0x08 });
    const enc = device.createCommandEncoder();
    t.encodeSteps(enc, 0, true);
    enc.copyBufferToBuffer(t.probeOut, 0, staging, 0, t.probeOut.size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    const gpuAdv = t.parseProbe(staging.getMappedRange().slice(0)).copyAdv;
    const cpuAdv = model.probe(copyProbe(t.c.V, t.c.probeB, t.c.probeT), t.c.probeB, [0, 0]).copyAdv;

    let w = '';
    const wBytes = new Uint8Array(P.buffer);
    for (let i = 0; i < wBytes.length; i += 8192) w += String.fromCharCode.apply(null, wBytes.subarray(i, i + 8192));

    const out = {
      adapter: adapterInfo(adapter),
      tEvent: first.tEvent,
      tPrev: first.tPrev,
      wallEvent: Math.round(first.wallEvent),
      secondEvent: second.tEvent,
      evals,
      frameSteps,
      frames,
      weights: btoa(w),
      checks,
      gpuAdv,
      cpuAdv,
      tokens: Array.from(copyProbe(t.c.V, 1, t.c.probeT)),
    };
    t.destroy();
    device.destroy();
    return out;
  }, { cfg, seed });

  for (const c of run.checks) console.log(`cpu vs gpu ${JSON.stringify(c.text)} masks ${c.masks}: worst logit gap ${c.worst.toExponential(2)}, same top ${c.sameTop}`);
  console.log(`copy advantage over the full probe batch: gpu ${run.gpuAdv.toFixed(4)}, cpu ${run.cpuAdv.toFixed(4)}`);
  assert.equal(run.secondEvent, run.tEvent, 'both passes hit the event at the same step');
  assert.equal(run.frames.length, run.frameSteps.length, 'one snapshot per chosen step');
  assert.ok(run.checks.every((c) => c.sameTop && c.worst < 1e-2), 'the JS forward pass agrees with t.predict');
  assert.ok(Math.abs(run.gpuAdv - run.cpuAdv) < 1e-3, 'the JS probe agrees with the GPU probe');

  fs.mkdirSync(OUT, { recursive: true });
  const frames = Buffer.concat(run.frames.map((f) => Buffer.from(f, 'base64')));
  const framesGz = zlib.gzipSync(frames, { level: 9 });
  const weights = Buffer.from(run.weights, 'base64');
  const meta = {
    machine: os.cpus()[0].model,
    browser: `Chrome ${browser.version().split('.')[0]}`,
    adapter: run.adapter,
    cfg,
    seed,
    tEvent: run.tEvent,
    tPrev: run.tPrev,
    wallEvent: run.wallEvent,
    tokens: run.tokens,
    frames: run.frameSteps,
    evals: run.evals,
  };
  const metaText = JSON.stringify(meta);
  fs.writeFileSync(path.join(OUT, 'meta.json'), metaText);
  fs.writeFileSync(path.join(OUT, 'frames.bin.gz'), framesGz);
  fs.writeFileSync(path.join(OUT, 'weights.bin'), weights);
  const total = framesGz.length + weights.length + metaText.length;
  console.log(`event at step ${run.tEvent} in ${(run.wallEvent / 1000).toFixed(1)} s on ${meta.machine}. ${run.frameSteps.length} frames.`);
  console.log(`meta ${metaText.length} B, frames ${framesGz.length} B gz of ${frames.length}, weights ${weights.length} B, total ${(total / 1048576).toFixed(2)} MB`);
  assert.ok(total < LIMIT_BYTES, 'recording stays under 3 MB');
} finally {
  await browser.close();
  srv.close();
}
