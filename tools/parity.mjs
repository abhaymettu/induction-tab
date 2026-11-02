// Runs the same weights and batches through PyTorch (autograd) and the WGSL trainer in headless Chrome,
// then compares every parameter tensor. One command: `npm run parity`.
import { chromium } from 'playwright';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { serve } from './serve.mjs';
import { layoutOf, resolve } from '../src/engine/layout.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const STEPS = Number(process.env.PARITY_STEPS ?? 20);
const CONFIGS = {
  small: { V: 128, T: 32, d: 32, H: 4, L: 2, unique: 2, B: 4, probeB: 8, probeT: 32 },
  looped: { V: 128, T: 32, d: 32, H: 2, L: 4, unique: 2, B: 4, probeB: 8, probeT: 32 },
  attnonly: { V: 128, T: 32, d: 32, H: 4, L: 2, unique: 2, B: 4, mlp: 0, ln: false, probeB: 8, probeT: 32 },
  demo: { V: 512, T: 128, d: 128, H: 4, L: 2, unique: 2, B: 8, probeB: 16, probeT: 128 },
  tab: JSON.parse(fs.readFileSync(new URL('../configs/tab.json', import.meta.url), 'utf8')),
  ...JSON.parse(process.env.PARITY_EXTRA ?? '{}'),
};
const names = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(CONFIGS);
const adapterArg = process.env.PARITY_ADAPTER === 'swiftshader'
  ? { headless: true, args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader'] }
  : { channel: process.env.PARITY_CHANNEL ?? 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] };

const srv = await serve(ROOT);
const browser = await chromium.launch(adapterArg);
const page = await browser.newPage();
page.on('console', (m) => { if (m.type() === 'error') console.error('browser:', m.text()); });
page.on('pageerror', (e) => console.error('pageerror:', e.message));
await page.goto(`${srv.url}/tools/harness.html`);
await page.waitForFunction('window.ready === true');

const fromFile = (f) => { const b = fs.readFileSync(f); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
const decode = (s) => { const b = Buffer.from(s, 'base64'); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };

function compare(lay, a, b) {
  return lay.specs.map((s) => {
    let num = 0; let den = 0; let maxAbs = 0; let maxB = 0;
    for (let i = s.off; i < s.off + s.size; i++) {
      const e = a[i] - b[i];
      num += e * e; den += b[i] * b[i];
      maxAbs = Math.max(maxAbs, Math.abs(e));
      maxB = Math.max(maxB, Math.abs(b[i]));
    }
    return { name: s.name, relL2: den ? Math.sqrt(num / den) : Math.sqrt(num), maxRel: maxB ? maxAbs / maxB : maxAbs };
  });
}

const report = { adapter: null, browser: browser.version(), steps: STEPS, configs: {} };
let worst = 0;
for (const name of names) {
  const cfg = CONFIGS[name];
  const c = resolve(cfg);
  const lay = layoutOf(c);
  const dir = path.join(ROOT, 'scratch/parity', name);
  fs.mkdirSync(dir, { recursive: true });
  execFileSync(path.join(ROOT, '.venv/bin/python'), [path.join(ROOT, 'ref/parity_ref.py'), '--cfg', JSON.stringify(cfg), '--steps', String(STEPS), '--out', dir], { stdio: 'inherit' });
  const ref = JSON.parse(fs.readFileSync(path.join(dir, 'ref.json')));
  const res = await page.evaluate(async ({ cfg, STEPS }) => {
    const b64 = (ab) => { let s = ''; const u = new Uint8Array(ab); for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode(...u.subarray(i, i + 32768)); return btoa(s); };
    const { adapter, device } = await window.engine.getDevice();
    const errors = [];
    device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
    const t = await window.engine.createTrainer(device, cfg);
    t.reset(t.c.seed);
    const run = (phases) => { const enc = device.createCommandEncoder(); const pass = enc.beginComputePass(); t.train.encode(pass, t.opsFor(phases)); pass.end(); device.queue.submit([enc.finish()]); };
    run(['data', 'fwd', 'loss', 'bwd', 'tick']);
    const tok0 = b64(await t.readback(t.tok));
    const grad0 = b64(await t.readback(t.G));
    run(['opt']);
    const p1 = b64(await t.readback(t.P));
    for (let i = 1; i < STEPS; i++) { const enc = device.createCommandEncoder(); t.encodeSteps(enc, 1); device.queue.submit([enc.finish()]); }
    const pN = b64(await t.readback(t.P));
    const hist = Array.from(new Float32Array(await t.readback(t.hist, STEPS * 4)));
    run(['data', 'fwd', 'loss', 'bwd']);
    const gradN = b64(await t.readback(t.G));
    const enc = device.createCommandEncoder(); const pass = enc.beginComputePass(); t.probe.encode(pass); pass.end(); device.queue.submit([enc.finish()]);
    const probe = t.parseProbe(await t.readback(t.probeOut));
    const info = adapter.info;
    t.destroy(); device.destroy();
    return { tok0, grad0, p1, pN, gradN, hist, probe, errors, adapter: `${info.vendor} ${info.architecture}` };
  }, { cfg, STEPS });
  if (res.errors.length) { console.error(res.errors.slice(0, 3).join('\n')); process.exitCode = 2; }
  report.adapter = res.adapter;
  const tokA = new Uint32Array(Buffer.from(res.tok0, 'base64').buffer.slice(0));
  const tb = fs.readFileSync(path.join(dir, 'tok0.bin'));
  const tokB = new Uint32Array(tb.buffer.slice(tb.byteOffset, tb.byteOffset + tb.length));
  let tokSame = 0;
  for (let i = 0; i < tokB.length; i++) if (Buffer.from(res.tok0, 'base64').readUInt32LE(i * 4) === tokB[i]) tokSame++;
  const stages = { grad0: compare(lay, decode(res.grad0), fromFile(path.join(dir, 'grad0.bin'))), p1: compare(lay, decode(res.p1), fromFile(path.join(dir, 'p1.bin'))), pN: compare(lay, decode(res.pN), fromFile(path.join(dir, 'pN.bin'))), gradN: compare(lay, decode(res.gradN), fromFile(path.join(dir, 'gradN.bin'))) };
  const summary = Object.fromEntries(Object.entries(stages).map(([k, v]) => [k, { worstRelL2: Math.max(...v.map((x) => x.relL2)), worstName: v.reduce((a, b) => (b.relL2 > a.relL2 ? b : a)).name }]));
  const lossErr = Math.max(...ref.losses.map((l, i) => Math.abs(l - res.hist[i])));
  const copyErr = Math.abs(ref.probe.copyAdv - res.probe.copyAdv);
  report.configs[name] = { cfg, params: lay.total, tokensEqual: `${tokSame}/${tokB.length}`, summary, lossMaxAbsErr: lossErr, copyAdvAbsErr: copyErr, perTensorGrad0: stages.grad0 };
  worst = Math.max(worst, summary.grad0.worstRelL2, summary.gradN.worstRelL2);
  console.log(`${name.padEnd(9)} params ${String(lay.total).padStart(8)} tokens ${tokSame}/${tokB.length} grad0 ${summary.grad0.worstRelL2.toExponential(2)} (${summary.grad0.worstName}) p1 ${summary.p1.worstRelL2.toExponential(2)} p${STEPS} ${summary.pN.worstRelL2.toExponential(2)} (${summary.pN.worstName}) grad${STEPS} ${summary.gradN.worstRelL2.toExponential(2)} (${summary.gradN.worstName}) loss ${lossErr.toExponential(1)} copyAdv ${copyErr.toExponential(1)}`);
}
await browser.close();
srv.close();
fs.mkdirSync(path.join(ROOT, 'results'), { recursive: true });
const out = path.join(ROOT, `results/parity-${report.adapter.replace(/\W+/g, '-')}.json`);
fs.writeFileSync(out, JSON.stringify(report, null, 1));
console.log(`adapter ${report.adapter}, browser ${report.browser}, worst gradient relative L2 ${worst.toExponential(2)}; wrote ${path.relative(ROOT, out)}`);
