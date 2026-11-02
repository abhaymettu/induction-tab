// Checks a benchmark log before it enters the record table. A row must survive all of these:
// the config hash is the frozen tab config, the log's own curve reproduces its event steps, the wall times agree
// with the step rate, and a rerun of the same seeds on this machine lands on the same event steps within tolerance.
// Wall time itself is confirmed only by a rerun on the same adapter, which the row must say.
// `node tools/verify-record.mjs log.json`
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { serve } from './serve.mjs';
import { configHash, BENCH_SEEDS } from '../src/engine/bench.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const TOLERANCE_STEPS = Number(process.env.TOLERANCE_STEPS ?? 100);
const cfg = JSON.parse(fs.readFileSync(path.join(ROOT, 'configs/tab.json'), 'utf8'));
const log = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const fails = [];
if (log.v !== 1) fails.push(`unknown log version ${log.v}`);
if (log.configHash !== configHash(cfg)) fails.push('config hash is not the frozen tab config');
const seeds = log.runs.map((r) => r.seed);
if (BENCH_SEEDS.some((s) => !seeds.includes(s))) fails.push(`missing seeds, need ${BENCH_SEEDS}`);
for (const r of log.runs) {
  let last = false;
  let ev = null;
  for (const [step, adv] of r.curve) {
    const ok = adv >= 2;
    if (ok && last) { ev = step; break; }
    last = ok;
  }
  if (ev !== r.tEvent) fails.push(`seed ${r.seed}: the curve gives event ${ev}, the log says ${r.tEvent}`);
  const expected = r.msPerStep * (r.tEvent + 1);
  if (r.wallEventMs < 0.7 * expected) fails.push(`seed ${r.seed}: ${r.wallEventMs} ms is too fast for ${r.tEvent} steps at ${r.msPerStep} ms/step`);
}
const srv = await serve(ROOT);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage();
await page.goto(`${srv.url}/tools/harness.html`);
await page.waitForFunction('window.ready === true');
const rerun = await page.evaluate(async ({ cfg, seeds }) => {
  const { benchSeed, adapterInfo } = await import('/src/engine/bench.js');
  const { adapter, device } = await window.engine.getDevice();
  const t = await window.engine.createTrainer(device, cfg);
  const runs = [];
  for (const seed of seeds) runs.push(await benchSeed(device, t, seed));
  return { adapter: adapterInfo(adapter), runs };
}, { cfg, seeds });
await browser.close();
srv.close();
for (const r of log.runs) {
  const again = rerun.runs.find((x) => x.seed === r.seed);
  const diff = Math.abs(again.tEvent - r.tEvent);
  console.log(`seed ${r.seed}: log step ${r.tEvent}, rerun step ${again.tEvent}, log ${(r.wallEventMs / 1000).toFixed(2)} s, rerun ${(again.wallEventMs / 1000).toFixed(2)} s`);
  if (diff > TOLERANCE_STEPS) fails.push(`seed ${r.seed}: event step differs by ${diff} from the rerun`);
}
const sameAdapter = rerun.adapter.vendor === log.adapter.vendor && rerun.adapter.architecture === log.adapter.architecture;
console.log(sameAdapter ? 'Rerun used the same adapter class, so the wall times are comparable.' : `Rerun used ${rerun.adapter.vendor} ${rerun.adapter.architecture}, so wall times are unconfirmed.`);
if (fails.length) { console.error(`REJECT\n${fails.join('\n')}`); process.exit(1); }
console.log('ACCEPT');
