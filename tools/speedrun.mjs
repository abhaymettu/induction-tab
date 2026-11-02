// Runs seeds through the induction event in headless Chrome on this machine's GPU and prints the scalar.
// `npm run speedrun -- '{"d":64}' 1,2,3 [budget]`
import { chromium } from 'playwright';
import path from 'node:path';
import fs from 'node:fs';
import { serve } from './serve.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const cfg = JSON.parse(process.argv[2] ?? fs.readFileSync(path.join(ROOT, 'configs/tab.json'), 'utf8'));
const seeds = (process.argv[3] ?? '1').split(',').map(Number);
const budget = Number(process.argv[4] ?? 8000);
const launch = process.env.ADAPTER === 'swiftshader'
  ? { headless: true, args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'] }
  : { channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] };
const srv = await serve(ROOT);
const browser = await chromium.launch(launch);
const page = await browser.newPage();
page.on('pageerror', (e) => console.error(e.message));
await page.goto(`${srv.url}/tools/harness.html`);
await page.waitForFunction('window.ready === true');
const out = [];
for (const seed of seeds) {
  const r = await page.evaluate(async ({ cfg, seed, budget }) => {
    const { trainLoop } = await import('/src/engine/run.js');
    const { adapter, device } = await window.engine.getDevice();
    const t = await window.engine.createTrainer(device, { ...cfg, seed });
    t.reset(seed);
    const curve = [];
    const res = await trainLoop(device, t, {
      budget,
      chunkSteps: () => 25,
      shouldStop: (s, done) => s.tEvent !== null && done > s.tEvent + 100,
      onEval: (step, p) => curve.push({ step, copyAdv: +p.copyAdv.toFixed(3), prev: +Math.max(...p.scores[0].map((h) => h.prev)).toFixed(3), prefix: +Math.max(...p.scores[t.c.L - 1].map((h) => h.prefix)).toFixed(3) }),
    });
    const info = adapter.info;
    t.destroy(); device.destroy();
    return { ...res, curve, adapter: `${info.vendor} ${info.architecture}` };
  }, { cfg, seed, budget });
  const { curve, ...row } = r;
  out.push({ seed, ...row });
  console.log(`seed ${seed}: event step ${r.tEvent}, precursor step ${r.tPrev}, ${r.wallEvent ? (r.wallEvent / 1000).toFixed(1) + ' s to event' : 'no event'}, ${(r.wallMs / r.steps).toFixed(2)} ms/step, ${r.adapter}`);
  if (process.env.CURVE) console.log(curve.filter((_, i) => i % Number(process.env.CURVE) === 0).map((x) => `${x.step}:${x.copyAdv}/${x.prev}/${x.prefix}`).join(' '));
}
await browser.close();
srv.close();
console.log(JSON.stringify({ browser: browser.version(), cfg, out }));
