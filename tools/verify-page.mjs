// Drives the real page on the real GPU: start, event, typed prompt, silencing, frame times, stills and a clip.
// `node tools/verify-page.mjs [--stills] [--clip] [--chunk N]`
import { chromium } from 'playwright';
import path from 'node:path';
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { serve } from './serve.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const SHOTS = path.join(ROOT, 'scratch/shots');
const flag = (name) => process.argv.includes(name);
fs.mkdirSync(SHOTS, { recursive: true });

const percentile = (xs, q) => [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(xs.length * q))];
const LAUNCH = { channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] };

const srv = await serve(ROOT);
const browser = await chromium.launch(LAUNCH);
try {
  const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2, colorScheme: 'dark' })).newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !/404/.test(m.text())) errors.push(m.text()); });
  await page.goto(`${srv.url}/`);
  await page.waitForFunction(() => document.documentElement.dataset.phase === 'ready', null, { timeout: 30000 });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(60);
  const bootMs = await page.evaluate(() => Math.round(performance.now()));
  if (flag('--stills')) await page.screenshot({ path: `${SHOTS}/ready-1440-dark.png` });

  await page.evaluate(() => {
    window.deltas = [];
    let last = performance.now();
    const f = (now) => { window.deltas.push(now - last); last = now; if (!window.stopFrames) requestAnimationFrame(f); };
    requestAnimationFrame(f);
  });
  await page.click('#start');
  await page.waitForFunction(() => 'arrived' in document.documentElement.dataset, null, { timeout: 60000 });
  const atEvent = await page.evaluate(() => ({ say: document.getElementById('say').textContent, clock: document.getElementById('clock').textContent, sub: document.getElementById('sub').textContent, n: window.deltas.length }));
  console.log('event:', JSON.stringify(atEvent));
  assert.match(atEvent.say, /^Induction in \d+\.\d seconds on your .*GPU\.$/);
  if (flag('--stills')) await page.screenshot({ path: `${SHOTS}/event-1440-dark.png` });

  const trainingDeltas = await page.evaluate(() => window.deltas.slice(2));
  const eventSeconds = Number(atEvent.clock);
  console.log(`frames to event: n=${trainingDeltas.length} p50=${percentile(trainingDeltas, 0.5).toFixed(1)} p95=${percentile(trainingDeltas, 0.95).toFixed(1)} p99=${percentile(trainingDeltas, 0.99).toFixed(1)} max=${Math.max(...trainingDeltas).toFixed(1)} ms, event at ${eventSeconds} s`);

  if (flag('--quick')) { console.log('quick stop'); process.exitCode = 0; await browser.close(); srv.close(); process.exit(0); }
  await page.waitForFunction(() => document.documentElement.dataset.phase === 'done', null, { timeout: 120000 });
  const doneSub = await page.textContent('#sub');
  console.log('done:', doneSub);
  await page.evaluate(() => { window.stopFrames = true; });
  const stepsSub = doneSub.match(/step (\d+)/);
  assert.ok(stepsSub && Number(stepsSub[1]) >= 3000, 'training stops itself near step 3000');

  await page.fill('#box', 'abcabcabcabcab');
  await page.waitForFunction(() => document.querySelectorAll('#ghost .g').length >= 24, null, { timeout: 10000 });
  const ghost = await page.textContent('#ghost');
  console.log('ghost:', JSON.stringify(ghost));
  assert.ok(ghost.startsWith('cabcabc'), `ghost continues the pattern, got ${ghost}`);
  const next = await page.$$eval('#next .n', (ns) => ns.map((n) => n.textContent));
  console.log('next:', next.join(' | '));
  assert.ok(next[0].startsWith('c'), 'top next character is c');
  await page.waitForFunction(() => document.querySelectorAll('#chips .chip').length >= 1, null, { timeout: 20000 });
  console.log('chips:', (await page.$$eval('#chips .chip', (c) => c.map((x) => x.textContent))).join(' | '));
  if (flag('--stills')) await page.screenshot({ path: `${SHOTS}/typed-1440-dark.png` });

  const advOf = async () => Number((await page.textContent('#sub')).match(/Copy advantage (-?\d+\.\d+)/)[1]);
  const base = await advOf();
  const silenceThenRead = async (selector, label) => {
    await page.click(selector);
    await page.waitForFunction(() => /Heads silenced/.test(document.getElementById('sub').textContent), null, { timeout: 5000 });
    const adv = await advOf();
    const g = await page.textContent('#ghost');
    console.log(`${label}: copy advantage ${adv} from ${base}; ghost ${JSON.stringify(g)}`);
    return adv;
  };
  const mapBox = await page.$$eval('.head', (hs) => hs.map((h) => { const r = h.getBoundingClientRect(); return { x: r.x + 2, y: r.y + 2, width: r.width - 4, height: r.height - 4 }; }));
  const brightness = async (i) => {
    const shot = await page.screenshot({ clip: mapBox[i] });
    return page.evaluate(async (b64) => {
      const bmp = await createImageBitmap(await (await fetch(`data:image/png;base64,${b64}`)).blob());
      const c = new OffscreenCanvas(bmp.width, bmp.height).getContext('2d', { willReadFrequently: true });
      c.drawImage(bmp, 0, 0);
      const d = c.getImageData(0, 0, bmp.width, bmp.height).data;
      let sum = 0;
      for (let k = 0; k < d.length; k += 4) sum += d[k];
      return sum / (d.length / 4);
    }, shot.toString('base64'));
  };
  const before = await brightness(4);
  const a1 = await silenceThenRead('[data-layer="1"]', 'silence layer 1');
  assert.ok(a1 < 1, `silencing layer 1 kills copying, got ${a1}`);
  await page.waitForTimeout(350);
  const after = await brightness(4);
  const background = await brightness(0) * 0 + await page.evaluate(() => { const c = document.createElement('canvas').getContext('2d'); c.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg'); c.fillRect(0, 0, 1, 1); return c.getImageData(0, 0, 1, 1).data[0]; });
  console.log(`layer 1 head 1 mean red: ${before.toFixed(1)} before, ${after.toFixed(1)} silenced (page background ${background})`);
  assert.ok(after - background < 1.5 && before - background > 5, 'a silenced map goes dark on screen');
  await page.waitForFunction(() => !document.getElementById('ghost').textContent.startsWith('cabcabc'), null, { timeout: 5000 });
  if (flag('--stills')) await page.screenshot({ path: `${SHOTS}/silenced-1440-dark.png` });
  await page.click('#restore');
  await page.waitForFunction(() => /All heads back/.test(document.getElementById('sub').textContent), null, { timeout: 5000 });
  const back = await advOf();
  console.log('restored:', back);
  assert.ok(Math.abs(back - base) < 0.01, `restoring brings the reading back, ${back} vs ${base}`);
  const a0 = await silenceThenRead('[data-layer="0"]', 'silence layer 0');
  assert.ok(a0 < 1, `silencing layer 0 kills copying, got ${a0}`);
  await page.click('#restore');
  await page.waitForFunction(() => /All heads back/.test(document.getElementById('sub').textContent), null, { timeout: 5000 });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
  assert.equal(overflow, 0, 'no horizontal scroll at 1440');
  assert.deepEqual(errors, [], 'no page errors');
  console.log('boot ms', bootMs, 'errors', errors.length, 'PASS');
} finally {
  await browser.close();
  srv.close();
}
