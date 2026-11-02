// The no-WebGPU path: a browser with no GPU plays the recording in the same composition and answers prompts with
// the real weights on the CPU. Also checks losing the device mid-run lands in the same place.
// `node tools/verify-fallback.mjs [--stills]`
import { chromium } from 'playwright';
import path from 'node:path';
import assert from 'node:assert/strict';
import { serve } from './serve.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const SHOTS = path.join(ROOT, 'scratch/shots');
const stills = process.argv.includes('--stills');
const srv = await serve(ROOT);

async function recordingFlow(browser, label, init) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2, colorScheme: 'dark' });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  if (init) await page.addInitScript(init);
  await page.goto(`${srv.url}/`);
  await page.waitForFunction(() => document.documentElement.dataset.phase === 'ready', null, { timeout: 30000 });
  const lede = await page.textContent('#lede');
  console.log(`${label}: lede ${JSON.stringify(lede)}`);
  assert.equal(await page.evaluate(() => document.documentElement.dataset.source), 'recording');
  assert.match(lede, /recent Chrome, Edge or Safari/);
  assert.match(await page.textContent('#gpu'), /^Recording from /);
  if (stills) await page.screenshot({ path: `${SHOTS}/recording-ready-1440.png` });
  await page.click('#start');
  await page.waitForFunction(() => 'arrived' in document.documentElement.dataset, null, { timeout: 60000 });
  console.log(`${label}: ${await page.textContent('#say')}`);
  assert.match(await page.textContent('#say'), /recorded on an Apple/);
  await page.fill('#box', 'abcabcabcabcab');
  await page.waitForFunction(() => document.querySelectorAll('#ghost .g').length >= 24, null, { timeout: 30000 });
  const ghost = await page.textContent('#ghost');
  console.log(`${label}: cpu ghost ${JSON.stringify(ghost)}`);
  assert.ok(ghost.startsWith('cabcabc'), 'CPU ghost text continues the pattern');
  if (stills) await page.screenshot({ path: `${SHOTS}/recording-typed-1440.png` });
  await page.click('[data-layer="1"]');
  await page.waitForFunction(() => /Heads silenced/.test(document.getElementById('sub').textContent), null, { timeout: 15000 });
  const sub = await page.textContent('#sub');
  console.log(`${label}: ${sub}`);
  assert.ok(Number(sub.match(/Copy advantage (-?\d+\.\d+)/)[1]) < 1, 'silencing layer 1 kills copying on the CPU');
  if (stills) { await page.waitForTimeout(400); await page.screenshot({ path: `${SHOTS}/recording-silenced-1440.png` }); }
  assert.deepEqual(errors, [], 'no page errors');
  await context.close();
}

try {
  const plain = await chromium.launch({ headless: true });
  await recordingFlow(plain, 'no navigator.gpu', () => { Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined }); });
  await plain.close();

  const noAdapter = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
  await recordingFlow(noAdapter, 'no adapter', () => { GPU.prototype.requestAdapter = async () => null; });

  const context = await noAdapter.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: 'dark' });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.addInitScript(() => {
    let lose;
    const lost = new Promise((r) => { lose = r; });
    window.loseDevice = () => lose({ reason: 'unknown', message: 'simulated' });
    Object.defineProperty(GPUDevice.prototype, 'lost', { get: () => lost });
  });
  await page.goto(`${srv.url}/`);
  await page.waitForFunction(() => document.documentElement.dataset.phase === 'ready', null, { timeout: 30000 });
  await page.click('#start');
  await page.waitForFunction(() => /Step [1-9]/.test(document.getElementById('sub').textContent), null, { timeout: 20000 });
  await page.evaluate(() => window.loseDevice());
  await page.waitForFunction(() => document.documentElement.dataset.source === 'recording' && document.documentElement.dataset.phase === 'ready', null, { timeout: 15000 });
  const lede = await page.textContent('#lede');
  console.log(`device lost mid-run: ${JSON.stringify(lede)}`);
  assert.match(lede, /lost its connection/);
  await page.click('#start');
  await page.waitForFunction(() => 'arrived' in document.documentElement.dataset, null, { timeout: 60000 });
  console.log(`device lost mid-run: ${await page.textContent('#say')}`);
  assert.deepEqual(errors, [], 'no page errors after losing the device');
  await context.close();
  await noAdapter.close();
  console.log('PASS');
} finally {
  srv.close();
}
