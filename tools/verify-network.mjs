// Drives the page through everything a reader does and fails if anything reaches a host other than the page's own.
// A link the reader clicks is not a request, so links are not checked. `node tools/verify-network.mjs [url]`, local serve by default.
import { chromium } from 'playwright';
import path from 'node:path';
import assert from 'node:assert/strict';
import { serve } from './serve.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const LOCAL_SCHEMES = new Set(['data:', 'blob:']);

const srv = process.argv[2] ? { url: process.argv[2].replace(/\/$/, ''), close() {} } : await serve(ROOT);
const pageHost = new URL(srv.url).host;
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const seen = [];
const pageErrors = [];
let hintsToOtherHosts = [];

async function openPage(label, init) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const note = (url, type) => seen.push({ label, url, type, failed: null });
  context.on('request', (r) => note(r.url(), r.resourceType()));
  context.on('requestfailed', (r) => {
    const row = seen.find((s) => s.url === r.url() && s.failed === null);
    if (row) row.failed = r.failure()?.errorText ?? 'failed';
    else seen.push({ label, url: r.url(), type: r.resourceType(), failed: r.failure()?.errorText ?? 'failed' });
  });
  const page = await context.newPage();
  page.on('websocket', (ws) => note(ws.url(), 'websocket'));
  page.on('pageerror', (e) => pageErrors.push(`${label}: ${e.message}`));
  if (init) await page.addInitScript(init);
  await page.goto(`${srv.url}/`);
  await page.waitForFunction(() => document.documentElement.dataset.phase === 'ready', null, { timeout: 30000 });
  return page;
}

// A preconnect opens a socket without ever firing a request event, so link and embed tags are read from the DOM as well.
const readHints = (page) => page.evaluate(() => [...document.querySelectorAll('link[href], script[src], img[src], iframe[src], source[src], audio[src], video[src], embed[src], object[data]')]
  .map((n) => n.href || n.src || n.data).filter((u) => u.startsWith('http') && new URL(u).host !== location.host));

try {
  const live = await openPage('live GPU');
  hintsToOtherHosts.push(...await readHints(live));
  await live.click('#start');
  await live.waitForFunction(() => 'arrived' in document.documentElement.dataset, null, { timeout: 60000 });
  console.log(`live GPU: ${await live.textContent('#say')}`);
  await live.fill('#box', 'abcabcabcabcab');
  await live.waitForFunction(() => document.getElementById('ghost').textContent.length > 0, null, { timeout: 30000 });
  console.log(`live GPU: typed abcabcabcabcab, the model says ${await live.textContent('#ghost')}`);
  await live.waitForFunction(() => document.documentElement.dataset.phase === 'done', null, { timeout: 120000 });
  await live.click('#benchBtn');
  await live.waitForFunction(() => !document.getElementById('benchOut').hidden, null, { timeout: 180000 });
  console.log(`live GPU: ${await live.textContent('#benchMedian')}`);
  await live.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await live.waitForTimeout(500);
  assert.equal(await live.evaluate(() => [...document.fonts].filter((f) => f.status === 'error').length), 0, 'a font face failed to load');
  console.log(`live GPU: font faces loaded: ${await live.evaluate(() => [...document.fonts].filter((f) => f.status === 'loaded').map((f) => `${f.weight} ${f.style}`).join(', '))}`);

  const recording = await openPage('no WebGPU', () => { Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined }); });
  hintsToOtherHosts.push(...await readHints(recording));
  await recording.click('#start');
  await recording.waitForFunction(() => 'arrived' in document.documentElement.dataset, null, { timeout: 60000 });
  await recording.fill('#box', 'abcabcabcabcab');
  await recording.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await recording.waitForTimeout(500);
} finally {
  await browser.close();
  srv.close();
}

const network = seen.filter((s) => !LOCAL_SCHEMES.has(new URL(s.url).protocol));
const byHost = new Map();
for (const s of network) byHost.set(new URL(s.url).host, (byHost.get(new URL(s.url).host) ?? 0) + 1);
const foreign = network.filter((s) => new URL(s.url).host !== pageHost);
const failed = network.filter((s) => s.failed);
hintsToOtherHosts = [...new Set(hintsToOtherHosts)];

console.log(`page host ${pageHost}`);
for (const [host, count] of byHost) console.log(`  ${host}: ${count} requests`);
console.log(`  ${seen.length - network.length} data or blob URLs, which never reach a network`);
console.log(`  failed: ${failed.length}`);
for (const s of failed) console.log(`    ${s.url} ${s.failed}`);
for (const s of foreign) console.log(`  FOREIGN ${s.label} ${s.type} ${s.url}${s.failed ? ` (failed: ${s.failed})` : ''}`);
for (const u of hintsToOtherHosts) console.log(`  FOREIGN TAG ${u}`);
for (const e of pageErrors) console.log(`  page error: ${e}`);

const pass = !foreign.length && !hintsToOtherHosts.length && !pageErrors.length;
console.log(pass ? `PASS: ${network.length} requests, all to the page's host` : `FAIL: ${foreign.length} requests and ${hintsToOtherHosts.length} tags point at another host, ${pageErrors.length} page errors`);
process.exit(pass ? 0 : 1);
