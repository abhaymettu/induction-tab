// Stills and a real-time clip for judging the page by eye. `node tools/verify-look.mjs [clip] [narrow] [light] [long]`
import { chromium } from 'playwright';
import path from 'node:path';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { serve } from './serve.mjs';
import { copyProbe } from '../src/engine/lang.js';
import { decode } from '../src/engine/text.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const SHOTS = path.join(ROOT, 'scratch/shots');
const want = (w) => process.argv.includes(w);
const srv = await serve(ROOT);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const ready = (page) => page.waitForFunction(() => document.documentElement.dataset.phase === 'ready', null, { timeout: 30000 });
const arrived = (page) => page.waitForFunction(() => 'arrived' in document.documentElement.dataset, null, { timeout: 60000 });
const done = (page) => page.waitForFunction(() => document.documentElement.dataset.phase === 'done', null, { timeout: 120000 });

try {
  if (want('clip')) {
    const dir = fs.mkdtempSync(path.join(SHOTS, 'video-'));
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: 'dark', recordVideo: { dir, size: { width: 1440, height: 900 } } });
    const t0 = Date.now();
    const page = await context.newPage();
    await page.goto(`${srv.url}/`);
    await ready(page);
    await page.waitForTimeout(500);
    const tStart = Date.now() - t0;
    await page.click('#start');
    await arrived(page);
    const tEvent = Date.now() - t0;
    await page.waitForTimeout(2000);
    const video = await page.video().path();
    await context.close();
    const out = path.join(SHOTS, 'transition.webm');
    fs.renameSync(video, out);
    fs.rmSync(dir, { recursive: true });
    const stamp = (ms) => (ms / 1000).toFixed(2);
    execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-ss', stamp(tStart - 100), '-t', '1.2', '-i', out, '-vf', 'fps=10,scale=720:-1,tile=3x4', '-frames:v', '1', path.join(SHOTS, 'start-sheet.png')]);
    execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-ss', stamp(tEvent - 1000), '-t', '2.4', '-i', out, '-vf', 'fps=5,scale=720:-1,tile=3x4', '-frames:v', '1', path.join(SHOTS, 'event-sheet.png')]);
    console.log(`clip ${out}: start at ${stamp(tStart)} s, event at ${stamp(tEvent)} s, ${(fs.statSync(out).size / 1e6).toFixed(1)} MB`);
  }

  if (want('narrow')) {
    for (const scheme of ['dark', 'light']) {
      const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true, colorScheme: scheme });
      const page = await context.newPage();
      await page.goto(`${srv.url}/`);
      await ready(page);
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(100);
      await page.screenshot({ path: `${SHOTS}/ready-390-${scheme}.png` });
      await page.tap('#start');
      await page.waitForTimeout(2200);
      await page.screenshot({ path: `${SHOTS}/training-390-${scheme}.png` });
      await arrived(page);
      await page.waitForTimeout(900);
      await page.screenshot({ path: `${SHOTS}/event-390-${scheme}.png` });
      await done(page);
      await page.fill('#box', 'abcabcabcabcab');
      await page.waitForFunction(() => document.querySelectorAll('#ghost .g').length >= 24, null, { timeout: 10000 });
      await page.waitForTimeout(1200);
      await page.screenshot({ path: `${SHOTS}/typed-390-${scheme}.png`, fullPage: true });
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      console.log(`390 ${scheme}: horizontal overflow ${overflow}`);
      await context.close();
    }
  }

  if (want('light') || want('long')) {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2, colorScheme: 'light' });
    const page = await context.newPage();
    await page.goto(`${srv.url}/`);
    await ready(page);
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(100);
    await page.screenshot({ path: `${SHOTS}/ready-1440-light.png` });
    await page.click('#start');
    await arrived(page);
    await page.waitForTimeout(700);
    await page.screenshot({ path: `${SHOTS}/event-1440-light.png` });
    await done(page);
    const toks = copyProbe(96, 1, 128);
    const prefix = [...toks.slice(0, 127)].map(decode).join('');
    await page.fill('#box', prefix);
    await page.waitForFunction(() => document.querySelector('#next').dataset.on === '1', null, { timeout: 10000 });
    await page.waitForTimeout(300);
    console.log('next after the 127 character prefix:', (await page.$$eval('#next .n', (ns) => ns.map((n) => n.textContent))).join(' | '));
    await page.fill('#box', 'abcabcabcabcab');
    await page.waitForFunction(() => { const g = document.getElementById('ghost').textContent; return g.startsWith('cabc') && g.length >= 24; }, null, { timeout: 10000 });
    await page.waitForFunction(() => document.querySelectorAll('#chips .chip').length >= 1, null, { timeout: 20000 });
    await page.screenshot({ path: `${SHOTS}/typed-1440-light.png` });
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.waitForTimeout(200);
    if (want('long')) await page.screenshot({ path: `${SHOTS}/full-1440-dark.png`, fullPage: true });
    await context.close();
  }
} finally {
  await browser.close();
  srv.close();
}
