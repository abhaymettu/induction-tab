// Everything the main run does not reach: pause and new seed, hover and keyboard, the benchmark and its log, the
// records and study sections with data in them, reduced motion, and other window sizes.
// `node tools/verify-flows.mjs [controls] [touch] [pages] [bench] [data] [motion] [sizes]`, all with no arguments.
import { chromium } from 'playwright';
import path from 'node:path';
import assert from 'node:assert/strict';
import { serve } from './serve.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const SHOTS = path.join(ROOT, 'scratch/shots');
const chosen = process.argv.slice(2);
const want = (w) => !chosen.length || chosen.includes(w);
const srv = await serve(ROOT);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });

const phaseIs = (page, p, timeout = 30000) => page.waitForFunction((x) => document.documentElement.dataset.phase === x, p, { timeout });
const arrived = (page) => page.waitForFunction(() => 'arrived' in document.documentElement.dataset, null, { timeout: 60000 });
const clock = (page) => page.$eval('#clock', (n) => Number(n.textContent));
const stepOf = async (page) => Number((await page.textContent('#sub')).match(/Step (\d+)/)?.[1]);
// The mark is drawn once the chart's eased head reaches the event, a moment after the event itself.
const eventStep = async (page) => {
  const handle = await page.waitForFunction(() => {
    const words = [...document.querySelectorAll('.curve-svg .mark')].filter((m) => m.style.display !== 'none').map((m) => m.textContent);
    return words.find((w) => w.startsWith('event'))?.match(/step (\d+)/)?.[1];
  }, null, { timeout: 5000 });
  return Number(await handle.jsonValue());
};

async function open(options, url = '/') {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: 'dark', ...options });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${srv.url}${url}`);
  return { context, page, errors };
}

try {
  if (want('controls')) {
    const { context, page, errors } = await open();
    await phaseIs(page, 'ready');

    await page.keyboard.press('Tab');
    const stops = [];
    for (let i = 0; i < 8; i++) {
      stops.push(await page.evaluate(() => document.activeElement.id || document.activeElement.className || document.activeElement.tagName));
      if (stops[stops.length - 1] === 'start') break;
      await page.keyboard.press('Tab');
    }
    assert.equal(stops[stops.length - 1], 'start', `the keyboard reaches Start: ${stops}`);
    await page.keyboard.press('Enter');
    await phaseIs(page, 'training');

    await page.waitForFunction(() => /Step [1-9]/.test(document.getElementById('sub').textContent));
    await page.click('#pause');
    await phaseIs(page, 'paused');
    assert.equal(await page.textContent('#pause'), 'Resume');
    const heldClock = await clock(page);
    await page.waitForTimeout(250);
    const heldStep = await stepOf(page);
    const heldWords = await page.textContent('#sub');
    await page.waitForTimeout(700);
    assert.equal(await clock(page), heldClock, 'the clock holds while paused');
    assert.equal(await stepOf(page), heldStep, 'training holds while paused');
    await page.click('#pause');
    await phaseIs(page, 'training');
    await arrived(page);
    assert.equal(await eventStep(page), 500, 'a pause does not change where seed 1 gets to the event');
    assert.ok((await clock(page)) < 9, 'the clock left out the pause');
    console.log(`pause held at step ${heldStep}, ${heldClock} s. event still at step 500, clock ${await clock(page)} s`);

    const box = await page.locator('#curve').boundingBox();
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await page.waitForTimeout(100);
    const tip = await page.$eval('.curve-svg .tip', (n) => ({ text: n.textContent, shown: n.style.display !== 'none' }));
    assert.ok(tip.shown && /^step \d+: loss/.test(tip.text), `the chart answers hover: ${tip.text}`);
    await page.mouse.move(10, 10);
    await page.waitForTimeout(100);
    assert.equal(await page.$eval('.curve-svg .tip', (n) => n.style.display), 'none', 'hover ends cleanly');
    console.log('chart hover:', tip.text);

    const row = await page.locator('#stream').boundingBox();
    await page.mouse.move(row.x + row.width * 0.2, row.y + row.height * 0.7);
    await page.waitForFunction(() => document.getElementById('caption').textContent.startsWith('Column'));
    console.log('stream hover:', await page.textContent('#caption'));
    await page.mouse.move(10, 10);

    const head = page.locator('.head').nth(5);
    await head.hover();
    await page.waitForFunction(() => document.getElementById('explain').textContent.length > 10);
    console.log('head hover:', await page.textContent('#explain'));
    await head.focus();
    const before = (await page.textContent('#sub'));
    await page.keyboard.press('Space');
    await page.waitForFunction(() => document.querySelector('.head[aria-pressed="true"]'));
    await page.waitForFunction(() => /silenced/i.test(document.getElementById('sub').textContent));
    console.log('head by keyboard:', before, '->', await page.textContent('#sub'));
    await page.click('#restore');
    await page.waitForFunction(() => !document.querySelector('.head[aria-pressed="true"]'));

    await page.click('#pause');
    await phaseIs(page, 'paused');
    await page.click('#reseed');
    await phaseIs(page, 'training');
    assert.match(await page.textContent('#sub'), /Seed \d+\. Step 0/);
    await page.waitForFunction((n) => document.getElementById('sub').textContent.startsWith(`Step ${n} of`), heldStep);
    const newWords = await page.textContent('#sub');
    assert.notEqual(newWords, heldWords, 'a new seed trains from different weights');
    console.log(`same step, two seeds: "${heldWords}" and "${newWords}"`);
    assert.ok(!(await page.evaluate(() => 'arrived' in document.documentElement.dataset)), 'a new seed starts before the event');
    assert.equal(await page.inputValue('#box'), '', 'a new seed clears the prompt');
    await arrived(page);
    const second = await eventStep(page);
    const sentence = await page.textContent('#say');
    assert.match(sentence, /^Induction in \d+\.\d seconds on your/);
    console.log(`new seed while paused: event at step ${second}. ${sentence}`);

    await page.locator('#after').scrollIntoViewIfNeeded();
    await page.click('#again');
    await page.waitForFunction(() => !('arrived' in document.documentElement.dataset));
    await page.waitForFunction(() => scrollY === 0, null, { timeout: 3000 });
    assert.equal(await page.getAttribute('html', 'data-phase'), 'training');
    console.log('run again goes back to the top and trains');

    await page.locator('#after').scrollIntoViewIfNeeded();
    await page.click('#benchBtn');
    await phaseIs(page, 'bench');
    assert.equal(await page.isDisabled('#reseed'), true, 'nothing else can start during the benchmark');
    assert.equal(await page.isDisabled('#box'), true, 'the box waits during the benchmark');
    await page.waitForFunction(() => !document.getElementById('benchOut').hidden, null, { timeout: 180000 });
    console.log('benchmark from the button, mid-run:', await page.textContent('#benchMedian'));
    await page.click('#again');
    await arrived(page);
    console.log('and a normal run after it:', await page.textContent('#say'));
    assert.deepEqual(errors, []);
    await context.close();
  }

  if (want('touch')) {
    const { context, page, errors } = await open({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    await phaseIs(page, 'ready');
    await page.tap('#start');
    await arrived(page);
    await page.locator('.head').nth(5).tap();
    await page.waitForFunction(() => document.querySelector('.head[aria-pressed="true"]'));
    const said = await page.textContent('#explain');
    assert.ok(said.startsWith('Layer 1, head 2'), `a tap explains: ${said}`);
    await page.locator('[data-layer="1"]').tap();
    await page.waitForFunction(() => document.querySelector('[data-layer="1"]').getAttribute('aria-pressed') === 'true');
    await page.locator('#restore').tap();
    await page.waitForFunction(() => !document.querySelector('.head[aria-pressed="true"]'));
    await page.locator('.head').nth(5).scrollIntoViewIfNeeded();
    await page.waitForTimeout(100);
    await page.screenshot({ path: `${SHOTS}/tapped-390-dark.png` });
    assert.deepEqual(errors, []);
    console.log('touch: a tap explains and silences, and the layer buttons stay reachable under the explanation');
    await context.close();
  }

  if (want('pages')) {
    const parent = await serve(path.dirname(ROOT));
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await context.newPage();
    const missing = [];
    page.on('response', (r) => { if (r.status() >= 400) missing.push(new URL(r.url()).pathname); });
    await page.goto(`${parent.url}/${path.basename(ROOT)}/`);
    await phaseIs(page, 'ready');
    await page.waitForSelector('#recordsTable tbody tr');
    await page.waitForSelector('#studyTable tbody tr');
    await page.evaluate(() => document.fonts.ready);
    assert.deepEqual(missing, [], 'under a project path every file resolves');
    assert.deepEqual(await page.evaluate(() => [...document.fonts].filter((f) => f.status === 'error').length), 0, 'no font face failed under a project path');
    console.log('served from a project path as on GitHub Pages: every file resolves, fonts included');
    await context.close();
    parent.close();
  }

  if (want('bench')) {
    const { context, page, errors } = await open({ permissions: ['clipboard-read', 'clipboard-write'] }, '/?bench');
    await page.waitForFunction(() => !document.getElementById('benchOut').hidden, null, { timeout: 180000 });
    const log = JSON.parse(await page.textContent('#benchLog'));
    assert.equal(log.v, 1);
    assert.equal(typeof log.configHash, 'string');
    assert.deepEqual(Object.keys(log.adapter).sort(), ['architecture', 'description', 'device', 'isFallback', 'vendor']);
    assert.equal(typeof log.userAgent, 'string');
    assert.deepEqual(log.runs.map((r) => r.seed), [1, 2, 3, 4, 5]);
    for (const r of log.runs) {
      assert.deepEqual(Object.keys(r).sort(), ['curve', 'msPerStep', 'seed', 'tEvent', 'tPrev', 'wallEventMs']);
      assert.ok(Array.isArray(r.curve) && r.curve.every((p) => p.length === 2));
    }
    const walls = log.runs.map((r) => r.wallEventMs).sort((a, b) => a - b);
    const median = (walls[2] / 1000).toFixed(2);
    const headline = await page.textContent('#benchMedian');
    assert.ok(headline.includes(`Median ${median} seconds`), `${headline} against ${median}`);
    console.log(headline);
    console.log(log.runs.map((r) => `seed ${r.seed}: event ${r.tEvent}, ${(r.wallEventMs / 1000).toFixed(2)} s, ${r.msPerStep.toFixed(2)} ms/step`).join('\n'));

    const href = await page.getAttribute('#issueLink', 'href');
    assert.ok(href.startsWith('https://github.com/abhaymettu/induction-tab/issues/new?'));
    assert.ok(href.length < 8000, `issue link is ${href.length} characters`);
    await page.click('#copyLog');
    await page.waitForFunction(() => document.getElementById('copyLog').textContent === 'Copied');
    assert.deepEqual(JSON.parse(await page.evaluate(() => navigator.clipboard.readText())), log);
    console.log(`copied, and the issue link is ${href.length} characters`);
    await page.screenshot({ path: `${SHOTS}/bench-1440-dark.png`, fullPage: false });

    await page.fill('#box', 'abcabcabcabcab');
    await page.waitForFunction(() => document.getElementById('ghost').textContent.startsWith('cab'), null, { timeout: 10000 });
    console.log('after the benchmark the box still answers:', await page.textContent('#ghost'));
    await page.locator('#live').scrollIntoViewIfNeeded();
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(200);
    await page.screenshot({ path: `${SHOTS}/bench-live-1440-dark.png` });
    assert.deepEqual(errors, []);
    await context.close();
  }

  if (want('data')) {
    const records = [
      { adapter: 'nvidia geforce rtx 4090', browser: 'Chrome 141', medianSeconds: 2.41, seeds: 5, submitter: '<img src=x onerror=alert(1)>', log: 'javascript:alert(1)' },
      { adapter: 'apple metal-3', browser: 'Chrome 141', medianSeconds: 6.62, seeds: 5, submitter: 'someone', log: 'https://github.com/abhaymettu/induction-tab/issues/1' },
      { adapter: 'intel gen-12lp', browser: 'Edge 140', medianSeconds: 31.7, seeds: 5, submitter: 'someone else', log: 'https://github.com/abhaymettu/induction-tab/issues/2' },
    ];
    for (const [width, height, name] of [[1440, 900, '1440'], [390, 844, '390']]) {
      const context = await browser.newContext({ viewport: { width, height }, colorScheme: 'dark', deviceScaleFactor: 2 });
      const page = await context.newPage();
      await page.route('**/records.json', (r) => r.fulfill({ json: records }));
      await page.goto(`${srv.url}/`);
      await phaseIs(page, 'ready');
      await page.waitForSelector('#recordsTable tbody tr');
      const rows = await page.$$eval('#recordsTable tbody tr', (trs) => trs.map((t) => [...t.cells].map((c) => c.textContent)));
      assert.deepEqual(rows.map((r) => r[0]), ['2.41', '6.62', '31.70'], 'fastest first');
      assert.equal(rows[0][4], records[0].submitter, 'a submitter is text, never markup');
      assert.equal(await page.$$eval('#recordsTable img', (n) => n.length), 0);
      assert.equal(await page.$$eval('#recordsTable a', (n) => n.length), 2, 'a javascript: link is dropped');
      assert.equal(await page.isVisible('#recordsEmpty'), false);
      assert.equal(await page.isVisible('#study'), true);
      await page.waitForSelector('#studyTable tbody tr');
      assert.equal(await page.$$eval('#studyCurriculum path.km', (n) => n.length), 4);
      assert.equal(await page.$$eval('#studyForest circle', (n) => n.length), 12);
      assert.match(await page.textContent('#studyForest'), /8 layers, 1 block.*no (prefix )?signature.*7150/);
      assert.equal(await page.$$eval('#studyTable tbody tr', (n) => n.length), 10);
      assert.match(await page.textContent('#studyTable'), /12 layers, 4 blocks against 12fit 1\.75 \(1\.68 to 1\.8\d\)KM 1\.74 \(1\.63 to 1\.87\)not run/);
      const over = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      assert.equal(over, 0, `no sideways scroll at ${width}`);
      await page.locator('#study').screenshot({ path: `${SHOTS}/study-${name}-dark.png` });
      await page.locator('#records').screenshot({ path: `${SHOTS}/records-${name}-dark.png` });
      console.log(`${width}: records sorted and escaped, study drawn, overflow ${over}`);
      await context.close();
    }
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await context.newPage();
    await page.route('**/records.json', (r) => r.fulfill({ status: 404, body: '' }));
    await page.route('**/results/study-summary.json', (r) => r.fulfill({ status: 404, body: '' }));
    await page.goto(`${srv.url}/`);
    await phaseIs(page, 'ready');
    await page.waitForFunction(() => !document.getElementById('recordsEmpty').hidden);
    assert.equal(await page.isVisible('#study'), false, 'the study section is absent with no file');
    console.log('empty records state:', await page.textContent('#recordsEmpty'));
    await context.close();
  }

  if (want('motion')) {
    for (const [width, height, name] of [[1440, 900, '1440'], [390, 844, '390']]) {
      const { context, page, errors } = await open({ viewport: { width, height }, reducedMotion: 'reduce', deviceScaleFactor: 2 });
      await phaseIs(page, 'ready');
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(100);
      await page.screenshot({ path: `${SHOTS}/reduced-ready-${name}.png` });
      const clash = await page.evaluate(() => document.getElementById('start').getBoundingClientRect().bottom - document.getElementById('band').getBoundingClientRect().top);
      assert.ok(clash <= 0, `the rows sit below the button, ${clash} px`);
      await page.click('#start');
      await page.waitForTimeout(60);
      const moving = await page.evaluate(() => document.getAnimations().filter((a) => a.transitionProperty === 'transform').map((a) => a.effect.target.id || a.effect.target.className));
      assert.deepEqual(moving, [], `nothing moves on start: ${moving}`);
      await arrived(page);
      await page.waitForTimeout(40);
      const rising = await page.evaluate(() => document.getAnimations().filter((a) => a.transitionProperty === 'transform' && !a.effect.target.classList.contains('bar')).map((a) => a.effect.target.id || a.effect.target.className));
      assert.deepEqual(rising, [], `nothing moves at the event: ${rising}`);
      await page.screenshot({ path: `${SHOTS}/reduced-event-${name}.png` });
      assert.deepEqual(errors, []);
      console.log(`reduced motion at ${width}: rows clear of the button by ${-clash} px, no transforms on start or at the event`);
      await context.close();
    }
  }

  if (want('sizes')) {
    for (const [width, height] of [[320, 568], [360, 740], [768, 1024], [1024, 768], [1280, 720], [1440, 790], [1920, 1080], [2560, 1440]]) {
      const { context, page, errors } = await open({ viewport: { width, height } });
      await phaseIs(page, 'ready');
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(100);
      const idle = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      const clash = await page.evaluate(() => document.getElementById('start').getBoundingClientRect().bottom - document.getElementById('band').getBoundingClientRect().top);
      await page.screenshot({ path: `${SHOTS}/size-${width}-ready.png` });
      await page.click('#start');
      await page.waitForTimeout(1100);
      const running = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      await page.screenshot({ path: `${SHOTS}/size-${width}-training.png` });
      const fold = await page.evaluate(() => document.querySelector('.mapfoot').getBoundingClientRect().bottom - innerHeight);
      const map = await page.evaluate(() => document.querySelector('.head').offsetWidth);
      console.log(`${width}x${height}: overflow ${idle} idle, ${running} running; rows clear of the button by ${-clash} px; maps ${map} px, their foot ${fold > 0 ? `${fold} px under` : 'above'} the fold`);
      assert.equal(idle, 0);
      assert.equal(running, 0);
      assert.ok(clash <= 0);
      assert.deepEqual(errors, []);
      await context.close();
    }
  }
  console.log('PASS');
} finally {
  await browser.close();
  srv.close();
}
