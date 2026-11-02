// Presses the page's own benchmark button in headless Chrome and saves the log it shows. `node tools/bench-log.mjs out.json`
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { serve } from './serve.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const srv = await serve(ROOT);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await page.goto(`${srv.url}/?bench`);
await page.waitForFunction(() => !document.getElementById('benchOut').hidden, null, { timeout: 300000 });
const log = JSON.parse(await page.textContent('#benchLog'));
fs.writeFileSync(path.resolve(process.argv[2]), JSON.stringify(log) + '\n');
console.log(await page.textContent('#benchMedian'));
await browser.close();
srv.close?.();
process.exit(0);
