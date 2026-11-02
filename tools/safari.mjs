// Runs the engine in the real Safari on this Mac, in a background tab, and collects the result over a local POST.
// Needs scratch/parity/small from `npm run parity -- small`. `node tools/safari.mjs`
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.bin': 'application/octet-stream' };
const srv = http.createServer((req, res) => {
  if (req.url === '/report') {
    let body = '';
    req.on('data', (d) => { body += d; });
    req.on('end', () => {
      res.end('ok');
      fs.mkdirSync(path.join(ROOT, 'results'), { recursive: true });
      fs.writeFileSync(path.join(ROOT, 'results/safari.json'), JSON.stringify(JSON.parse(body), null, 1));
      console.log(body);
      try { execFileSync('osascript', ['-e', `tell application "Safari" to close (every tab of every window whose URL contains "127.0.0.1:${srv.address().port}")`]); } catch { console.error('could not close the Safari tab'); }
      srv.close();
      process.exit(0);
    });
    return;
  }
  const file = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.setHeader('content-type', TYPES[path.extname(file)] ?? 'application/octet-stream');
  res.setHeader('cache-control', 'no-store');
  fs.createReadStream(file).pipe(res);
}).listen(0, '127.0.0.1', () => {
  execFileSync('open', ['-g', '-a', 'Safari', `http://127.0.0.1:${srv.address().port}/tools/safari-check.html`]);
});
setTimeout(() => { console.error('timed out'); process.exit(1); }, 150000);
