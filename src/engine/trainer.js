// Wires the kernels into fixed dispatch lists. A net is a list of ops over dedicated buffers. A training
// step is DATA, forward, loss, backward, TICK, AdamW, and the whole list replays with no CPU in the loop,
// so one submit can carry many steps.

import * as K from './kernels.js';
import { layoutOf, resolve } from './layout.js';
import { initParams } from './init.js';
import { buildLanguage, copyProbe, SLOTS } from './lang.js';
import { STREAM } from './prng.js';

const SLOT = 256;
const STORAGE = 0x80 | 0x04 | 0x08; // STORAGE | COPY_SRC | COPY_DST
const HIST_CAP = 1 << 16;
const NO_PARAM = 0xffffffff;

export async function getDevice() {
  if (!navigator.gpu) throw new Error('no-webgpu');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no-adapter');
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
    },
  });
  return { adapter, device };
}

class Graph {
  constructor(device, pipelines) {
    this.device = device;
    this.pipelines = pipelines;
    this.ops = [];
    this.uniforms = [];
  }

  // fields: u32 unless given as {f: value}
  op(phase, kernel, fields, bindings, x, y = 1, z = 1) {
    const slot = new ArrayBuffer(SLOT);
    const u32 = new Uint32Array(slot);
    const f32 = new Float32Array(slot);
    fields.forEach((v, i) => {
      if (typeof v === 'object') f32[i] = v.f;
      else u32[i] = v;
    });
    const index = this.uniforms.length;
    this.uniforms.push(slot);
    this.ops.push({ phase, kernel, bindings, x, y, z, index });
    return index;
  }

  gemm(phase, { A, B, C, D, M, N, K: KK, a = {}, b = {}, c = {}, d = {}, zCount = 1, zInner = 1, alpha = 1, beta = 1, bias = -1, acc = false, ones = false }) {
    const v = (o) => [o.off ?? 0, o.ld ?? 1, o.so ?? 0, o.si ?? 0];
    const flags = (D ? 1 : 0) | (bias >= 0 ? 2 : 0) | (acc ? 4 : 0) | (ones ? 8 : 0);
    const fields = [M, N, KK, zInner, ...v(a), ...v(b), ...v(c), ...v(d), a.trans ? 1 : 0, b.trans ? 1 : 0, flags, Math.max(bias, 0), { f: alpha }, { f: beta }, 0, 0];
    this.op(phase, 'GEMM', fields, [A, B, C, D ?? A], Math.ceil(N / 32), Math.ceil(M / 32), zCount);
  }

  finalize() {
    const { device } = this;
    this.ubuf = device.createBuffer({ size: SLOT * this.uniforms.length, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const all = new Uint8Array(SLOT * this.uniforms.length);
    this.uniforms.forEach((s, i) => all.set(new Uint8Array(s), i * SLOT));
    device.queue.writeBuffer(this.ubuf, 0, all);
    for (const op of this.ops) {
      op.pipe = this.pipelines[op.kernel];
      op.bind = device.createBindGroup({
        layout: op.pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.ubuf, offset: op.index * SLOT, size: 64 + 64 } },
          ...op.bindings.map((buffer, i) => ({ binding: i + 1, resource: { buffer } })),
        ],
      });
    }
  }

  encode(pass, ops = this.ops) {
    for (const op of ops) {
      pass.setPipeline(op.pipe);
      pass.setBindGroup(0, op.bind);
      pass.dispatchWorkgroups(op.x, op.y, op.z);
    }
  }

  setField(index, field, value) {
    this.device.queue.writeBuffer(this.ubuf, index * SLOT + field * 4, new Uint32Array([value]));
  }
}

const groups = (n) => Math.ceil(n / K.WG);

// Forward pass of [B, T] token sequences, and the matching backward pass when withGrad is set.
function buildNet(g, ctx, { B, T, tok, withGrad, phase = 'fwd' }) {
  const { c, lay, P, G, buf } = ctx;
  const { d, H, dh, f, V, L } = c;
  const N = B * T;
  const scale = 1 / Math.sqrt(dh);
  const by = lay.by;
  const HTT = H * T * T;
  const head = { so: T * 3 * d, si: dh, ld: 3 * d };
  const headS = { ld: T, so: HTT, si: T * T };
  const headO = { ld: d, so: T * d, si: dh };
  const attnZ = { zCount: B * H, zInner: H, M: T };

  const X = [buf(N * d)];
  const lnStats = [];
  const layers = [];

  const ln = (input, gName, bName) => {
    if (!c.ln) return { out: input };
    const out = buf(N * d);
    const stats = buf(N * 2);
    const gOff = c.lnAffine ? by[gName].off : NO_PARAM;
    const bOff = c.lnAffine ? by[bName].off : NO_PARAM;
    g.op(phase, 'LN_FWD', [N, d, gOff, bOff], [P, input, out, stats], groups(N));
    return { out, stats, gOff, bOff };
  };

  g.op(phase, 'EMBED_FWD', [N, T, d, by.wte.off, by.wpe.off], [P, tok, X[0]], groups(N));
  for (let l = 0; l < L; l++) {
    const p = `b${l % c.unique}.`;
    const lay1 = { p };
    lay1.xIn = X[X.length - 1];
    lay1.n1 = ln(lay1.xIn, p + 'ln1g', p + 'ln1b');
    lay1.qkv = buf(N * 3 * d);
    g.gemm(phase, { A: lay1.n1.out, B: P, C: lay1.qkv, M: N, N: 3 * d, K: d, a: { ld: d }, b: { off: by[p + 'wqkv'].off, ld: 3 * d }, c: { ld: 3 * d } });
    lay1.S = buf(B * HTT);
    g.gemm(phase, { A: lay1.qkv, B: lay1.qkv, C: lay1.S, ...attnZ, N: T, K: dh, a: { off: 0, ...head }, b: { off: d, ...head, trans: 1 }, c: headS, alpha: scale });
    lay1.softmax = g.op(phase, 'SOFTMAX_FWD', [B * H * T, T, H, 0], [lay1.S], groups(B * H * T));
    lay1.O = buf(N * d);
    g.gemm(phase, { A: lay1.S, B: lay1.qkv, C: lay1.O, ...attnZ, N: dh, K: T, a: headS, b: { off: 2 * d, ...head }, c: headO });
    lay1.xMid = buf(N * d);
    g.gemm(phase, { A: lay1.O, B: P, C: lay1.xMid, D: lay1.xIn, M: N, N: d, K: d, a: { ld: d }, b: { off: by[p + 'wo'].off, ld: d }, c: { ld: d }, d: { ld: d } });
    if (c.mlp) {
      lay1.n2 = ln(lay1.xMid, p + 'ln2g', p + 'ln2b');
      lay1.H1 = buf(N * f);
      g.gemm(phase, { A: lay1.n2.out, B: P, C: lay1.H1, M: N, N: f, K: d, a: { ld: d }, b: { off: by[p + 'w1'].off, ld: f }, c: { ld: f }, bias: by[p + 'b1'].off });
      lay1.Gl = buf(N * f);
      g.op(phase, 'GELU_FWD', [N, f], [lay1.H1, lay1.Gl], groups(N));
      lay1.xOut = buf(N * d);
      g.gemm(phase, { A: lay1.Gl, B: P, C: lay1.xOut, D: lay1.xMid, M: N, N: d, K: f, a: { ld: f }, b: { off: by[p + 'w2'].off, ld: d }, c: { ld: d }, d: { ld: d }, bias: by[p + 'b2'].off });
    } else {
      lay1.xOut = lay1.xMid;
    }
    X.push(lay1.xOut);
    layers.push(lay1);
  }
  const nF = ln(X[L], 'lnfg', 'lnfb');
  const logits = buf(N * V);
  g.gemm(phase, { A: nF.out, B: P, C: logits, M: N, N: V, K: d, a: { ld: d }, b: { off: by.wu.off, ld: V }, c: { ld: V } });
  const lossBuf = buf(N);
  const forwardEnd = g.ops.length;
  g.op('loss', 'XENT', [N, V, T, withGrad ? 1 : 0], [logits, tok, lossBuf], groups(N));

  if (!withGrad) return { layers, logits, lossBuf, N, B, T };

  const bwd = 'bwd';
  const dStream = buf(N * d);
  const dNorm = buf(N * d);
  const dO = buf(N * d);
  const dQKV = buf(N * 3 * d);
  const dS = buf(B * HTT);
  const dG = c.mlp ? buf(N * f) : null;
  let started = false;

  // Pull dNorm through a LayerNorm (or straight) into the residual stream.
  const intoStream = (n, xBuf) => {
    if (c.ln) {
      if (c.lnAffine) g.op(bwd, 'LN_BWD_P', [N, d, n.gOff, n.bOff], [G, xBuf, dNorm, n.stats], groups(d));
      g.op(bwd, 'LN_BWD_X', [N, d, n.gOff, started ? 1 : 0], [P, xBuf, dNorm, n.stats, dStream], groups(N));
    } else {
      g.op(bwd, 'ADD', [N * d, started ? 1 : 0], [dNorm, dStream], groups(N * d));
    }
    started = true;
  };

  const acc = true;
  g.gemm(bwd, { A: nF.out, B: logits, C: G, M: d, N: V, K: N, a: { ld: d, trans: 1 }, b: { ld: V }, c: { off: by.wu.off, ld: V }, acc });
  g.gemm(bwd, { A: logits, B: P, C: dNorm, M: N, N: d, K: V, a: { ld: V }, b: { off: by.wu.off, ld: V, trans: 1 }, c: { ld: d } });
  intoStream(nF, X[L]);

  for (let l = L - 1; l >= 0; l--) {
    const y = layers[l];
    const p = y.p;
    if (c.mlp) {
      g.gemm(bwd, { A: y.Gl, B: dStream, C: G, M: f + 1, N: d, K: N, a: { ld: f, trans: 1 }, b: { ld: d }, c: { off: by[p + 'w2'].off, ld: d }, acc, ones: true });
      g.gemm(bwd, { A: dStream, B: P, C: dG, M: N, N: f, K: d, a: { ld: d }, b: { off: by[p + 'w2'].off, ld: d, trans: 1 }, c: { ld: f } });
      g.op(bwd, 'GELU_BWD', [N, f], [y.H1, dG], groups(N));
      g.gemm(bwd, { A: y.n2.out, B: dG, C: G, M: d + 1, N: f, K: N, a: { ld: d, trans: 1 }, b: { ld: f }, c: { off: by[p + 'w1'].off, ld: f }, acc, ones: true });
      g.gemm(bwd, { A: dG, B: P, C: dNorm, M: N, N: d, K: f, a: { ld: f }, b: { off: by[p + 'w1'].off, ld: f, trans: 1 }, c: { ld: d } });
      intoStream(y.n2, y.xMid);
    }
    g.gemm(bwd, { A: y.O, B: dStream, C: G, M: d, N: d, K: N, a: { ld: d, trans: 1 }, b: { ld: d }, c: { off: by[p + 'wo'].off, ld: d }, acc });
    g.gemm(bwd, { A: dStream, B: P, C: dO, M: N, N: d, K: d, a: { ld: d }, b: { off: by[p + 'wo'].off, ld: d, trans: 1 }, c: { ld: d } });
    g.gemm(bwd, { A: dO, B: y.qkv, C: dS, ...attnZ, N: T, K: dh, a: headO, b: { off: 2 * d, ...head, trans: 1 }, c: headS });
    g.gemm(bwd, { A: y.S, B: dO, C: dQKV, ...attnZ, N: dh, K: T, a: { ...headS, trans: 1 }, b: headO, c: { off: 2 * d, ...head } });
    g.op(bwd, 'SOFTMAX_BWD', [B * H * T, T], [y.S, dS], groups(B * H * T));
    g.gemm(bwd, { A: dS, B: y.qkv, C: dQKV, ...attnZ, N: dh, K: T, a: headS, b: { off: d, ...head }, c: { off: 0, ...head }, alpha: scale });
    g.gemm(bwd, { A: dS, B: y.qkv, C: dQKV, ...attnZ, N: dh, K: T, a: { ...headS, trans: 1 }, b: { off: 0, ...head }, c: { off: d, ...head }, alpha: scale });
    g.gemm(bwd, { A: y.n1.out, B: dQKV, C: G, M: d, N: 3 * d, K: N, a: { ld: d, trans: 1 }, b: { ld: 3 * d }, c: { off: by[p + 'wqkv'].off, ld: 3 * d }, acc });
    g.gemm(bwd, { A: dQKV, B: P, C: dNorm, M: N, N: d, K: 3 * d, a: { ld: 3 * d }, b: { off: by[p + 'wqkv'].off, ld: 3 * d, trans: 1 }, c: { ld: d } });
    intoStream(y.n1, y.xIn);
  }
  const head1 = buf(V);
  const nxt = buf(N);
  g.op(bwd, 'TOKEN_INDEX', [N, V], [tok, head1, nxt], 1);
  g.op(bwd, 'EMBED_BWD', [N, T, d, by.wte.off, by.wpe.off, V], [G, head1, nxt, dStream], groups(V + T));
  return { layers, logits, lossBuf, N, B, T, forwardEnd };
}

const KERNELS = ['GEMM', 'TOKEN_INDEX', 'EMBED_FWD', 'EMBED_BWD', 'SOFTMAX_FWD', 'SOFTMAX_BWD', 'LN_FWD', 'LN_BWD_X', 'LN_BWD_P', 'GELU_FWD', 'GELU_BWD', 'XENT', 'LOSS_REDUCE', 'ADD', 'DATA', 'PROBE_SCORE', 'PROBE_CE', 'PROBE_VIEW', 'ADAMW', 'TICK'];

export async function createTrainer(device, cfg) {
  const c = resolve(cfg);
  const lay = layoutOf(c);
  const pipelines = {};
  await Promise.all(KERNELS.map(async (name) => {
    pipelines[name] = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: K[name], label: name }), entryPoint: 'main' },
    });
  }));

  const owned = [];
  const buf = (n) => {
    const b = device.createBuffer({ size: Math.max(16, n * 4), usage: STORAGE });
    owned.push(b);
    return b;
  };
  const P = buf(lay.total);
  const G = buf(lay.total);
  const M = buf(lay.total);
  const Vv = buf(lay.total);
  const state = buf(4);
  const hist = buf(HIST_CAP);
  const ctx = { c, lay, P, G, buf };

  const lang = buildLanguage(c.V, c.langSeed);
  const succ = buf(lang.succ.length);
  const cum = buf(lang.cum.length);
  device.queue.writeBuffer(succ, 0, lang.succ);
  device.queue.writeBuffer(cum, 0, lang.cum);

  const train = new Graph(device, pipelines);
  const tok = buf(c.B * c.T);
  const dataFields = [c.B, c.T, c.V, SLOTS, c.seed, c.rMin, c.rMax, STREAM.data, { f: c.pRep }, c.uniform ? 1 : 0, c.tile ? 1 : 0];
  const dataIndex = train.op('data', 'DATA', dataFields, [succ, cum, tok, state], c.B);
  const net = buildNet(train, ctx, { B: c.B, T: c.T, tok, withGrad: true });
  train.op('tick', 'TICK', [1], [state], 1);
  train.op('loss', 'LOSS_REDUCE', [net.N, c.T, HIST_CAP], [net.lossBuf, hist, state], 1);
  const adamFields = [lay.total, c.warmup, 0, 0, { f: c.lr }, { f: c.b1 }, { f: c.b2 }, { f: c.eps }, { f: c.wd }];
  train.op('opt', 'ADAMW', adamFields, [P, G, M, Vv, state], groups(lay.total));
  train.finalize();

  // Copy probe on fixed tokens, with attention kept for the instruments and the live view.
  const pBk = c.probeB;
  const pT = c.probeT;
  const probe = new Graph(device, pipelines);
  const probeTok = buf(pBk * pT);
  device.queue.writeBuffer(probeTok, 0, copyProbe(c.V, pBk, pT));
  const probeNet = buildNet(probe, ctx, { B: pBk, T: pT, tok: probeTok, withGrad: false, phase: 'probe' });
  const nScores = c.L * c.H * 2;
  const probeOut = buf(nScores + pT);
  probeNet.layers.forEach((y, l) => {
    probe.op('probe', 'PROBE_SCORE', [pBk, pT, c.H, l * c.H * 2], [y.S, probeOut], c.H);
  });
  probe.op('probe', 'PROBE_CE', [pBk, pT, nScores], [probeNet.lossBuf, probeOut], groups(pT));
  const view = buf(c.L * c.H * pT * pT);
  probeNet.layers.forEach((y, l) => {
    probe.op('probe', 'PROBE_VIEW', [pBk, pT, c.H, l * c.H * pT * pT], [y.S, view], Math.ceil((c.H * pT * pT) / 16));
  });
  probe.finalize();

  // One sequence, forward only, for whatever the reader types. Unused positions hold token 0; attention is
  // causal, so they cannot reach the position being read.
  const steer = new Graph(device, pipelines);
  const steerTok = buf(c.T);
  const steerNet = buildNet(steer, ctx, { B: 1, T: c.T, tok: steerTok, withGrad: false, phase: 'steer' });
  steer.finalize();

  const readback = async (buffer, size = buffer.size) => {
    const staging = device.createBuffer({ size, usage: 0x01 | 0x08 }); // MAP_READ | COPY_DST
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    const out = staging.getMappedRange().slice(0);
    staging.destroy();
    return out;
  };

  const zero = (b) => {
    const enc = device.createCommandEncoder();
    enc.clearBuffer(b);
    device.queue.submit([enc.finish()]);
  };

  const reset = (seed) => {
    device.queue.writeBuffer(P, 0, initParams(c, seed));
    [G, M, Vv, state, hist].forEach(zero);
    train.setField(dataIndex, 4, seed);
  };

  // Bit h of masks[l] silences head h of layer l in the probe and the prompt, never in training.
  const setAblation = (masks) => {
    masks.forEach((mask, l) => {
      probe.setField(probeNet.layers[l].softmax, 3, mask);
      steer.setField(steerNet.layers[l].softmax, 3, mask);
    });
  };

  // Logits for the token after `tokens`, plus each head's attention from the last position when asked.
  const predict = async (tokens) => {
    const ids = new Uint32Array(c.T);
    const n = Math.min(tokens.length, c.T);
    ids.set(tokens.slice(-n));
    device.queue.writeBuffer(steerTok, 0, ids);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    steer.encode(pass);
    pass.end();
    const staging = device.createBuffer({ size: c.V * 4, usage: 0x01 | 0x08 });
    enc.copyBufferToBuffer(steerNet.logits, (n - 1) * c.V * 4, staging, 0, c.V * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    const logits = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    return logits;
  };

  const opsFor = (phases) => train.ops.filter((o) => phases.includes(o.phase));

  // Replays `n` steps into one compute pass. The optional probe runs after the last of them.
  const encodeSteps = (enc, n, withProbe = false) => {
    const pass = enc.beginComputePass();
    for (let i = 0; i < n; i++) train.encode(pass);
    if (withProbe) probe.encode(pass);
    pass.end();
  };

  const parseProbe = (ab) => {
    const f = new Float32Array(ab);
    const scores = [];
    for (let l = 0; l < c.L; l++) {
      const heads = [];
      for (let h = 0; h < c.H; h++) heads.push({ prev: f[(l * c.H + h) * 2], prefix: f[(l * c.H + h) * 2 + 1] });
      scores.push(heads);
    }
    const ce = f.subarray(nScores, nScores + pT);
    const half = pT / 2;
    let first = 0;
    let second = 0;
    for (let q = 0; q < half - 1; q++) first += ce[q];
    for (let q = half - 1; q < pT - 1; q++) second += ce[q];
    // Appended by the training loop when present: the first probe sequence's per-position loss, then train loss.
    const seqLoss = f.length > nScores + pT ? f.subarray(nScores + pT, nScores + 2 * pT) : null;
    const trainLoss = f.length > nScores + pT ? f[nScores + 2 * pT] : null;
    return { scores, ce: Array.from(ce), firstPass: first / (half - 1), secondPass: second / half, copyAdv: first / (half - 1) - second / half, seqLoss, trainLoss };
  };

  const destroy = () => {
    owned.forEach((b) => b.destroy());
    train.ubuf.destroy();
    probe.ubuf.destroy();
    steer.ubuf.destroy();
  };

  return {
    c, lay, P, G, M, Vv, state, hist, tok, train, probe, net, probeNet, probeOut, probeTok, view, readback, reset, opsFor, encodeSteps, parseProbe, destroy, setAblation, predict,
  };
}
