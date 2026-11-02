"""Same bigram language, copy probe and batch stream as src/engine/lang.js and the DATA kernel."""
import numpy as np
from .prng import hash3, unit, STREAM

SLOTS = 20


def build_language(V, lang_seed):
    j = np.arange(V * SLOTS, dtype=np.uint64) * np.uint64(3)
    succ = (hash3(lang_seed, STREAM["bigram"], j) % np.uint64(V)).astype(np.int64).reshape(V, SLOTS)
    h1 = hash3(lang_seed, STREAM["bigram"], j + np.uint64(1))
    u1 = ((h1 >> np.uint64(8)).astype(np.float64) + 1) * 5.9604644775390625e-8
    u2 = unit(hash3(lang_seed, STREAM["bigram"], j + np.uint64(2)))
    w = np.exp(1.5 * np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)).reshape(V, SLOTS)
    cum = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1).astype(np.float32)
    return succ, cum


def copy_probe(V, B, T, seed=1234):
    L = T // 2
    lo = 10
    i = np.arange(B * L, dtype=np.uint64)
    x = (lo + (hash3(seed, STREAM["eval"], i) % np.uint64(V - 2 * lo)).astype(np.int64)).reshape(B, L)
    return np.concatenate([x, x], axis=1)


def sample_batch(c, succ, cum, step):
    """Token batch for one training step, identical to the DATA kernel."""
    B, T, V = c["B"], c["T"], c["V"]
    seed, data = c["seed"], STREAM["data"]
    base = (np.uint64(step) * np.uint64(B) + np.arange(B, dtype=np.uint64)) * np.uint64(T + 3)
    base &= np.uint64(0xFFFFFFFF)
    M = np.uint64(0xFFFFFFFF)
    tok = np.zeros((B, T), dtype=np.int64)
    cur = (hash3(seed, data, base) % np.uint64(V)).astype(np.int64)
    rows = np.arange(B)
    for t in range(T):
        tok[:, t] = cur
        x = unit(hash3(seed, data, (base + np.uint64(1 + t)) & M)).astype(np.float32)
        hits = x[:, None] < cum[cur]
        slot = np.where(hits.any(axis=1), hits.argmax(axis=1), SLOTS - 1)
        cur = succ[cur, slot]
    rep = unit(hash3(seed, data, (base + np.uint64(T + 1)) & M)).astype(np.float32) < np.float32(c["pRep"])
    r = c["rMin"] + (hash3(seed, data, (base + np.uint64(T + 2)) & M) % np.uint64(c["rMax"] - c["rMin"])).astype(np.int64)
    for b in rows[rep]:
        tok[b, r[b]:] = tok[b, : T - r[b]].copy()
    return tok
