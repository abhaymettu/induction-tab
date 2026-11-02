"""Counter-based PRNG, bit-identical to src/engine/prng.js and the WGSL copy."""
import numpy as np

STREAM = {"init": 0x1000, "data": 0x2000, "bigram": 0x3000, "eval": 0x4000}
_M = np.uint64(0xFFFFFFFF)


def pcg(v):
    v = np.asarray(v, dtype=np.uint64) & _M
    state = (v * np.uint64(747796405) + np.uint64(2891336453)) & _M
    word = (((state >> ((state >> np.uint64(28)) + np.uint64(4))) ^ state) * np.uint64(277803737)) & _M
    return ((word >> np.uint64(22)) ^ word) & _M


def hash3(seed, stream, i):
    seed = np.uint64(seed) & _M
    return pcg(seed ^ pcg(np.uint64(stream) ^ pcg(i)))


def unit(h):
    return (np.asarray(h, dtype=np.uint64) >> np.uint64(8)).astype(np.float64) * 5.9604644775390625e-8
