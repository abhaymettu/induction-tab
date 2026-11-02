"""Same init as src/engine/init.js."""
import numpy as np
from .prng import hash3, unit, STREAM
from .layout import layout_of


def init_params(cfg, seed):
    lay = layout_of(cfg)
    out = np.zeros(lay["total"], dtype=np.float32)
    for ti, s in enumerate(lay["specs"]):
        stream = STREAM["init"] + ti
        n = s["size"]
        if s["init"] == "one":
            v = np.ones(n)
        elif s["init"] == "zero":
            v = np.zeros(n)
        elif s["init"] == "kaiming":
            v = (2 * unit(hash3(seed, stream, np.arange(n, dtype=np.uint64))) - 1) / np.sqrt(s["fanIn"])
        else:
            i = np.arange(n, dtype=np.uint64)
            h1 = hash3(seed, stream, 2 * i)
            u1 = ((h1 >> np.uint64(8)).astype(np.float64) + 1) * 5.9604644775390625e-8
            u2 = unit(hash3(seed, stream, 2 * i + np.uint64(1)))
            v = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        out[s["off"]:s["off"] + n] = v.astype(np.float32)
    return out
