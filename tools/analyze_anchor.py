"""Second plan, prediction 1: how the gap moves with the anchor threshold and with eval spacing. Writes results/anchor-summary.json.

    python tools/analyze_anchor.py results/p2-T64-L2-u2.jsonl results/p2-T64-L2-u1.jsonl results/p2-T128-L2-u2.jsonl results/p2-T128-L2-u1.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np

ENV = (0.735, 0.901)
BOOT = 2000


def baseline(T):
    return (sum(1 / i for i in range(1, T + 1)) - 1) / (T - 1)


def anchor(curve, threshold, early):
    for c in curve:
        if max(c["prev"][:early]) >= threshold:
            return c["step"]
    return None


def fractions(rows, threshold, every=5):
    """Anchor over event. These runs stop at the fine event, so a coarser eval spacing is derived rather than replayed:
    each time moves up to the next multiple of the spacing, and the event needs one more eval after its first crossing."""
    out = []
    for r in rows:
        an = anchor(r["curve"], threshold, max(1, r["cfg"]["L"] // 2))
        if r["t_event"] is None or an is None:
            continue
        ev = r["t_event"] - 5
        out.append((np.ceil(an / every) * every) / (np.ceil(ev / every) * every + every))
    return np.array(out)


def describe(frac, rng):
    boot = [np.median(frac[rng.integers(0, len(frac), len(frac))]) for _ in range(BOOT)]
    return {
        "n": len(frac), "median": float(np.median(frac)), "ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "inEnvelopeShare": float(np.mean((frac >= ENV[0]) & (frac <= ENV[1]))), "medianInside": bool(ENV[0] <= np.median(frac) <= ENV[1]),
    }


def main(paths):
    rng = np.random.default_rng(0)
    out = []
    for p in paths:
        rows = [json.loads(line) for line in Path(p).read_text().splitlines()]
        c = rows[0]["cfg"]
        T = c["T"]
        b = baseline(T)
        out.append({
            "T": T, "L": c["L"], "u": c["unique"], "seeds": len(rows), "events": sum(r["t_event"] is not None for r in rows), "baseline": b,
            "A, 0.10, every 5": describe(fractions(rows, 0.10), rng),
            "A, 0.10, every 25": describe(fractions(rows, 0.10, 25), rng),
            "B, 2.86 baseline, every 5": describe(fractions(rows, 2.86 * b), rng),
        })
    Path("results/anchor-summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for a in out:
        print(f"T={a['T']} ({a['L']},{a['u']}) seeds {a['seeds']} events {a['events']} b={a['baseline']:.4f}")
        for k in ("A, 0.10, every 5", "A, 0.10, every 25", "B, 2.86 baseline, every 5"):
            d = a[k]
            print(f"  {k}: {d['median']:.3f} ({d['ci'][0]:.3f} to {d['ci'][1]:.3f}), in envelope {d['inEnvelopeShare']:.2f}, median inside {d['medianInside']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
