"""Browser WGSL trainer against native PyTorch, seed by seed, per research/PREREG.md. Writes results/native-vs-browser.json.

    python tools/compare_native.py results/study-lr3e-3.jsonl results/browser-2-2.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main(native_path, browser_path):
    browser = {r["seed"]: r for r in rows(browser_path)}
    cfg = next(iter(browser.values()))["cfg"]
    native = {r["seed"]: r for r in rows(native_path) if r["cfg"] == cfg}
    seeds = sorted(set(browser) & set(native))
    t_native = np.array([native[s]["t_event"] for s in seeds])
    t_browser = np.array([browser[s]["t_event"] for s in seeds])
    worst_copy, worst_loss, evals = 0.0, 0.0, 0
    for s in seeds:
        for a, b in zip(native[s]["curve"], browser[s]["curve"]):
            assert a["step"] == b["step"]
            worst_copy = max(worst_copy, abs(a["copyAdv"] - b["copyAdv"]))
            worst_loss = max(worst_loss, abs(a["loss"] - b["loss"]))
            evals += 1
    out = {
        "seeds": len(seeds), "evals": evals,
        "medianNative": float(np.median(t_native)), "medianBrowser": float(np.median(t_browser)),
        "pairedMedianDifference": float(np.median(t_browser - t_native)),
        "sameEventStep": int((t_browser == t_native).sum()),
        "sameAnchorStep": sum(native[s]["t_pv"] == browser[s]["t_pv"] for s in seeds),
        "worstCopyAdvDifference": worst_copy, "worstLossDifference": worst_loss,
        "pass": bool(abs(np.median(t_browser - t_native)) <= 25),
    }
    Path("results/native-vs-browser.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:])
