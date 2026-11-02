"""Third plan, research/PREREG-3.md. Writes results/third-summary.json.

    python tools/analyze_third.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_study import ENV, finite, km_median

BOOT = 4000
R = Path("results")
LN96 = float(np.log(96))


def rows(name):
    return [json.loads(line) for line in (R / f"{name}.jsonl").read_text().splitlines()]


def arm(table, **want):
    return [r for r in rows(table) if all(r["cfg"].get(k) == v for k, v in want.items())]


def te(rs):
    t = np.array([r["t_event"] if r["t_event"] is not None else r["budget"] for r in rs], float)
    e = np.array([r["t_event"] is not None for r in rs])
    return t, e


def summary(rs):
    t, e = te(rs)
    return {"n": len(rs), "events": int(e.sum()), "medianStep": km_median(t, e) if e.any() else None}


def ratio(a, b, rng):
    ta, ea = te(a)
    tb, eb = te(b)
    if not ea.any() or not eb.any():
        return {"ratio": None, "a": summary(a), "b": summary(b)}
    boot = [km_median(ta[i], ea[i]) / km_median(tb[j], eb[j]) for i, j in ((rng.integers(0, len(ta), len(ta)), rng.integers(0, len(tb), len(tb))) for _ in range(BOOT))]
    boot = np.array(boot)
    finite = boot[np.isfinite(boot)]
    return {"ratio": km_median(ta, ea) / km_median(tb, eb), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "bootFinite": len(finite) / len(boot), "a": summary(a), "b": summary(b)}


def read_every(r, every):
    last, ev, an = False, None, None
    for c in r["curve"]:
        if c["step"] % every:
            continue
        if an is None and max(c["prev"][: max(1, r["cfg"]["L"] // 2)]) >= 0.10:
            an = c["step"]
        ok = c["copyAdv"] >= 2
        if ok and last:
            ev = c["step"]
            break
        last = ok
    return ev, an


def main():
    rng = np.random.default_rng(0)
    out = {}

    main_arm = [r for r in rows("study-lr3e-3") if r["cfg"]["L"] == 2 and r["cfg"]["unique"] == 2]
    cur = rows("study-curriculum")
    pick = lambda uniform, tile: [r for r in cur if r["cfg"]["uniform"] == uniform and r["cfg"]["tile"] == tile]
    out["curriculum"] = {
        "rule, Howe's language, pRep 1, periods [4, 32): clone against tiled": ratio(rows("crossed-rule-bigram"), pick(False, True), rng),
        "rule, uniform, pRep 0.75, periods [4, 60): tiled against clone": ratio(rows("crossed-rule-uniform"), pick(True, False), rng),
        "range, uniform tiled, pRep 1: [4, 60) against [4, 32)": ratio(rows("crossed-range-uniform"), main_arm, rng),
        "range, Howe's language clone, pRep 0.75: [4, 32) against [4, 60)": ratio(rows("crossed-range-bigram"), pick(False, False), rng),
    }

    out["negatives"] = {n: {**summary(rows(n)), "maxCopyAdv": max(c["copyAdv"] for r in rows(n) for c in r["curve"])} for n in ("tab-neg-L1", "tab-neg-norep")}

    tab = rows("tab-browser")
    fired = [r for r in tab if r["t_event"] is not None]
    at = lambda r, s: next(c for c in r["curve"] if c["step"] == s)
    first = np.array([at(r, r["t_event"])["firstPass"] for r in fired])
    second = np.array([at(r, r["t_event"])["secondPass"] for r in fired])
    adv = first - second
    ev = np.array([r["t_event"] for r in fired])
    out["tab"] = {
        **summary(tab), "eventRange": [int(ev.min()), int(ev.max())], "stalled": [r["seed"] for r in tab if r["t_event"] is None],
        "atEvent": {"firstPassMedian": float(np.median(first)), "secondPassMedian": float(np.median(second)), "advantageMedian": float(np.median(adv)), "shareFromFirstPassMedian": float(np.median((first - LN96) / adv)), "chance": LN96},
    }

    fine = {r["seed"]: r for r in rows("fine-2-2")}
    coarse = {r["seed"]: r for r in main_arm}
    seeds = sorted(fine)
    read25 = {s: read_every(fine[s], 25) for s in seeds}
    unread = [s for s in seeds if read25[s][0] is None]
    seeds = [s for s in seeds if read25[s][0] is not None]
    frac5 = np.array([fine[s]["t_pv"] / fine[s]["t_event"] for s in seeds])
    frac25 = np.array([read25[s][1] / read25[s][0] for s in seeds])
    out["spacing"] = {
        "seeds": len(seeds), "stoppedBeforeTheCoarseEvent": unread, "sameEventAsStudy": sum(read25[s][0] == coarse[s]["t_event"] for s in seeds), "sameAnchorAsStudy": sum(read25[s][1] == coarse[s]["t_pv"] for s in seeds),
        "gapEvery5": float(np.median(frac5)), "gapEvery25": float(np.median(frac25)), "pairedMedianDifference": float(np.median(frac5 - frac25)),
        "eventEvery5": float(np.median([fine[s]["t_event"] for s in seeds])), "eventEvery25": float(np.median([read25[s][0] for s in seeds])),
        "inEnvelopeEvery5": float(np.mean((frac5 >= ENV[0]) & (frac5 <= ENV[1]))), "inEnvelopeEvery25": float(np.mean((frac25 >= ENV[0]) & (frac25 <= ENV[1]))),
    }

    long8 = rows("deep-8-1-long")
    out["depth"] = {
        "(12,4) against (12,12)": ratio(rows("deep-12-4"), rows("deep-12-12"), rng),
        "(8,1) to 16000": {**summary(long8), "events": [r["t_event"] for r in long8], "finalPrefix": [round(max(r["curve"][-1]["prefix"]), 3) for r in long8], "finalCopyAdv": [round(r["curve"][-1]["copyAdv"], 3) for r in long8]},
    }

    how = rows("tab-howe-curriculum")
    out["tabOnHowe"] = {**summary(how), "finalCopyAdv": [round(r["curve"][-1]["copyAdv"], 3) for r in how], "finalPrefix": [round(max(r["curve"][-1]["prefix"]), 3) for r in how]}

    (R / "third-summary.json").write_text(json.dumps(finite(out), indent=1, allow_nan=False) + "\n")
    print(json.dumps(finite(out), indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
