"""How far the tying intervals hold up, and what a percentile bootstrap on a 25-step grid covers.

    python tools/analyze_contrasts.py

Writes results/contrast-correction.json. Numpy only. Two parts:

Bonferroni: every ratio interval read in results/study-summary.json and results/third-summary.json counts as one
comparison, and each tying contrast's bootstrap is read at the adjusted level, for both estimators.

Lattice: draws event times from a log-normal with the (2,2) arm's median and spread, rounds them up to the eval grid,
and asks how often the percentile interval for the Kaplan-Meier median covers the continuous median and the grid median.
"""
import json
from pathlib import Path

import numpy as np

R = Path("results")
HI, LO = 0.003, 0.001
FILES = {HI: ["study-lr3e-3", "study-deep", "deep-12-12", "deep-12-4"], LO: ["study-lr1e-3"]}
TYING = [((2, 1), (2, 2)), ((3, 1), (3, 3)), ((4, 1), (4, 4)), ((4, 2), (4, 4)), ((8, 2), (8, 8)), ((12, 4), (12, 12))]
RESAMPLES = 40000
EVAL = 25
LATTICE_REPS = 2000
LATTICE_BOOT = 2000
LATTICE_N = 64
OFFSETS = [i / 10 for i in range(10)]


def rows(name):
    return [json.loads(line) for line in (R / f"{name}.jsonl").read_text().splitlines()]


def arm(shape, lr):
    runs = [r for name in FILES[lr] for r in rows(name) if (r["cfg"]["L"], r["cfg"]["unique"]) == shape and r["cfg"]["lr"] == lr]
    return np.array([r["t_event"] if r["t_event"] is not None else r["budget"] for r in runs], float), np.array([r["t_event"] is not None for r in runs])


def km_median_matrix(t, fired):
    """Kaplan-Meier median per row. Every censored run sits at the budget, past every event, so the median is an order statistic."""
    t = np.where(fired, t, np.inf)
    return np.sort(t, axis=-1)[..., (t.shape[-1] + 1) // 2 - 1]


def censored_mu(t, fired, iters=60):
    """Censored log-normal mu per row, by expectation maximization. Matches the maximum likelihood fit in tools/analyze_study.py."""
    lt = np.log(t)
    n = fired.sum(-1)
    m = n + (~fired).sum(-1)
    mu = (lt * fired).sum(-1) / np.maximum(n, 1)
    var = (((lt - mu[..., None]) ** 2) * fired).sum(-1) / np.maximum(n, 1) + 1e-6
    if (~fired).any():
        c = lt[~fired].flat[0]
        ncens = (~fired).sum(-1)
        sum_e = (lt * fired).sum(-1)
        sum_e2 = (lt ** 2 * fired).sum(-1)
        for _ in range(iters):
            s = np.sqrt(var)
            z = (c - mu) / s
            h = hazard(z)
            e1 = mu + s * h
            e2 = var * (1 + z * h) + 2 * mu * s * h + mu ** 2
            mu = (sum_e + ncens * e1) / m
            var = np.maximum((sum_e2 + ncens * e2) / m - mu ** 2, 1e-8)
    return mu


def hazard(z):
    from math import erfc

    q = np.vectorize(erfc)(z / np.sqrt(2)) / 2
    return np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi) / np.maximum(q, 1e-300)


def resample(rng, t, fired, n):
    idx = rng.integers(0, len(t), (n, len(t)))
    return t[idx], fired[idx]


def counted_intervals():
    study = json.loads((R / "study-summary.json").read_text())["contrasts"]
    third = json.loads((R / "third-summary.json").read_text())
    groups = {
        "study contrasts, fit": sum(c.get("ci") is not None for c in study),
        "study contrasts, medians": sum(c.get("kmCi") is not None for c in study),
        "third plan single factors and depth": sum(v.get("ci") is not None for part in ("curriculum", "depth") for v in third[part].values()),
        "repeat rate": 1,
    }
    return groups, sum(groups.values())


def bonferroni(rng, m):
    tail = 100 * 0.05 / m / 2
    out = []
    for a, b in TYING:
        for lr in (HI, LO):
            if lr == LO and a[0] > 4:
                continue
            ta, ea = arm(a, lr)
            tb, eb = arm(b, lr)
            xa, fa = resample(rng, ta, ea, RESAMPLES)
            xb, fb = resample(rng, tb, eb, RESAMPLES)
            km = km_median_matrix(xa, fa) / km_median_matrix(xb, fb)
            fit = np.exp(censored_mu(xa, fa) - censored_mu(xb, fb))
            out.append({
                "a": a, "b": b, "lr": lr, "seeds": [len(ta), len(tb)],
                "fit": float(np.exp(censored_mu(ta, ea) - censored_mu(tb, eb))), "km": float(km_median_matrix(ta, ea) / km_median_matrix(tb, eb)),
                "fit95": [float(x) for x in np.percentile(fit, [2.5, 97.5])], "km95": [float(x) for x in np.percentile(km, [2.5, 97.5])],
                "fitAdjusted": [float(x) for x in np.percentile(fit, [tail, 100 - tail])], "kmAdjusted": [float(x) for x in np.percentile(km, [tail, 100 - tail])],
            })
    return out


def lattice(rng, median, sigma):
    runs = []
    for offset in OFFSETS:
        true_median = median + offset * EVAL
        grid_median = np.ceil(true_median / EVAL) * EVAL
        continuous, grid = 0, 0
        for _ in range(LATTICE_REPS):
            sample = np.ceil(np.exp(rng.normal(np.log(true_median), sigma, LATTICE_N)) / EVAL) * EVAL
            boot = np.sort(sample[rng.integers(0, LATTICE_N, (LATTICE_BOOT, LATTICE_N))], axis=-1)[:, LATTICE_N // 2 - 1]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            continuous += lo <= true_median <= hi
            grid += lo <= grid_median <= hi
        runs.append({"offsetInCell": offset, "continuousMedian": float(true_median), "gridMedian": float(grid_median), "coversContinuous": continuous / LATTICE_REPS, "coversGrid": grid / LATTICE_REPS})
    return runs


def main():
    rng = np.random.default_rng(0)
    groups, m = counted_intervals()
    tying = bonferroni(rng, m)
    t, fired = arm((2, 2), HI)
    assert fired.all()
    sigma = float(np.log(t).std(ddof=1))
    median = float(km_median_matrix(t, fired))
    runs = lattice(rng, median, sigma)
    out = {
        "intervalsRead": groups, "intervalsReadTotal": m, "adjustedLevel": 0.05 / m, "resamples": RESAMPLES, "tying": tying,
        "lowestAdjustedBound": min(min(c["fitAdjusted"][0], c["kmAdjusted"][0]) for c in tying), "tyingIntervals": 2 * len(tying),
        "lattice": {"n": LATTICE_N, "evalEvery": EVAL, "reps": LATTICE_REPS, "boot": LATTICE_BOOT, "median": median, "logSd": sigma, "runs": runs,
                    "coversContinuousRange": [min(r["coversContinuous"] for r in runs), max(r["coversContinuous"] for r in runs)], "coversGridRange": [min(r["coversGrid"] for r in runs), max(r["coversGrid"] for r in runs)]},
    }
    (R / "contrast-correction.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("lattice",)}, indent=None)[:1500])
    print(out["lattice"]["coversContinuousRange"], out["lattice"]["coversGridRange"], [(r["offsetInCell"], r["coversContinuous"], r["coversGrid"]) for r in runs])


if __name__ == "__main__":
    main()
