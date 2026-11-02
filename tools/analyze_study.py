"""Survival analysis of the seeds study, per research/PREREG.md. Writes results/study-summary.json and results/study-table.md.

    python tools/analyze_study.py results/study-lr3e-3.jsonl results/study-lr1e-3.jsonl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

BOOT = 2000
ENV = (0.735, 0.901)


def key(c):
    return (c["L"], c["unique"], c["lr"])


def flops(c):
    """Floating point multiplies per training step: forward matmuls, tripled for backward."""
    N = c["B"] * c["T"]
    d, f, V, L = c["d"], c["mlp"] * c["d"], c["V"], c["L"]
    attn = c["B"] * c["H"] * c["T"] ** 2 * (c["d"] // c["H"]) * 2
    per = N * d * 3 * d + N * d * d + 2 * N * d * f + attn
    return 3 * (L * per + N * d * V)


def km(t, e):
    order = np.argsort(t)
    t, e = t[order], e[order]
    n, s, xs, ys = len(t), 1.0, [0], [1.0]
    for i, (ti, ei) in enumerate(zip(t, e)):
        if ei:
            s *= 1 - 1 / (n - i)
            xs.append(int(ti))
            ys.append(round(s, 5))
    return xs, ys


def km_median(t, e):
    xs, ys = km(t, e)
    for x, y in zip(xs, ys):
        if y <= 0.5:
            return x
    return float("inf")


def lognormal_fit(t, e):
    """Censored log-normal maximum likelihood. Returns mu, sigma of log time."""
    lt = torch.tensor(np.log(t), dtype=torch.float64)
    ev = torch.tensor(e, dtype=torch.bool)
    mu = lt[ev].mean().clone().requires_grad_(True) if ev.any() else lt.mean().clone().requires_grad_(True)
    ls = torch.log(lt[ev].std() + 1e-3).clone().requires_grad_(True) if ev.sum() > 1 else torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
    normal = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    opt = torch.optim.LBFGS([mu, ls], max_iter=60, line_search_fn="strong_wolfe")

    def loss():
        opt.zero_grad()
        sig = ls.exp()
        z = (lt - mu) / sig
        ll = normal.log_prob(z[ev]).sum() - ev.sum() * ls
        if (~ev).any():
            ll = ll + torch.log(torch.clamp(1 - normal.cdf(z[~ev]), min=1e-12)).sum()
        nll = -ll
        nll.backward()
        return nll

    opt.step(loss)
    return mu.item(), ls.exp().item()


def spearman(a, b):
    def ranks(x):
        order = np.argsort(x)
        r = np.empty(len(x))
        r[order] = np.arange(len(x), dtype=float)
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
        return r

    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def load(paths):
    arms = defaultdict(list)
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            arms[key(r["cfg"])].append(r)
    return arms


def main(paths):
    rng = np.random.default_rng(0)
    arms = load(paths)
    summary = {}
    for k, rows in sorted(arms.items()):
        c = rows[0]["cfg"]
        t = np.array([r["t_event"] if r["t_event"] is not None else r["budget"] for r in rows], dtype=float)
        e = np.array([r["t_event"] is not None for r in rows])
        boot = [km_median(t[i], e[i]) for i in rng.integers(0, len(t), (BOOT, len(t)))]
        pairs = [(r["t_pv"], r["t_event"]) for r in rows if r["t_event"] is not None and r["t_pv"] is not None]
        frac = np.array([a / b for a, b in pairs]) if pairs else np.array([])
        xs, ys = km(t, e)
        summary[k] = {
            "L": k[0], "u": k[1], "lr": k[2], "n": len(rows), "events": int(e.sum()), "medianStep": km_median(t, e),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "curve": list(zip(xs, ys)), "flopsPerStep": flops(c),
            "anchorNoFire": int(sum(r["t_event"] is not None and r["t_pv"] is None for r in rows)),
            "gap": None if not len(frac) else {
                "median": float(np.median(frac)), "iqr": [float(np.percentile(frac, 25)), float(np.percentile(frac, 75))], "range": [float(frac.min()), float(frac.max())],
                "inEnvelope": float(np.mean((frac >= ENV[0]) & (frac <= ENV[1]))), "n": len(frac),
                "rho": spearman(np.array([a for a, _ in pairs], float), np.array([b for _, b in pairs], float)) if len(pairs) > 2 else None,
                "leadMedian": float(np.median([b - a for a, b in pairs])),
            },
            "_t": t, "_e": e, "_cost": flops(c),
        }

    contrasts = [("tying", (2, 1), (2, 2)), ("tying", (3, 1), (3, 3)), ("tying", (4, 1), (4, 4)), ("tying", (4, 2), (4, 4)), ("depth at equal unique", (4, 2), (2, 2)), ("depth at equal unique", (3, 1), (2, 1)), ("depth at equal unique", (4, 1), (2, 1)), ("untied depth", (3, 3), (2, 2)), ("untied depth", (4, 4), (2, 2))]
    out = []
    for kind, a, b in contrasts:
        for lr in sorted({k[2] for k in summary}):
            ka, kb = (*a, lr), (*b, lr)
            if ka not in summary or kb not in summary:
                continue
            A, B = summary[ka], summary[kb]
            mua, _ = lognormal_fit(A["_t"], A["_e"])
            mub, _ = lognormal_fit(B["_t"], B["_e"])
            ratios = []
            for _ in range(BOOT // 4):
                ia, ib = rng.integers(0, len(A["_t"]), len(A["_t"])), rng.integers(0, len(B["_t"]), len(B["_t"]))
                if A["_e"][ia].sum() < 2 or B["_e"][ib].sum() < 2:
                    continue
                ratios.append(lognormal_fit(A["_t"][ia], A["_e"][ia])[0] - lognormal_fit(B["_t"][ib], B["_e"][ib])[0])
            tr = float(np.exp(mua - mub))
            lo, hi = np.exp(np.percentile(ratios, [2.5, 97.5])) if ratios else (float("nan"), float("nan"))
            fr = A["_cost"] / B["_cost"]
            out.append({"kind": kind, "a": a, "b": b, "lr": lr, "timeRatio": tr, "ci": [float(lo), float(hi)], "flopRatio": tr * fr, "flopCi": [float(lo * fr), float(hi * fr)]})

    public = [{**{k: v for k, v in s.items() if not k.startswith("_")}, "label": f"{s['L']} layers, {s['u']} unique, lr {s['lr']:g}"} for s in summary.values()]
    Path("results").mkdir(exist_ok=True)
    Path("results/study-summary.json").write_text(json.dumps({"arms": public, "contrasts": out}, indent=1))

    lines = ["| layers | unique | lr | n | events | median step (95% CI) | gap median (IQR) | in envelope | rho | anchor never fired |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in public:
        g = s["gap"]
        lines.append(f"| {s['L']} | {s['u']} | {s['lr']:g} | {s['n']} | {s['events']} | {s['medianStep']} ({s['ci'][0]:.0f} to {s['ci'][1]:.0f}) | " + (f"{g['median']:.3f} ({g['iqr'][0]:.3f} to {g['iqr'][1]:.3f}) | {g['inEnvelope']:.2f} | {g['rho']:.2f} | " if g else "n/a | n/a | n/a | ") + f"{s['anchorNoFire']} |")
    lines += ["", "| contrast | lr | time ratio, steps (95% CI) | time ratio, flops (95% CI) |", "|---|---|---|---|"]
    for r in out:
        lines.append(f"| {r['kind']}: {r['a']} vs {r['b']} | {r['lr']:g} | {r['timeRatio']:.3f} ({r['ci'][0]:.3f} to {r['ci'][1]:.3f}) | {r['flopRatio']:.3f} ({r['flopCi'][0]:.3f} to {r['flopCi'][1]:.3f}) |")
    Path("results/study-table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
