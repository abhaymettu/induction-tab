"""Survival analysis of the seeds study, per research/PREREG.md. Writes results/study-summary.json and results/study-table.md.

    python tools/analyze_study.py results/study-lr3e-3.jsonl results/study-lr1e-3.jsonl results/study-deep.jsonl results/study-curriculum.jsonl results/deep-12-12.jsonl results/deep-12-4.jsonl results/deep-8-1-long.jsonl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

BOOT = 2000
ENV = (0.735, 0.901)


TILED = "uniform tiled"

# A longer rerun of an arm whose budget ended before any event. Its rows replace that arm, never merge with it.
SUPERSEDING = {"deep-8-1-long"}


def curriculum(c):
    return ("uniform" if c["uniform"] else "bigram") + (" tiled" if c["tile"] else " clone")


def key(c):
    return (c["L"], c["unique"], c["lr"], curriculum(c))


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
            ll = ll + torch.special.log_ndtr(-z[~ev]).sum()
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
    """Returns the arms and, for each superseded arm, what it held before."""
    arms, superseding = defaultdict(list), defaultdict(list)
    for p in paths:
        target = superseding if Path(p).stem in SUPERSEDING else arms
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            target[key(r["cfg"])].append(r)
    replaced = {k: {"n": len(arms[k]), "events": sum(r["t_event"] is not None for r in arms[k]), "budget": max(r["budget"] for r in arms[k])} for k in superseding if k in arms}
    return {**arms, **superseding}, replaced


def finite(o):
    """JSON has no Infinity or NaN, so a median never reached or an interval with no upper end becomes null."""
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main(paths):
    rng = np.random.default_rng(0)
    arms, replaced = load(paths)
    summary = {}
    for k, rows in sorted(arms.items()):
        c = rows[0]["cfg"]
        t = np.array([r["t_event"] if r["t_event"] is not None else r["budget"] for r in rows], dtype=float)
        e = np.array([r["t_event"] is not None for r in rows])
        boot = [km_median(t[i], e[i]) for i in rng.integers(0, len(t), (BOOT, len(t)))]
        pairs = [(r["t_pv"], r["t_event"]) for r in rows if r["t_event"] is not None and r["t_pv"] is not None]
        frac = np.array([a / b for a, b in pairs]) if pairs else np.array([])
        xs, ys = km(t, e)
        prefix = [max(c["prefix"]) for r in rows if r["t_event"] is not None for c in r["curve"] if c["step"] == r["t_event"]]
        summary[k] = {
            "L": k[0], "u": k[1], "lr": k[2], "curriculum": k[3], "n": len(rows), "budget": max(r["budget"] for r in rows), "events": int(e.sum()), "medianStep": km_median(t, e) if e.any() else None,
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if np.isfinite(boot).all() else None, "curve": list(zip(xs, ys)), "flopsPerStep": flops(c),
            "anchorNoFire": int(sum(r["t_event"] is not None and r["t_pv"] is None for r in rows)),
            "prefixAtEvent": [min(prefix), max(prefix)] if prefix else None,
            "gap": None if not len(frac) else {
                "median": float(np.median(frac)), "iqr": [float(np.percentile(frac, 25)), float(np.percentile(frac, 75))], "range": [float(frac.min()), float(frac.max())],
                "inEnvelope": float(np.mean((frac >= ENV[0]) & (frac <= ENV[1]))), "n": len(frac),
                "rho": spearman(np.array([a for a, _ in pairs], float), np.array([b for _, b in pairs], float)) if len(pairs) > 2 else None,
                "leadMedian": float(np.median([b - a for a, b in pairs])),
            },
            "_t": t, "_e": e, "_cost": flops(c),
        }
        if k in replaced:
            summary[k]["replaces"] = replaced[k]

    contrasts = [("tying", (2, 1), (2, 2)), ("tying", (3, 1), (3, 3)), ("tying", (4, 1), (4, 4)), ("tying", (4, 2), (4, 4)), ("depth at equal unique", (4, 2), (2, 2)), ("depth at equal unique", (3, 1), (2, 1)), ("depth at equal unique", (4, 1), (2, 1)), ("untied depth", (3, 3), (2, 2)), ("untied depth", (4, 4), (2, 2)), ("tying", (8, 1), (8, 8)), ("tying", (8, 2), (8, 8)), ("depth at equal unique", (8, 2), (4, 2)), ("untied depth", (8, 8), (4, 4)), ("tying", (12, 4), (12, 12))]
    pairs = [(kind, (*a, lr, TILED), (*b, lr, TILED)) for kind, a, b in contrasts for lr in sorted({k[2] for k in summary})]
    pairs += [("curriculum", (2, 2, 0.003, c), (2, 2, 0.003, TILED)) for c in ("bigram tiled", "bigram clone", "uniform clone")]
    out = []
    for kind, ka, kb in pairs:
        if ka not in summary or kb not in summary:
            continue
        A, B = summary[ka], summary[kb]
        a, b, lr = ka[:2] if kind != "curriculum" else ka[3], kb[:2] if kind != "curriculum" else kb[3], ka[2]
        if not A["events"]:
            # An arm with no events has no location to fit, so only the bound is reported.
            out.append({"kind": kind, "a": a, "b": b, "lr": lr, "timeRatio": None, "atLeast": float(A["_t"].min() / B["medianStep"]), "ci": None, "flopRatio": None, "flopCi": None})
            continue
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
        km_boot = [km_median(A["_t"][ia], A["_e"][ia]) / km_median(B["_t"][ib], B["_e"][ib]) for ia, ib in ((rng.integers(0, len(A["_t"]), len(A["_t"])), rng.integers(0, len(B["_t"]), len(B["_t"]))) for _ in range(BOOT))]
        km_lo, km_hi = np.percentile(km_boot, [2.5, 97.5])
        km_ratio = A["medianStep"] / B["medianStep"]
        fr = A["_cost"] / B["_cost"]
        out.append({"kind": kind, "a": a, "b": b, "lr": lr, "timeRatio": tr, "ci": [float(lo), float(hi)], "kmRatio": km_ratio, "kmCi": [float(km_lo), float(km_hi)], "flopRatio": tr * fr, "flopCi": [float(lo * fr), float(hi * fr)]})

    public = [{**{k: v for k, v in s.items() if not k.startswith("_")}, "label": f"{s['L']} layers, {s['u']} unique, lr {s['lr']:g}" + ("" if s["curriculum"] == TILED else f", {s['curriculum']}")} for s in summary.values()]
    Path("results").mkdir(exist_ok=True)
    Path("results/study-summary.json").write_text(json.dumps(finite({"arms": public, "contrasts": out}), indent=1, allow_nan=False))

    lines = ["| layers | unique | lr | curriculum | n | events | median step (95% CI) | gap median (IQR) | in envelope | rho | anchor never fired | top prefix score at the event |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in public:
        g = s["gap"]
        lines.append(f"| {s['L']} | {s['u']} | {s['lr']:g} | {s['curriculum']} | {s['n']} | {s['events']} | " + (f"{s['medianStep']} ({s['ci'][0]:.0f} to {s['ci'][1]:.0f})" if s["ci"] else f"{s['medianStep']} (interval unbounded)" if s["medianStep"] else "not reached") + " | " + (f"{g['median']:.3f} ({g['iqr'][0]:.3f} to {g['iqr'][1]:.3f}) | {g['inEnvelope']:.2f} | {g['rho']:.2f} | " if g else "n/a | n/a | n/a | ") + f"{s['anchorNoFire']} | " + (f"{s['prefixAtEvent'][0]:.3f} to {s['prefixAtEvent'][1]:.3f}" if s["prefixAtEvent"] else "n/a") + " |")
    for s in public:
        if "replaces" in s:
            was = s["replaces"]
            lines += ["", f"({s['L']},{s['u']}) at lr {s['lr']:g} is the longer rerun, {s['n']} seeds to step {s['budget']}. The earlier {was['n']} seeds had {was['events']} events by step {was['budget']} and are not merged into it."]
    lines += ["", "| contrast | lr | time ratio, steps (95% CI) | ratio of KM medians (95% CI) | time ratio, flops (95% CI) |", "|---|---|---|---|---|"]
    for r in out:
        if r["timeRatio"] is None:
            lines.append(f"| {r['kind']}: {r['a']} vs {r['b']} | {r['lr']:g} | above {r['atLeast']:.2f}, no events in budget | n/a | n/a |")
            continue
        lines.append(f"| {r['kind']}: {r['a']} vs {r['b']} | {r['lr']:g} | {r['timeRatio']:.3f} ({r['ci'][0]:.3f} to {r['ci'][1]:.3f}) | {r['kmRatio']:.3f} ({r['kmCi'][0]:.3f} to {r['kmCi'][1]:.3f}) | {r['flopRatio']:.3f} ({r['flopCi'][0]:.3f} to {r['flopCi'][1]:.3f}) |")
    Path("results/study-table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
