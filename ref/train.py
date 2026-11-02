"""Native reference trainer: same language, batches, init and instruments as the WGSL trainer.

    python ref/train.py --cfg '{"d":64}' --seeds 1-8 --budget 8000 --stop-after-event 200 --jobs 4 --out results/x.jsonl

One JSON line per run. It holds t_event (copy advantage >= 2 nats on two consecutive evals, reported at
the second, as in 2609.19000), t_pv (first eval whose first-half-of-layers previous-token score is >= 0.10),
and the eval curves when --curves is set.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from induction_ref.init import init_params
from induction_ref.lang import build_language, copy_probe, sample_batch
from induction_ref.model import AdamW, Flat, forward, loss_of, make_config, probes


def run_one(job):
    cfg, seed, budget, stop_after, curves, threads = job
    torch.set_num_threads(threads)
    c = make_config({**cfg, "seed": seed})
    succ, cum = build_language(c["V"], c["langSeed"])
    flat = Flat(c)
    p = torch.from_numpy(init_params(c, seed).copy())
    opt = AdamW(c, p)
    ptok = torch.from_numpy(copy_probe(c["V"], c["probeB"], c["probeT"]))
    t0 = time.perf_counter()
    prev_ok = False
    t_event = t_pv = None
    early = max(1, c["L"] // 2)
    rows = []
    last_loss = None
    final = None
    for step in range(budget):
        tok = torch.from_numpy(sample_batch(c, succ, cum, step))
        leaf = p.clone().requires_grad_(True)
        loss = loss_of(forward(c, flat.views(leaf), tok)[0], tok)
        loss.backward()
        last_loss = loss.item()
        if not np.isfinite(last_loss):
            break
        opt.step(leaf.grad)
        if step % c["evalEvery"] == 0:
            pr = probes(c, flat.views(p), ptok)
            pv = max(max(h) for h in pr["prev"][:early])
            final = {"step": step, "loss": last_loss, "copyAdv": pr["copyAdv"], "prev": [max(h) for h in pr["prev"]], "prefix": [max(h) for h in pr["prefix"]]}
            if curves:
                rows.append(final)
            if t_pv is None and pv >= 0.10:
                t_pv = step
            ok = pr["copyAdv"] >= 2.0
            if t_event is None and ok and prev_ok:
                t_event = step
            prev_ok = ok
            if t_event is not None and step >= t_event + stop_after:
                break
    out = {"cfg": cfg, "seed": seed, "t_event": t_event, "t_pv": t_pv, "budget": budget, "steps_run": step + 1, "wall_s": time.perf_counter() - t0, "final": final, "params": flat.lay["total"]}
    if curves:
        out["curve"] = rows
    return out


def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="JSON object, or @file holding a JSON list of arms")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--budget", type=int, default=8000)
    ap.add_argument("--stop-after-event", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = json.loads(Path(a.cfg[1:]).read_text()) if a.cfg.startswith("@") else [json.loads(a.cfg)]
    jobs = [(arm, s, a.budget, a.stop_after_event, a.curves, a.threads) for arm in arms for s in parse_seeds(a.seeds)]
    done = set()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((json.dumps(r["cfg"], sort_keys=True), r["seed"]))
    jobs = [j for j in jobs if (json.dumps(j[0], sort_keys=True), j[1]) not in done]
    with mp.get_context("spawn").Pool(a.jobs) as pool, out.open("a") as f:
        for r in pool.imap_unordered(run_one, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["seed"], r["t_event"], r["t_pv"], round(r["wall_s"], 1), r["final"]["copyAdv"] if r["final"] else None, flush=True)
