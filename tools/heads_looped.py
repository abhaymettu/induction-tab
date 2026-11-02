"""Exploratory: in a one-block model applied twice, does one head play both roles, or do heads split? Prints per-head scores by pass."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "ref"))
from induction_ref.init import init_params
from induction_ref.lang import build_language, copy_probe, sample_batch
from induction_ref.model import AdamW, Flat, forward, loss_of, make_config, probes

torch.set_num_threads(2)
base = json.loads(sys.argv[1])
steps = int(sys.argv[2])
rows = []
for seed in range(1, int(sys.argv[3]) + 1):
    c = make_config({**base, "seed": seed})
    succ, cum = build_language(c["V"], c["langSeed"])
    flat = Flat(c)
    p = torch.from_numpy(init_params(c, seed).copy())
    opt = AdamW(c, p)
    ptok = torch.from_numpy(copy_probe(c["V"], c["probeB"], c["probeT"]))
    for step in range(steps):
        tok = torch.from_numpy(sample_batch(c, succ, cum, step))
        leaf = p.clone().requires_grad_(True)
        loss_of(forward(c, flat.views(leaf), tok)[0], tok).backward()
        opt.step(leaf.grad)
    pr = probes(c, flat.views(p), ptok)
    prev, pre = pr["prev"], pr["prefix"]
    both = sum(1 for h in range(c["H"]) if prev[0][h] >= 0.1 and pre[1][h] >= 0.1)
    print(seed, "copy", round(pr["copyAdv"], 2), "pass0 prev", [round(x, 2) for x in prev[0]], "pass1 prefix", [round(x, 2) for x in pre[1]], "heads doing both", both, flush=True)
    rows.append({"seed": seed, "copyAdv": pr["copyAdv"], "prev": prev, "prefix": pre, "both": both})
if len(sys.argv) > 4:
    Path(sys.argv[4]).write_text(json.dumps({"cfg": base, "steps": steps, "rows": rows}))
