"""Writes the PyTorch side of the gradient and trajectory parity check. Driven by tools/parity.mjs."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from induction_ref.init import init_params
from induction_ref.lang import build_language, copy_probe, sample_batch
from induction_ref.model import AdamW, Flat, forward, loss_of, make_config, probes

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--out", required=True)
a = ap.parse_args()

torch.set_num_threads(4)
c = make_config(json.loads(a.cfg))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
succ, cum = build_language(c["V"], c["langSeed"])
flat = Flat(c)
p = torch.from_numpy(init_params(c, c["seed"]).copy())
opt = AdamW(c, p)
losses = []
for step in range(a.steps):
    tok = torch.from_numpy(sample_batch(c, succ, cum, step))
    leaf = p.clone().requires_grad_(True)
    loss = loss_of(forward(c, flat.views(leaf), tok)[0], tok)
    loss.backward()
    losses.append(loss.item())
    if step == 0:
        tok.numpy().astype(np.uint32).tofile(out / "tok0.bin")
        leaf.grad.numpy().astype(np.float32).tofile(out / "grad0.bin")
    opt.step(leaf.grad)
    if step == 0:
        p.numpy().astype(np.float32).tofile(out / "p1.bin")
p.numpy().astype(np.float32).tofile(out / "pN.bin")
leaf = p.clone().requires_grad_(True)
tok = torch.from_numpy(sample_batch(c, succ, cum, a.steps))
loss_of(forward(c, flat.views(leaf), tok)[0], tok).backward()
leaf.grad.numpy().astype(np.float32).tofile(out / "gradN.bin")
ptok = torch.from_numpy(copy_probe(c["V"], c["probeB"], c["probeT"]))
pr = probes(c, flat.views(p), ptok)
(out / "ref.json").write_text(json.dumps({"losses": losses, "probe": pr}))
