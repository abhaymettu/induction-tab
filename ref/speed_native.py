"""Native PyTorch step time on this machine, for comparison with the browser trainer. Writes results/native-speed.json.

    python ref/speed_native.py configs/tab.json cpu mps
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from induction_ref.init import init_params
from induction_ref.lang import build_language, sample_batch
from induction_ref.model import AdamW, Flat, forward, loss_of, make_config

WARM, STEPS = 20, 200


def timed(cfg, device):
    c = make_config({**cfg, "seed": 1})
    succ, cum = build_language(c["V"], c["langSeed"])
    flat = Flat(c)
    p = torch.from_numpy(init_params(c, 1).copy()).to(device)
    opt = AdamW(c, p)
    batches = [torch.from_numpy(sample_batch(c, succ, cum, s)) for s in range(WARM + STEPS)]
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    times = []
    for step, tok in enumerate(batches):
        t0 = time.perf_counter()
        tok = tok.to(device)
        leaf = p.clone().requires_grad_(True)
        loss_of(forward(c, flat.views(leaf), tok)[0], tok).backward()
        opt.step(leaf.grad)
        sync()
        if step >= WARM:
            times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    out = {"cfg": sys.argv[1], "torch": torch.__version__, "machine": platform.processor(), "threads": torch.get_num_threads()}
    for device in sys.argv[2:]:
        out[device + "MsPerStep"] = timed(cfg, device)
    Path("results/native-speed.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
