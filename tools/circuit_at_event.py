"""Exploratory, from stored curves: which applied layer holds which circuit at the event, and how the gap moves with the anchor threshold."""
import json
import sys
from collections import defaultdict

import numpy as np

arms = defaultdict(list)
for line in open(sys.argv[1]):
    r = json.loads(line)
    arms[(r["cfg"]["L"], r["cfg"]["unique"])].append(r)
THRESH = (0.08, 0.10, 0.125, 0.15, 0.20, 0.30)
print("arm | prev by layer at event | prefix by layer at event | step-0 prev | lead median | gap by anchor threshold " + str(THRESH))
for k, rows in sorted(arms.items()):
    ev = [r for r in rows if r["t_event"] is not None]
    last = [r["curve"][-1] for r in ev]
    prev = np.median([x["prev"] for x in last], axis=0).round(2).tolist()
    pre = np.median([x["prefix"] for x in last], axis=0).round(2).tolist()
    p0 = np.median([max(r["curve"][0]["prev"][: max(1, k[0] // 2)]) for r in ev])
    lead = np.median([r["t_event"] - r["t_pv"] for r in ev])
    gaps = []
    for th in THRESH:
        fr = []
        for r in ev:
            early = max(1, k[0] // 2)
            t = next((x["step"] for x in r["curve"] if max(x["prev"][:early]) >= th), None)
            if t is not None:
                fr.append(t / r["t_event"])
        gaps.append(round(float(np.median(fr)), 3) if len(fr) > len(ev) / 2 else None)
    print(k, prev, pre, round(float(p0), 3), lead, gaps)
