"""One-factor-at-a-time arms around a centre config, for the speed search."""
import json
import sys

centre = json.loads(sys.argv[1])
grid = json.loads(sys.argv[2])
arms = [centre]
for k, vals in grid.items():
    for v in vals:
        if centre.get(k) != v:
            a = {**centre, k: v}
            if "T" in a:
                a["probeT"] = min(a.get("probeT", 128), a["T"])
            arms.append(a)
print(json.dumps(arms))
