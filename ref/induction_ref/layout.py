"""Mirror of src/engine/layout.js. tools/parity.mjs asserts the two produce the same layout."""
import json
from pathlib import Path

DEFAULTS = json.loads((Path(__file__).resolve().parents[2] / "configs" / "defaults.json").read_text())


def resolve(cfg):
    c = {**DEFAULTS, **cfg}
    assert c["d"] % c["H"] == 0 and c["L"] % c["unique"] == 0
    c["dh"] = c["d"] // c["H"]
    c["f"] = c["mlp"] * c["d"]
    c.setdefault("rMin", c["T"] // 16)
    c.setdefault("rMax", c["T"] - c["T"] // 16)
    c["probeT"] = min(c["probeT"], c["T"])
    return c


def param_specs(cfg):
    c = resolve(cfg)
    s = []

    def add(name, shape, init, fan_in=0):
        s.append({"name": name, "shape": shape, "init": init, "fanIn": fan_in})

    add("wte", [c["V"], c["d"]], "normal")
    add("wpe", [c["T"], c["d"]], "normal")
    d, f = c["d"], c["f"]
    for u in range(c["unique"]):
        p = f"b{u}."
        if c["ln"] and c["lnAffine"]:
            add(p + "ln1g", [d], "one"); add(p + "ln1b", [d], "zero")
        add(p + "wqkv", [d, 3 * d], "kaiming", d)
        add(p + "wo", [d, d], "kaiming", d)
        if c["mlp"]:
            if c["ln"] and c["lnAffine"]:
                add(p + "ln2g", [d], "one"); add(p + "ln2b", [d], "zero")
            add(p + "w1", [d, f], "kaiming", d)
            add(p + "b1", [f], "kaiming", d)
            add(p + "w2", [f, d], "kaiming", f)
            add(p + "b2", [d], "kaiming", f)
    if c["ln"] and c["lnAffine"]:
        add("lnfg", [d], "one"); add("lnfb", [d], "zero")
    add("wu", [d, c["V"]], "kaiming", d)
    off = 0
    out = []
    for x in s:
        size = 1
        for k in x["shape"]:
            size *= k
        out.append({**x, "off": off, "size": size})
        off += size
    return out


def layout_of(cfg):
    specs = param_specs(cfg)
    total = sum(x["size"] for x in specs)
    return {"specs": specs, "by": {x["name"]: x for x in specs}, "total": total}
