"""The reference model: plain PyTorch over the same flat parameter layout as the WGSL trainer.

Autograd supplies the gradients the hand-written WGSL backward pass is checked against, and the same
module runs the CPU seed studies.
"""
import math

import torch
import torch.nn.functional as F

from .layout import layout_of, resolve


class Flat:
    def __init__(self, c):
        self.lay = layout_of(c)

    def views(self, p):
        return {s["name"]: p[s["off"]:s["off"] + s["size"]].view(*s["shape"]) for s in self.lay["specs"]}


def layer_norm(x, w, g, b):
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    y = (x - mean) * (var + 1e-5).pow(-0.5)
    return y * w[g] + w[b] if g in w else y


def forward(c, w, tok, ablate=None, keep_attn=False):
    B, T = tok.shape
    d, H, dh = c["d"], c["H"], c["dh"]
    x = w["wte"][tok] + w["wpe"][torch.arange(T, device=tok.device)]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tok.device), 1)
    attn = []
    for l in range(c["L"]):
        p = f"b{l % c['unique']}."
        n = layer_norm(x, w, p + "ln1g", p + "ln1b") if c["ln"] else x
        q, k, v = (n @ w[p + "wqkv"]).chunk(3, -1)
        q = q.view(B, T, H, dh).transpose(1, 2)
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(dh)
        a = s.masked_fill(mask, float("-inf")).softmax(-1)
        if keep_attn:
            attn.append(a.detach())
        o = (a @ v).transpose(1, 2).reshape(B, T, d)
        x = x + o @ w[p + "wo"]
        if c["mlp"]:
            n = layer_norm(x, w, p + "ln2g", p + "ln2b") if c["ln"] else x
            x = x + F.gelu(n @ w[p + "w1"] + w[p + "b1"]) @ w[p + "w2"] + w[p + "b2"]
    if c["ln"]:
        x = layer_norm(x, w, "lnfg", "lnfb")
    return x @ w["wu"], attn


def loss_of(logits, tok):
    V = logits.shape[-1]
    return F.cross_entropy(logits[:, :-1].reshape(-1, V), tok[:, 1:].reshape(-1))


class AdamW:
    """torch.optim.AdamW semantics on the flat vector, in float32, with the linear warmup."""

    def __init__(self, c, p):
        self.c, self.p, self.t = c, p, 0
        self.m = torch.zeros_like(p)
        self.v = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, g):
        c = self.c
        self.t += 1
        lr = c["lr"] * min(1.0, self.t / c["warmup"]) if c["warmup"] else c["lr"]
        self.m.mul_(c["b1"]).add_(g, alpha=1 - c["b1"])
        self.v.mul_(c["b2"]).addcmul_(g, g, value=1 - c["b2"])
        mh = self.m / (1 - c["b1"] ** self.t)
        vh = self.v / (1 - c["b2"] ** self.t)
        self.p.mul_(1 - lr * c["wd"]).sub_(lr * mh / (vh.sqrt() + c["eps"]))


@torch.no_grad()
def probes(c, w, tok):
    """Instruments of 2609.19000 on a [x ; x] batch: copy advantage, previous-token and prefix-match scores."""
    logits, attn = forward(c, w, tok, keep_attn=True)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, c["V"]), tok[:, 1:].reshape(-1), reduction="none").view(tok.shape[0], -1)
    T = tok.shape[1]
    L = T // 2
    copy_adv = (ce[:, : L - 1].mean() - ce[:, L - 1:].mean()).item()
    q = torch.arange(1, T, device=tok.device)
    qp = torch.arange(L, T - 1, device=tok.device)
    prev, prefix = [], []
    for a in attn:
        prev.append(a[:, :, q, q - 1].mean(dim=(0, 2)).tolist())
        prefix.append(a[:, :, qp, qp - L + 1].mean(dim=(0, 2)).tolist())
    return {"copyAdv": copy_adv, "prev": prev, "prefix": prefix}


def make_config(cfg):
    return resolve(cfg)
