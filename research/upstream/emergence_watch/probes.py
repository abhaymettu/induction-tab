# Source: github.com/gunnerhowe/Research emergence-watch/emergence_watch/probes.py at commit 493c4bf38bf0ddcc56986d9003b4a448c32439e4. MIT License, Copyright (c) 2026 Gunner Levi Howe (full text in LICENSE beside this file). Everything below this line is verbatim.
"""Probe instruments: the measurements behind emergence forecasting.

Framework-agnostic core: every scorer consumes plain tensors (attention weights, logits,
token ids), so any training stack can feed them. A HuggingFace convenience layer
(`capture_attentions`) handles the eager-attention wrinkle: production kernels
(flash/sdpa) never materialize attention weights, so the probe forward — a single small
batch — temporarily requests eager attention. Validated against Pythia, OLMo-1 and
OLMo-2 public checkpoints (see the paper's cross-family replication).

All scores are in [0, 1] except copy_advantage (nats; positive once in-context copying
works). Conventions match the paper exactly:
  prefix_matching_score : induction pattern — attention from second-occurrence queries to
                          (previous occurrence + 1). The capability's own circuit.
  prev_token_score      : attention to the immediately preceding position. The known
                          mechanistic PRECURSOR of induction (Olsson et al., 2022).
  copy_advantage        : CE(first half) - CE(second half) on repeated random sequences.
"""
from __future__ import annotations

import contextlib
from typing import Callable, Sequence

import torch
import torch.nn.functional as F


def repeated_probe_batch(vocab_size: int, batch: int = 64, half_len: int = 64,
                         seed: int = 1234, lo: int = 10,
                         device: str | torch.device = "cpu") -> torch.Tensor:
    """Fixed, deterministic batch of repeated random sequences: [x ; x] with x uniform.

    The same seed must be used for every probe call within a run (and across runs you
    intend to compare) — the probe batch is part of the instrument.
    """
    g = torch.Generator().manual_seed(seed)
    hi = max(lo + 1, vocab_size - lo)
    half = torch.randint(lo, hi, (batch, half_len), generator=g)
    return torch.cat([half, half], dim=1).to(device)


def ce_per_position(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Per-position next-token cross-entropy. logits (B,T,V), ids (B,T) -> (B,T-1)."""
    lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    return -lp.gather(2, ids[:, 1:, None]).squeeze(2)


def copy_advantage(logits: torch.Tensor, ids: torch.Tensor, half_len: int) -> float:
    """CE(first half) - CE(second half) on a repeated batch. Positive = copying works."""
    ce = ce_per_position(logits, ids)
    first = ce[:, : half_len - 1].mean().item()
    second = ce[:, half_len - 1:].mean().item()
    return first - second


def prefix_matching_score(attn: torch.Tensor, half_len: int) -> float:
    """Max-over-heads induction score for one layer's attention (B,H,T,T) on a repeated
    batch of half length L: mean attention from query position L+i to key position i+1."""
    B, H, T, _ = attn.shape
    L = half_len
    q = torch.arange(L, 2 * L - 1, device=attn.device)
    tgt = q - L + 1
    a = attn.float()[:, :, q, :]
    score = a.gather(3, tgt.view(1, 1, -1, 1).expand(B, H, len(q), 1)).squeeze(3)
    return float(score.mean(dim=(0, 2)).max())


def prev_token_score(attn: torch.Tensor) -> float:
    """Max-over-heads previous-token score for one layer's attention (B,H,T,T)."""
    B, H, T, _ = attn.shape
    qq = torch.arange(1, T, device=attn.device)
    a = attn.float()[:, :, qq, :]
    score = a.gather(3, (qq - 1).view(1, 1, -1, 1).expand(B, H, len(qq), 1)).squeeze(3)
    return float(score.mean(dim=(0, 2)).max())


def early_layer_prev_token_score(attentions: Sequence[torch.Tensor]) -> float:
    """The paper's frozen precursor signal: max prev-token score over the FIRST HALF of
    layers (the precursor circuit lives below the induction layer it feeds)."""
    n = len(attentions)
    early = attentions[: max(1, n // 2)]
    return max(prev_token_score(a) for a in early)


@contextlib.contextmanager
def eager_attention(model):
    """Temporarily request eager attention on a HuggingFace model so attention weights
    are materialized for the probe forward. No-op for models without the knob."""
    cfgs = [model.config] + [getattr(model.config, "text_config", None)]
    saved = []
    for c in cfgs:
        if c is not None and hasattr(c, "_attn_implementation"):
            saved.append((c, c._attn_implementation))
            c._attn_implementation = "eager"
    try:
        yield
    finally:
        for c, v in saved:
            c._attn_implementation = v


@torch.no_grad()
def capture_attentions(model, ids: torch.Tensor):
    """HF-model probe forward: returns (logits, [per-layer attention (B,H,T,T)]).

    For non-HF models, pass your own capture function to the monitor instead — anything
    returning this pair works.
    """
    was_training = model.training
    model.eval()
    try:
        with eager_attention(model):
            out = model(ids, output_attentions=True)
    finally:
        if was_training:
            model.train()
    return out.logits, list(out.attentions)


def induction_probe(model, ids: torch.Tensor, half_len: int,
                    capture: Callable | None = None) -> dict:
    """One probe evaluation: the three paper signals on a repeated batch."""
    cap = capture or capture_attentions
    logits, attentions = cap(model, ids)
    return {
        "copy_adv": copy_advantage(logits, ids, half_len),
        "prefix_max": max(prefix_matching_score(a, half_len) for a in attentions),
        "prevtok_early": early_layer_prev_token_score(attentions),
    }
