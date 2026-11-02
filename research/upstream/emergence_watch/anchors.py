# Source: github.com/gunnerhowe/Research emergence-watch/emergence_watch/anchors.py at commit 493c4bf38bf0ddcc56986d9003b4a448c32439e4. MIT License, Copyright (c) 2026 Gunner Levi Howe (full text in LICENSE beside this file). Everything below this line is verbatim.
"""Anchor state machines: when does the alarm fire?

Lessons encoded as defaults (each bought with a dead result in the paper):
- Anchors are COMPOSED by default: a circuit signal alone false-alarms wherever that
  circuit pays for the task itself (the trap-language rung: 10/10 false alarms); a
  behavioral ramp alone false-alarms on early transients. The same-evaluation
  conjunction was certified at 0 false alarms across 33 manufactured negatives.
- Crossings can require SUSTAINED evaluations (absolute-criterion events use 2).
- An anchor fires ONCE per run; that moment is the forecast anchor time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Signal:
    """A named threshold condition over a probe value."""
    name: str
    threshold: float
    direction: str = ">="          # ">=" or "<="
    sustain: int = 1               # consecutive evaluations required

    _streak: int = field(default=0, repr=False)

    def crossed(self, value: float) -> bool:
        ok = value >= self.threshold if self.direction == ">=" else value <= self.threshold
        self._streak = self._streak + 1 if ok else 0
        return self._streak >= self.sustain

    def reset(self):
        self._streak = 0


@dataclass
class ComposedAnchor:
    """Fires at the first evaluation where ALL signals hold simultaneously (each with its
    own sustain). The paper's certified default is the two-signal conjunction:
    circuit precursor AND behavioral ramp, same evaluation."""
    signals: list[Signal]
    fired_at: Optional[int] = None

    def update(self, step: int, values: dict[str, float]) -> bool:
        """Feed one probe evaluation; returns True on the firing evaluation only."""
        if self.fired_at is not None:
            return False
        oks = []
        for s in self.signals:
            if s.name not in values:
                raise KeyError(f"anchor signal '{s.name}' missing from probe values "
                               f"{sorted(values)}")
            oks.append(s.crossed(values[s.name]))
        if all(oks):
            self.fired_at = step
            return True
        return False

    def reset(self):
        self.fired_at = None
        for s in self.signals:
            s.reset()


def paper_default_anchor(precursor_threshold: float = 0.10,
                         behavioral_threshold: float = 0.10) -> ComposedAnchor:
    """The composed anchor certified in the paper: early-layer previous-token score AND
    an in-distribution behavioral ramp, at the same evaluation. Thresholds are the
    paper's constants for its setting — CALIBRATE YOUR OWN on your fleet; constants are
    regime-local (that non-transfer is itself a pre-registered result)."""
    return ComposedAnchor([
        Signal("prevtok_early", precursor_threshold),
        Signal("behavioral_ramp", behavioral_threshold),
    ])
