# Source: github.com/gunnerhowe/Research emergence-watch/emergence_watch/events.py at commit 493c4bf38bf0ddcc56986d9003b4a448c32439e4. MIT License, Copyright (c) 2026 Gunner Levi Howe (full text in LICENSE beside this file). Everything below this line is verbatim.
"""Event criteria: when has the capability actually arrived?

Hard-won rule, encoded here after it broke twice in our own frozen specs: events are
ABSOLUTE behavioral crossings with a sustain requirement, never a percentage of a
run's own maximum, which is silently corrupted by any late-training outlier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AbsoluteEvent:
    """Fires at the evaluation where `metric >= threshold` has held for `sustain`
    consecutive evaluations; `fired_at` is that completing evaluation's step (the
    convention under which every number in the paper was scored). The paper's induction
    event: copy_adv >= 2.0 nats, sustain=2."""
    metric: str
    threshold: float
    sustain: int = 2
    fired_at: Optional[int] = None

    _streak: int = field(default=0, repr=False)

    def update(self, step: int, values: dict[str, float]) -> bool:
        if self.fired_at is not None:
            return False
        v = values.get(self.metric)
        if v is None:
            raise KeyError(f"event metric '{self.metric}' missing from {sorted(values)}")
        if v >= self.threshold:
            self._streak += 1
            if self._streak >= self.sustain:
                self.fired_at = step
                return True
        else:
            self._streak = 0
        return False

    def reset(self):
        self.fired_at = None
        self._streak = 0


def paper_induction_event() -> AbsoluteEvent:
    """The frozen induction event from the paper: copy advantage >= 2.0 nats (far from
    both the ~0 noise floor and the ~10-12 nat plateau), sustained 2 evaluations."""
    return AbsoluteEvent(metric="copy_adv", threshold=2.0, sustain=2)
