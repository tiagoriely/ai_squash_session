# src/grammar_tools/dedup/exhaustion.py
from __future__ import annotations

class ExhaustionStopper:
    """
    Tracks consecutive duplicate (or reject) events to decide when we're 'exhausted'.

    Usage:
        stopper = ExhaustionStopper(limit=2500)
        ...
        if duplicate_or_rejected:
            stopper.register_dup()
            if stopper.exhausted:
                break
        else:
            stopper.register_unique()
    """

    def __init__(self, limit: int = 2500):
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self.limit = int(limit)
        self._consec = 0

    def register_dup(self) -> None:
        self._consec += 1

    def register_unique(self) -> None:
        self._consec = 0

    @property
    def consec(self) -> int:
        return self._consec

    @property
    def exhausted(self) -> bool:
        return self._consec >= self.limit
