"""Module 7: trading-window and exposure filters."""
from __future__ import annotations

import pandas as pd

from .config import Params

TF_15M = pd.Timedelta(minutes=15)


def in_session(bar_open: pd.Timestamp, p: Params = Params()) -> bool:
    """Entry bar must lie fully inside Mon-Fri [06:00, 23:00) UTC (so its close is <= 23:00)."""
    if bar_open.weekday() >= 5:
        return False
    mins = bar_open.hour * 60 + bar_open.minute
    return p.session_start_hour * 60 <= mins and mins + 15 <= p.session_end_hour * 60


class EntryGate:
    """Sequential gate: session window, max 1 entry per UTC day, one position at a time."""

    def __init__(self, p: Params = Params()):
        self.p = p
        self.days: set = set()
        self.busy_until: pd.Timestamp | None = None

    def check(self, bar_open: pd.Timestamp) -> str | None:
        """None if an entry is allowed, else the reason it is refused."""
        if not in_session(bar_open, self.p):
            return "outside_session"
        if bar_open.normalize() in self.days:
            return "daily_limit"
        if self.busy_until is not None and bar_open + TF_15M < self.busy_until:
            return "position_open"
        return None

    def register(self, bar_open: pd.Timestamp, exit_bar_open: pd.Timestamp) -> None:
        self.days.add(bar_open.normalize())
        self.busy_until = exit_bar_open + TF_15M  # position is flat once the exit bar closes
