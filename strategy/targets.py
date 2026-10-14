"""Module 5: 24h-extreme target and nearest-liquidity TP levels for a triggered trade."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Params
from .liquidity import untested_at

TF_15M = pd.Timedelta(minutes=15)


@dataclass(frozen=True)
class Plan:
    direction: str
    entry: float
    stop: float
    target: float                 # 24h high (long) / 24h low (short) at entry time
    tp1: float | None             # nearest untested liquidity in the profit direction
    tp2: float | None             # next one, only if it is still before `target`
    tp1_frac: float               # fraction of the ORIGINAL size closed at tp1
    tp2_frac: float               # fraction of the ORIGINAL size closed at tp2
    runner_frac: float            # what is left for the target / trail

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)


def extreme_24h(df15: pd.DataFrame, bar_open: pd.Timestamp, p: Params = Params()) -> tuple[float, float]:
    """(24h high, 24h low) over the 96 15m bars ending with (and including) the entry bar."""
    w = df15.loc[bar_open - (p.lookback_24h_bars - 1) * TF_15M: bar_open]
    return float(w["high"].max()), float(w["low"].min())


def make_plan(trigger: pd.Series, levels: pd.DataFrame, df15: pd.DataFrame,
              p: Params = Params()) -> Plan | None:
    """Build the exit plan, or None if the 24h extreme is not beyond the entry."""
    long = trigger["direction"] == "long"
    entry, stop = float(trigger["entry"]), float(trigger["stop"])
    hi24, lo24 = extreme_24h(df15, trigger["bar_open"], p)
    target = hi24 if long else lo24
    if (target <= entry) if long else (target >= entry):
        return None
    lv = untested_at(levels, trigger["time"], side="high" if long else "low")
    lo_b, hi_b = (entry, target) if long else (target, entry)
    prices = sorted({float(x) for x in lv["price"] if lo_b < x < hi_b}, reverse=not long)
    tp1 = prices[0] if prices else None
    tp2 = prices[1] if len(prices) > 1 else None
    if tp1 is None:
        f1, f2 = 0.0, 0.0
    elif tp2 is None:
        f1, f2 = p.tp1_fraction, 0.0
    else:
        f1 = p.tp1_fraction
        f2 = (1 - f1) * p.tp2_fraction
    return Plan(trigger["direction"], entry, stop, target, tp1, tp2, f1, f2, 1 - f1 - f2)
