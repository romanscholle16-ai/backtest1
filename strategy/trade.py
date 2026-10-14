"""Module 6: trade management — TP1 50%, break-even, TP2, trail by prior 15m candle.

Bar-by-bar on 15m candles, starting with the bar AFTER the entry bar. Rules inside a bar:
  1. The stop in force at the START of the bar is checked first (conservative: if stop and a
     target are both inside one bar, the stop wins). A bar opening beyond the stop fills at the open.
  2. Otherwise targets are filled in order (nearest first) at their limit prices. Several can
     fill in the same bar.
  3. A stop change (break-even after TP1, trail after TP2) only takes effect from the NEXT bar.
Trail: after TP2 the stop ratchets to the extreme of the previous completed 15m candle
(low for longs, high for shorts), never loosening.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import Params
from .targets import Plan

TF_15M = pd.Timedelta(minutes=15)


@dataclass
class Leg:
    time: pd.Timestamp
    price: float
    frac: float
    reason: str


@dataclass
class TradeResult:
    direction: str
    entry_time: pd.Timestamp
    entry: float
    stop0: float
    legs: list[Leg] = field(default_factory=list)
    gross_r: float = 0.0
    fee_r: float = 0.0
    open_at_end: bool = False

    @property
    def net_r(self) -> float:
        return self.gross_r - self.fee_r

    @property
    def exit_time(self) -> pd.Timestamp:
        return self.legs[-1].time


def simulate(plan: Plan, df15: pd.DataFrame, entry_time: pd.Timestamp,
             p: Params = Params()) -> TradeResult:
    """Run the trade. `entry_time` = close time of the trigger bar (first managed bar opens then)."""
    sgn = 1 if plan.direction == "long" else -1
    # long-orientation arrays
    o = (sgn * df15["open"]).to_numpy()
    h = (df15["high"] if sgn == 1 else -df15["low"]).to_numpy()
    l = (df15["low"] if sgn == 1 else -df15["high"]).to_numpy()
    c = (sgn * df15["close"]).to_numpy()
    idx = df15.index
    entry, stop = sgn * plan.entry, sgn * plan.stop
    target = sgn * plan.target
    tp = [None if plan.tp1 is None else sgn * plan.tp1, None if plan.tp2 is None else sgn * plan.tp2]
    fr = [plan.tp1_frac, plan.tp2_frac]
    risk = abs(entry - stop)

    res = TradeResult(plan.direction, entry_time, plan.entry, plan.stop)
    remaining, stage = 1.0, 0   # stage = number of liquidity TPs already filled
    i = int(idx.searchsorted(entry_time, side="left"))
    cur_stop, trail_on = stop, False
    last = len(idx) - 1

    def record(t, price, frac, reason):
        res.legs.append(Leg(t, sgn * price, frac, reason))
        res.gross_r += frac * (price - entry) / risk

    while i <= last and remaining > 1e-12:
        t = idx[i]
        # 1. stop in force at the start of this bar
        if l[i] <= cur_stop:
            reason = "stop" if stage == 0 else ("be" if not trail_on else "trail")
            record(t, min(o[i], cur_stop), remaining, reason)
            remaining = 0
            break
        # 2. targets, nearest first
        while remaining > 1e-12:
            if stage < 2 and tp[stage] is not None:
                px, frac, reason = tp[stage], fr[stage], f"tp{stage + 1}"
            else:
                px, frac, reason = target, remaining, "target"
            if h[i] >= px:
                frac = min(frac, remaining)
                record(t, px, frac, reason)
                remaining -= frac
                if reason == "target":
                    remaining = 0
                else:
                    stage += 1
                    if stage == 2:
                        trail_on = True
            else:
                break
        if remaining <= 1e-12:
            break
        # 3. stop updates for next bar
        if stage >= 1:
            cur_stop = max(cur_stop, entry)
        if trail_on:
            cur_stop = max(cur_stop, l[i])
        i += 1
    if remaining > 1e-12:  # data ended with the trade still open
        record(idx[last], c[last], remaining, "eod")
        res.open_at_end = True
    # fees: entry + every exit leg, on notional sized so a stop-out = 1R
    res.fee_r = p.fee_per_side * (abs(entry) + sum(leg.frac * abs(leg.price) for leg in res.legs)) / risk
    return res
