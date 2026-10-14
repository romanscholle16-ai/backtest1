"""Independent auditor for a finished backtest. Deliberately shares no logic with engine/trade."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Params

M15 = pd.Timedelta(minutes=15)


def naive_sim(d15: pd.DataFrame, t: pd.Series) -> tuple[float, list[tuple[float, float]]]:
    """Plain re-implementation of the management rules for one logged trade."""
    long = t.direction == "long"
    p = Params()
    entry, stop0, target = t.entry, t.stop, t.target
    risk = abs(entry - stop0)
    pnl, rem, stop, stage, trailing = 0.0, 1.0, stop0, 0, False
    ex: list[tuple[float, float]] = []
    tp1 = None if pd.isna(t.tp1) else t.tp1
    tp2 = None if pd.isna(t.tp2) else t.tp2
    f1 = 0.0 if tp1 is None else p.tp1_fraction
    f2 = 0.0 if tp2 is None else (1 - f1) * p.tp2_fraction

    def take(price, frac):
        nonlocal pnl, rem
        pnl += frac * ((price - entry) if long else (entry - price)) / risk
        rem -= frac
        ex.append((price, frac))

    for _, b in d15[d15.index >= t.entry_time].iterrows():
        if (b.low <= stop) if long else (b.high >= stop):
            take(min(b.open, stop) if long else max(b.open, stop), rem)
            break
        while rem > 1e-12:
            if stage == 0 and tp1 is not None:
                lvl, fr, nm = tp1, f1, 1
            elif stage == 1 and tp2 is not None:
                lvl, fr, nm = tp2, f2, 2
            else:
                lvl, fr, nm = target, rem, 0
            if not ((b.high >= lvl) if long else (b.low <= lvl)):
                break
            take(lvl, min(fr, rem))
            if nm == 1:
                stage = 1
            elif nm == 2:
                stage, trailing = 2, True
        if rem <= 1e-12:
            break
        if stage >= 1:
            stop = max(stop, entry) if long else min(stop, entry)
        if trailing:
            stop = max(stop, b.low) if long else min(stop, b.high)
    if rem > 1e-12:
        take(d15.close.iloc[-1], rem)
    return pnl, ex


def audit(result, p: Params = Params()) -> list[str]:
    d15, t = result.prep.d15, result.trades
    bad: list[str] = []
    if t.empty:
        return bad
    # --- rules on entries
    eb = t.entry_time - M15                       # entry bar open
    if (eb.dt.dayofweek >= 5).any():
        bad.append("weekend entry")
    mins = eb.dt.hour * 60 + eb.dt.minute
    if not ((mins >= 6 * 60) & (mins + 15 <= 23 * 60)).all():
        bad.append("entry outside 06:00-23:00 UTC")
    if eb.dt.normalize().duplicated().any():
        bad.append("more than one entry in a UTC day")
    if (t.entry_time.iloc[1:].values < t.exit_time.iloc[:-1].values).any():
        bad.append("overlapping trades")
    # --- per trade
    eq = p.start_equity
    for i, x in t.iterrows():
        trg = result.prep.triggers[(result.prep.triggers.setup_id == x.setup_id)
                                   & (result.prep.triggers.time == x.entry_time)]
        if len(trg) != 1:
            bad.append(f"trade {i}: no unique trigger"); continue
        g, s = trg.iloc[0], result.prep.setups.loc[x.setup_id]
        bar = d15.loc[g.bar_open]
        long = x.direction == "long"
        if x.entry != bar.close:
            bad.append(f"trade {i}: entry != trigger bar close")
        if long and not (g.stop == g.fvg_bottom < g.fvg_top < x.entry):
            bad.append(f"trade {i}: long stop/FVG geometry")
        if not long and not (x.entry < g.fvg_bottom < g.fvg_top == g.stop):
            bad.append(f"trade {i}: short stop/FVG geometry")
        if not (s.armed_at <= g.bar_open < s.expires_at):
            bad.append(f"trade {i}: trigger outside setup window")
        w = d15[(d15.index > g.bar_open - pd.Timedelta(hours=24)) & (d15.index <= g.bar_open)]
        want = w.high.max() if long else w.low.min()
        if x.target != want:
            bad.append(f"trade {i}: target {x.target} != 24h extreme {want}")
        pnl, ex = naive_sim(d15, x)
        if abs(pnl - x.gross_r) > 1e-9:
            bad.append(f"trade {i}: gross R {x.gross_r} != naive {pnl}")
        fee = p.fee_per_side * (x.entry + sum(f * px for px, f in ex)) / abs(x.entry - x.stop)
        if abs(fee - x.fee_r) > 1e-9:
            bad.append(f"trade {i}: fee R mismatch")
        risk_amt = eq * p.risk_pct
        eq += (pnl - fee) * risk_amt
        if abs(eq - x.equity) > 1e-6 * eq:
            bad.append(f"trade {i}: equity chain mismatch")
    return bad
