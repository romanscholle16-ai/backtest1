import sys; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from collections import Counter
from strategy import data, liquidity as L, reaction as R, fvg as F, targets as T, trade as TR
d15 = pd.read_parquet("data/ETHUSDT_15m.parquet"); d45 = data.resample(d15)
lv = L.build_levels(d15, d45); st = R.find_setups(d45, lv); tr = F.find_triggers(d15, st, F.detect_fvgs(d15))

def naive(plan, entry_time):
    """Independent, deliberately plain re-implementation using explicit state and per-direction branches."""
    rows = d15[d15.index >= entry_time]
    long = plan.direction == "long"
    better = (lambda a, b: a >= b) if long else (lambda a, b: a <= b)   # a is at/over b in profit direction
    worse = (lambda a, b: a <= b) if long else (lambda a, b: a >= b)
    pnl, rem, stop, stage, trailing = 0.0, 1.0, plan.stop, 0, False
    risk = abs(plan.entry - plan.stop)
    exits = []
    def take(price, frac, why):
        nonlocal pnl, rem
        pnl += frac * ((price - plan.entry) if long else (plan.entry - price)) / risk
        rem -= frac; exits.append((price, frac, why))
    for t, b in rows.iterrows():
        hi, lo, op = b.high, b.low, b.open
        adverse = lo if long else hi; favour = hi if long else lo
        if worse(adverse, stop) :  # stop touched
            fill = min(op, stop) if long else max(op, stop)
            take(fill, rem, "stop"); break
        while rem > 1e-12:
            if stage == 0 and plan.tp1 is not None: lvl, fr, nm = plan.tp1, plan.tp1_frac, "tp1"
            elif stage == 1 and plan.tp2 is not None: lvl, fr, nm = plan.tp2, plan.tp2_frac, "tp2"
            else: lvl, fr, nm = plan.target, rem, "target"
            if not better(favour, lvl): break
            take(lvl, min(fr, rem), nm)
            if nm == "tp1": stage = 1
            elif nm == "tp2": stage = 2; trailing = True
        if rem <= 1e-12: break
        if stage >= 1: stop = max(stop, plan.entry) if long else min(stop, plan.entry)
        if trailing: stop = max(stop, lo) if long else min(stop, hi)
    else:
        pass
    if rem > 1e-12:
        take(d15.close.iloc[-1], rem, "eod")
    return pnl, exits

n = 0; reasons = Counter(); rs = []
for sid, t in tr.iterrows():
    pl = T.make_plan(t, lv, d15)
    res = TR.simulate(pl, d15, t.time)
    pnl, ex = naive(pl, t.time)
    assert abs(pnl - res.gross_r) < 1e-9, (t.time, pnl, res.gross_r, ex, res.legs)
    assert abs(sum(g.frac for g in res.legs) - 1) < 1e-9
    for g in res.legs:   # every fill price lies inside its bar's range (or equals open on gaps)
        if g.reason == "eod": continue
        b = d15.loc[g.time]
        assert b.low - 1e-9 <= g.price <= b.high + 1e-9, (g, b.to_dict())
        assert g.time > t.bar_open
    assert res.legs[-1].time >= t.time
    reasons.update(g.reason for g in res.legs); rs.append(res.net_r); n += 1
print("trades simulated:", n, "- naive re-implementation agrees on all")
print("leg reasons:", dict(reasons))
rs = np.array(rs)
print("net R: mean %.3f  median %.3f  win%% %.1f  (informational only — overlaps/filters not applied yet)" % (rs.mean(), np.median(rs), (rs > 0).mean() * 100))
