import sys; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from strategy import data, liquidity as L, reaction as R, fvg as F, targets as T
d15 = pd.read_parquet("data/ETHUSDT_15m.parquet"); d45 = data.resample(d15)
lv = L.build_levels(d15, d45); st = R.find_setups(d45, lv); tr = F.find_triggers(d15, st, F.detect_fvgs(d15))
H, Lo, ix = d15.high.to_numpy(), d15.low.to_numpy(), d15.index
def untested_bruteforce(side, t):
    """Independent: levels known by t that no 15m bar before t has traded through."""
    c = lv[(lv.side == side) & (lv.known_at <= t)]
    out = []
    for price, known in zip(c.price, c.known_at):
        a, b = ix.searchsorted(known), ix.searchsorted(t)
        seg = H[a:b] if side == "high" else Lo[a:b]
        if not ((seg >= price).any() if side == "high" else (seg <= price).any()): out.append(price)
    return sorted(set(out))
plans, none = [], 0
for _, t in tr.iterrows():
    pl = T.make_plan(t, lv, d15)
    if pl is None: none += 1; continue
    long = pl.direction == "long"
    w = d15[(d15.index > t.bar_open - pd.Timedelta(hours=24)) & (d15.index <= t.bar_open)]
    assert len(w) == 96 and pl.target == (w.high.max() if long else w.low.min())
    cands = [x for x in untested_bruteforce("high" if long else "low", t.time)
             if (pl.entry < x < pl.target if long else pl.target < x < pl.entry)]
    cands = sorted(cands) if long else sorted(cands, reverse=True)
    assert (pl.tp1, pl.tp2) == (cands[0] if cands else None, cands[1] if len(cands) > 1 else None), (pl, cands[:3])
    assert abs(pl.tp1_frac + pl.tp2_frac + pl.runner_frac - 1) < 1e-12
    plans.append(pl)
print("triggers", len(tr), "| plans", len(plans), "| dropped (24h extreme not beyond entry)", none)
print("plans with 0/1/2 liquidity TPs:", [sum(1 for p in plans if (p.tp1 is not None) + (p.tp2 is not None) == k) for k in (0, 1, 2)])
rr = np.array([abs(p.target - p.entry) / p.risk for p in plans])
print("reward:risk to 24h extreme  median %.2f  p10 %.2f  p90 %.2f" % (np.median(rr), *np.percentile(rr, [10, 90])))
print("all plans match brute-force TP/target derivation")
