import sys; sys.path.insert(0, ".")
import pandas as pd
from strategy import data, liquidity as L, reaction as R
d15 = pd.read_parquet("data/ETHUSDT_15m.parquet"); d45 = data.resample(d15)
lv = L.build_levels(d15, d45); s = R.find_setups(d45, lv)
print("setups:", len(s)); print(s.groupby(["direction", "level_kind"]).size().to_string())
print("per month avg:", round(len(s) / (len(d45) / 32 / 30.4), 1))
# independent re-check of each setup's invariants straight from the candles
atr = R.atr(d45, 14).shift(1)
for _, r in s.iterrows():
    sgn = 1 if r.direction == "short" else -1
    j = d45.index.get_loc(r.sweep_time); d = d45.index.get_loc(r.disp_time)
    bar = d45.iloc[j]
    if sgn == 1:
        assert bar.high >= r.level_price > bar.close                      # swept + rejected
        assert (d45.close.iloc[j:d + 1] <= r.extreme).all()               # never closed above extreme
        assert d45.close.iloc[d] < d45.low.iloc[d - 1] and d45.open.iloc[d] - d45.close.iloc[d] >= atr.iloc[d]
    else:
        assert bar.low <= r.level_price < bar.close
        assert (d45.close.iloc[j:d + 1] >= r.extreme).all()
        assert d45.close.iloc[d] > d45.high.iloc[d - 1] and d45.close.iloc[d] - d45.open.iloc[d] >= atr.iloc[d]
    # level untested before the sweep bar (brute force on 15m)
    prior = d15[d15.index < r.sweep_time]
    cand = lv[(lv.side == ("high" if sgn == 1 else "low")) & (lv.price == r.level_price) & (lv.known_at <= r.sweep_time)]
    assert len(cand), "level not found"
    ok = False
    for _, c in cand.iterrows():
        pr = prior[prior.index >= c.known_at]
        if not ((pr.high >= c.price).any() if sgn == 1 else (pr.low <= c.price).any()):
            ok = True
    assert ok, "level was already tested"
print("all", len(s), "setups pass independent invariant checks")
print(s.head(5).to_string())
