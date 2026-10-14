import sys, random; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from strategy import data, liquidity as L
d15 = pd.read_parquet("data/ETHUSDT_15m.parquet"); d45 = data.resample(d15)
lv = L.build_levels(d15, d45)
print(lv.groupby(["kind", "side"]).size().to_string())
print("untested overall:", lv.tested_at.isna().sum(), "of", len(lv))
# independent swing check with rolling max/min
n = 3
rh = d45.high.rolling(2*n+1, center=True).max(); rl = d45.low.rolling(2*n+1, center=True).min()
cnt_h = ((d45.high == rh) & (d45.high.shift(1) < d45.high) & (d45.high.rolling(2*n+1, center=True).apply(lambda w: (w == w.max()).sum(), raw=True) == 1)).sum()
cnt_l = ((d45.low == rl) & (d45.low.rolling(2*n+1, center=True).apply(lambda w: (w == w.min()).sum(), raw=True) == 1)).sum()
sw = lv[lv.kind == "swing"]
assert (sw.side == "high").sum() == cnt_h and (sw.side == "low").sum() == cnt_l, (cnt_h, cnt_l)
print("swing counts match independent rolling-window method:", cnt_h, cnt_l)
# independent tested_at recheck on random levels
random.seed(2)
for i in random.sample(range(len(lv)), 400):
    r = lv.loc[i]
    after = d15[d15.index >= r.known_at]
    hit = after[after.high >= r.price] if r.side == "high" else after[after.low <= r.price]
    exp = hit.index[0] if len(hit) else pd.NaT
    assert (pd.isna(exp) and pd.isna(r.tested_at)) or exp == r.tested_at
print("tested_at matches brute force on 400 random levels")
# no level is already traded through between formed and known (swings must be intact)
for i in random.sample(list(sw.index), 300):
    r = sw.loc[i]
    seg = d15[(d15.index >= r.formed) & (d15.index < r.known_at)]
    assert (seg.high.max() == r.price) if r.side == "high" else (seg.low.min() == r.price)
print("swing prices are the extreme of their formation window")
