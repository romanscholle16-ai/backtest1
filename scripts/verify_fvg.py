import sys; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from strategy import data, liquidity as L, reaction as R, fvg as F
from strategy.config import Params
p = Params()
d15 = pd.read_parquet("data/ETHUSDT_15m.parquet"); d45 = data.resample(d15)
lv = L.build_levels(d15, d45); st = R.find_setups(d45, lv); fv = F.detect_fvgs(d15)
tr = F.find_triggers(d15, st, fv)
print("fvgs", len(fv), "| setups", len(st), "| triggers", len(tr), "| setups with >=1 trigger", tr.setup_id.nunique())
print(tr.groupby("direction").size().to_string())
# independent FVG rebuild
o = d15
for _, f in fv.sample(300, random_state=1).iterrows():
    i = o.index.get_loc(f.start)
    a, c = o.iloc[i], o.iloc[i + 2]
    if f.direction == "bear": assert c.high < a.low and (f.bottom, f.top) == (c.high, a.low)
    else: assert c.low > a.high and (f.bottom, f.top) == (a.high, c.low)
    assert f.known_at == o.index[i + 2] + pd.Timedelta(minutes=15)
print("300 sampled FVGs verified from raw candles")
# every trigger re-derived by hand
for sid, r in tr.iterrows():
    s = st.loc[r.setup_id]; short = r.direction == "short"
    bar = d15.loc[r.bar_open]
    assert r.time == r.bar_open + pd.Timedelta(minutes=15) and r.time > s.armed_at and r.time <= s.expires_at
    assert r.entry == bar.close
    if short:
        assert bar.close < r.fvg_bottom and r.stop == r.fvg_top and r.stop > r.entry
    else:
        assert bar.close > r.fvg_top and r.stop == r.fvg_bottom and r.stop < r.entry
    # walk bars from armed_at to the trigger: gap never closed through, setup alive, penetration reached
    w = d15[(d15.index >= s.armed_at) & (d15.index <= r.bar_open)]
    w = w[w.index >= r.fvg_known_at]
    size = r.fvg_top - r.fvg_bottom
    if short:
        assert (w.close.iloc[:-1] <= r.fvg_top).all() and (d15[(d15.index>=s.armed_at)&(d15.index<=r.bar_open)].close <= s.extreme).all()
        pen = (np.minimum(w.high, r.fvg_top) - r.fvg_bottom).clip(lower=0).max() / size
    else:
        assert (w.close.iloc[:-1] >= r.fvg_bottom).all() and (d15[(d15.index>=s.armed_at)&(d15.index<=r.bar_open)].close >= s.extreme).all()
        pen = (r.fvg_top - np.maximum(w.low, r.fvg_bottom)).clip(lower=0).max() / size
    assert pen >= p.fvg_penetration - 1e-9 and abs(pen - r.penetration) < 1e-9, (pen, r.penetration)
    # gap untouched before arming
    pre = d15[(d15.index >= r.fvg_known_at) & (d15.index < s.armed_at)]
    assert not ((pre.high >= r.fvg_bottom).any() if short else (pre.low <= r.fvg_top).any())
print("all", len(tr), "triggers pass independent checks")
print("stop distance % of entry: ", (abs(tr.entry - tr.stop) / tr.entry * 100).describe().round(3).to_dict())
