import sys; sys.path.insert(0, ".")
import pandas as pd
from strategy import data
d15 = pd.read_parquet("data/ETHUSDT_15m.parquet")
d45 = data.resample(d15)
print("15m bars", len(d15), "| 45m bars", len(d45), "| expected", len(d15)//3)
assert len(d45) == len(d15)//3 and data.validate(d15) == []
assert (d45.index.hour * 60 + d45.index.minute).isin(range(0, 1440, 45)).all()
# independent cross-check: random 45m bars rebuilt by hand from 15m
import random; random.seed(1)
for t in random.sample(list(d45.index), 200):
    w = d15.loc[t: t + pd.Timedelta(minutes=44)]
    r = d45.loc[t]
    assert (w.open.iloc[0], w.high.max(), w.low.min(), w.close.iloc[-1]) == (r.open, r.high, r.low, r.close)
print("price range", d15.low.min(), d15.high.max(), "| OK: 200 random 45m bars match manual rebuild")
