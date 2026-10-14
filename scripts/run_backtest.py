"""Run the full backtest on real data, audit it, and write reports/."""
import sys, time
sys.path.insert(0, ".")
from pathlib import Path
import pandas as pd
from strategy import engine, audit, metrics
from strategy.config import Params

p = Params()
d15 = pd.read_parquet("data/ETHUSDT_15m.parquet")
t0 = time.time()
res = engine.run_backtest(d15, p)
print(f"pipeline: {len(res.prep.levels)} levels, {len(res.prep.setups)} setups, "
      f"{len(res.prep.triggers)} triggers -> {len(res.trades)} trades ({time.time()-t0:.0f}s)")
print("skipped triggers:", res.skipped.reason.value_counts().to_dict())
problems = audit.audit(res, p)
print("AUDIT:", "clean" if not problems else problems[:10])
for k, v in metrics.summarize(res.trades, p.start_equity).items():
    print(f"  {k:18s} {v:,.3f}" if isinstance(v, float) else f"  {k:18s} {v}")
Path("reports").mkdir(exist_ok=True)
res.trades.to_csv("reports/trades.csv", index=False)
print("exit combos:", res.trades.exits.value_counts().to_dict())
