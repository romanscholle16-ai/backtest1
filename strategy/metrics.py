"""Summary statistics for a trade log."""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(trades: pd.DataFrame, start_equity: float) -> dict:
    if trades.empty:
        return {"trades": 0}
    r = trades["net_r"]
    eq = pd.concat([pd.Series([start_equity]), trades["equity"]], ignore_index=True)
    dd = (eq / eq.cummax() - 1).min()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    months = max((trades["exit_time"].max() - trades["entry_time"].min()).days / 30.44, 1e-9)
    return {
        "trades": len(trades), "trades_per_month": len(trades) / months,
        "win_rate_%": (r > 0).mean() * 100, "avg_R": r.mean(), "median_R": r.median(),
        "total_R": r.sum(), "profit_factor": gp / gl if gl else np.inf,
        "return_%": (eq.iloc[-1] / start_equity - 1) * 100, "max_drawdown_%": dd * 100,
        "avg_fee_R": trades["fee_r"].mean(), "avg_stop_%": trades["risk_pct_of_price"].mean(),
        "longs": int((trades.direction == "long").sum()), "shorts": int((trades.direction == "short").sum()),
    }
