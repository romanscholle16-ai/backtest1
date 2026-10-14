"""Module 2: untested liquidity levels (45m swings + previous day/week high/low)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Params

TF_45M = pd.Timedelta(minutes=45)
TF_15M = pd.Timedelta(minutes=15)
LEVEL_COLS = ["kind", "side", "price", "formed", "known_at"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"kind": [], "side": [], "price": [], "formed": pd.to_datetime([], utc=True),
                         "known_at": pd.to_datetime([], utc=True)})


def swing_levels(df45: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Strict fractal swings: higher (lower) than `n` bars on each side.

    A swing at bar i is only knowable once bar i+n has closed.
    """
    h, l = df45["high"].to_numpy(), df45["low"].to_numpy()
    idx = df45.index
    rows = []
    for i in range(n, len(df45) - n):
        win_h = np.concatenate([h[i - n:i], h[i + 1:i + n + 1]])
        win_l = np.concatenate([l[i - n:i], l[i + 1:i + n + 1]])
        known = idx[i + n] + TF_45M
        if h[i] > win_h.max():
            rows.append(("swing", "high", h[i], idx[i], known))
        if l[i] < win_l.min():
            rows.append(("swing", "low", l[i], idx[i], known))
    return pd.DataFrame(rows, columns=LEVEL_COLS) if rows else _empty()


def prior_period_levels(df15: pd.DataFrame, period: str) -> pd.DataFrame:
    """High/low of each *complete* UTC day ('D') or Mon-Sun week ('W').

    Known at the end of the period.
    """
    if period == "D":
        start = df15.index.normalize()
        length, per, kind = pd.Timedelta(days=1), 96, "day"
    elif period == "W":
        start = df15.index.normalize() - pd.to_timedelta(df15.index.weekday, unit="D")
        length, per, kind = pd.Timedelta(days=7), 672, "week"
    else:
        raise ValueError(period)
    g = df15.groupby(start)
    hi, lo, cnt = g["high"].max(), g["low"].min(), g["high"].count()
    rows = []
    for s in hi.index[cnt == per]:
        known = s + length
        rows.append(("%s" % kind, "high", hi[s], s, known))
        rows.append((kind, "low", lo[s], s, known))
    return pd.DataFrame(rows, columns=LEVEL_COLS) if rows else _empty()


def add_tested_at(levels: pd.DataFrame, df15: pd.DataFrame, chunk: int = 4000) -> pd.DataFrame:
    """Add `tested_at`: open time of the first 15m bar (at/after known_at) that traded
    at or through the level. NaT if never tested."""
    idx = df15.index
    hi, lo = df15["high"].to_numpy(), df15["low"].to_numpy()
    out = []
    for side, price, known in zip(levels["side"], levels["price"], levels["known_at"]):
        i = int(idx.searchsorted(known, side="left"))
        found = pd.NaT
        while i < len(idx):
            seg = (hi[i:i + chunk] >= price) if side == "high" else (lo[i:i + chunk] <= price)
            if seg.any():
                found = idx[i + int(seg.argmax())]
                break
            i += chunk
        out.append(found)
    res = levels.copy()
    res["tested_at"] = pd.to_datetime(pd.Series(out, index=levels.index), utc=True)
    return res


def build_levels(df15: pd.DataFrame, df45: pd.DataFrame, p: Params = Params()) -> pd.DataFrame:
    """All liquidity levels with known_at and tested_at, sorted by known_at."""
    parts = [swing_levels(df45, p.swing_n), prior_period_levels(df15, "D"),
             prior_period_levels(df15, "W")]
    lv = pd.concat([x for x in parts if len(x)], ignore_index=True)
    lv = lv.sort_values(["known_at", "price"], kind="stable").reset_index(drop=True)
    return add_tested_at(lv, df15)


def untested_at(levels: pd.DataFrame, t: pd.Timestamp, side: str | None = None) -> pd.DataFrame:
    """Levels known and still untested at decision time `t` (a bar-close time).

    A 15m bar opening before `t` is complete; so a level is tested if tested_at < t.
    """
    m = (levels["known_at"] <= t) & (levels["tested_at"].isna() | (levels["tested_at"] >= t))
    if side:
        m &= levels["side"] == side
    return levels[m]
