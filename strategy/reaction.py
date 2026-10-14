"""Module 3: sharp reaction off untested liquidity -> retrace -> continued rejection (45m).

Written once in "short" orientation. Long setups run the same code on a sign-flipped
copy of the data (prices negated, highs<->lows), so both directions are guaranteed
symmetric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Params

TF_45M = pd.Timedelta(minutes=45)
SETUP_COLS = ["direction", "level_price", "level_kind", "sweep_time", "extreme", "reaction_low",
              "retrace_time", "disp_time", "armed_at", "expires_at"]


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Wilder ATR, causal. Value at bar i uses bars <= i."""
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                   axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _scan(O, H, L, C, idx, atr_prev, swept, p: Params, direction: str, sign: int) -> list[dict]:
    """Short-orientation scan. `swept[j]` = list of (ref_price, kind) for levels swept in bar j."""
    out = []
    n = len(C)
    for j, lv in swept.items():
        if np.isnan(atr_prev[j]) or not lv:
            continue
        ref, kind = min(lv, key=lambda x: x[0])  # first level touched (lowest in short frame)
        if not C[j] < ref:  # must close back below the swept level
            continue
        extreme = H[j]
        rl = C[j]  # lowest of (close of sweep bar, later lows)
        thr = p.react_atr * atr_prev[j]
        # --- 1. sharp reaction within react_bars
        k = None
        for b in range(j, min(j + p.react_bars, n)):
            if b > j:
                if C[b] > extreme:
                    break
                extreme = max(extreme, H[b])
                rl = min(rl, L[b])
            if extreme - rl >= thr:
                k = b
                break
        if k is None:
            continue
        # --- 2. retrace toward the level, then 3. displacement
        retr = None
        disp = None
        for b in range(k + 1, min(k + 1 + p.max_retrace_bars + p.max_disp_bars, n)):
            if C[b] > extreme:  # closed above the sweep high: rejection failed
                break
            if retr is None:
                if b > k + p.max_retrace_bars:
                    break
                if H[b] >= rl + p.retrace_frac * (extreme - rl):
                    retr = b
                else:
                    rl = min(rl, L[b])
                    extreme = max(extreme, H[b])
                    continue
            if b - retr > p.max_disp_bars:
                break
            body = O[b] - C[b]
            if (body >= p.disp_atr * atr_prev[b] and C[b] < L[b - 1]):
                disp = b
                extreme = max(extreme, H[b])
                break
            extreme = max(extreme, H[b])
        if disp is None:
            continue
        armed = idx[disp] + TF_45M
        out.append(dict(
            direction=direction, level_price=sign * ref, level_kind=kind, sweep_time=idx[j],
            extreme=sign * extreme, reaction_low=sign * rl, retrace_time=idx[retr],
            disp_time=idx[disp], armed_at=armed, expires_at=armed + p.setup_expiry_bars * TF_45M))
    return out


def _swept_by_bar(levels: pd.DataFrame, idx: pd.DatetimeIndex, side: str, sign: int):
    lv = levels[(levels["side"] == side) & levels["tested_at"].notna()]
    pos = idx.get_indexer(lv["tested_at"].dt.floor("45min"))
    swept: dict[int, list] = {}
    for j, price, kind in zip(pos, lv["price"], lv["kind"]):
        if j >= 0:
            swept.setdefault(int(j), []).append((sign * price, kind))
    return swept


def find_setups(df45: pd.DataFrame, levels: pd.DataFrame, p: Params = Params()) -> pd.DataFrame:
    """All armed setups, sorted by armed_at. `levels` must carry known_at / tested_at."""
    idx = df45.index
    a = atr(df45, p.atr_len).shift(1).to_numpy()
    res = []
    for direction, side, s in (("short", "high", 1), ("long", "low", -1)):
        if s == 1:
            O, H, L, C = (df45[c].to_numpy() for c in ("open", "high", "low", "close"))
        else:
            O, H, L, C = (-df45["open"].to_numpy(), -df45["low"].to_numpy(),
                          -df45["high"].to_numpy(), -df45["close"].to_numpy())
        swept = _swept_by_bar(levels, idx, side, s)
        res += _scan(O, H, L, C, idx, a, swept, p, direction, s)
    if not res:
        return pd.DataFrame(columns=SETUP_COLS)
    df = pd.DataFrame(res, columns=SETUP_COLS).sort_values(["armed_at", "direction"], kind="stable")
    return df.reset_index(drop=True)
