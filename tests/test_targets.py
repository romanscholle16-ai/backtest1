import pandas as pd
import pytest
from strategy import targets as T, liquidity as L
from strategy.config import Params
from tests.helpers import bars, T0

P = Params()
M15 = pd.Timedelta(minutes=15)


def lv(rows):
    """rows: (side, price, known_idx, tested_idx|None) in 15m-bar units"""
    out = [("swing", s, pr, T0, T0 + k * M15, (T0 + t * M15) if t is not None else pd.NaT)
           for s, pr, k, t in rows]
    df = pd.DataFrame(out, columns=L.LEVEL_COLS + ["tested_at"])
    df["tested_at"] = pd.to_datetime(df["tested_at"], utc=True)
    return df


def market(n=200, hi_at=None, lo_at=None, px=100.0):
    rows = [(px, px + 1, px - 1, px)] * n
    rows = list(rows)
    for k, v in (hi_at or {}).items():
        rows[k] = (px, v, px - 1, px)
    for k, v in (lo_at or {}).items():
        rows[k] = (px, px + 1, v, px)
    return bars(rows)


def trig(df, i, direction, entry, stop):
    return pd.Series(dict(direction=direction, entry=entry, stop=stop,
                          bar_open=df.index[i], time=df.index[i] + M15))


def test_24h_window_is_96_bars_inclusive_of_entry_bar():
    df = market(hi_at={100: 150, 4: 160, 5: 140}, lo_at={99: 60})
    # entry bar 100: window = bars 5..100
    hi, lo = T.extreme_24h(df, df.index[100])
    assert hi == 150 and lo == 60
    df = market(hi_at={4: 160, 5: 140}, lo_at={3: 10, 6: 70})
    hi, lo = T.extreme_24h(df, df.index[100])
    assert hi == 140 and lo == 70   # bars 3,4 (97+ back) excluded; bars 5,6 included


def test_window_ignores_future_bars():
    df = market(hi_at={101: 999})
    assert T.extreme_24h(df, df.index[100])[0] == 101


def test_long_no_liquidity_everything_to_target():
    df = market(hi_at={50: 120})
    pl = T.make_plan(trig(df, 100, "long", 100, 98), lv([]), df)
    assert pl.target == 120 and pl.tp1 is None and pl.tp2 is None
    assert (pl.tp1_frac, pl.tp2_frac, pl.runner_frac) == (0, 0, 1)


def test_long_one_level():
    df = market(hi_at={50: 120})
    levels = lv([("high", 110, 10, None), ("low", 105, 10, None)])  # low side ignored
    pl = T.make_plan(trig(df, 100, "long", 100, 98), levels, df)
    assert pl.tp1 == 110 and pl.tp2 is None
    assert (pl.tp1_frac, pl.tp2_frac, pl.runner_frac) == (0.5, 0, 0.5)


def test_long_two_levels_sorted_nearest_first():
    df = market(hi_at={50: 130})
    levels = lv([("high", 120, 10, None), ("high", 108, 10, None), ("high", 112, 10, None)])
    pl = T.make_plan(trig(df, 100, "long", 100, 98), levels, df)
    assert (pl.tp1, pl.tp2) == (108, 112)
    assert (pl.tp1_frac, pl.tp2_frac, pl.runner_frac) == (0.5, 0.25, 0.25)
    assert pl.risk == 2


def test_levels_must_be_strictly_between_entry_and_target():
    df = market(hi_at={50: 120})
    levels = lv([("high", 100, 10, None), ("high", 99, 10, None),   # <= entry
                 ("high", 120, 10, None), ("high", 125, 10, None)])  # == / beyond target
    pl = T.make_plan(trig(df, 100, "long", 100, 98), levels, df)
    assert pl.tp1 is None and pl.runner_frac == 1


def test_tested_and_unknown_levels_excluded():
    df = market(hi_at={50: 120})
    levels = lv([("high", 105, 10, 60),     # tested at bar 60 < entry time -> excluded
                 ("high", 106, 10, 100),    # tested during entry bar -> already traded, excluded
                 ("high", 107, 10, 101),    # tested after entry -> still untested at entry
                 ("high", 108, 102, None)])  # not yet known
    pl = T.make_plan(trig(df, 100, "long", 100, 98), levels, df)
    assert pl.tp1 == 107 and pl.tp2 is None


def test_short_mirror():
    df = market(lo_at={50: 80})
    levels = lv([("low", 90, 10, None), ("low", 94, 10, None), ("high", 95, 10, None)])
    pl = T.make_plan(trig(df, 100, "short", 100, 102), levels, df)
    assert pl.target == 80 and (pl.tp1, pl.tp2) == (94, 90)
    assert (pl.tp1_frac, pl.tp2_frac, pl.runner_frac) == (0.5, 0.25, 0.25)


def test_no_plan_if_target_not_beyond_entry():
    df = market()  # 24h high = 101
    assert T.make_plan(trig(df, 100, "long", 101, 99), lv([]), df) is None
    assert T.make_plan(trig(df, 100, "short", 99, 101), lv([]), df) is None


def test_tp2_fraction_param():
    df = market(hi_at={50: 130})
    levels = lv([("high", 108, 10, None), ("high", 112, 10, None)])
    from dataclasses import replace
    pl = T.make_plan(trig(df, 100, "long", 100, 98), levels, df, replace(P, tp2_fraction=1.0))
    assert (pl.tp1_frac, pl.tp2_frac, pl.runner_frac) == (0.5, 0.5, 0.0)


def test_no_lookahead_future_data_does_not_change_plan():
    df = market(hi_at={50: 120})
    levels = lv([("high", 110, 10, None)])
    a = T.make_plan(trig(df, 100, "long", 100, 98), levels, df)
    b = T.make_plan(trig(df, 100, "long", 100, 98), levels, df.iloc[:101])
    assert a == b
