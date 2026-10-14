"""Full pipeline (prepare -> execute) on a hand-built market with one textbook setup."""
import pandas as pd
import pytest
from strategy import engine as E
from strategy.config import Params
from tests.helpers import bars, T0

P = Params()
Q = (100.0, 101.0, 99.0, 100.0)


def build(n_before):
    """n_before quiet 45m bars (with a swing high 105 at bar 3), then the textbook sequence."""
    rows = []
    for k in range(n_before):
        rows += [Q, (100.0, 105.0, 99.0, 100.0), Q] if k == 3 else [Q, Q, Q]
    rows += [  # 45m bar = 3 x 15m bars
        (100, 101, 99.5, 100.5), (100.5, 106, 100.5, 104), (104, 104.5, 103, 103.5),       # sweep 105, close 103.5
        (103.5, 103.5, 102, 102.2), (102.2, 102.5, 100, 100.4), (100.4, 101, 100.2, 100.5),  # sharp drop (low 100)
        (100.5, 101, 100.5, 100.8), (100.8, 103.5, 100.8, 102.8), (102.8, 103, 101.9, 102),  # retrace to 103.5
        (102.5, 103, 101.5, 101.8), (101.8, 102, 100, 100.2), (100.2, 100.5, 99, 99.5),      # displacement; bear FVG [100.5,101.5]
        # --- 15m trade bars
        (99.5, 101.3, 99.4, 100.0),    # pokes to 101.3 (80% of the gap) and closes under 100.5 -> TRIGGER, entry 100
        (100.0, 100.2, 98.9, 99.0),    # hits the 24h low target 99.0
        Q, Q, Q, Q,
    ]
    return bars(rows)


def test_textbook_short_in_session():
    n = 74                       # armed Wed 10:30 UTC
    df = build(n)
    assert len(df) % 3 == 0
    r = E.run_backtest(df, P)
    s = r.prep.setups
    assert len(s) == 1 and s.iloc[0].direction == "short" and s.iloc[0].level_price == 105
    assert s.iloc[0].armed_at == pd.Timestamp("2024-01-03 10:30", tz="UTC")
    tr = r.prep.triggers
    assert len(tr) == 1
    assert tr.iloc[0].entry == 100.0 and tr.iloc[0].stop == 101.5
    t = r.trades
    assert len(t) == 1
    x = t.iloc[0]
    assert x.direction == "short" and x.entry == 100.0 and x.stop == 101.5 and x.target == 99.0
    assert x.exits == "target" and x.gross_r == pytest.approx(1.0 / 1.5)
    assert x.entry_time == pd.Timestamp("2024-01-03 10:45", tz="UTC")
    assert x.net_r == pytest.approx(x.gross_r - P.fee_per_side * (100 + 99) / 1.5)
    assert r.skipped.empty


def test_same_pattern_outside_session_is_skipped():
    df = build(60)               # armed Thu 00:00 -> trigger bar before 06:00
    r = E.run_backtest(df, P)
    assert len(r.prep.triggers) == 1 and r.trades.empty
    assert list(r.skipped.reason) == ["outside_session"]


def test_same_pattern_on_weekend_is_skipped():
    df = build(74 + 32 * 3)      # shifted +3 days: armed Saturday 10:30
    r = E.run_backtest(df, P)
    assert len(r.prep.triggers) == 1 and r.trades.empty
    assert list(r.skipped.reason) == ["outside_session"]
    assert r.prep.setups.iloc[0].armed_at.dayofweek == 5


def test_mirror_long_gives_mirrored_trade():
    df = build(74)
    m = 200.0
    mir = df.copy()
    mir["open"], mir["high"], mir["low"], mir["close"] = m - df.open, m - df.low, m - df.high, m - df.close
    r = E.run_backtest(mir, P)
    assert len(r.trades) == 1
    x = r.trades.iloc[0]
    assert x.direction == "long" and x.entry == 100.0 and x.stop == 98.5 and x.target == 101.0
    assert x.gross_r == pytest.approx(1.0 / 1.5)
