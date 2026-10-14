import numpy as np
import pandas as pd
import pytest
from strategy import engine as E, liquidity as L
from strategy.config import Params
from tests.helpers import bars, T0

M15 = pd.Timedelta(minutes=15)
P = Params()
DAY = 96


def market(n_days=5, overrides=None):
    rows = [(100.0, 101.0, 99.0, 100.0)] * (DAY * n_days)
    rows = list(rows)
    for d in range(n_days):                   # a 120 spike each day -> 24h high = 120 always
        rows[d * DAY + 2] = (100.0, 120.0, 99.0, 100.0)
    for k, v in (overrides or {}).items():
        rows[k] = v
    return bars(rows)


def trig_row(df, bar, sid, direction="long", entry=100.0, stop=98.0):
    o = df.index[bar]
    return dict(setup_id=sid, direction=direction, bar_open=o, time=o + M15, entry=entry, stop=stop)


def prep(df, trigs):
    lv = pd.DataFrame(columns=L.LEVEL_COLS + ["tested_at"])
    lv["tested_at"] = pd.to_datetime(lv["tested_at"], utc=True)
    return E.Prepared(df, df.iloc[:0], lv, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(trigs))


WIN = (100.0, 121.0, 100.0, 120.0)    # hits the 120 target
LOSS = (100.0, 100.0, 97.0, 97.5)     # hits stop 98


def hour(h, day=0):
    return day * DAY + h * 4


def test_win_loss_sizing_and_compounding():
    df = market(overrides={hour(9) + 1: WIN, hour(9, 1) + 1: LOSS})
    trigs = [trig_row(df, hour(9), 1), trig_row(df, hour(9, 1), 2)]
    r = E.execute(prep(df, trigs), P)
    t = r.trades
    assert len(t) == 2 and r.skipped.empty
    fee = lambda exits: P.fee_per_side * (100 + sum(f * x for f, x in exits)) / 2
    assert t.gross_r.iloc[0] == pytest.approx(10.0) and t.gross_r.iloc[1] == pytest.approx(-1.0)
    assert t.net_r.iloc[0] == pytest.approx(10 - fee([(1, 120)]))
    assert t.net_r.iloc[1] == pytest.approx(-1 - fee([(1, 98)]))
    # risk 1% of CURRENT equity, compounding
    assert t.risk_amt.iloc[0] == pytest.approx(100.0)
    e1 = 10_000 + t.net_r.iloc[0] * 100
    assert t.equity.iloc[0] == pytest.approx(e1)
    assert t.risk_amt.iloc[1] == pytest.approx(e1 * 0.01)
    assert t.equity.iloc[1] == pytest.approx(e1 + t.net_r.iloc[1] * e1 * 0.01)
    assert t.leverage.iloc[0] == pytest.approx(0.01 * 100 / 2)


def test_daily_limit_session_and_setup_reuse_skips_with_reasons():
    df = market(overrides={hour(9) + 1: WIN})
    trigs = [trig_row(df, hour(5, 0), 1),          # 05:00 -> outside session
             trig_row(df, hour(9), 2),             # taken
             trig_row(df, hour(14), 3),            # same day -> daily limit
             trig_row(df, hour(9), 2),             # duplicate of a traded setup (same bar)
             trig_row(df, hour(23, 0), 4),         # 23:00 -> outside
             ]
    r = E.execute(prep(df, trigs), P)
    assert len(r.trades) == 1 and r.trades.setup_id.iloc[0] == 2
    reasons = dict(zip(r.skipped.setup_id, r.skipped.reason))
    assert reasons[1] == "outside_session" and reasons[3] == "daily_limit"
    assert reasons[4] == "outside_session"


def test_weekend_skipped_and_monday_taken():
    # T0 is Monday: day 5 = Saturday, day 6 = Sunday, day 7 = Monday
    df = market(8, overrides={hour(9, 5) + 1: WIN, hour(9, 6) + 1: WIN, hour(9, 7) + 1: WIN})
    trigs = [trig_row(df, hour(9, 5), 1), trig_row(df, hour(9, 6), 2), trig_row(df, hour(9, 7), 3)]
    r = E.execute(prep(df, trigs), P)
    assert list(r.trades.setup_id) == [3]
    assert list(r.skipped.reason) == ["outside_session", "outside_session"]


def test_one_position_at_a_time_and_no_overlap():
    # trade on day0 never exits (flat bars) until day 3; day1/day2 triggers refused
    df = market(5, overrides={hour(9, 3) + 5: WIN, DAY + 2: (100.0, 119.0, 99.0, 100.0),
                              2 * DAY + 2: (100.0, 119.0, 99.0, 100.0), 3 * DAY + 2: (100.0, 119.0, 99.0, 100.0)})
    trigs = [trig_row(df, hour(9, 0), 1), trig_row(df, hour(9, 1), 2), trig_row(df, hour(9, 2), 3)]
    r = E.execute(prep(df, trigs), P)
    assert list(r.trades.setup_id) == [1]
    assert list(r.skipped.reason) == ["position_open", "position_open"]
    t = r.trades.iloc[0]
    assert t.exit_time == df.index[hour(9, 3) + 5] + M15


def test_trade_after_exit_allowed_next_day_and_no_overlap_invariant():
    df = market(6, overrides={hour(9, 0) + 1: WIN, hour(9, 1) + 1: LOSS, hour(9, 2) + 1: WIN})
    trigs = [trig_row(df, hour(9, d), d + 1) for d in range(3)]
    t = E.execute(prep(df, trigs), P).trades
    assert len(t) == 3
    assert (t.entry_time.iloc[1:].values >= t.exit_time.iloc[:-1].values).all()
    assert t.entry_time.dt.normalize().is_unique


def test_no_target_when_24h_high_not_above_entry():
    df = market(overrides={i: (100.0, 101.0, 99.0, 100.0) for i in range(0, DAY * 5, DAY)} | {
        d * DAY + 2: (100.0, 101.0, 99.0, 100.0) for d in range(5)})
    r = E.execute(prep(df, [trig_row(df, hour(9), 1, entry=101.0, stop=99.0)]), P)
    assert r.trades.empty and list(r.skipped.reason) == ["no_target"]


def test_short_trade_end_to_end():
    rows = [(100.0, 101.0, 99.0, 100.0)] * (DAY * 3)
    rows = list(rows)
    for d in range(3):
        rows[d * DAY + 2] = (100.0, 101.0, 80.0, 100.0)     # 24h low = 80
    rows[hour(9) + 1] = (100.0, 100.0, 79.0, 80.0)          # target hit
    df = bars(rows)
    r = E.execute(prep(df, [trig_row(df, hour(9), 1, "short", 100.0, 102.0)]), P)
    assert r.trades.gross_r.iloc[0] == pytest.approx(10.0) and r.trades.direction.iloc[0] == "short"
