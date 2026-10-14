import pandas as pd
import pytest
from strategy import data
from tests.helpers import bars, flat, T0


def test_parse_klines_ms_and_us():
    ms = b"1704067200000,1,2,0.5,1.5,10,1704068099999,0,0,0,0,0\n"
    us = b"1704067200000000,1,2,0.5,1.5,10,1704068099999999,0,0,0,0,0\n"
    a, b = data.parse_klines(ms), data.parse_klines(us)
    assert a.index[0] == b.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert a.iloc[0].tolist() == [1, 2, 0.5, 1.5, 10]


def test_validate_clean_and_dirty():
    ok = bars(flat(10))
    assert data.validate(ok) == []
    gap = ok.drop(ok.index[4])
    assert any("gap" in p for p in data.validate(gap))
    bad = ok.copy()
    bad.iloc[3, bad.columns.get_loc("high")] = 50  # high below close
    assert any("high below" in p for p in data.validate(bad))
    dup = pd.concat([ok, ok.iloc[:1]]).sort_index()
    assert any("duplicate" in p for p in data.validate(dup))


def test_resample_45m_values_and_alignment():
    # 6 bars -> two 45m bars starting 00:00 and 00:45
    rows = [(1, 5, 0, 2), (2, 6, 1, 3), (3, 4, 2, 4),
            (10, 20, 9, 11), (11, 12, 8, 12), (12, 30, 10, 13)]
    out = data.resample(bars(rows))
    assert list(out.index) == [T0, T0 + pd.Timedelta(minutes=45)]
    assert out.iloc[0][["open", "high", "low", "close"]].tolist() == [1, 6, 0, 4]
    assert out.iloc[1][["open", "high", "low", "close"]].tolist() == [10, 30, 8, 13]
    assert out.iloc[0]["volume"] == 3


def test_resample_drops_incomplete_trailing_bar():
    out = data.resample(bars(flat(8)))  # 2 full + 2 stray 15m bars
    assert len(out) == 2


def test_resample_anchored_to_midnight_32_per_day():
    out = data.resample(bars(flat(96 * 2)))
    assert len(out) == 64
    assert out.index[32] == T0 + pd.Timedelta(days=1)


def test_resample_drops_bar_with_missing_constituent():
    df = bars(flat(6)).drop(T0 + pd.Timedelta(minutes=15))
    out = data.resample(df)
    assert list(out.index) == [T0 + pd.Timedelta(minutes=45)]


def test_resample_no_lookahead_prefix_property():
    import numpy as np
    rng = np.random.default_rng(0)
    c = 100 + rng.normal(0, 1, 300).cumsum()
    df = bars([(x, x + 1, x - 1, x + 0.2) for x in c])
    full = data.resample(df)
    for cut in (50, 101, 200):
        part = data.resample(df.iloc[:cut])
        pd.testing.assert_frame_equal(part, full.loc[part.index])
