import pandas as pd
import pytest
from strategy import filters as F

M15 = pd.Timedelta(minutes=15)


def ts(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.mark.parametrize("t,ok", [
    ("2024-01-01 05:45", False),   # closes 06:00 - opens before window
    ("2024-01-01 06:00", True),    # Monday first bar
    ("2024-01-01 12:00", True),
    ("2024-01-01 22:30", True),
    ("2024-01-01 22:45", True),    # closes exactly 23:00
    ("2024-01-01 23:00", False),
    ("2024-01-01 00:00", False),
    ("2024-01-05 22:45", True),    # Friday
    ("2024-01-05 23:00", False),
    ("2024-01-06 12:00", False),   # Saturday
    ("2024-01-07 12:00", False),   # Sunday
    ("2024-01-08 06:00", True),    # next Monday
])
def test_session_window(t, ok):
    assert F.in_session(ts(t)) is ok


def test_every_weekday_hour_matrix():
    # brute force a whole week of 15m bars against an independent rule
    idx = pd.date_range("2024-01-01", periods=96 * 7, freq="15min", tz="UTC")
    for t in idx:
        expect = t.dayofweek < 5 and 6 <= t.hour <= 22  # bars opening 06:00..22:45
        assert F.in_session(t) == expect, t


def test_daily_limit_resets_at_utc_midnight():
    g = F.EntryGate()
    t = ts("2024-01-01 09:00")
    assert g.check(t) is None
    g.register(t, t + 2 * M15)
    assert g.check(ts("2024-01-01 15:00")) == "daily_limit"
    assert g.check(ts("2024-01-02 06:00")) is None


def test_position_open_blocks_entries_until_exit_bar_closes():
    g = F.EntryGate()
    t = ts("2024-01-01 09:00")
    g.register(t, ts("2024-01-03 10:00"))   # holds across days
    # next day entry whose trigger bar closes before the exit bar closes is refused
    assert g.check(ts("2024-01-02 10:00")) == "position_open"
    assert g.check(ts("2024-01-03 09:30")) == "position_open"
    assert g.check(ts("2024-01-03 10:00")) is None   # same bar as the exit: entry happens at its close, after the exit


def test_session_checked_before_other_reasons():
    g = F.EntryGate()
    g.register(ts("2024-01-01 09:00"), ts("2024-01-01 09:15"))
    assert g.check(ts("2024-01-01 23:00")) == "outside_session"
