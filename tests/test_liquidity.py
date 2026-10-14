import numpy as np
import pandas as pd
from strategy import liquidity as L
from strategy.data import resample
from tests.helpers import bars, flat, T0

M45 = pd.Timedelta(minutes=45)


def b45(highs, lows=None):
    lows = lows or [h - 1 for h in highs]
    return bars([(l + 0.5, h, l, h - 0.5) for h, l in zip(highs, lows)], freq="45min")


def test_swing_high_detected_with_correct_known_time():
    df = b45([10, 11, 12, 20, 12, 11, 10, 9])  # peak at i=3
    sw = L.swing_levels(df, n=3)
    hi = sw[sw.side == "high"]
    assert len(hi) == 1 and hi.price.iloc[0] == 20
    assert hi.formed.iloc[0] == T0 + 3 * M45
    # confirmed once bar i+3=6 has closed
    assert hi.known_at.iloc[0] == T0 + 6 * M45 + M45


def test_swing_low_detected():
    df = b45([20] * 8, lows=[10, 9, 8, 1, 8, 9, 10, 11])
    sw = L.swing_levels(df, n=3)
    lo = sw[sw.side == "low"]
    assert len(lo) == 1 and lo.price.iloc[0] == 1


def test_equal_highs_are_not_strict_swings():
    df = b45([10, 11, 12, 20, 20, 11, 10, 9])
    assert L.swing_levels(df, 3)[lambda x: x.side == "high"].empty


def test_edges_cannot_be_swings():
    df = b45([30, 11, 12, 13, 14, 15, 16, 40])
    assert L.swing_levels(df, 3)[lambda x: x.side == "high"].empty


def test_not_a_swing_if_higher_bar_within_n():
    df = b45([10, 11, 25, 20, 12, 11, 10, 9, 8])  # 20 has 25 two bars before
    sw = L.swing_levels(df, 3)
    assert 20 not in sw.price.values and sw[sw.side == "high"].empty


def _days(n_days, day_hi_lo):
    rows = []
    for hi, lo in day_hi_lo:
        r = [(100, 100.5, 99.5, 100)] * 96
        r[10] = (100, hi, 99.5, 100)
        r[20] = (100, 100.5, lo, 100)
        rows += r
    return bars(rows)


def test_prior_day_levels_only_complete_days():
    df = _days(2, [(110, 90), (120, 80)])
    df = pd.concat([df, bars(flat(30), start=df.index[-1] + pd.Timedelta(minutes=15))])
    lv = L.prior_period_levels(df, "D")
    assert len(lv) == 4  # 2 complete days x (high, low); partial day 3 excluded
    d1 = lv[lv.formed == T0]
    assert set(d1.price) == {110, 90}
    assert (d1.known_at == T0 + pd.Timedelta(days=1)).all()


def test_prior_week_requires_full_week_and_starts_monday():
    df = bars(flat(672 + 100))  # T0 is a Monday
    lv = L.prior_period_levels(df, "W")
    assert len(lv) == 2 and (lv.formed == T0).all()
    assert (lv.known_at == T0 + pd.Timedelta(days=7)).all()
    # data beginning mid-week: first (partial) week excluded
    mid = bars(flat(672 * 2), start=T0 + pd.Timedelta(days=2))
    lv2 = L.prior_period_levels(mid, "W")
    assert (lv2.formed >= T0 + pd.Timedelta(days=7)).all() and len(lv2) == 2


def _level(side, price, known):
    return pd.DataFrame([("swing", side, price, known - M45, known)], columns=L.LEVEL_COLS)


def test_tested_at_wick_touch_counts_and_exact_equal():
    df = bars(flat(10, 100, 0.5))
    rows = list(zip(df.open, df.high, df.low, df.close))
    rows[6] = (100, 105, 99.5, 100)  # wick to exactly 105
    df = bars(rows)
    lv = L.add_tested_at(_level("high", 105, T0), df)
    assert lv.tested_at.iloc[0] == df.index[6]
    lv = L.add_tested_at(_level("high", 105.01, T0), df)
    assert pd.isna(lv.tested_at.iloc[0])


def test_tested_ignores_bars_before_known():
    rows = [(100, 100.5, 99.5, 100)] * 10
    rows[2] = (100, 120, 99.5, 100)  # spike before the level is known
    df = bars(rows)
    lv = L.add_tested_at(_level("high", 110, df.index[5]), df)
    assert pd.isna(lv.tested_at.iloc[0])


def test_tested_at_low_side():
    rows = [(100, 100.5, 99.5, 100)] * 10
    rows[7] = (100, 100.5, 90, 100)
    df = bars(rows)
    lv = L.add_tested_at(_level("low", 91, T0), df)
    assert lv.tested_at.iloc[0] == df.index[7]


def test_untested_at_boundaries():
    t = T0 + pd.Timedelta(hours=5)
    lv = pd.concat([_level("high", 1, t), _level("high", 2, t), _level("low", 3, t + M45)],
                   ignore_index=True)
    lv["tested_at"] = [pd.NaT, t + M45, pd.NaT]
    lv["tested_at"] = pd.to_datetime(lv["tested_at"], utc=True)
    # at t: first two known; #2 tested later -> still untested; #3 not yet known
    assert list(L.untested_at(lv, t).price) == [1, 2]
    # tested_at == t means the bar opening at t has not closed -> still untested at t
    lv.loc[1, "tested_at"] = t
    assert list(L.untested_at(lv, t).price) == [1, 2]
    # one bar later, #2 (tested by bar opening at t) is tested; #3 is now known
    assert list(L.untested_at(lv, t + M45).price) == [1, 3]
    assert list(L.untested_at(lv, t + M45, side="low").price) == [3]


def test_no_lookahead_truncation():
    rng = np.random.default_rng(3)
    c = 100 + rng.normal(0, 1, 96 * 20).cumsum()
    df15 = bars([(x, x + abs(rng.normal(0, .5)), x - abs(rng.normal(0, .5)), x) for x in c])
    df45 = resample(df15)
    full = L.build_levels(df15, df45)
    for cut_bars in (96 * 9 + 3 * 17, 96 * 14, 96 * 16 + 9):
        p15 = df15.iloc[:cut_bars]
        p45 = resample(p15)
        part = L.build_levels(p15, p45)
        T = p15.index[-1] + pd.Timedelta(minutes=15)  # decision time = end of prefix
        f = full[full.known_at <= T].reset_index(drop=True)
        q = part[part.known_at <= T].reset_index(drop=True)
        key = ["kind", "side", "price", "formed", "known_at"]
        pd.testing.assert_frame_equal(f[key], q[key])
        # status as of T is identical
        fa = L.untested_at(full, T)[key].reset_index(drop=True)
        qa = L.untested_at(part, T)[key].reset_index(drop=True)
        pd.testing.assert_frame_equal(fa, qa)
