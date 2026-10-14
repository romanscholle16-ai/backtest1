import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
from strategy import fvg as F, liquidity as L, reaction as R
from strategy.config import Params
from strategy.data import resample
from tests.helpers import bars, T0

M15 = pd.Timedelta(minutes=15)
P = Params()


def at(i):
    return T0 + i * M15


def test_detect_bear_and_bull_zones():
    rows = [(10, 11, 9, 10),      # 0
            (9.5, 9.8, 7, 7.5),   # 1 big down
            (7.5, 8, 7, 7.8)]     # 2: high 8 < low[0]=9 -> bear gap [8, 9]
    f = F.detect_fvgs(bars(rows))
    assert len(f) == 1 and f.iloc[0].direction == "bear"
    assert (f.iloc[0].bottom, f.iloc[0].top) == (8, 9)
    assert f.iloc[0].start == at(0) and f.iloc[0].known_at == at(3)
    rows = [(10, 11, 9, 10), (10.5, 14, 10.5, 13.5), (13.5, 15, 12, 14)]  # low[2]=12 > high[0]=11
    f = F.detect_fvgs(bars(rows))
    assert len(f) == 1 and f.iloc[0].direction == "bull" and (f.iloc[0].bottom, f.iloc[0].top) == (11, 12)


def test_no_fvg_when_touching_or_overlapping():
    rows = [(10, 11, 9, 10), (10, 12, 9, 11), (11, 12, 11, 11.5)]  # low[2]=11 == high[0]=11
    assert F.detect_fvgs(bars(rows)).empty
    rows = [(10, 11, 9, 10), (10, 12, 9, 11), (11, 12, 10.5, 11.5)]
    assert F.detect_fvgs(bars(rows)).empty


# ---- trigger scenarios (short). Bear FVG [100, 104] size 4: 75% = 103.
def setup_row(direction="short", sweep=0, armed=3, expires=60, extreme=130.0):
    return pd.DataFrame([dict(direction=direction, level_price=0, level_kind="swing",
                              sweep_time=at(sweep), extreme=extreme, reaction_low=0,
                              retrace_time=at(1), disp_time=at(2), armed_at=at(armed),
                              expires_at=at(expires))])


# candle0 low=104, candle2 high=100 -> bear gap [100,104], known at at(3)
LEG = [(110, 111, 104, 105), (105, 105, 99, 99.5), (99.5, 100, 95, 96)]


def run(path, setup=None, params=P, lead=None):
    rows = (lead or []) + LEG + path
    df = bars(rows)
    f = F.detect_fvgs(df)
    s = setup if setup is not None else setup_row(armed=len(lead or []) + 3)
    return F.find_triggers(df, s, f, params), df, f


def test_happy_short_trigger_values():
    # bar3 pokes to 103.5 (87%) and closes under the gap bottom at 99
    t, df, f = run([(96, 103.5, 96, 99)])
    assert len(f) == 1 and len(t) == 1
    r = t.iloc[0]
    assert r.entry == 99 and r.stop == 104 and r.time == at(4) and r.bar_open == at(3)
    assert (r.fvg_bottom, r.fvg_top) == (100, 104) and r.penetration == pytest.approx(0.875)


def test_penetration_threshold_boundary():
    assert run([(96, 102.99, 96, 99)])[0].empty         # 74.75%
    assert len(run([(96, 103.0, 96, 99)])[0]) == 1      # exactly 75%


def test_no_trigger_if_close_stays_inside_gap():
    t, *_ = run([(96, 103.5, 96, 101)])
    assert t.empty
    # ... but a later close below the bottom completes the rejection
    t, *_ = run([(96, 103.5, 96, 101), (101, 101.5, 98, 99)])
    assert len(t) == 1 and t.iloc[0].bar_open == at(4)


def test_close_above_gap_invalidates_it_forever():
    t, *_ = run([(96, 106, 96, 105), (105, 105, 96, 99)])
    assert t.empty


def test_wick_through_whole_gap_but_close_below_is_valid():
    t, *_ = run([(96, 108, 96, 99)])  # wick above the gap top, close below the bottom
    assert len(t) == 1 and t.iloc[0].penetration == 1.0 and t.iloc[0].stop == 104


def test_gap_touched_before_arming_is_used_up():
    # setup armed 3 bars after the gap formed; price already traded into the gap meanwhile
    t, *_ = run([(96, 101, 96, 97), (97, 98, 96, 97), (97, 98, 96, 97), (96, 103.5, 96, 99)],
                setup=setup_row(armed=6))
    assert t.empty
    # untouched while waiting -> still valid
    t, *_ = run([(96, 98, 96, 97), (97, 98, 96, 97), (97, 98, 96, 97), (96, 103.5, 96, 99)],
                setup=setup_row(armed=6))
    assert len(t) == 1


def test_gap_before_sweep_bar_is_ineligible():
    t, *_ = run([(96, 103.5, 96, 99)], setup=setup_row(sweep=20, armed=3))
    assert t.empty


def test_no_trigger_before_armed_at():
    # the rejecting bar closes before the setup is armed -> not usable
    t, *_ = run([(96, 103.5, 96, 99)], setup=setup_row(armed=5))
    assert t.empty


def test_setup_killed_by_close_beyond_extreme():
    t, *_ = run([(96, 131, 96, 130.5), (96, 103.5, 96, 99)], setup=setup_row(extreme=130))
    assert t.empty


def test_expiry():
    t, *_ = run([(96, 103.5, 96, 99)], setup=setup_row(expires=4))   # bar 3 opens before expiry
    assert len(t) == 1
    t, *_ = run([(96, 103.5, 96, 99)], setup=setup_row(expires=3))
    assert t.empty


def test_wrong_direction_gap_ignored():
    # only a bull gap exists but the setup is short
    rows = [(10, 11, 9, 10), (10.5, 14, 10.5, 13.5), (13.5, 15, 12, 14), (14, 14, 11.2, 11.5)]
    df = bars(rows)
    t = F.find_triggers(df, setup_row(extreme=1e9, armed=3), F.detect_fvgs(df), P)
    assert t.empty


def test_second_fvg_forms_after_arming():
    path = [(96, 96.5, 90, 91), (91, 91.5, 88, 89), (89, 91, 85, 86),   # new bear gap [91, 91.5?]
            ]
    df = bars(LEG + path)
    f = F.detect_fvgs(df)
    assert (f.direction == "bear").sum() >= 2


def mirror(rows, axis=200):
    return [(axis - o, axis - l, axis - h, axis - c) for o, h, l, c in rows]


def test_long_is_exact_mirror():
    path = [(96, 103.5, 96, 99)]
    t_s, _, _ = run(path)
    rows = mirror(LEG + path)
    df = bars(rows)
    s = setup_row("long", extreme=200 - 130)
    t_l = F.find_triggers(df, s, F.detect_fvgs(df), P)
    assert len(t_l) == 1
    a, b = t_s.iloc[0], t_l.iloc[0]
    assert b.entry == 200 - a.entry and b.stop == 200 - a.stop and b.time == a.time
    assert (b.fvg_bottom, b.fvg_top) == (200 - a.fvg_top, 200 - a.fvg_bottom)
    assert b.penetration == pytest.approx(a.penetration)


def test_multiple_triggers_sorted():
    # two successive bear gaps, each swept and rejected
    path = [(96, 103.5, 96, 99), (99, 99, 92, 93), (93, 93.5, 90, 91), (91, 91.5, 89, 89.5),
            (89.5, 92.2, 89, 90)]
    t, df, f = run(path)
    assert t.time.is_monotonic_increasing and len(t) >= 1


def test_no_lookahead_truncation_random():
    rng = np.random.default_rng(5)
    n = 96 * 3 * 40
    c = 100 + np.cumsum(rng.normal(0, 0.7, n))
    rows = [(x, x + abs(rng.normal(0, .6)), x - abs(rng.normal(0, .6)), x + rng.normal(0, .3)) for x in c]
    d15 = bars([(o, max(h, o, cl), min(l, o, cl), cl) for o, h, l, cl in rows])
    d45 = resample(d15)
    loose = replace(P, react_atr=0.8, disp_atr=0.5, swing_n=2)
    lv = L.build_levels(d15, d45, loose)
    st = R.find_setups(d45, lv, loose)
    fv = F.detect_fvgs(d15)
    full = F.find_triggers(d15, st, fv, loose)
    assert len(full) >= 20, len(full)
    for cut in (96 * 30 + 3, 96 * 60, 96 * 85 + 7):
        p15 = d15.iloc[:cut]; p45 = resample(p15)
        T = p15.index[-1] + M15
        part_st = R.find_setups(p45, L.build_levels(p15, p45, loose), loose)
        part = F.find_triggers(p15, part_st, F.detect_fvgs(p15), loose)
        f = full[full.time <= T].drop(columns="setup_id").reset_index(drop=True)
        q = part[part.time <= T].drop(columns="setup_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(f, q)
