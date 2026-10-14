import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
from strategy import reaction as R, liquidity as L
from strategy.config import Params
from strategy.data import resample
from tests.helpers import bars, T0

M45 = pd.Timedelta(minutes=45)
P = Params()


def base(n=20):
    """n quiet bars: range 2, ATR ~ 2."""
    return [(100, 101, 99, 100)] * n


def df45(rows):
    return bars(rows, freq="45min")


def level(side, price, tested_bar, df):
    t = df.index[tested_bar] + pd.Timedelta(minutes=15)  # tested inside that 45m bar
    return pd.DataFrame([("swing", side, price, T0, T0, t)],
                        columns=L.LEVEL_COLS + ["tested_at"]).astype({"tested_at": "datetime64[ns, UTC]"})


# bar 20 = sweep of 105 (close 103 back below); 21 = sharp drop; 22 = retrace; 23 = displacement
HAPPY = base() + [
    (100, 106, 99.5, 103.5),   # 20 sweep + close back under 105 (drop 2.5 < 3: not yet sharp)
    (103.5, 103.5, 100, 100.5),  # 21 low 100 -> drop 6 >= 1.5*2
    (100.5, 103.5, 100.5, 102),  # 22 high 103.5 >= 100 + .5*6 = 103 -> retrace
    (102.5, 103, 99, 99.5),    # 23 bearish body 3 >= 2, close < low[22]=100.5
]


def run(rows, lv_price=105, side="high", tested=20, params=P):
    df = df45(rows)
    return R.find_setups(df, level(side, lv_price, tested, df), params), df


def test_atr_is_causal_wilder():
    df = df45(base(30))
    a = R.atr(df, 14)
    assert a.iloc[:13].isna().all() and a.iloc[13] == pytest.approx(2.0)
    df2 = df.copy(); df2.iloc[-1, df2.columns.get_loc("high")] = 999
    assert R.atr(df2, 14).iloc[:-1].equals(a.iloc[:-1])


def test_happy_path_short():
    s, df = run(HAPPY)
    assert len(s) == 1
    r = s.iloc[0]
    assert r.direction == "short" and r.level_price == 105
    assert r.sweep_time == df.index[20] and r.retrace_time == df.index[22] and r.disp_time == df.index[23]
    assert r.armed_at == df.index[23] + M45
    assert r.extreme == 106 and r.reaction_low == 100
    assert r.expires_at == r.armed_at + 24 * M45


def mirror(rows, axis=200):
    return [(axis - o, axis - l, axis - h, axis - c) for o, h, l, c in rows]


def test_long_is_exact_mirror_of_short():
    s, _ = run(HAPPY)
    m_rows = mirror(HAPPY)
    l, _ = run(m_rows, lv_price=95, side="low")
    assert len(l) == 1
    a, b = s.iloc[0], l.iloc[0]
    assert b.direction == "long" and b.level_price == 200 - a.level_price
    assert b.extreme == 200 - a.extreme and b.reaction_low == 200 - a.reaction_low
    assert (a.sweep_time, a.retrace_time, a.disp_time, a.armed_at) == \
           (b.sweep_time, b.retrace_time, b.disp_time, b.armed_at)


def test_no_setup_if_sweep_bar_closes_above_level():
    rows = list(HAPPY); rows[20] = (100, 106, 99.5, 105.5)
    assert run(rows)[0].empty


def test_no_setup_if_reaction_not_sharp():
    rows = list(HAPPY); rows[21] = (103, 103.5, 102, 102.5)  # drop only 4 < 6? thr=3 -> 106-102=4 passes
    s, _ = run(rows, params=replace(P, react_atr=3.0))        # thr=6, drop=4 -> fail
    assert s.empty


def test_no_setup_if_price_never_retraces():
    rows = HAPPY[:22] + [(100.5, 101.5, 99, 100)] * 12  # never reaches 103
    assert run(rows)[0].empty


def test_no_setup_if_retrace_too_late():
    quiet = (100.5, 101, 100, 100.5)  # tight range keeps ATR low
    rows = HAPPY[:22] + [quiet] * 16 + [HAPPY[22], HAPPY[23]]
    assert run(rows)[0].empty
    rows = HAPPY[:22] + [quiet] * 15 + [HAPPY[22], HAPPY[23]]  # retrace on the last allowed bar
    assert len(run(rows)[0]) == 1


def test_invalidated_by_close_above_sweep_high():
    rows = list(HAPPY); rows[22] = (100.5, 107.5, 100.5, 107)  # closes above 106
    assert run(rows)[0].empty


def test_weak_displacement_rejected():
    rows = list(HAPPY); rows[23] = (102.5, 103, 101.5, 101.4)  # body 1.1 < 2
    assert run(rows)[0].empty
    rows[23] = (102.5, 103, 98.5, 100.6)  # body 1.9 < 2 (ATR~2)
    assert run(rows)[0].empty


def test_displacement_must_close_below_previous_low():
    rows = list(HAPPY); rows[23] = (104, 104.2, 100.9, 100.6)  # body 3.4, close 100.6 > low[22]=100.5
    assert run(rows)[0].empty


def test_displacement_can_come_later_within_window():
    rows = HAPPY[:23] + [(102, 102.5, 101, 101.5)] * 3 + [HAPPY[23]]
    # low[-1 before disp] must be above close of disp: prior low=101 > 99.5 ok
    assert len(run(rows)[0]) == 1
    rows = HAPPY[:23] + [(102, 102.5, 101, 101.5)] * 9 + [HAPPY[23]]  # >8 bars after retrace
    assert run(rows)[0].empty


def test_already_tested_level_does_not_trigger():
    # level was swept earlier (tested in bar 10), so the bar-20 spike is not its first test
    s, _ = run(HAPPY, tested=10)
    assert s.empty


def test_needs_atr_history():
    rows = base(10) + HAPPY[20:]
    assert run(rows, tested=10)[0].empty


def test_no_lookahead_truncation_random():
    rng = np.random.default_rng(11)
    n = 96 * 3 * 60
    c = 100 + np.cumsum(rng.normal(0, 0.6, n))
    rows = []
    for x in c:
        w = abs(rng.normal(0, .5))
        rows.append((x, x + w, x - w, x + rng.normal(0, .2)))
    d15 = bars([(o, max(h, o, cl), min(l, o, cl), cl) for o, h, l, cl in rows])
    d45 = resample(d15)
    loose = replace(P, react_atr=0.8, disp_atr=0.5, swing_n=2)
    lv = L.build_levels(d15, d45, loose)
    full = R.find_setups(d45, lv, loose)
    assert len(full) >= 15, len(full)  # test must be meaningful
    for cut in (96 * 40 + 3, 96 * 110, 96 * 150 + 21):
        p15 = d15.iloc[:cut]; p45 = resample(p15)
        part = R.find_setups(p45, L.build_levels(p15, p45, loose), loose)
        T = p15.index[-1] + pd.Timedelta(minutes=15)
        f = full[full.armed_at <= T].reset_index(drop=True)
        q = part[part.armed_at <= T].reset_index(drop=True)
        pd.testing.assert_frame_equal(f, q)
