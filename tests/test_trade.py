import pandas as pd
import pytest
from strategy import trade as TR
from strategy.config import Params
from strategy.targets import Plan
from tests.helpers import bars, T0

M15 = pd.Timedelta(minutes=15)
P = Params()
AXIS = 200.0
# Long scenarios: entry 100, stop 98 (risk 2). Bar 0 is the entry bar; management starts at bar 1.


def plan(target=120, tp1=None, tp2=None, f1=0.0, f2=0.0, entry=100, stop=98):
    return Plan("long", entry, stop, target, tp1, tp2, f1, f2, 1 - f1 - f2)


def mir_plan(pl):
    f = lambda x: None if x is None else AXIS - x
    return Plan("short", f(pl.entry), f(pl.stop), f(pl.target), f(pl.tp1), f(pl.tp2),
                pl.tp1_frac, pl.tp2_frac, pl.runner_frac)


def mir_rows(rows):
    return [(AXIS - o, AXIS - l, AXIS - h, AXIS - c) for o, h, l, c in rows]


E = (100, 100.5, 99.5, 100)  # entry bar
Q = (100, 101, 99.5, 100.5)  # quiet bar, inside stop/targets


def both(pl, rows):
    """Run long and mirrored short; both must produce identical R / reasons."""
    a = TR.simulate(pl, bars(rows), T0 + M15, P)
    b = TR.simulate(mir_plan(pl), bars(mir_rows(rows)), T0 + M15, P)
    assert [(g.time, g.frac, g.reason) for g in a.legs] == [(g.time, g.frac, g.reason) for g in b.legs]
    assert [AXIS - g.price for g in a.legs] == pytest.approx([g.price for g in b.legs])
    assert a.gross_r == pytest.approx(b.gross_r)
    return a


def test_simple_stop_out_is_minus_one_r():
    r = both(plan(), [E, Q, (100, 100.2, 97.5, 98)])
    assert r.gross_r == pytest.approx(-1.0) and r.legs[0].reason == "stop" and r.legs[0].price == 98
    assert r.legs[0].time == T0 + 2 * M15


def test_no_liquidity_full_position_to_target():
    r = both(plan(target=110), [E, Q, (100, 110.5, 100, 110)])
    assert r.gross_r == pytest.approx(5.0) and [g.reason for g in r.legs] == ["target"]


def test_stop_wins_when_stop_and_target_in_same_bar():
    r = both(plan(target=110), [E, (100, 111, 97, 100)])
    assert r.gross_r == pytest.approx(-1.0) and r.legs[0].reason == "stop"


def test_gap_through_stop_fills_at_open():
    r = both(plan(), [E, (95, 96, 94, 95)])
    assert r.legs[0].price == 95 and r.gross_r == pytest.approx(-2.5)


def test_tp1_then_breakeven_stop():
    pl = plan(target=120, tp1=106, f1=0.5)
    r = both(pl, [E, (100, 106.5, 100, 106), (106, 106.5, 99.9, 100.5), (100.5, 100.6, 99.9, 100)])
    # 50% at 106 (+3R each => +1.5R), remaining 50% stopped at entry (0R)
    assert [g.reason for g in r.legs] == ["tp1", "be"]
    assert r.gross_r == pytest.approx(1.5)
    assert r.legs[1].price == 100


def test_breakeven_not_active_in_same_bar_as_tp1():
    # bar fills TP1 and also trades down through entry; BE only applies from next bar
    pl = plan(target=120, tp1=106, f1=0.5)
    r = both(pl, [E, (100, 106.5, 99.0, 100), (100, 100.5, 99.9, 100)])
    assert r.legs[0].reason == "tp1"
    assert r.legs[1].reason == "be" and r.legs[1].time == T0 + 2 * M15  # the bar AFTER the tp1 bar


def test_one_level_remaining_half_runs_to_target():
    pl = plan(target=112, tp1=106, f1=0.5)
    r = both(pl, [E, (100, 106.5, 100, 106), (106, 107, 105, 106.5), (106, 112.5, 105, 112)])
    assert [g.reason for g in r.legs] == ["tp1", "target"]
    assert r.gross_r == pytest.approx(0.5 * 3 + 0.5 * 6)


def test_tp2_then_trail_by_prior_candle_low():
    pl = plan(target=130, tp1=106, tp2=110, f1=0.5, f2=0.25)
    rows = [E,
            (100, 106.5, 100, 106),            # 1: TP1
            (106, 110.5, 105, 110),            # 2: TP2 (low 105)
            (110, 114, 108, 113),              # 3: trail stop = low of bar 2 = 105 (not hit, low 108)
            (113, 118, 111, 117),              # 4: trail now 108 (bar 3 low); low 111 holds
            (117, 117.5, 107.5, 108)]          # 5: trail = bar 4 low 111; low 107.5 <= 111 -> out at 111
    r = both(pl, rows)
    assert [g.reason for g in r.legs] == ["tp1", "tp2", "trail"]
    assert r.legs[2].price == 111 and r.legs[2].frac == pytest.approx(0.25)
    assert r.gross_r == pytest.approx(0.5 * 3 + 0.25 * 5 + 0.25 * 5.5)


def test_trail_never_loosens():
    pl = plan(target=130, tp1=106, tp2=110, f1=0.5, f2=0.25)
    rows = [E, (100, 106.5, 100, 106), (106, 110.5, 107, 110),
            (110, 114, 109, 113),       # trail -> 107 (bar2 low) ; this bar low 109
            (113, 114, 100.5, 101),     # low 100.5 <= trail 109 -> out at 109
            ]
    r = both(pl, rows)
    assert r.legs[-1].reason == "trail" and r.legs[-1].price == 109
    # a lower-low bar in the middle must not drag the stop down
    rows = [E, (100, 106.5, 100, 106), (106, 110.5, 107, 110),
            (110, 111, 108.5, 109),     # trail=107, low 108.5 -> trail becomes 108.5 next
            (109, 111, 108.6, 110),     # low 108.6 > stop 108.5 ok; trail -> max(108.5,108.6)=108.6
            (110, 110.5, 108.55, 109)]  # low 108.55 <= 108.6 -> out at 108.6
    r = both(pl, rows)
    assert r.legs[-1].price == 108.6


def test_trailing_runner_can_still_hit_24h_target():
    pl = plan(target=115, tp1=106, tp2=110, f1=0.5, f2=0.25)
    rows = [E, (100, 106.5, 100, 106), (106, 110.5, 107, 110), (110, 115.5, 109, 115)]
    r = both(pl, rows)
    assert [g.reason for g in r.legs] == ["tp1", "tp2", "target"]
    assert r.gross_r == pytest.approx(0.5 * 3 + 0.25 * 5 + 0.25 * 7.5)


def test_multiple_targets_filled_in_one_bar_in_order():
    pl = plan(target=115, tp1=106, tp2=110, f1=0.5, f2=0.25)
    r = both(pl, [E, (100, 116, 100, 115)])
    assert [g.reason for g in r.legs] == ["tp1", "tp2", "target"]
    assert sum(g.frac for g in r.legs) == pytest.approx(1.0)


def test_no_trail_without_second_level():
    # only one liquidity level: after TP1 stop stays at BE (does not trail)
    pl = plan(target=130, tp1=106, f1=0.5)
    rows = [E, (100, 106.5, 100, 106), (106, 120, 112, 119), (119, 119.5, 100.5, 101),
            (101, 101.2, 99.9, 100)]
    r = both(pl, rows)
    assert r.legs[-1].reason == "be" and r.legs[-1].price == 100


def test_open_at_end_marks_to_last_close():
    pl = plan(target=130)
    r = both(pl, [E, Q, (100.5, 102, 100, 101)])
    assert r.open_at_end and r.legs[-1].reason == "eod" and r.legs[-1].price == 101
    assert r.gross_r == pytest.approx(0.5)


def test_fee_math():
    # stop-out: fee = 0.05% * (100 + 98) / 2R
    r = TR.simulate(plan(), bars([E, (100, 100, 97, 97.5)]), T0 + M15, P)
    assert r.gross_r == pytest.approx(-1.0)
    assert r.fee_r == pytest.approx(0.0005 * (100 + 98) / 2)
    assert r.net_r == pytest.approx(-1.0 - r.fee_r)


def test_only_bars_after_entry_are_used_and_future_is_irrelevant():
    pl = plan(target=110)
    base = [E, Q, (100, 110.5, 100, 110)]
    a = TR.simulate(pl, bars(base), T0 + M15, P)
    junk = base + [(100, 500, 1, 100)] * 10      # data after the exit must not matter
    b = TR.simulate(pl, bars(junk), T0 + M15, P)
    assert a.gross_r == b.gross_r and a.legs == b.legs
    # the entry bar itself (a stop-sized wick) is not part of management
    c = TR.simulate(pl, bars([(100, 100.5, 90, 100), Q, (100, 110.5, 100, 110)]), T0 + M15, P)
    assert c.gross_r == pytest.approx(5.0)
