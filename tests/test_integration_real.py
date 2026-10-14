"""Combined-system checks on real ETHUSDT data (skipped if data/ETHUSDT_15m.parquet is absent)."""
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from strategy import audit, engine
from strategy.config import Params

DATA = Path(__file__).resolve().parents[1] / "data" / "ETHUSDT_15m.parquet"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run scripts/fetch_data.py first")
P = Params()
KEY = ["setup_id", "direction", "entry_time", "entry", "stop", "tp1", "tp2", "target"]


@pytest.fixture(scope="module")
def d15():
    return pd.read_parquet(DATA)


@pytest.fixture(scope="module")
def full(d15):
    return engine.run_backtest(d15, P)


def test_full_run_audit_is_clean(full):
    assert len(full.trades) > 50
    assert audit.audit(full, P) == []


def test_every_rule_visible_in_the_trade_log(full):
    t = full.trades
    assert (t.entry_time.dt.normalize().value_counts() <= 1).all()
    assert t.entry_time.is_monotonic_increasing and (t.entry_time.iloc[1:].values >= t.exit_time.iloc[:-1].values).all()
    assert (t.risk_amt > 0).all() and (t.exits.str.len() > 0).all()


@pytest.mark.parametrize("cut", ["2024-06-01", "2025-03-15 13:30", "2026-01-10 04:15"])
def test_no_lookahead_walk_forward(d15, full, cut):
    T = pd.Timestamp(cut, tz="UTC")
    part = engine.run_backtest(d15[d15.index < T], P)
    f, q = full.trades, part.trades
    # trades that were fully closed before the cut are identical in every column
    fd = f[f.exit_time <= T].reset_index(drop=True)
    qd = q[(q.exit_time <= T) & ~q.open_at_end].reset_index(drop=True)  # eod-closed = still open at cut
    pd.testing.assert_frame_equal(fd, qd)
    # trades merely OPENED before the cut have identical entry/stop/targets
    fo = f[f.entry_time <= T][KEY].reset_index(drop=True)
    qo = q[q.entry_time <= T][KEY].reset_index(drop=True)
    pd.testing.assert_frame_equal(fo, qo)


def test_zero_fee_run_only_changes_costs_not_trades(d15, full):
    z = engine.run_backtest(d15, replace(P, fee_per_side=0.0))
    assert list(z.trades.entry_time) == list(full.trades.entry_time)
    assert (z.trades.gross_r.values == full.trades.gross_r.values).all()
    assert (z.trades.fee_r == 0).all()
