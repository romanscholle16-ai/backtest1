"""Module 8: combine all modules into the backtest."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import data, filters, fvg, liquidity, reaction, targets, trade
from .config import Params


@dataclass
class Prepared:
    d15: pd.DataFrame
    d45: pd.DataFrame
    levels: pd.DataFrame
    setups: pd.DataFrame
    fvgs: pd.DataFrame
    triggers: pd.DataFrame


@dataclass
class Result:
    trades: pd.DataFrame
    skipped: pd.DataFrame
    prep: Prepared


def prepare(d15: pd.DataFrame, p: Params = Params()) -> Prepared:
    d45 = data.resample(d15)
    levels = liquidity.build_levels(d15, d45, p)
    setups = reaction.find_setups(d45, levels, p)
    fvgs = fvg.detect_fvgs(d15)
    triggers = fvg.find_triggers(d15, setups, fvgs, p)
    return Prepared(d15, d45, levels, setups, fvgs, triggers)


def execute(prep: Prepared, p: Params = Params()) -> Result:
    """Walk triggers chronologically, apply filters, size and simulate each trade."""
    d15 = prep.d15
    gate = filters.EntryGate(p)
    used_setups: set = set()
    equity = p.start_equity
    rows, skipped = [], []
    for _, trg in prep.triggers.sort_values(["time", "setup_id"], kind="stable").iterrows():
        sid = trg["setup_id"]
        if sid in used_setups:
            skipped.append((trg["time"], sid, "setup_already_traded"))
            continue
        why = gate.check(trg["bar_open"])
        if why:
            skipped.append((trg["time"], sid, why))
            continue
        plan = targets.make_plan(trg, prep.levels, d15, p)
        if plan is None:
            skipped.append((trg["time"], sid, "no_target"))
            continue
        res = trade.simulate(plan, d15, trg["time"], p)
        risk_amt = equity * p.risk_pct
        pnl = res.net_r * risk_amt
        gate.register(trg["bar_open"], res.legs[-1].time)
        used_setups.add(sid)
        equity += pnl
        rows.append(dict(
            setup_id=sid, direction=plan.direction, entry_time=trg["time"], exit_time=res.legs[-1].time + trade.TF_15M,
            entry=plan.entry, stop=plan.stop, tp1=plan.tp1, tp2=plan.tp2, target=plan.target,
            risk_pct_of_price=plan.risk / plan.entry * 100, gross_r=res.gross_r, fee_r=res.fee_r,
            net_r=res.net_r, exits="/".join(f"{g.reason}" for g in res.legs), open_at_end=res.open_at_end,
            legs=";".join(f"{g.reason}@{g.price:.2f}@{g.time.isoformat()}@{g.frac:.4f}" for g in res.legs),
            risk_amt=risk_amt, pnl=pnl, equity=equity, leverage=p.risk_pct * plan.entry / plan.risk))
    trades = pd.DataFrame(rows)
    sk = pd.DataFrame(skipped, columns=["time", "setup_id", "reason"])
    return Result(trades, sk, prep)


def run_backtest(d15: pd.DataFrame, p: Params = Params()) -> Result:
    return execute(prepare(d15, p), p)
