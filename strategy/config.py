"""All tunable parameters in one place."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # liquidity
    swing_n: int = 3
    # setup (45m)
    atr_len: int = 14
    react_bars: int = 3
    react_atr: float = 1.5
    retrace_frac: float = 0.5
    max_retrace_bars: int = 16
    disp_atr: float = 1.0
    max_disp_bars: int = 8
    setup_expiry_bars: int = 24
    # fvg (15m)
    fvg_penetration: float = 0.75
    # targets / management
    lookback_24h_bars: int = 96
    tp1_fraction: float = 0.5
    tp2_fraction: float = 0.5  # of the remaining position
    # filters
    session_start_hour: int = 6
    session_end_hour: int = 23
    # sizing / costs
    risk_pct: float = 0.01
    fee_per_side: float = 0.0005
    start_equity: float = 10_000.0
