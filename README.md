# backtest1 — ETHUSDT liquidity-rejection → 15m FVG strategy

Rules, interpretations and parameters: see [SPEC.md](SPEC.md).

```
pip install -r requirements.txt
python scripts/fetch_data.py          # ETHUSDT 15m from data.binance.vision -> data/
python -m pytest tests -q             # unit + pipeline + real-data integration tests
python scripts/run_backtest.py        # full backtest + audit -> reports/trades.csv
python scripts/verify_<module>.py     # per-module checks against real data
```

Build order (each module has its own tests and a real-data verification script):
`data → liquidity → reaction → fvg → targets → trade → filters → engine`, then
`audit.py` (independent re-simulation) and a walk-forward no-look-ahead test on the
combined system.
