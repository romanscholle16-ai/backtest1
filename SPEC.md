# ETHUSDT Liquidity-Rejection → 15m FVG Strategy — Spec

Every rule is built as its own module, unit-tested on hand-built synthetic candles, and
checked for **look-ahead** (a result computed on data truncated at time T must equal the
result computed on the full data, restricted to what was known at T). Only after all
modules pass are they combined (`strategy/engine.py`) and audited on real data.

All times are **UTC**. A bar is identified by its *open* time; it is "known" at its close
(`open + timeframe`). 45m bars are anchored at 00:00 UTC (32 per day), built from 15m data.

## Modules (build order)

| # | Module | Rule it implements |
|---|--------|--------------------|
| 1 | `data.py` | ETHUSDT 15m klines (Binance spot), clean, 45m resample |
| 2 | `liquidity.py` | "Mark out untested liquidity" on 45m |
| 3 | `reaction.py` | "Sharp reaction off liquidity → retrace → continued rejection with displacement" (the *setup*) |
| 4 | `fvg.py` | 15m FVG detection + "sweep ≥ 75% and reject, entry on close beyond FVG" (the *trigger*) |
| 5 | `targets.py` | 24h high/low target, nearest-liquidity TP1/TP2 |
| 6 | `trade.py` | Trade management: 50% TP1, BE, TP2, 15m-candle trail |
| 7 | `filters.py` | Mon–Fri 06:00–23:00, max 1 trade/day, one position at a time |
| 8 | `engine.py` | Combines 2–7, sizing, fees, trade log |

## Interpretations (the rules left room; these are the defaults, all in `config.py`)

**Liquidity (user choice: swings + prior day/week H/L).**
- Swing high/low on 45m: bar whose high (low) is strictly above (below) the `swing_n`=3 bars each side. Known at close of the 3rd bar after.
- Previous UTC day high/low (known at 00:00 next day) and previous UTC week (Mon–Sun) high/low (known Monday 00:00).
- *Untested* = no 15m bar has traded at/through the price since the level became known. `tested_at` is measured on 15m bars (finer than 45m).
- Highs = buy-side liquidity (above price); lows = sell-side (below price).

**Setup (45m)** — for a high level (mirror for low level):
1. *Reaction bar*: first bar that trades up through the untested level and **closes back below it**.
2. *Sharp*: within `react_bars`=3 bars from the reaction bar, (sweep high − lowest low) ≥ `react_atr`=1.5 × ATR(14).
3. *Retrace*: a later bar trades back up to ≥ `retrace_frac`=50% of that drop (within `max_retrace_bars`=16), without any bar closing above the sweep high.
4. *Continuation displacement*: a bearish bar (on or after the retrace bar, within `max_disp_bars`=8) with body ≥ `disp_atr`=1.0 × ATR and close below the previous bar's low.
5. Setup is **armed at the close of the displacement bar**; it expires after `setup_expiry_bars`=24 45m-bars or when a 15m bar closes beyond the sweep high.
Direction: high swept → **short**; low swept → **long**.

**FVG / trigger (15m)** — short example (mirror for long):
- Bearish FVG: `high[i] < low[i-2]`, zone = [high[i], low[i-2]]. Eligible if it formed at/after the sweep bar (so an FVG already created by the displacement counts; otherwise wait for one).
- After the setup is armed, price must trade up into the zone ≥ `fvg_penetration`=75% of its height; trigger = **15m close below the zone's bottom**, entry at that close.
- Stop = just beyond the far edge of the FVG (above top for shorts). If any 15m bar closes beyond the far edge, the FVG is invalid.

**Targets.** Final target = 24h high (longs) / 24h low (shorts) over the 96 15m bars ending at the entry bar. TP1 = nearest untested liquidity between entry and that target.

**Management.**
- TP1: close `tp1_fraction`=50% at nearest liquidity; stop → break even (from the next bar).
- TP2: only if another untested liquidity level exists before the 24h target: close `tp2_fraction`=50% **of the remaining** (= 25% of original) there. Then the remaining runner trails the low (long) / high (short) of the *previous* 15m candle, ratcheting only, with the 24h extreme as its limit target.
- If no second level exists, the remaining 50% simply targets the 24h extreme with the BE stop.
- If no liquidity exists before the 24h target, the whole position targets the 24h extreme.
- Same-bar ambiguity (stop and target both inside one 15m bar): stop is assumed hit first (conservative).

**Filters.** Entry bar must open Mon–Fri between 06:00 and 22:45 UTC (close ≤ 23:00). Max 1 entry per UTC day. A new entry is not taken while a position is open. A trigger outside the window or on a used-up day is skipped (that FVG is consumed).

**Sizing / costs (user choice).** Risk 1% of equity at the stop, compounding; fee 0.05% per side on notional; results in R and %.

## Open assumptions worth confirming
1. TP2 = 50% of the *remaining* (so a runner is left to trail). If you meant 50% + 50% = flat at TP2, set `tp2_fraction=1.0`.
2. Stop sits at the FVG far edge, not beyond the sweep wick.
3. Numeric thresholds for "sharp", "displacement", "retrace" (above) are my starting points to be tuned.
