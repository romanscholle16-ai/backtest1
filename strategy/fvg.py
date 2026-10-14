"""Module 4: 15m fair value gaps and the sweep-and-reject entry trigger."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Params

TF_15M = pd.Timedelta(minutes=15)
FVG_COLS = ["direction", "bottom", "top", "start", "known_at"]
TRIG_COLS = ["setup_id", "direction", "bar_open", "time", "entry", "stop", "fvg_bottom", "fvg_top",
             "fvg_known_at", "penetration"]


def detect_fvgs(df15: pd.DataFrame) -> pd.DataFrame:
    """Three-candle gaps. Bull: low[i] > high[i-2]; bear: high[i] < low[i-2] (strict).

    `start` = open of candle i-2; known at the close of candle i.
    """
    h, l = df15["high"].to_numpy(), df15["low"].to_numpy()
    idx = df15.index
    rows = []
    bull = np.where(l[2:] > h[:-2])[0]
    for k in bull:
        rows.append(("bull", h[k], l[k + 2], idx[k], idx[k + 2] + TF_15M))
    bear = np.where(h[2:] < l[:-2])[0]
    for k in bear:
        rows.append(("bear", h[k + 2], l[k], idx[k], idx[k + 2] + TF_15M))
    out = pd.DataFrame(rows, columns=FVG_COLS)
    return out.sort_values("known_at", kind="stable").reset_index(drop=True)


def find_triggers(df15: pd.DataFrame, setups: pd.DataFrame, fvgs: pd.DataFrame,
                  p: Params = Params()) -> pd.DataFrame:
    """For each armed setup: every FVG that is swept >= `fvg_penetration` and rejected.

    Rules (short orientation; long is the exact mirror):
      * FVG is a bear gap that began forming at/after the setup's sweep bar and was still
        completely untouched when the setup was armed (a mitigated gap is used up).
      * From `armed_at`, a 15m bar that closes ABOVE the gap invalidates it.
      * Trigger = a bar that closes BELOW the gap after the gap has been entered to
        >= penetration; entry = that close, stop = top of the gap.
      * A 15m close beyond the setup's sweep extreme kills the setup; scanning also stops at
        `expires_at`.
      * If one candle completes the rejection of several gaps, only the earliest-formed gap
        produces the trigger.
    """
    idx = df15.index
    O = df15["open"].to_numpy(); H = df15["high"].to_numpy()
    L = df15["low"].to_numpy(); C = df15["close"].to_numpy()
    rows = []
    for sid, s in setups.iterrows():
        short = s.direction == "short"
        sgn = 1 if short else -1
        # short orientation arrays
        h = H if short else -L
        l = L if short else -H
        c = C if short else -C
        extreme = sgn * s.extreme
        want = "bear" if short else "bull"
        cand = fvgs[(fvgs.direction == want) & (fvgs.start + TF_15M >= s.sweep_time)
                    & (fvgs.known_at < s.expires_at)]
        i0 = int(idx.searchsorted(s.armed_at, side="left"))
        i1 = int(idx.searchsorted(s.expires_at, side="left"))  # bars opening before expiry
        zones = []
        for _, f in cand.iterrows():
            bot, top = (f.bottom, f.top) if short else (-f.top, -f.bottom)
            j0 = int(idx.searchsorted(f.known_at, side="left"))
            # must be untouched while waiting for the setup to arm
            if j0 < i0 and (h[j0:i0] >= bot).any():
                continue
            zones.append(dict(bot=bot, top=top, known=f.known_at, j0=max(j0, i0), maxpen=0.0,
                              done=False, ts=f.known_at))
        if not zones:
            continue
        for t in range(i0, i1):
            if c[t] > extreme:  # sweep high lost: setup dead
                break
            for z in zones:
                if z["done"] or t < z["j0"]:
                    continue
                size = z["top"] - z["bot"]
                if h[t] >= z["bot"]:
                    z["maxpen"] = max(z["maxpen"], (min(h[t], z["top"]) - z["bot"]) / size)
                if c[t] > z["top"]:
                    z["done"] = True  # closed through the gap: invalid
                elif c[t] < z["bot"] and z["maxpen"] >= p.fvg_penetration - 1e-12:
                    z["done"] = True
                    rows.append(dict(
                        setup_id=sid, direction=s.direction, bar_open=idx[t], time=idx[t] + TF_15M,
                        entry=sgn * c[t], stop=sgn * z["top"],
                        fvg_bottom=min(sgn * z["bot"], sgn * z["top"]),
                        fvg_top=max(sgn * z["bot"], sgn * z["top"]),
                        fvg_known_at=z["known"], penetration=z["maxpen"]))
    if not rows:
        return pd.DataFrame(columns=TRIG_COLS)
    out = pd.DataFrame(rows, columns=TRIG_COLS)
    # one entry per setup per bar: if one candle rejects several gaps, the earliest-formed gap wins
    out = out.sort_values(["time", "setup_id", "fvg_known_at"], kind="stable")
    out = out.drop_duplicates(["setup_id", "time"], keep="first")
    return out.reset_index(drop=True)
