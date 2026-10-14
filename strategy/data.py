"""Module 1: ETHUSDT data loading, validation and resampling."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/spot"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "tb_base", "tb_quote", "ignore"]
OHLC = ["open", "high", "low", "close"]
TF_15M = pd.Timedelta(minutes=15)


def parse_klines(raw: bytes) -> pd.DataFrame:
    """Parse one Binance kline CSV. Handles ms and µs timestamps."""
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLS)
    if df.empty:
        return pd.DataFrame(columns=OHLC + ["volume"], index=pd.DatetimeIndex([], tz="UTC"))
    t = df["open_time"].astype("int64")
    unit = "us" if t.iloc[0] > 10**14 else "ms"
    df.index = pd.to_datetime(t, unit=unit, utc=True)
    df.index.name = "open_time"
    return df[OHLC + ["volume"]].astype(float)


def _fetch_zip(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
            return z.read(z.namelist()[0])
    except Exception:
        return None


def download(symbol: str, start: str, end: str, cache: str | Path = "data") -> pd.DataFrame:
    """Download 15m klines [start, end] (dates) using monthly files, then daily for the tail."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    parts = []
    for m in pd.period_range(s, e, freq="M"):
        last_day = m.end_time.normalize()
        f = cache / f"{symbol}-15m-{m}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f))
            continue
        raw = _fetch_zip(f"{BASE}/monthly/klines/{symbol}/15m/{symbol}-15m-{m}.zip")
        if raw is not None:
            df = parse_klines(raw)
            df.to_parquet(f)
            parts.append(df)
            continue
        # month not published yet: fall back to daily files
        days = []
        for d in pd.date_range(m.start_time, min(last_day, e), freq="D"):
            raw = _fetch_zip(f"{BASE}/daily/klines/{symbol}/15m/{symbol}-15m-{d:%Y-%m-%d}.zip")
            if raw is not None:
                days.append(parse_klines(raw))
        if days:
            parts.append(pd.concat(days))
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.loc[s.tz_localize("UTC"): (e + pd.Timedelta(days=1)).tz_localize("UTC")
                  - TF_15M]


def validate(df: pd.DataFrame, freq: pd.Timedelta = TF_15M) -> list[str]:
    """Return a list of problems found (empty list = clean)."""
    problems = []
    if not df.index.is_monotonic_increasing:
        problems.append("index not sorted")
    if df.index.has_duplicates:
        problems.append("duplicate timestamps")
    if df[OHLC].isna().any().any():
        problems.append("NaN in OHLC")
    if (df.high < df[["open", "close", "low"]].max(axis=1)).any():
        problems.append("high below open/close/low")
    if (df.low > df[["open", "close", "high"]].min(axis=1)).any():
        problems.append("low above open/close/high")
    if (df[OHLC] <= 0).any().any():
        problems.append("non-positive price")
    gaps = df.index.to_series().diff().dropna()
    n_gaps = int((gaps != freq).sum())
    if n_gaps:
        problems.append(f"{n_gaps} gaps/irregular steps (largest {gaps.max()})")
    return problems


def resample(df15: pd.DataFrame, rule: str = "45min") -> pd.DataFrame:
    """Aggregate 15m bars into `rule` bars anchored at 00:00 UTC.

    Only *complete* bars are returned (every constituent 15m bar present), so a
    partially-formed trailing bar can never leak in.
    """
    step = pd.Timedelta(rule)
    per = int(step / TF_15M)
    g = df15.resample(rule, origin="epoch", label="left", closed="left")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last",
                 "volume": "sum"})
    out["n"] = g["close"].count()
    out = out[out["n"] == per].drop(columns="n")
    return out


def bar_close_time(index: pd.DatetimeIndex, tf: pd.Timedelta) -> pd.DatetimeIndex:
    return index + tf
