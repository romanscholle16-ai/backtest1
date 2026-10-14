"""Builders for hand-made candle data."""
import pandas as pd

T0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")  # a Monday


def bars(rows, start=T0, freq="15min"):
    """rows: list of (open, high, low, close). Returns OHLCV frame."""
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    df.index.name = "open_time"
    return df


def flat(n, price=100.0, rng=0.5):
    return [(price, price + rng, price - rng, price)] * n
