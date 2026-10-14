"""Download ETHUSDT 15m data. Usage: python scripts/fetch_data.py [start] [end]"""
import sys
sys.path.insert(0, ".")
import pandas as pd
from strategy import data

start = sys.argv[1] if len(sys.argv) > 1 else "2023-10-01"
end = sys.argv[2] if len(sys.argv) > 2 else (pd.Timestamp.utcnow() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
df = data.download("ETHUSDT", start, end)
df.to_parquet("data/ETHUSDT_15m.parquet")
print(df.index[0], df.index[-1], len(df))
print("problems:", data.validate(df))
