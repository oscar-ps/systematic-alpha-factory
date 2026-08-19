"""
Data loading & cleaning for the Alpha Factory.

Reads raw MetaTrader M1 exports (tab-separated), cleans them, and resamples
to a working timeframe (default H1). Caches resampled output as parquet so
the rest of the pipeline never has to re-touch the multi-million-row M1 files.

Design choices (documented, not hidden):
- Working timeframe: H1. Chosen because both root ideas (Section 3.1) are
  framed as session/day-scale phenomena; M1 is too noisy/expensive for a
  30+ instance x 4-layer gate within the timebox, and the assignment
  explicitly names H1 as "a sensible default".
- Duplicate timestamps: dropped (keep first).
- Non-monotonic / out-of-order rows: sorted by timestamp before resampling.
- Weekend / no-trade gaps: resampling with OHLC aggregation naturally drops
  empty bars (no synthetic fill), so weekends simply produce no bars rather
  than fabricated flat candles.
- Zero/negative price or OHLC-violating rows (e.g. high < low) in the raw
  M1 data: dropped before resampling (data integrity, not a strategy choice).
- <SPREAD> column on the resampled bar = mean spread (points) of the M1 bars
  that rolled into it. This is what we feed the cost model.
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(os.environ.get("ALPHA_RAW_DIR", "data"))
CACHE_DIR = Path(os.environ.get("ALPHA_CACHE_DIR", "data_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL_FILES = {
    "SPXUSD": "SPXUSD_M1_202001012300_202605220914.csv",
    "ETHUSD": "ETHUSD_M1_202001012205_202606050000.csv",
    "USDJPY": "USDJPY_l_M1_202001012200_202606050000.csv",
    "XAUUSD": "XAUUSD_l_M1_202001012300_202606050000.csv",
}

TICK_SIZE = {
    "SPXUSD": 0.01,
    "USDJPY": 0.001,
    "XAUUSD": 0.001,
    "ETHUSD": 0.001,
}

ASSET_CLASS = {
    "SPXUSD": "index",
    "USDJPY": "fx",
    "XAUUSD": "gold",
    "ETHUSD": "crypto",
}


def _load_raw_m1(symbol: str) -> pd.DataFrame:
    path = RAW_DIR / SYMBOL_FILES[symbol]
    df = pd.read_csv(
        path,
        sep="\t",
        names=["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"],
        header=0,
        dtype={"date": str, "time": str},
        engine="c",
    )
    ts = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S", errors="coerce")
    df.index = ts
    df = df.drop(columns=["date", "time"])
    df = df[~df.index.isna()]

    n0 = len(df)
    # Drop duplicate timestamps (keep first)
    df = df[~df.index.duplicated(keep="first")]
    # Sort chronologically
    df = df.sort_index()
    # Drop OHLC integrity violations and non-positive prices
    bad = (
        (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    df = df[~bad]
    n1 = len(df)
    if n0 - n1 > 0:
        print(f"[{symbol}] dropped {n0 - n1} raw rows ({(n0-n1)/n0:.4%}) to data-integrity issues / dupes")
    return df


def load_h1(symbol: str, timeframe: str = "1h", force_reload: bool = False) -> pd.DataFrame:
    """Load resampled OHLC bars for a symbol, using a parquet cache."""
    cache_path = CACHE_DIR / f"{symbol}_{timeframe}.pkl"
    if cache_path.exists() and not force_reload:
        return pd.read_pickle(cache_path)

    raw = _load_raw_m1(symbol)
    agg = raw.resample(timeframe, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tickvol": "sum",
            "vol": "sum",
            "spread": "mean",
        }
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"])

    # Convert mean spread (points) -> fraction of price using instrument tick size
    tick = TICK_SIZE[symbol]
    agg["spread_frac"] = (agg["spread"] * tick) / agg["close"]

    agg["symbol"] = symbol
    agg.to_pickle(cache_path)
    return agg


def load_all(timeframe: str = "1h", force_reload: bool = False) -> dict[str, pd.DataFrame]:
    return {s: load_h1(s, timeframe, force_reload) for s in SYMBOL_FILES}


if __name__ == "__main__":
    for s in SYMBOL_FILES:
        df = load_h1(s, force_reload=True)
        print(s, df.shape, df.index.min(), df.index.max(), "mean spread_frac(bps)=", (df["spread_frac"].mean()*1e4).round(3))
