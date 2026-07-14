"""
EGX30 benchmark loader — replaces tvDatafeed entirely.

Self-refreshing cache:
  get_egx30()  -> checks if the cached CSV is stale (older than today).
                  If stale, re-pulls live daily prices from Investing.com and rewrites the cache.
                  If the pull fails for ANY reason, it serves the cached data instead of crashing.

You never edit the CSV by hand. Ever.
Real EGX30 index values. No ETF proxy, no FX synthesis.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "data", "egx30.csv")

_INVESTING_ID = 12860  # EGX 30
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "domain-id": "www",
    "Accept": "application/json",
    "Referer": "https://www.investing.com/indices/egx30-historical-data",
}


def _download(start: str = "2018-01-01", attempts: int = 4) -> pd.DataFrame:
    url = (
        f"https://api.investing.com/api/financialdata/historical/{_INVESTING_ID}"
        f"?start-date={start}&end-date={date.today():%Y-%m-%d}"
        f"&time-frame=Daily&add-missing-rows=false"
    )
    rows, last_err = None, None
    for i in range(attempts):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=30)
            rows = r.json()["data"]
            break
        except Exception as e:  # throttled / transient — back off and retry
            last_err = e
            time.sleep(1.5 * (i + 1))
    if rows is None:
        raise RuntimeError(f"EGX30 download failed after {attempts} attempts: {last_err}")

    df = pd.DataFrame(rows)[["rowDateTimestamp", "last_open", "last_max", "last_min", "last_close"]]
    df.columns = ["date", "open", "high", "low", "close"]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(str).str.replace(",", "").astype(float)

    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if len(df) < 200:
        raise ValueError(f"EGX30 download returned only {len(df)} rows — refusing to overwrite cache")
    return df


def _yahoo_today() -> tuple[date, float] | None:
    """Fallback source: Yahoo has a LIVE ^CASE30 quote (no history, but today's close is enough
    to append to the cache). Independent of Investing.com — if one is blocked, the other works."""
    try:
        u = "https://query1.finance.yahoo.com/v8/finance/chart/%5ECASE30?range=5d&interval=1d"
        j = requests.get(u, headers={"User-Agent": _HEADERS["User-Agent"]}, timeout=20).json()
        m = j["chart"]["result"][0]["meta"]
        px = float(m["regularMarketPrice"])
        d = pd.to_datetime(m["regularMarketTime"], unit="s").date()
        return d, px
    except Exception:
        return None


def _read_cache() -> pd.DataFrame | None:
    if not os.path.exists(CSV_PATH):
        return None
    try:
        df = pd.read_csv(CSV_PATH)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.sort_values("date")
    except Exception:
        return None


def _write_cache(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False)


def _is_stale(df: pd.DataFrame) -> bool:
    """Stale if the newest cached bar is older than the last EGX trading day.
    EGX trades Sun-Thu. Fri/Sat are weekends."""
    today = date.today()
    last_session = today
    while last_session.weekday() in (4, 5):  # Fri=4, Sat=5
        last_session -= timedelta(days=1)
    return df["date"].max() < last_session


def get_egx30(force_refresh: bool = False) -> pd.Series:
    """
    THE ONLY FUNCTION THE ENGINE SHOULD CALL.
    Returns EGX30 close prices as a Series indexed by DatetimeIndex (tz-naive).
    Auto-refreshes daily. Never raises unless there is no cache AND no network.
    """
    cached = _read_cache()

    need = force_refresh or cached is None or _is_stale(cached)
    if need:
        try:
            fresh = _download()
            _write_cache(fresh)
            cached = fresh
        except Exception as e:
            if cached is None:
                raise RuntimeError(f"EGX30 unavailable: no cache and download failed ({e})")
            print(f"[egx30] primary source failed ({e}); trying Yahoo fallback")
            y = _yahoo_today()
            if y and y[0] > cached["date"].max():
                d, px = y
                row = pd.DataFrame([{"date": d, "open": px, "high": px, "low": px, "close": px}])
                cached = pd.concat([cached, row], ignore_index=True).sort_values("date")
                _write_cache(cached)
                print(f"[egx30] appended {d} close {px:,.2f} from Yahoo")
            else:
                print(f"[egx30] serving cache from {cached['date'].max()}")

    s = cached.set_index(pd.to_datetime(cached["date"]))["close"].astype(float)
    s.index.name = "Date"
    s.name = "EGX30"
    return s


def egx30_returns() -> pd.Series:
    """Daily simple returns of the benchmark."""
    return get_egx30().pct_change().dropna()


if __name__ == "__main__":
    s = get_egx30(force_refresh=True)
    print(f"OK — {len(s)} rows, {s.index.min().date()} -> {s.index.max().date()}, last close {s.iloc[-1]:,.2f}")
