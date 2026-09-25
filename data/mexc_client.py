"""
MEXC Spot REST API v3 Client.

Handles:
- exchangeInfo (symbol listings and permissions)
- ticker/24hr (24h volume and quote metrics)
- klines (OHLCV historical data with pagination and rate-limit backoff)
"""

import time
import logging
import requests
import pandas as pd
from typing import Dict, List, Any, Optional

logger = logging.getLogger("MEXCClient")


class MEXCClient:
    """
    Client for public MEXC Spot v3 REST endpoints.
    """

    BASE_URL = "https://api.mexc.com/api/v3"

    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "60m",
        "60m": "60m",
        "4h": "4h",
        "1d": "1d",
        "1D": "1d",
    }

    def __init__(self, request_delay_sec: float = 0.1, timeout: int = 15):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MEXC-Williams-Scanner/1.0",
            "Content-Type": "application/json"
        })
        self.request_delay_sec = request_delay_sec
        self.timeout = timeout

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> Any:
        url = f"{self.BASE_URL}{endpoint}"
        for attempt in range(1, max_retries + 1):
            try:
                time.sleep(self.request_delay_sec)
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait_time = attempt * 2.0
                    logger.warning(f"Rate limited (429) on {url}. Backing off {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {resp.status_code} on {url}: {resp.text[:200]}")
                    time.sleep(attempt * 1.0)
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed for {url}: {e}")
                time.sleep(attempt * 1.5)
        
        logger.error(f"Failed to fetch {url} after {max_retries} attempts.")
        return None

    def get_exchange_info(self) -> List[Dict[str, Any]]:
        """Fetches symbol definitions and status from MEXC."""
        data = self._get("/exchangeInfo")
        if data and "symbols" in data:
            return data["symbols"]
        return []

    def get_24hr_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Fetches 24-hour volume and price stats for all pairs."""
        data = self._get("/ticker/24hr")
        if not data or not isinstance(data, list):
            return {}
        tickers = {item["symbol"]: item for item in data if "symbol" in item}
        return tickers

    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetches OHLCV klines for symbol.
        MEXC response format:
        [
          [
            openTime,       # 0
            open,           # 1
            high,           # 2
            low,            # 3
            close,          # 4
            volume,         # 5
            closeTime,      # 6
            quoteVolume     # 7
          ]
        ]
        """
        mex_interval = self.INTERVAL_MAP.get(interval, interval)
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": mex_interval,
            "limit": min(limit, 1000)
        }
        if start_time:
            params["startTime"] = int(start_time)
        if end_time:
            params["endTime"] = int(end_time)

        raw = self._get("/klines", params=params)
        if not raw or not isinstance(raw, list):
            return pd.DataFrame()

        records = []
        for bar in raw:
            try:
                records.append({
                    "timestamp": pd.to_datetime(bar[0], unit="ms", utc=True),
                    "open": float(bar[1]),
                    "high": float(bar[2]),
                    "low": float(bar[3]),
                    "close": float(bar[4]),
                    "volume": float(bar[5]),
                    "quote_volume": float(bar[7]) if len(bar) > 7 else float(bar[5]) * float(bar[4])
                })
            except (ValueError, IndexError):
                continue

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_historical_klines_paginated(
        self,
        symbol: str,
        interval: str = "1h",
        total_bars: int = 1000
    ) -> pd.DataFrame:
        """
        Fetches up to `total_bars` by paginating backwards in time.
        """
        all_dfs = []
        needed = total_bars
        current_end_time = None

        while needed > 0:
            fetch_limit = min(needed, 1000)
            df = self.get_klines(symbol, interval=interval, limit=fetch_limit, end_time=current_end_time)
            if df.empty:
                break
            all_dfs.append(df)
            needed -= len(df)
            # Oldest timestamp becomes new end_time
            oldest_ms = int(df["timestamp"].iloc[0].timestamp() * 1000)
            if current_end_time and oldest_ms >= current_end_time:
                break
            current_end_time = oldest_ms - 1
            if len(df) < fetch_limit:
                break

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return combined.tail(total_bars).reset_index(drop=True)
