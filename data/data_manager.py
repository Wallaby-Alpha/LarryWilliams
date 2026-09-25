"""
Data Storage, Synchronization, Resampling, and Synthetic Generator Module.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .validator import DataQualityValidator

logger = logging.getLogger("DataManager")


class DataManager:
    """
    Manages local storage, resampling, and synthetic data generation.
    """

    def __init__(self, cache_dir: str = "data_storage"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.validator = DataQualityValidator()

    def _get_file_path(self, symbol: str, timeframe: str) -> str:
        safe_sym = symbol.replace("/", "_").replace(":", "_")
        return os.path.join(self.cache_dir, f"{safe_sym}_{timeframe}.parquet")

    def save_klines(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Saves validated klines to disk."""
        if df.empty:
            return
        is_valid, issues, clean_df = self.validator.validate_ohlcv(df, symbol=symbol)
        file_path = self._get_file_path(symbol, timeframe)
        try:
            clean_df.to_parquet(file_path, index=False)
        except Exception:
            # Fallback to CSV if pyarrow/fastparquet not available
            csv_path = file_path.replace(".parquet", ".csv")
            clean_df.to_csv(csv_path, index=False)

    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Loads klines from local cache if available."""
        file_path = self._get_file_path(symbol, timeframe)
        csv_path = file_path.replace(".parquet", ".csv")

        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
            except Exception:
                pass

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
            except Exception:
                pass

        return pd.DataFrame()

    @staticmethod
    def resample_ohlcv(df_1h: pd.DataFrame, target_rule: str = "4h") -> pd.DataFrame:
        """
        Resamples 1-hour candles into higher timeframes (e.g. '4h', '1d').
        Aggregates:
            open: first
            high: max
            low: min
            close: last
            volume: sum
            quote_volume: sum
        """
        if df_1h.empty:
            return pd.DataFrame()

        df = df_1h.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = df.set_index("timestamp").sort_index()

        rule_map = {"4h": "4h", "1d": "1D", "1D": "1D", "12h": "12h"}
        pandas_rule = rule_map.get(target_rule.lower(), target_rule)

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }
        if "quote_volume" in df.columns:
            agg_dict["quote_volume"] = "sum"

        resampled = df.resample(pandas_rule).agg(agg_dict).dropna().reset_index()
        return resampled

    @staticmethod
    def generate_synthetic_universe(
        num_coins: int = 25,
        num_days: int = 180,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Generates realistic synthetic multi-asset OHLCV dataset covering multiple market regimes:
        - Bull market regime (first 60 days)
        - High-volatility choppy/sideways regime (next 60 days)
        - Pullback and resumption test regime (last 60 days)
        Returns:
            (btc_1h_df, {symbol: coin_1h_df})
        """
        np.random.seed(seed)
        total_hours = num_days * 24
        start_date = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        timestamps = pd.date_range(start=start_date, periods=total_hours, freq="1h")

        # 1. Synthesize BTC with bull, sideways, and recovery phases
        btc_log_returns = []
        for i in range(total_hours):
            day = i / 24.0
            if day < 60:
                # Strong bull regime
                drift = 0.0006  # upward drift
                vol = 0.008
            elif day < 120:
                # Sideways / choppy regime
                drift = -0.0001
                vol = 0.014
            else:
                # Trend resumption
                drift = 0.0004
                vol = 0.010
            ret = np.random.normal(drift, vol)
            btc_log_returns.append(ret)

        btc_close_prices = 40000.0 * np.exp(np.cumsum(btc_log_returns))
        btc_records = []
        for i, ts in enumerate(timestamps):
            c = btc_close_prices[i]
            spread = c * 0.004 * (1.0 + np.random.rand())
            o = c + np.random.normal(0, spread * 0.4)
            h = max(o, c) + abs(np.random.normal(0, spread * 0.6))
            l = min(o, c) - abs(np.random.normal(0, spread * 0.6))
            v = float(np.random.lognormal(mean=14, sigma=0.5))
            btc_records.append({
                "timestamp": ts, "open": o, "high": h, "low": l, "close": c,
                "volume": v, "quote_volume": v * c
            })
        btc_1h_df = pd.DataFrame(btc_records)

        # 2. Synthesize Altcoins with varying beta, alpha, and pullback behaviors
        symbols = [f"COIN{i+1:02d}USDT" for i in range(num_coins)]
        universe_data: Dict[str, pd.DataFrame] = {}

        for idx, sym in enumerate(symbols):
            beta = 0.8 + 0.08 * idx  # Some low beta, some high beta leaders
            alpha = 0.0003 if idx < 7 else (-0.0001 if idx > 18 else 0.0) # idx < 7 are persistent leaders
            idiosyncratic_vol = 0.012 + 0.004 * (idx % 3)

            coin_records = []
            c = 10.0 + idx * 5.0
            for i, ts in enumerate(timestamps):
                btc_ret = btc_log_returns[i]
                coin_ret = alpha + (beta * btc_ret) + np.random.normal(0, idiosyncratic_vol)
                
                # Introduce occasional sharp pullback followed by Williams %R momentum failure
                if (i % 240) in [50, 51, 52]:  # periodic pullbacks
                    coin_ret -= 0.015
                elif (i % 240) in [53, 54]:   # reversal candle
                    coin_ret += 0.020

                c = max(0.1, c * np.exp(coin_ret))
                spread = c * 0.008 * (1.0 + np.random.rand())
                o = c + np.random.normal(0, spread * 0.4)
                h = max(o, c) + abs(np.random.normal(0, spread * 0.7))
                l = min(o, c) - abs(np.random.normal(0, spread * 0.7))
                v = float(np.random.lognormal(mean=13, sigma=0.6))
                coin_records.append({
                    "timestamp": ts, "open": o, "high": h, "low": l, "close": c,
                    "volume": v, "quote_volume": v * c
                })

            universe_data[sym] = pd.DataFrame(coin_records)

        return btc_1h_df, universe_data
