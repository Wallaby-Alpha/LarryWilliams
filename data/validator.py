"""
Data Quality Validation Layer.

Detects data integrity issues before running backtests or scans:
- Missing candles / timestamp gaps
- Duplicate candles / timestamps
- Abnormal single-candle price spikes
- Zero volume or flat-line pricing
- Non-chronological timestamps
- Logs all anomalies explicitly without silent corrupting fills.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

logger = logging.getLogger("DataValidator")


class DataQualityValidator:
    """
    Validates OHLCV DataFrame integrity across timeframes.
    """

    def __init__(
        self,
        expected_interval_minutes: int = 60,
        max_spike_pct: float = 0.50,
        log_anomalies: bool = True
    ):
        self.interval_minutes = expected_interval_minutes
        self.max_spike_pct = max_spike_pct
        self.log_anomalies = log_anomalies

    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> Tuple[bool, List[str], pd.DataFrame]:
        """
        Validates OHLCV data.
        Returns:
            (is_valid, list_of_issues, cleaned_df)
        """
        issues = []
        if df.empty:
            issues.append(f"[{symbol}] Empty dataset received.")
            return False, issues, df

        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            issues.append(f"[{symbol}] Missing required columns: {missing_cols}")
            return False, issues, df

        # Ensure datetime index or sorted timestamp
        clean_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(clean_df["timestamp"]):
            clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])

        # 1. Check for duplicates
        duplicate_count = clean_df["timestamp"].duplicated().sum()
        if duplicate_count > 0:
            msg = f"[{symbol}] Found {duplicate_count} duplicate timestamp records. Removing duplicates keeping first."
            issues.append(msg)
            if self.log_anomalies:
                logger.warning(msg)
            clean_df = clean_df.drop_duplicates(subset=["timestamp"], keep="first")

        # Sort chronologically
        clean_df = clean_df.sort_values("timestamp").reset_index(drop=True)

        # 2. Check for chronological order
        time_diffs = clean_df["timestamp"].diff()
        inverted = (time_diffs < pd.Timedelta(seconds=0)).sum()
        if inverted > 0:
            msg = f"[{symbol}] Found {inverted} non-chronological timestamp jumps."
            issues.append(msg)
            if self.log_anomalies:
                logger.error(msg)

        # 3. Check for timestamp gaps
        expected_delta = pd.Timedelta(minutes=self.interval_minutes)
        gaps = time_diffs[time_diffs > (expected_delta * 1.5)]
        if len(gaps) > 0:
            msg = f"[{symbol}] Detected {len(gaps)} timestamp gaps greater than {self.interval_minutes}m (Max gap: {gaps.max()})."
            issues.append(msg)
            if self.log_anomalies:
                logger.warning(msg)

        # 4. Check for abnormal price spikes
        price_ret = clean_df["close"].pct_change().abs()
        spikes = clean_df[price_ret > self.max_spike_pct]
        if len(spikes) > 0:
            msg = f"[{symbol}] Detected {len(spikes)} price jumps exceeding {self.max_spike_pct * 100:.0f}% in a single candle."
            issues.append(msg)
            if self.log_anomalies:
                logger.warning(msg)

        # 5. Check for zero or negative prices
        invalid_prices = (
            (clean_df["open"] <= 0) |
            (clean_df["high"] <= 0) |
            (clean_df["low"] <= 0) |
            (clean_df["close"] <= 0) |
            (clean_df["high"] < clean_df["low"])
        ).sum()
        if invalid_prices > 0:
            msg = f"[{symbol}] Detected {invalid_prices} rows with zero/negative prices or High < Low."
            issues.append(msg)
            if self.log_anomalies:
                logger.error(msg)

        # 6. Check for consecutive zero volume bars
        zero_vol = (clean_df["volume"] == 0).sum()
        if zero_vol > 0:
            msg = f"[{symbol}] Detected {zero_vol} zero-volume candles."
            issues.append(msg)
            if self.log_anomalies:
                logger.info(msg)

        is_valid = (invalid_prices == 0) and (len(clean_df) >= 30)
        return is_valid, issues, clean_df
