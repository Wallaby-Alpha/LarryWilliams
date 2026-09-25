"""
Technical Indicators Module for MEXC Larry Williams Strategy System.

Implements exact formulas for:
- Williams %R: (Highest High_N - Close) / (Highest High_N - Lowest Low_N) * -100
- SMA / EMA
- Average True Range (ATR)
- Moving Average Slopes
- Extension metrics (distance from 20D EMA, 50D SMA, ATR distance)
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes Williams %R over `period` bars.
    Formula:
        %R = (Highest High_N - Close) / (Highest High_N - Lowest Low_N) * -100
    Range: [0, -100]
    Oversold typically < -80, Overbought > -20.
    """
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    
    denominator = highest_high - lowest_low
    # Avoid zero division when high equals low
    denominator = denominator.replace(0, np.nan)
    
    wr = ((highest_high - close) / denominator) * -100.0
    # Fill cases where range is zero with -50 (midpoint) or forward fill
    wr = wr.fillna(-50.0)
    return wr


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range using Wilder's smoothing.
    TR = max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder's smoothing (equivalent to ewm with alpha=1/period)
    atr_series = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr_series


def ma_slope(ma_series: pd.Series, lookback: int = 5) -> pd.Series:
    """
    Calculates percentage slope of moving average over `lookback` bars.
    Slope > 0 indicates rising MA.
    """
    past_ma = ma_series.shift(lookback)
    slope = (ma_series - past_ma) / past_ma.replace(0, np.nan)
    return slope


def distance_from_ma(close: pd.Series, ma_series: pd.Series) -> pd.Series:
    """
    Returns fractional distance: (Close - MA) / MA.
    e.g. 0.15 means price is 15% above MA.
    """
    return (close - ma_series) / ma_series.replace(0, np.nan)


def atr_distance_from_ma(close: pd.Series, ma_series: pd.Series, atr_series: pd.Series) -> pd.Series:
    """
    Returns distance in terms of ATR multiples: (Close - MA) / ATR.
    """
    return (close - ma_series) / atr_series.replace(0, np.nan)


def check_crossover(series: pd.Series, threshold: float) -> pd.Series:
    """
    Detects when series crosses ABOVE threshold:
    Previous bar <= threshold and Current bar > threshold.
    """
    prev_series = series.shift(1)
    return (prev_series <= threshold) & (series > threshold)


def check_crossunder(series: pd.Series, threshold: float) -> pd.Series:
    """
    Detects when series crosses BELOW threshold:
    Previous bar >= threshold and Current bar < threshold.
    """
    prev_series = series.shift(1)
    return (prev_series >= threshold) & (series < threshold)
