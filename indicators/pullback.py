"""
Pullback Detection and Analysis Module.

Detects and quantifies temporary weakness/resumption in strong uptrends:
- Drawdown from recent local swing high (percent and ATR multiples)
- Multiple configurable pullback definitions (Model A, B, C, D)
- Pullback volume contraction vs 20-period average
- Relative strength during pullback vs BTC (1H, 4H, 24H excess return)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


def find_swing_high(high_series: pd.Series, lookback: int = 24) -> pd.Series:
    """
    Returns rolling maximum high over past `lookback` bars (excluding current bar if strictly prior).
    """
    return high_series.shift(1).rolling(window=lookback, min_periods=lookback // 2).max()


def calculate_pullback_metrics(
    df: pd.DataFrame,
    lookback: int = 24,
    atr_period: int = 14,
    vol_avg_period: int = 20
) -> pd.DataFrame:
    """
    Adds pullback metrics to an OHLCV DataFrame:
    - swing_high: recent local high
    - pullback_low: minimum low since the swing high was established
    - pullback_depth_pct: (swing_high - low) / swing_high
    - pullback_depth_atr: (swing_high - low) / atr
    - volume_avg: 20-period rolling average volume
    - volume_ratio: current volume / volume_avg
    """
    res = df.copy()
    
    # Calculate ATR if not present
    if "atr" not in res.columns:
        prev_close = res["close"].shift(1)
        tr = pd.concat([
            res["high"] - res["low"],
            (res["high"] - prev_close).abs(),
            (res["low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        res["atr"] = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()

    # Swing high over prior lookback bars
    res["swing_high"] = res["high"].shift(1).rolling(window=lookback, min_periods=lookback // 2).max()
    res["pullback_depth_pct"] = (res["swing_high"] - res["low"]) / res["swing_high"].replace(0, np.nan)
    res["pullback_depth_atr"] = (res["swing_high"] - res["low"]) / res["atr"].replace(0, np.nan)
    
    # Volume metrics
    if "volume" in res.columns:
        res["volume_avg"] = res["volume"].rolling(window=vol_avg_period, min_periods=vol_avg_period // 2).mean()
        res["volume_ratio"] = res["volume"] / res["volume_avg"].replace(0, np.nan)
    else:
        res["volume_avg"] = 1.0
        res["volume_ratio"] = 1.0

    return res


def evaluate_pullback_qualification(
    row: pd.Series,
    model: str = "model_a",
    min_pullback_atr: float = 1.0,
    min_pullback_pct: float = 0.02,
    max_pullback_pct: float = 0.08,
    model_c_min_atr: float = 0.5,
    model_c_max_atr: float = 1.8
) -> Tuple[bool, str]:
    """
    Evaluates whether the candle qualifies as an acceptable pullback under the specified model.
    Models:
      - model_a: falls at least 1 ATR from recent swing high
      - model_b: falls between min_pullback_pct and max_pullback_pct (e.g. 2% to 8%)
      - model_c: retraces between 0.5 and 1.8 ATR
      - model_d: falls below 1H EMA20 but remains above 1H SMA50
    """
    depth_atr = row.get("pullback_depth_atr", 0.0)
    depth_pct = row.get("pullback_depth_pct", 0.0)
    close = row.get("close", 0.0)
    low = row.get("low", 0.0)
    ema20 = row.get("ema20", 0.0)
    sma50 = row.get("sma50", 0.0)

    if model == "model_a":
        qualified = depth_atr >= min_pullback_atr
        detail = f"Depth {depth_atr:.2f} ATR (min {min_pullback_atr:.2f} ATR)"
        return bool(qualified), detail

    elif model == "model_b":
        qualified = (depth_pct >= min_pullback_pct) and (depth_pct <= max_pullback_pct)
        detail = f"Depth {depth_pct * 100:.2f}% (range {min_pullback_pct * 100:.1f}% - {max_pullback_pct * 100:.1f}%)"
        return bool(qualified), detail

    elif model == "model_c":
        qualified = (depth_atr >= model_c_min_atr) and (depth_atr <= model_c_max_atr)
        detail = f"Retrace {depth_atr:.2f} ATR (range {model_c_min_atr:.1f} - {model_c_max_atr:.1f} ATR)"
        return bool(qualified), detail

    elif model == "model_d":
        # Price falls below EMA20 but remains above SMA50
        qualified = (low < ema20) and (low >= sma50)
        detail = f"Low {low:.4f} between SMA50 ({sma50:.4f}) and EMA20 ({ema20:.4f})"
        return bool(qualified), detail

    # Default fallback to model_a
    qualified = depth_atr >= min_pullback_atr
    return bool(qualified), f"Fallback Depth {depth_atr:.2f} ATR"


def compute_pullback_relative_strength(
    coin_returns: Dict[str, float],
    btc_returns: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculates excess performance during pullback:
    coin 1H vs BTC 1H, 4H vs BTC 4H, 24H vs BTC 24H.
    """
    ret_1h_excess = coin_returns.get("1h", 0.0) - btc_returns.get("1h", 0.0)
    ret_4h_excess = coin_returns.get("4h", 0.0) - btc_returns.get("4h", 0.0)
    ret_24h_excess = coin_returns.get("24h", 0.0) - btc_returns.get("24h", 0.0)

    is_strong = (ret_1h_excess > 0) or (ret_24h_excess > 0)

    return {
        "excess_1h": ret_1h_excess,
        "excess_4h": ret_4h_excess,
        "excess_24h": ret_24h_excess,
        "pullback_relative_strength_flag": is_strong,
    }
