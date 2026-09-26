"""
Unified Larry Williams-Style Crypto Signal Engine.

This SINGLE engine generates signals for both:
1. Historical Backtesting
2. Live Scanner

Zero look-ahead bias: At bar t, decisions only use data up to bar t.
Execution fills on bar t+1 open.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from indicators.technical import (
    williams_r, sma, ema, atr, ma_slope, distance_from_ma, check_crossover
)
from indicators.relative_strength import compute_composite_leadership_score
from indicators.pullback import calculate_pullback_metrics, evaluate_pullback_qualification

logger = logging.getLogger("SignalEngine")


class WilliamsSignalEngine:
    """
    Modular Signal Engine implementing Larry Williams-style
    Trend + Relative Strength + Pullback + %R Reversal.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.btc_cfg = config.get("btc_regime", {})
        self.rs_cfg = config.get("relative_strength", {})
        self.trend_cfg = config.get("trend", {})
        self.ext_cfg = config.get("extension_filter", {})
        self.pb_cfg = config.get("pullback", {})
        self.wr_cfg = config.get("williams_r", {})
        self.vol_cfg = config.get("volume_confirmation", {})
        self.pa_cfg = config.get("price_action_confirmation", {})
        self.risk_cfg = config.get("risk_and_sizing", {})

    # ==========================================
    # 1. BTC Market Regime
    # ==========================================
    def evaluate_btc_regime(self, btc_daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates BTC regime on daily DataFrame:
        - Price > 50-day SMA
        - 50-day SMA is rising (slope > 0)
        - BTC 30-day return > 0%
        Returns DataFrame with 'btc_long_regime' boolean column.
        """
        df = btc_daily_df.copy()
        slow_period = self.btc_cfg.get("slow_sma", 50)
        slope_lookback = self.btc_cfg.get("slope_lookback", 5)
        ret_lookback = self.btc_cfg.get("return_lookback", 30)
        min_ret = self.btc_cfg.get("min_return_pct", 0.0)

        df["btc_sma_slow"] = sma(df["close"], slow_period)
        df["btc_sma_slope"] = ma_slope(df["btc_sma_slow"], slope_lookback)
        df["btc_n_day_ret"] = (df["close"] - df["close"].shift(ret_lookback)) / df["close"].shift(ret_lookback).replace(0, np.nan)

        cond_price = df["close"] > df["btc_sma_slow"]
        cond_rising = df["btc_sma_slope"] > 0 if self.btc_cfg.get("require_rising", True) else True
        cond_return = df["btc_n_day_ret"] > (min_ret / 100.0 if min_ret > 1.0 else min_ret)

        if not self.btc_cfg.get("enabled", True):
            df["btc_long_regime"] = True
        else:
            df["btc_long_regime"] = cond_price & cond_rising & cond_return

        return df

    # ==========================================
    # 2. Individual Coin Trend Indicators
    # ==========================================
    def prepare_daily_indicators(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Computes 20D SMA, 50D SMA, 20D EMA, and 50D slope."""
        df = daily_df.copy()
        fast_p = self.trend_cfg.get("daily_fast_ma", 20)
        slow_p = self.trend_cfg.get("daily_slow_ma", 50)
        ema_p = self.ext_cfg.get("ma_period", 20)

        df["daily_sma_fast"] = sma(df["close"], fast_p)
        df["daily_sma_slow"] = sma(df["close"], slow_p)
        df["daily_ema_fast"] = ema(df["close"], ema_p)
        df["daily_sma_slow_slope"] = ma_slope(df["daily_sma_slow"], 5)
        df["ext_from_ema20"] = distance_from_ma(df["close"], df["daily_ema_fast"])

        # Daily trend condition
        cond_price = df["close"] > df["daily_sma_slow"] if self.trend_cfg.get("require_price_above_slow", True) else True
        cond_fast = df["daily_sma_fast"] > df["daily_sma_slow"] if self.trend_cfg.get("require_fast_above_slow", True) else True
        cond_slope = df["daily_sma_slow_slope"] > 0 if self.trend_cfg.get("require_slow_rising", True) else True

        df["daily_uptrend"] = cond_price & cond_fast & cond_slope
        return df

    def prepare_4h_indicators(self, four_hour_df: pd.DataFrame) -> pd.DataFrame:
        """Computes 4H 50-period SMA and slope."""
        df = four_hour_df.copy()
        ma_p = self.trend_cfg.get("four_hour_ma", 50)
        df["sma_4h"] = sma(df["close"], ma_p)
        df["sma_4h_slope"] = ma_slope(df["sma_4h"], 6)

        cond_price = df["close"] > df["sma_4h"] if self.trend_cfg.get("require_4h_price_above_ma", True) else True
        cond_slope = df["sma_4h_slope"] > 0 if self.trend_cfg.get("require_4h_ma_rising", True) else True

        df["four_hour_uptrend"] = cond_price & cond_slope
        return df

    # ==========================================
    # 3. 1H Setup & Entry Indicators
    # ==========================================
    def prepare_1h_indicators(self, one_hour_df: pd.DataFrame) -> pd.DataFrame:
        """Prepares Williams %R, ATR, pullback metrics, moving averages."""
        df = one_hour_df.copy()
        wr_period = self.wr_cfg.get("period", 14)
        pb_lookback = self.pb_cfg.get("swing_high_lookback", 24)
        vol_avg_p = self.vol_cfg.get("avg_period", 20)

        # Technical indicators
        df["williams_r"] = williams_r(df["high"], df["low"], df["close"], period=wr_period)
        df["atr"] = atr(df["high"], df["low"], df["close"], period=14)
        df["ema20"] = ema(df["close"], 20)
        df["sma50"] = sma(df["close"], 50)

        # Pullback metrics
        df = calculate_pullback_metrics(df, lookback=pb_lookback, atr_period=14, vol_avg_period=vol_avg_p)

        # %R Crossover condition
        threshold = self.wr_cfg.get("oversold_threshold", -80.0)
        df["wr_oversold"] = df["williams_r"] < threshold
        df["wr_cross_above"] = (df["williams_r"].shift(1) <= threshold) & (df["williams_r"] > threshold)

        # Detect Williams %R momentum failure divergence
        # 1st pullback %R <= -90, then recovers, 2nd pullback higher low in %R (>= -80) while price makes swing low
        df["wr_min_recent"] = df["williams_r"].rolling(window=12, min_periods=6).min()
        df["momentum_failure_flag"] = (df["wr_min_recent"].shift(6) <= -90.0) & (df["williams_r"].shift(1) >= -80.0) & (df["williams_r"] > df["williams_r"].shift(1))

        return df

    # ==========================================
    # 4. Multi-Timeframe Feature Merge
    # ==========================================
    def build_coin_feature_table(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_daily: pd.DataFrame,
        rs_metrics: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Merges multi-timeframe indicators without look-ahead bias:
        1H bar at time t can only access daily/4H indicators from the most recently COMPLETED bar.
        """
        f_1h = self.prepare_1h_indicators(df_1h)
        f_4h = self.prepare_4h_indicators(df_4h)
        f_daily = self.prepare_daily_indicators(df_daily)

        # Ensure timestamps are datetimes
        f_1h["timestamp"] = pd.to_datetime(f_1h["timestamp"])
        f_4h["timestamp"] = pd.to_datetime(f_4h["timestamp"])
        f_daily["timestamp"] = pd.to_datetime(f_daily["timestamp"])

        # Shift higher timeframes by 1 bar to prevent look-ahead bias
        # (Only completed 4H/Daily bars can be known to a 1H candle)
        f_4h_shifted = f_4h.copy()
        f_4h_cols = ["timestamp", "four_hour_uptrend", "sma_4h", "sma_4h_slope"]
        f_4h_shifted = f_4h_shifted[f_4h_cols]
        # Shift indicator values
        for c in ["four_hour_uptrend", "sma_4h", "sma_4h_slope"]:
            f_4h_shifted[c] = f_4h_shifted[c].shift(1)

        f_daily_shifted = f_daily.copy()
        f_daily_cols = ["timestamp", "daily_uptrend", "daily_sma_slow", "daily_ema_fast", "ext_from_ema20"]
        f_daily_shifted = f_daily_shifted[f_daily_cols]
        for c in ["daily_uptrend", "daily_sma_slow", "daily_ema_fast", "ext_from_ema20"]:
            f_daily_shifted[c] = f_daily_shifted[c].shift(1)

        # Merge asof backward in time
        merged = pd.merge_asof(
            f_1h.sort_values("timestamp"),
            f_4h_shifted.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            f_daily_shifted.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

        # Add Relative Strength metrics
        for col_name in ["rs_7d_pctile", "rs_30d_pctile", "rs_90d_pctile", "leadership_score"]:
            if col_name in rs_metrics:
                s = rs_metrics[col_name]
                if isinstance(s, pd.Series):
                    # Align by date
                    s_shifted = s.shift(1) # completed daily RS
                    s_df = pd.DataFrame({"timestamp": s_shifted.index, col_name: s_shifted.values})
                    s_df["timestamp"] = pd.to_datetime(s_df["timestamp"])
                    merged = pd.merge_asof(merged.sort_values("timestamp"), s_df.sort_values("timestamp"), on="timestamp", direction="backward")
                else:
                    merged[col_name] = float(s)
            else:
                merged[col_name] = 50.0

        # Fill forward any initial NA
        merged = merged.ffill().bfill()
        return merged

    # ==========================================
    # 5. Core Signal Evaluation Logic
    # ==========================================
    def evaluate_bar(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        btc_regime: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates a single bar's setup conditions and state:
        States:
        - NO_SETUP
        - WATCH (Leader & trend positive)
        - SETUP_DEVELOPING (Pullback in progress, %R oversold)
        - READY (All conditions met, pending buy on next open)
        """
        # A. BTC Regime
        if self.btc_cfg.get("enabled", True) and not btc_regime:
            return {"state": "NO_SETUP", "is_ready": False, "reason": "BTC Market Regime Bearish/Off"}

        # B. Relative Strength / Leadership
        rs_7d = row.get("rs_7d_pctile", 0.0)
        rs_30d = row.get("rs_30d_pctile", 0.0)
        lead_score = row.get("leadership_score", 0.0)

        min_composite = self.rs_cfg.get("composite_score_min", 70.0)
        min_30d = self.rs_cfg.get("rs_30d_min_percentile", 70.0)
        min_7d = self.rs_cfg.get("rs_7d_min_percentile", 40.0)

        # Leader qualification:
        # A coin qualifies if:
        # 1. Composite Leadership Score >= min_composite AND 7D RS >= min_7d (macro leader in a healthy pullback)
        # OR 2. Both 30D RS >= min_30d AND 7D RS >= min_7d
        is_leader = (lead_score >= min_composite and rs_7d >= min_7d) or (rs_30d >= min_30d and rs_7d >= min_7d)
        if not is_leader:
            return {"state": "NO_SETUP", "is_ready": False, "reason": f"RS below threshold (Score: {lead_score:.1f}, 7D: {rs_7d:.1f}, 30D: {rs_30d:.1f})"}

        # C. Trend Filters (Daily and 4H)
        daily_trend = bool(row.get("daily_uptrend", False))
        four_h_trend = bool(row.get("four_hour_uptrend", False))
        if not (daily_trend and four_h_trend):
            return {"state": "WATCH", "is_ready": False, "reason": "Leader but Daily/4H Trend not aligned"}

        # D. Extension Filter
        if self.ext_cfg.get("enabled", True):
            ext = row.get("ext_from_ema20", 0.0)
            max_ext = self.ext_cfg.get("max_extension_pct", 0.15)
            if ext > max_ext:
                return {"state": "WATCH", "is_ready": False, "reason": f"Extended ({ext*100:.1f}% > {max_ext*100:.1f}% max)"}

        # E. Pullback Detection
        pb_model = self.pb_cfg.get("model", "model_a")
        min_pb_atr = self.pb_cfg.get("min_pullback_atr", 1.0)
        min_pb_pct = self.pb_cfg.get("min_pullback_pct", 0.02)
        max_pb_pct = self.pb_cfg.get("max_pullback_pct", 0.08)
        pb_qual, pb_detail = evaluate_pullback_qualification(
            row,
            model=pb_model,
            min_pullback_atr=min_pb_atr,
            min_pullback_pct=min_pb_pct,
            max_pullback_pct=max_pb_pct
        )
        if not pb_qual:
            return {"state": "WATCH", "is_ready": False, "reason": f"No qualified pullback ({pb_detail})"}

        # F. Williams %R Check
        wr_threshold = self.wr_cfg.get("oversold_threshold", -80.0)
        wr_curr = row.get("williams_r", -50.0)
        wr_prev = prev_row.get("williams_r", -50.0) if prev_row is not None else -50.0

        # Recovery condition: prev was oversold (< threshold) and curr crosses back above threshold
        wr_crossed = (wr_prev <= wr_threshold) and (wr_curr > wr_threshold)

        if not wr_crossed:
            if wr_curr <= wr_threshold:
                return {"state": "SETUP_DEVELOPING", "is_ready": False, "reason": f"Pullback active, %R oversold at {wr_curr:.1f}"}
            else:
                return {"state": "WATCH", "is_ready": False, "reason": f"%R at {wr_curr:.1f}, not crossing above {wr_threshold:.1f}"}

        # G. Volume Confirmation
        vol_model = self.vol_cfg.get("model", "none")
        vol_ratio = row.get("volume_ratio", 1.0)
        if vol_model == "contracting_pullback":
            if vol_ratio > self.vol_cfg.get("pullback_vol_ratio_max", 1.0):
                return {"state": "SETUP_DEVELOPING", "is_ready": False, "reason": "Pullback volume expanding, failed contraction filter"}
        elif vol_model == "expanding_reversal":
            if vol_ratio < self.vol_cfg.get("reversal_vol_ratio_min", 1.05):
                return {"state": "SETUP_DEVELOPING", "is_ready": False, "reason": "Reversal volume below 20-period average"}
        elif vol_model == "both":
            if vol_ratio < self.vol_cfg.get("reversal_vol_ratio_min", 1.05):
                return {"state": "SETUP_DEVELOPING", "is_ready": False, "reason": "Reversal volume did not expand"}

        # H. Price Action Entry Confirmation
        pa_model = self.pa_cfg.get("model", "close_above_prev_high")
        close_curr = row.get("close", 0.0)
        high_prev = prev_row.get("high", 0.0) if prev_row is not None else 0.0
        open_prev = prev_row.get("open", 0.0) if prev_row is not None else 0.0
        close_prev = prev_row.get("close", 0.0) if prev_row is not None else 0.0
        swing_high = row.get("swing_high", close_curr * 1.05)
        ema20_1h = row.get("ema20", 0.0)

        pa_confirmed = True
        pa_reason = ""

        if pa_model == "close_above_prev_high":
            pa_confirmed = close_curr > high_prev
            pa_reason = f"Close {close_curr:.4f} > Prev High {high_prev:.4f}"
        elif pa_model == "break_swing_high":
            pa_confirmed = close_curr > swing_high
            pa_reason = f"Close {close_curr:.4f} > Swing High {swing_high:.4f}"
        elif pa_model == "close_above_ema20":
            pa_confirmed = close_curr > ema20_1h
            pa_reason = f"Close {close_curr:.4f} > 1H EMA20 {ema20_1h:.4f}"
        elif pa_model == "bullish_engulfing":
            pa_confirmed = (close_curr > open_prev) and (row.get("open", 0.0) <= close_prev)
            pa_reason = "Bullish Engulfing Candle"
        elif pa_model == "williams_cross_only":
            pa_confirmed = True
            pa_reason = "Williams %R Cross Only (No price confirmation required)"

        if not pa_confirmed:
            return {"state": "SETUP_DEVELOPING", "is_ready": False, "reason": f"Waiting for price action confirmation ({pa_model})"}

        # All conditions satisfied -> READY
        narrative = (
            f"Top-decile 7D/30D RS (7D: {rs_7d:.0f}%, 30D: {rs_30d:.0f}%). "
            f"Daily & 4H trends positive. 1H pullback of {row.get('pullback_depth_atr', 0):.2f} ATR. "
            f"Williams %R reached {wr_prev:.1f} and crossed above {wr_threshold:.0f} to {wr_curr:.1f}. "
            f"Price confirmed: {pa_reason}."
        )

        return {
            "state": "READY",
            "is_ready": True,
            "reason": narrative,
            "price": close_curr,
            "atr": row.get("atr", close_curr * 0.02),
            "swing_low": row.get("low", close_curr * 0.98),
            "rs_7d": rs_7d,
            "rs_30d": rs_30d,
            "rs_90d": row.get("rs_90d_pctile", 50.0),
            "leadership_score": row.get("leadership_score", 70.0),
            "wr_curr": wr_curr,
            "wr_min": row.get("wr_min_recent", wr_prev),
            "momentum_failure": bool(row.get("momentum_failure_flag", False)),
            "volume_ratio": vol_ratio,
            "pullback_atr": row.get("pullback_depth_atr", 0.0)
        }
