"""
Unit Tests for Strategy Signals, Regime Filters, and Pullback Detection.
"""

import unittest
import numpy as np
import pandas as pd

from strategies.williams_signal_engine import WilliamsSignalEngine
from indicators.pullback import evaluate_pullback_qualification


class TestStrategySignals(unittest.TestCase):

    def setUp(self):
        self.config = {
            "btc_regime": {
                "enabled": True,
                "slow_sma": 50,
                "require_rising": True,
                "slope_lookback": 5,
                "return_lookback": 30,
                "min_return_pct": 0.0
            },
            "relative_strength": {
                "rs_7d_min_percentile": 80.0,
                "rs_30d_min_percentile": 80.0
            },
            "trend": {
                "daily_fast_ma": 20,
                "daily_slow_ma": 50,
                "require_price_above_slow": True,
                "require_fast_above_slow": True,
                "require_slow_rising": True,
                "four_hour_ma": 50,
                "require_4h_price_above_ma": True,
                "require_4h_ma_rising": True
            },
            "extension_filter": {
                "enabled": True,
                "max_extension_pct": 0.15
            },
            "pullback": {
                "model": "model_a",
                "min_pullback_atr": 1.0,
                "min_pullback_pct": 0.02,
                "max_pullback_pct": 0.08
            },
            "williams_r": {
                "period": 14,
                "oversold_threshold": -80.0
            },
            "volume_confirmation": {
                "model": "none"
            },
            "price_action_confirmation": {
                "model": "close_above_prev_high"
            },
            "risk_and_sizing": {
                "risk_per_trade_pct": 0.005,
                "max_position_equity_pct": 0.20,
                "stop_loss_model": "atr",
                "atr_multiplier": 1.5
            }
        }
        self.engine = WilliamsSignalEngine(self.config)

    def test_btc_regime_filtering(self):
        """Tests BTC regime boolean calculation."""
        # Upward sloping prices
        prices = [40000.0 + i * 200.0 for i in range(70)]
        dates = pd.date_range("2024-01-01", periods=70, freq="1D")
        btc_df = pd.DataFrame({"timestamp": dates, "close": prices})

        res_df = self.engine.evaluate_btc_regime(btc_df)
        self.assertTrue(bool(res_df["btc_long_regime"].iloc[-1]))

        # Downward sloping prices
        prices_down = [60000.0 - i * 300.0 for i in range(70)]
        btc_down = pd.DataFrame({"timestamp": dates, "close": prices_down})
        res_down = self.engine.evaluate_btc_regime(btc_down)
        self.assertFalse(bool(res_down["btc_long_regime"].iloc[-1]))

    def test_pullback_qualification_models(self):
        """Tests Pullback Models A, B, C, D."""
        # Model A: Depth >= 1.0 ATR
        row_a = pd.Series({"pullback_depth_atr": 1.25, "pullback_depth_pct": 0.03, "close": 100.0, "low": 98.0})
        qual_a, _ = evaluate_pullback_qualification(row_a, model="model_a", min_pullback_atr=1.0)
        self.assertTrue(qual_a)

        # Model A fail (< 1.0 ATR)
        row_a_fail = pd.Series({"pullback_depth_atr": 0.70})
        qual_fail, _ = evaluate_pullback_qualification(row_a_fail, model="model_a", min_pullback_atr=1.0)
        self.assertFalse(qual_fail)

        # Model B: 2% to 8% drop
        row_b = pd.Series({"pullback_depth_pct": 0.04})
        qual_b, _ = evaluate_pullback_qualification(row_b, model="model_b", min_pullback_pct=0.02, max_pullback_pct=0.08)
        self.assertTrue(qual_b)

        # Model D: Low below EMA20 and above SMA50
        row_d = pd.Series({"low": 95.0, "ema20": 98.0, "sma50": 90.0})
        qual_d, _ = evaluate_pullback_qualification(row_d, model="model_d")
        self.assertTrue(qual_d)

    def test_williams_crossover_signal(self):
        """Tests that %R crossing back above -80 triggers entry."""
        curr_row = pd.Series({
            "rs_7d_pctile": 85.0,
            "rs_30d_pctile": 88.0,
            "daily_uptrend": True,
            "four_hour_uptrend": True,
            "ext_from_ema20": 0.05,
            "pullback_depth_atr": 1.3,
            "williams_r": -75.0, # Crossed above -80
            "close": 105.0,
            "low": 99.0,
            "atr": 2.0,
            "volume_ratio": 1.0
        })
        prev_row = pd.Series({
            "williams_r": -85.0, # Was oversold
            "high": 103.0,
            "low": 98.0
        })

        sig = self.engine.evaluate_bar(curr_row, prev_row, btc_regime=True)
        self.assertEqual(sig["state"], "READY")
        self.assertTrue(sig["is_ready"])


if __name__ == "__main__":
    unittest.main()
