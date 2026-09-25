"""
Look-Ahead Bias Verification Unit Test.

Demonstrates that changing future candles has zero impact on signals at bar t.
"""

import unittest
import copy
import numpy as np
import pandas as pd

from data.data_manager import DataManager
from strategies.williams_signal_engine import WilliamsSignalEngine


class TestLookAheadBias(unittest.TestCase):

    def test_zero_future_leakage(self):
        """
        Generates dataset, evaluates signal at bar t.
        Then perturbs future candles (t+1, t+2, ...) radically.
        Verifies that bar t signal and metrics remain 100% identical.
        """
        b_1h, u_1h = DataManager.generate_synthetic_universe(num_coins=5, num_days=30, seed=123)
        sym = list(u_1h.keys())[0]
        df_1h = u_1h[sym]

        config = {
            "btc_regime": {"enabled": False},
            "relative_strength": {"rs_7d_min_percentile": 0, "rs_30d_min_percentile": 0},
            "trend": {"require_price_above_slow": False, "require_fast_above_slow": False, "require_slow_rising": False, "require_4h_price_above_ma": False, "require_4h_ma_rising": False},
            "extension_filter": {"enabled": False},
            "pullback": {"model": "model_a", "min_pullback_atr": 0.5},
            "williams_r": {"period": 14, "oversold_threshold": -80.0},
            "volume_confirmation": {"model": "none"},
            "price_action_confirmation": {"model": "williams_cross_only"}
        }

        engine = WilliamsSignalEngine(config)
        df_prepared = engine.prepare_1h_indicators(df_1h)

        test_bar_idx = 100
        sig_original = engine.evaluate_bar(df_prepared.iloc[test_bar_idx], df_prepared.iloc[test_bar_idx - 1], btc_regime=True)

        # Perturb future candles radically
        df_perturbed = df_1h.copy()
        df_perturbed.loc[test_bar_idx + 1:, ["open", "high", "low", "close"]] *= 10.0 # 1000% jump in future
        df_prepared_perturbed = engine.prepare_1h_indicators(df_perturbed)

        sig_perturbed = engine.evaluate_bar(df_prepared_perturbed.iloc[test_bar_idx], df_prepared_perturbed.iloc[test_bar_idx - 1], btc_regime=True)

        self.assertEqual(sig_original["is_ready"], sig_perturbed["is_ready"])
        self.assertEqual(sig_original["state"], sig_perturbed["state"])
        self.assertEqual(sig_original["reason"], sig_perturbed["reason"])
        self.assertAlmostEqual(
            df_prepared.iloc[test_bar_idx]["williams_r"],
            df_prepared_perturbed.iloc[test_bar_idx]["williams_r"],
            places=6
        )
        self.assertAlmostEqual(
            df_prepared.iloc[test_bar_idx]["close"],
            df_prepared_perturbed.iloc[test_bar_idx]["close"],
            places=6
        )


if __name__ == "__main__":
    unittest.main()
