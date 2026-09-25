"""
Unit Tests for Indicators and Mathematical Correctness.
"""

import unittest
import numpy as np
import pandas as pd

from indicators.technical import (
    williams_r, sma, ema, atr, ma_slope, distance_from_ma
)
from indicators.relative_strength import (
    compute_n_day_return, compute_excess_return, calculate_cross_sectional_percentiles, compute_composite_leadership_score
)


class TestIndicators(unittest.TestCase):

    def test_williams_r_formula(self):
        """
        Tests Williams %R exact formula:
        %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        Known values:
        Highs = [100], Lows = [50], Close = [75] -> %R = (100 - 75) / (100 - 50) * -100 = -50.0
        """
        highs = pd.Series([100.0] * 14)
        lows = pd.Series([50.0] * 14)
        closes = pd.Series([75.0] * 14)

        wr = williams_r(highs, lows, closes, period=14)
        self.assertAlmostEqual(wr.iloc[-1], -50.0, places=4)

        # Extreme High: Close == High -> %R = 0.0
        closes_high = pd.Series([100.0] * 14)
        wr_high = williams_r(highs, lows, closes_high, period=14)
        self.assertAlmostEqual(wr_high.iloc[-1], 0.0, places=4)

        # Extreme Low: Close == Low -> %R = -100.0
        closes_low = pd.Series([50.0] * 14)
        wr_low = williams_r(highs, lows, closes_low, period=14)
        self.assertAlmostEqual(wr_low.iloc[-1], -100.0, places=4)

    def test_sma_and_ema(self):
        """Tests Simple and Exponential moving averages."""
        data = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        s = sma(data, period=3)
        self.assertAlmostEqual(s.iloc[2], 20.0, places=4)
        self.assertAlmostEqual(s.iloc[4], 40.0, places=4)

        e = ema(data, period=3)
        self.assertTrue(len(e) == len(data))
        self.assertTrue(e.iloc[-1] > e.iloc[0])

    def test_atr_calculation(self):
        """Tests ATR values on known fixed range candles."""
        highs = pd.Series([110.0] * 20)
        lows = pd.Series([90.0] * 20)
        closes = pd.Series([100.0] * 20)

        atr_val = atr(highs, lows, closes, period=14)
        # TR is constant 20.0 (High - Low)
        self.assertAlmostEqual(atr_val.iloc[-1], 20.0, places=2)

    def test_relative_strength_percentiles(self):
        """Tests cross-sectional percentile rankings."""
        df = pd.DataFrame({
            "COIN_A": [0.10, 0.20],
            "COIN_B": [0.05, 0.10],
            "COIN_C": [0.30, 0.40],
            "COIN_D": [-0.05, -0.10]
        })
        pctiles = calculate_cross_sectional_percentiles(df)
        # COIN_C is highest in both rows -> percentile 100.0
        self.assertAlmostEqual(pctiles.loc[0, "COIN_C"], 100.0, places=1)
        # COIN_D is lowest -> percentile 25.0
        self.assertAlmostEqual(pctiles.loc[0, "COIN_D"], 25.0, places=1)

    def test_composite_leadership_score(self):
        """Tests 40% 30D + 35% 7D + 25% 90D weighting."""
        score = compute_composite_leadership_score(
            rs_7d_pctile=80.0,
            rs_30d_pctile=90.0,
            rs_90d_pctile=70.0
        )
        expected = (0.40 * 90.0) + (0.35 * 80.0) + (0.25 * 70.0)
        self.assertAlmostEqual(score, expected, places=2)


if __name__ == "__main__":
    unittest.main()
