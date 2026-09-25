"""
Unit Tests for Risk Management, Execution Simulation, Fees, and Sizing.
"""

import unittest
from execution.risk_manager import RiskManager
from execution.simulator import ExecutionSimulator


class TestExecutionRisk(unittest.TestCase):

    def setUp(self):
        self.config = {
            "execution": {
                "taker_fee_pct": 0.0005,
                "maker_fee_pct": 0.0000,
                "slippage_pct": 0.0005,
                "spread_pct": 0.0002,
                "collision_resolution": "conservative_stop_first"
            },
            "risk_and_sizing": {
                "risk_per_trade_pct": 0.005,      # 0.5% risk
                "max_position_equity_pct": 0.20,  # 20% equity cap
                "stop_loss_model": "atr",
                "atr_multiplier": 1.5,
                "structural_atr_buffer": 0.5
            },
            "portfolio_constraints": {
                "max_open_positions": 5,
                "max_portfolio_risk_pct": 0.025
            }
        }
        self.risk_mgr = RiskManager(self.config)
        self.simulator = ExecutionSimulator(self.config)

    def test_stop_calculation(self):
        """Tests ATR stop loss formula: entry - 1.5 * ATR."""
        entry = 100.0
        atr_val = 4.0
        stop = self.risk_mgr.calculate_stop_loss(entry, swing_low=95.0, atr_val=atr_val, model="atr")
        self.assertEqual(stop, 100.0 - (4.0 * 1.5)) # 94.0

    def test_position_sizing(self):
        """
        Tests risk-based position sizing:
        Equity = $100,000
        Risk = 0.5% -> $500
        Entry = $100, Stop = $95 -> Stop distance = $5
        Units = $500 / $5 = 100 units ($10,000 position, 10% of portfolio < 20% cap)
        """
        equity = 100000.0
        entry = 100.0
        stop = 95.0
        units, dollar_pos, dollar_risk = self.risk_mgr.calculate_position_size(equity, entry, stop)

        self.assertAlmostEqual(units, 100.0, places=2)
        self.assertAlmostEqual(dollar_pos, 10000.0, places=2)
        self.assertAlmostEqual(dollar_risk, 500.0, places=2)

    def test_max_position_equity_cap(self):
        """Tests that if stop is extremely tight, position does not exceed 20% cap."""
        equity = 100000.0
        entry = 100.0
        stop = 99.9  # $0.10 stop distance -> would want 5,000 units = $500,000 (500% of equity!)
        units, dollar_pos, dollar_risk = self.risk_mgr.calculate_position_size(equity, entry, stop)

        # Capped at 20% of $100k = $20,000
        self.assertAlmostEqual(dollar_pos, 20000.0, places=2)
        self.assertAlmostEqual(units, 200.0, places=2)

    def test_fee_and_slippage_accounting(self):
        """Tests net P&L after MEXC fees and slippage."""
        entry = 100.0
        exit_p = 110.0
        units = 10.0
        dollar_risk = 50.0

        pnl = self.simulator.compute_trade_pnl(entry, exit_p, units, dollar_risk)
        gross = 10.0 * 10.0 # $100
        self.assertEqual(pnl["gross_pnl"], gross)
        # Net must be less than gross due to taker fee, slippage, and spread
        self.assertTrue(pnl["net_pnl"] < gross)
        self.assertTrue(pnl["fees"] > 0)
        self.assertTrue(pnl["slippage"] > 0)


if __name__ == "__main__":
    unittest.main()
