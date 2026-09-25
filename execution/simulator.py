"""
Realistic Execution Simulator with MEXC Fee Structures, Slippage, and Collision Resolution.

Ensures:
- Conservative fill prices on next candle open
- Accurate maker/taker fees + slippage + spread calculation
- Conservative intra-candle collision resolution (Stop vs Target in same bar)
- Net P&L prominently distinguished from Gross P&L
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger("ExecutionSimulator")


class ExecutionSimulator:
    """
    Simulates realistic exchange execution.
    """

    def __init__(self, config: Dict[str, Any]):
        self.exec_cfg = config.get("execution", {})
        self.taker_fee_pct = float(self.exec_cfg.get("taker_fee_pct", 0.0005))
        self.maker_fee_pct = float(self.exec_cfg.get("maker_fee_pct", 0.0000))
        self.slippage_pct = float(self.exec_cfg.get("slippage_pct", 0.0005))
        self.spread_pct = float(self.exec_cfg.get("spread_pct", 0.0002))
        self.collision_resolution = self.exec_cfg.get("collision_resolution", "conservative_stop_first")

    def simulate_entry_fill(self, candle_open: float) -> Tuple[float, float, float]:
        """
        Calculates execution price and entry costs for a market buy order:
        Fill Price = candle_open * (1 + slippage + spread / 2)
        Returns:
            (executed_fill_price, slippage_cost_pct, fee_pct)
        """
        cost_multiplier = 1.0 + self.slippage_pct + (self.spread_pct / 2.0)
        fill_price = candle_open * cost_multiplier
        slippage_pct = self.slippage_pct + (self.spread_pct / 2.0)
        fee_pct = self.taker_fee_pct
        return round(fill_price, 8), slippage_pct, fee_pct

    def simulate_exit_fill(self, theoretical_exit_price: float, is_stop: bool = False) -> Tuple[float, float, float]:
        """
        Calculates exit fill price and costs:
        For market sell or stop market:
        Fill Price = theoretical_exit_price * (1 - slippage - spread / 2)
        """
        slip = self.slippage_pct * (1.5 if is_stop else 1.0)  # slightly higher slippage on stop-loss market fills
        cost_multiplier = 1.0 - slip - (self.spread_pct / 2.0)
        fill_price = theoretical_exit_price * cost_multiplier
        fee_pct = self.taker_fee_pct
        return round(fill_price, 8), slip, fee_pct

    def resolve_candle_collisions(
        self,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        stop_price: float,
        target_price: float
    ) -> Tuple[bool, bool, str]:
        """
        Determines whether Stop Loss or Take Profit was triggered within a bar.
        Handles cases where both [low <= stop] and [high >= target] occur in the same candle.
        Returns:
            (stop_hit, target_hit, primary_trigger)
        """
        stop_hit = bar_low <= stop_price
        target_hit = bar_high >= target_price

        if stop_hit and target_hit:
            # Collision detected
            if self.collision_resolution == "conservative_stop_first":
                return True, False, "STOP_FIRST_CONSERVATIVE"
            elif self.collision_resolution == "target_first":
                return False, True, "TARGET_FIRST"
            else:
                # If bar was bearish (close < open), assume stop hit first; otherwise target
                if bar_close < bar_open:
                    return True, False, "STOP_BEARISH_BAR"
                else:
                    return False, True, "TARGET_BULLISH_BAR"

        if stop_hit:
            return True, False, "STOP_HIT"
        if target_hit:
            return False, True, "TARGET_HIT"

        return False, False, "NONE"

    def compute_trade_pnl(
        self,
        entry_price: float,
        exit_price: float,
        units: float,
        dollar_risk: float
    ) -> Dict[str, float]:
        """
        Calculates Gross P&L, Total Fees, Slippage Cost, Net P&L, and R-Multiple.
        """
        entry_notional = units * entry_price
        exit_notional = units * exit_price

        gross_pnl = exit_notional - entry_notional
        fees = (entry_notional * self.taker_fee_pct) + (exit_notional * self.taker_fee_pct)
        slippage_cost = (entry_notional * self.slippage_pct) + (exit_notional * self.slippage_pct)
        spread_cost = (entry_notional + exit_notional) * (self.spread_pct / 2.0)

        net_pnl = gross_pnl - fees - slippage_cost - spread_cost
        
        # Calculate R-Multiple
        r_multiple = net_pnl / dollar_risk if dollar_risk > 0 else 0.0

        return {
            "gross_pnl": round(gross_pnl, 4),
            "fees": round(fees, 4),
            "slippage": round(slippage_cost, 4),
            "spread": round(spread_cost, 4),
            "net_pnl": round(net_pnl, 4),
            "r_multiple": round(r_multiple, 3),
            "return_pct": round((net_pnl / entry_notional) if entry_notional > 0 else 0.0, 4)
        }
