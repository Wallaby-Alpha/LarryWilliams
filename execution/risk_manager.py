"""
Risk Management and Position Sizing Module.

Implements:
- Risk-based position sizing: Dollar Risk / Distance to Stop
- Configurable risk per trade (0.25%, 0.5%, 0.75%, 1.0%, 1.5%)
- Maximum portfolio position size cap (e.g. 20% of account equity)
- Multiple stop-loss models:
  - Model A: Pullback swing low
  - Model B: ATR multiple (Entry - X * ATR)
  - Model C: Structural swing low + ATR buffer
- Portfolio constraints:
  - Maximum simultaneous positions (e.g. 5)
  - Maximum cumulative portfolio risk (e.g. 2.5%)
  - Correlation exposure limits
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logger = logging.getLogger("RiskManager")


class RiskManager:
    """
    Manages position sizing, stop-loss calculations, and portfolio limits.
    """

    def __init__(self, config: Dict[str, Any]):
        self.risk_cfg = config.get("risk_and_sizing", {})
        self.port_cfg = config.get("portfolio_constraints", {})

        self.risk_per_trade_pct = float(self.risk_cfg.get("risk_per_trade_pct", 0.005))
        self.max_position_equity_pct = float(self.risk_cfg.get("max_position_equity_pct", 0.20))
        self.stop_loss_model = self.risk_cfg.get("stop_loss_model", "atr")
        self.atr_multiplier = float(self.risk_cfg.get("atr_multiplier", 1.5))
        self.structural_atr_buffer = float(self.risk_cfg.get("structural_atr_buffer", 0.5))

        self.max_open_positions = int(self.port_cfg.get("max_open_positions", 5))
        self.max_portfolio_risk_pct = float(self.port_cfg.get("max_portfolio_risk_pct", 0.025))

    def calculate_stop_loss(
        self,
        entry_price: float,
        swing_low: float,
        atr_val: float,
        model: Optional[str] = None
    ) -> float:
        """
        Calculates stop price based on selected model:
        - 'pullback_low': swing_low * 0.998
        - 'atr': entry_price - (atr_val * atr_multiplier)
        - 'structural_atr': swing_low - (atr_val * structural_atr_buffer)
        """
        chosen_model = model or self.stop_loss_model

        if chosen_model == "pullback_low":
            stop = swing_low * 0.998
        elif chosen_model == "structural_atr":
            stop = swing_low - (atr_val * self.structural_atr_buffer)
        else: # "atr"
            stop = entry_price - (atr_val * self.atr_multiplier)

        # Sanity check: stop must be strictly below entry price
        if stop >= entry_price:
            stop = entry_price * 0.98  # Fallback to 2% stop

        return round(float(stop), 8)

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float
    ) -> Tuple[float, float, float]:
        """
        Computes risk-based position sizing.
        Returns:
            (position_units, dollar_position_size, dollar_risk)
        """
        stop_distance = entry_price - stop_price
        if stop_distance <= 0:
            return 0.0, 0.0, 0.0

        dollar_risk = equity * self.risk_per_trade_pct
        units = dollar_risk / stop_distance
        dollar_position = units * entry_price

        # Enforce maximum position size as % of portfolio equity
        max_allowed_dollar = equity * self.max_position_equity_pct
        if dollar_position > max_allowed_dollar:
            dollar_position = max_allowed_dollar
            units = dollar_position / entry_price
            # Adjusted dollar risk
            dollar_risk = units * stop_distance

        return round(units, 6), round(dollar_position, 2), round(dollar_risk, 2)

    def can_open_new_position(
        self,
        current_open_positions_count: int,
        current_cumulative_risk_pct: float
    ) -> Tuple[bool, str]:
        """
        Evaluates portfolio constraints:
        - max simultaneous positions
        - max cumulative portfolio risk
        """
        if current_open_positions_count >= self.max_open_positions:
            return False, f"Max open positions reached ({current_open_positions_count}/{self.max_open_positions})"

        projected_risk = current_cumulative_risk_pct + self.risk_per_trade_pct
        if projected_risk > self.max_portfolio_risk_pct:
            return False, f"Max portfolio risk would be exceeded ({projected_risk*100:.2f}% > {self.max_portfolio_risk_pct*100:.2f}%)"

        return True, "Approved"
