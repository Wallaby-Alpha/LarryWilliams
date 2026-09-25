"""
Feature Analysis and Trade Performance Attribution Module.

Evaluates which strategy rules provide genuine incremental edge:
- RS Decile breakdown (7D > 90th vs 80-90th)
- Williams %R depth (< -90 vs -80 to -90)
- Momentum Failure contribution
- Volume Confirmation impact
- Performance by Market Regime (Bull, Bear, Sideways)
- Performance by Calendar Year and Coin
"""

import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd

logger = logging.getLogger("FeatureAnalysis")


class FeatureAnalyzer:
    """
    Analyzes which trade characteristics correlate with higher win rates and expectancy.
    """

    def __init__(self, trade_log: pd.DataFrame):
        self.trade_log = trade_log.copy()

    def generate_full_analysis(self) -> Dict[str, Any]:
        """Runs all feature correlation breakdowns."""
        if self.trade_log.empty or len(self.trade_log) < 3:
            return {"status": "INSUFFICIENT_TRADES", "message": "At least 3 completed trades required."}

        df = self.trade_log.copy()
        df["is_win"] = df["net_pnl"] > 0

        # 1. Relative Strength Breakdown (RS 7D > 90 vs 80-90)
        df["rs_7d_bucket"] = pd.cut(
            df["rs_7d"], bins=[0, 79.9, 89.9, 100], labels=["< 80th", "80th - 90th", "Top 10% (>90th)"]
        )
        rs_7d_perf = self._group_stats(df, "rs_7d_bucket")

        # 2. Williams %R Entry Depth Breakdown (Deep <= -90 vs Regular -80 to -90)
        df["wr_depth_bucket"] = pd.cut(
            df["williams_r_min"], bins=[-101, -90, -80, 0], labels=["Extreme (< -90)", "Oversold (-80 to -90)", "Mild (> -80)"]
        )
        wr_perf = self._group_stats(df, "wr_depth_bucket")

        # 3. Momentum Failure Flag Performance
        mom_perf = self._group_stats(df, "momentum_failure_flag")

        # 4. Pullback Depth in ATR multiples
        df["pb_atr_bucket"] = pd.cut(
            df["pullback_depth_atr"], bins=[0, 1.0, 1.5, 2.0, 10.0], labels=["< 1.0 ATR", "1.0 - 1.5 ATR", "1.5 - 2.0 ATR", "> 2.0 ATR"]
        )
        pb_perf = self._group_stats(df, "pb_atr_bucket")

        # 5. Performance by Symbol
        sym_perf = self._group_stats(df, "symbol")

        # 6. Performance by Exit Reason
        exit_perf = self._group_stats(df, "reason_for_exit")

        return {
            "status": "SUCCESS",
            "total_trades_analyzed": len(df),
            "rs_7d_performance": rs_7d_perf.to_dict(orient="records"),
            "williams_depth_performance": wr_perf.to_dict(orient="records"),
            "momentum_failure_performance": mom_perf.to_dict(orient="records"),
            "pullback_atr_performance": pb_perf.to_dict(orient="records"),
            "exit_reason_performance": exit_perf.to_dict(orient="records"),
            "symbol_performance": sym_perf.head(10).to_dict(orient="records"),
        }

    @staticmethod
    def _group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """Helper to compute win rate, avg R, and profit factor per group."""
        groups = []
        for val, grp in df.groupby(group_col, observed=False):
            n = len(grp)
            if n == 0:
                continue
            wins = grp[grp["net_pnl"] > 0]
            losses = grp[grp["net_pnl"] < 0]
            win_rate = (len(wins) / n) * 100.0
            avg_r = grp["r_multiple"].mean()
            median_r = grp["r_multiple"].median()
            total_net = grp["net_pnl"].sum()
            profit_factor = (wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())) if len(losses) > 0 and losses["net_pnl"].sum() != 0 else (99.0 if len(wins) > 0 else 0.0)

            groups.append({
                group_col: str(val),
                "trades": n,
                "win_rate_pct": round(win_rate, 1),
                "avg_r": round(float(avg_r), 2),
                "median_r": round(float(median_r), 2),
                "total_net_pnl": round(float(total_net), 2),
                "profit_factor": round(float(profit_factor), 2),
            })

        return pd.DataFrame(groups)
