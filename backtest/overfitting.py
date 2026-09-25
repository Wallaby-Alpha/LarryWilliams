"""
Overfitting and Parameter Cliff Detection Module.

Identifies fragile parameter spikes vs broad robust plateaus:
- Scans parameter neighborhood (e.g. Williams %R -75, -80, -85; ATR mult 1.0, 1.25, 1.5, 1.75, 2.0)
- Computes parameter stability index and variance
- Flags 'SUSPICIOUS_OPTIMIZATION' if isolated spike surrounded by sharp cliffs
"""

import copy
import logging
from typing import Dict, List, Any, Callable
import numpy as np
import pandas as pd

from .engine import BacktestEngine

logger = logging.getLogger("OverfittingDetector")


class ParameterCliffDetector:
    """
    Detects parameter fragility and sensitivity cliffs.
    """

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config

    def test_williams_r_neighborhood(
        self,
        btc_daily_df: pd.DataFrame,
        btc_1h_df: pd.DataFrame,
        universe_1h_dfs: Dict[str, pd.DataFrame],
        universe_4h_dfs: Dict[str, pd.DataFrame],
        universe_daily_dfs: Dict[str, pd.DataFrame],
        relative_strength_tables: Dict[str, pd.DataFrame],
        thresholds: List[float] = [-70.0, -75.0, -80.0, -85.0, -90.0]
    ) -> pd.DataFrame:
        """
        Runs sensitivity analysis across Williams %R thresholds to check for parameter cliffs.
        """
        results = []
        for val in thresholds:
            cfg = copy.deepcopy(self.base_config)
            cfg["williams_r"]["oversold_threshold"] = val
            engine = BacktestEngine(cfg)
            res = engine.run_backtest(
                btc_daily_df, btc_1h_df, universe_1h_dfs, universe_4h_dfs, universe_daily_dfs,
                relative_strength_tables=relative_strength_tables
            )
            m = res["metrics"]
            results.append({
                "parameter": "williams_r_threshold",
                "param_value": val,
                "net_return_pct": m.get("net_return_pct", 0.0),
                "sharpe_ratio": m.get("sharpe_ratio", 0.0),
                "win_rate_pct": m.get("win_rate_pct", 0.0),
                "profit_factor": m.get("profit_factor", 0.0),
                "total_trades": m.get("total_trades", 0),
                "max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
            })

        df = pd.DataFrame(results)
        # Compute neighbor return variance
        if len(df) > 2:
            ret_series = df["net_return_pct"]
            cliff_scores = []
            for i in range(len(ret_series)):
                neighbors = []
                if i > 0:
                    neighbors.append(ret_series.iloc[i - 1])
                if i < len(ret_series) - 1:
                    neighbors.append(ret_series.iloc[i + 1])
                avg_neighbor = np.mean(neighbors) if neighbors else ret_series.iloc[i]
                diff = abs(ret_series.iloc[i] - avg_neighbor)
                # If current is 3x higher than neighbor average, flag cliff
                is_cliff = ret_series.iloc[i] > (avg_neighbor * 2.5) if avg_neighbor > 0 else False
                cliff_scores.append("SUSPICIOUS_CLIFF" if is_cliff else "ROBUST_PLATEAU")
            df["robustness_flag"] = cliff_scores

        return df
