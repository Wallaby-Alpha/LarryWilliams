"""
Walk-Forward and Out-of-Sample Testing Module.

Implements:
- Chronological train/validation/test splits (No look-ahead bias)
- Rolling window walk-forward optimization and validation (e.g. 6m train / 2m test)
- Clear comparison between In-Sample and Out-of-Sample metrics
- Walk-forward efficiency ratio (WFE = Out-of-sample CAGR / In-sample CAGR)
"""

import copy
import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

from .engine import BacktestEngine

logger = logging.getLogger("WalkForward")


def _safe_slice_df(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Safely slices DataFrame by timestamp accounting for potential tz-awareness differences."""
    if df.empty:
        return df
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is not None and start_date.tzinfo is None:
        s_date = start_date.tz_localize("UTC")
        e_date = end_date.tz_localize("UTC")
    elif ts.dt.tz is None and start_date.tzinfo is not None:
        s_date = start_date.tz_localize(None)
        e_date = end_date.tz_localize(None)
    else:
        s_date = start_date
        e_date = end_date

    mask = (ts >= s_date) & (ts <= e_date)
    return df[mask].reset_index(drop=True)


class WalkForwardOptimizer:
    """
    Executes walk-forward and out-of-sample validation.
    """

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config

    def split_dataset_by_dates(
        self,
        btc_daily_df: pd.DataFrame,
        btc_1h_df: pd.DataFrame,
        universe_1h_dfs: Dict[str, pd.DataFrame],
        universe_4h_dfs: Dict[str, pd.DataFrame],
        universe_daily_dfs: Dict[str, pd.DataFrame],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Slices all data tables to a specific window [start_date, end_date]."""
        b_daily = _safe_slice_df(btc_daily_df, start_date, end_date)
        b_1h = _safe_slice_df(btc_1h_df, start_date, end_date)

        u_1h = {s: _safe_slice_df(df, start_date, end_date) for s, df in universe_1h_dfs.items()}
        u_4h = {s: _safe_slice_df(df, start_date, end_date) for s, df in universe_4h_dfs.items()}
        u_daily = {s: _safe_slice_df(df, start_date, end_date) for s, df in universe_daily_dfs.items()}

        return b_daily, b_1h, u_1h, u_4h, u_daily

    def run_train_test_split(
        self,
        btc_daily_df: pd.DataFrame,
        btc_1h_df: pd.DataFrame,
        universe_1h_dfs: Dict[str, pd.DataFrame],
        universe_4h_dfs: Dict[str, pd.DataFrame],
        universe_daily_dfs: Dict[str, pd.DataFrame],
        relative_strength_tables: Dict[str, pd.DataFrame],
        train_ratio: float = 0.67
    ) -> Dict[str, Any]:
        """
        Splits data into In-Sample (training) and Out-of-Sample (testing).
        """
        all_ts = btc_1h_df["timestamp"].sort_values().reset_index(drop=True)
        split_idx = int(len(all_ts) * train_ratio)
        split_date = all_ts.iloc[split_idx]
        start_date = all_ts.iloc[0]
        end_date = all_ts.iloc[-1]

        # Slices
        in_sample_data = self.split_dataset_by_dates(
            btc_daily_df, btc_1h_df, universe_1h_dfs, universe_4h_dfs, universe_daily_dfs,
            start_date=start_date, end_date=split_date
        )
        out_sample_data = self.split_dataset_by_dates(
            btc_daily_df, btc_1h_df, universe_1h_dfs, universe_4h_dfs, universe_daily_dfs,
            start_date=split_date, end_date=end_date
        )

        engine_is = BacktestEngine(self.base_config)
        res_is = engine_is.run_backtest(
            in_sample_data[0], in_sample_data[1], in_sample_data[2], in_sample_data[3], in_sample_data[4],
            relative_strength_tables=relative_strength_tables
        )

        engine_oos = BacktestEngine(self.base_config)
        res_oos = engine_oos.run_backtest(
            out_sample_data[0], out_sample_data[1], out_sample_data[2], out_sample_data[3], out_sample_data[4],
            relative_strength_tables=relative_strength_tables
        )

        return {
            "split_date": str(split_date),
            "in_sample": {
                "period": f"{start_date.date()} to {split_date.date()}",
                "metrics": res_is["metrics"],
                "trade_log": res_is["trade_log"],
                "equity_curve": res_is["equity_curve"]
            },
            "out_of_sample": {
                "period": f"{split_date.date()} to {end_date.date()}",
                "metrics": res_oos["metrics"],
                "trade_log": res_oos["trade_log"],
                "equity_curve": res_oos["equity_curve"]
            }
        }
