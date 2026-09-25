"""
Relative Strength & Leadership Engine for Crypto Universe.

Calculates:
- 7D, 30D, 90D returns for each coin and BTC
- 7D, 30D, 90D excess return vs BTC
- Cross-sectional percentile ranking across the eligible liquid universe
- Composite LEADERSHIP_SCORE: 40% 30D + 35% 7D + 25% 90D
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def compute_n_day_return(close_series: pd.Series, days: int, bars_per_day: int = 1) -> pd.Series:
    """
    Computes return over `days`.
    bars_per_day: 1 for daily data, 24 for 1H data, 6 for 4H data.
    """
    shift_periods = days * bars_per_day
    past_close = close_series.shift(shift_periods)
    ret = (close_series - past_close) / past_close.replace(0, np.nan)
    return ret


def compute_excess_return(coin_return: pd.Series, btc_return: pd.Series) -> pd.Series:
    """
    Excess return: Coin Return - BTC Return.
    """
    return coin_return - btc_return


def calculate_cross_sectional_percentiles(
    universe_returns_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Given a DataFrame where columns are coin symbols and rows are timestamps (or a single row snapshot),
    calculates percentile rank (0 to 100) for each coin along each row.
    """
    # rank(axis=1, pct=True) gives values in (0, 1]
    percentile_df = universe_returns_df.rank(axis=1, pct=True, ascending=True) * 100.0
    return percentile_df


def compute_composite_leadership_score(
    rs_7d_pctile: float,
    rs_30d_pctile: float,
    rs_90d_pctile: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Computes composite leadership score:
    Default weights: 40% 30-day, 35% 7-day, 25% 90-day.
    """
    if weights is None:
        weights = {"weight_30d": 0.40, "weight_7d": 0.35, "weight_90d": 0.25}
    
    score = (
        weights.get("weight_30d", 0.40) * rs_30d_pctile +
        weights.get("weight_7d", 0.35) * rs_7d_pctile +
        weights.get("weight_90d", 0.25) * rs_90d_pctile
    )
    return float(np.round(score, 2))


class RelativeStrengthEngine:
    """
    Engine to manage historical cross-sectional relative strength calculations across coins.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {"weight_30d": 0.40, "weight_7d": 0.35, "weight_90d": 0.25}

    def compute_universe_metrics(
        self,
        coin_daily_closes: Dict[str, pd.Series],
        btc_daily_close: pd.Series
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates time series of returns, excess returns, and cross-sectional percentiles.
        Returns a dict of DataFrames keyed by metric:
        - 'ret_7d', 'ret_30d', 'ret_90d'
        - 'excess_7d', 'excess_30d', 'excess_90d'
        - 'pctile_7d', 'pctile_30d', 'pctile_90d'
        - 'leadership_score'
        """
        # Align all coin daily closes into a single DataFrame
        df_all = pd.DataFrame(coin_daily_closes).sort_index()
        btc_close = btc_daily_close.reindex(df_all.index).ffill()

        # Compute coin returns
        ret_7d = (df_all - df_all.shift(7)) / df_all.shift(7).replace(0, np.nan)
        ret_30d = (df_all - df_all.shift(30)) / df_all.shift(30).replace(0, np.nan)
        ret_90d = (df_all - df_all.shift(90)) / df_all.shift(90).replace(0, np.nan)

        # Compute BTC returns
        btc_ret_7d = (btc_close - btc_close.shift(7)) / btc_close.shift(7).replace(0, np.nan)
        btc_ret_30d = (btc_close - btc_close.shift(30)) / btc_close.shift(30).replace(0, np.nan)
        btc_ret_90d = (btc_close - btc_close.shift(90)) / btc_close.shift(90).replace(0, np.nan)

        # Excess returns
        excess_7d = ret_7d.sub(btc_ret_7d, axis=0)
        excess_30d = ret_30d.sub(btc_ret_30d, axis=0)
        excess_90d = ret_90d.sub(btc_ret_90d, axis=0)

        # Cross-sectional percentiles (using excess returns across universe at each timestamp)
        pctile_7d = calculate_cross_sectional_percentiles(excess_7d)
        pctile_30d = calculate_cross_sectional_percentiles(excess_30d)
        pctile_90d = calculate_cross_sectional_percentiles(excess_90d)

        # Composite leadership score
        w30 = self.weights.get("weight_30d", 0.40)
        w7 = self.weights.get("weight_7d", 0.35)
        w90 = self.weights.get("weight_90d", 0.25)
        
        leadership_score = (pctile_30d * w30) + (pctile_7d * w7) + (pctile_90d * w90)

        return {
            "ret_7d": ret_7d,
            "ret_30d": ret_30d,
            "ret_90d": ret_90d,
            "excess_7d": excess_7d,
            "excess_30d": excess_30d,
            "excess_90d": excess_90d,
            "pctile_7d": pctile_7d,
            "pctile_30d": pctile_30d,
            "pctile_90d": pctile_90d,
            "leadership_score": leadership_score,
            "btc_ret_7d": btc_ret_7d,
            "btc_ret_30d": btc_ret_30d,
            "btc_ret_90d": btc_ret_90d,
        }
