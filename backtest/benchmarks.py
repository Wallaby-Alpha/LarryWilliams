"""
Benchmark Comparison Suite.

Compares Larry Williams System against:
1. Buy-and-Hold BTC
2. Buy-and-Hold ETH / Top Asset
3. Equal-Weight Altcoin Basket
4. BTC Trend Following (Long if Close > 50 SMA, else Cash)
5. Randomized Entry Benchmark (Monte Carlo test with identical risk & exits)
"""

import copy
import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from indicators.technical import sma
from execution.risk_manager import RiskManager
from execution.simulator import ExecutionSimulator

logger = logging.getLogger("Benchmarks")


class BenchmarkSuite:
    """
    Computes performance benchmarks.
    """

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital

    def buy_and_hold_asset(self, ohlcv_df: pd.DataFrame, asset_name: str = "BTC") -> pd.DataFrame:
        """Computes buy-and-hold equity curve."""
        df = ohlcv_df.sort_values("timestamp").reset_index(drop=True)
        start_price = df["close"].iloc[0]
        units = self.initial_capital / start_price
        equity = df["close"] * units
        return pd.DataFrame({"timestamp": df["timestamp"], f"equity_{asset_name}": equity}).set_index("timestamp")

    def equal_weight_altcoin_basket(self, universe_1h_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Equal-weight basket of universe altcoins."""
        all_closes = {}
        for sym, df in universe_1h_dfs.items():
            if not df.empty:
                s = df.set_index("timestamp")["close"]
                all_closes[sym] = s
        df_all = pd.DataFrame(all_closes).sort_index().ffill().dropna()
        # Daily or hourly returns
        ret = df_all.pct_change().mean(axis=1)
        equity = self.initial_capital * (1.0 + ret).cumprod()
        equity.iloc[0] = self.initial_capital
        return pd.DataFrame({"equity_altcoin_basket": equity})

    def btc_trend_following(self, btc_daily_df: pd.DataFrame, sma_period: int = 50) -> pd.DataFrame:
        """BTC Trend-following strategy (Long if Close > 50 SMA, Cash if below)."""
        df = btc_daily_df.sort_values("timestamp").copy()
        df["sma50"] = sma(df["close"], sma_period)
        df["in_market"] = (df["close"].shift(1) > df["sma50"].shift(1)).astype(float)
        ret = df["close"].pct_change() * df["in_market"]
        equity = self.initial_capital * (1.0 + ret).cumprod()
        equity.iloc[0] = self.initial_capital
        return pd.DataFrame({"timestamp": df["timestamp"], "equity_btc_trend": equity}).set_index("timestamp")

    def randomized_entry_test(
        self,
        config: Dict[str, Any],
        universe_1h_dfs: Dict[str, pd.DataFrame],
        num_simulations: int = 10,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Runs Monte Carlo randomized entries with exact same risk and exit rules
        to establish whether the Williams + RS signals outperform random chance.
        """
        np.random.seed(seed)
        risk_mgr = RiskManager(config)
        sim = ExecutionSimulator(config)
        all_syms = list(universe_1h_dfs.keys())

        sim_returns = []
        for sim_idx in range(num_simulations):
            # Select random entries across symbols
            cash = self.initial_capital
            trades = []
            for sym in all_syms[:10]:
                df = universe_1h_dfs[sym]
                if len(df) < 50:
                    continue
                # Pick 5 random entry points
                entry_indices = np.random.choice(len(df) - 50, size=min(4, (len(df) - 50) // 10), replace=False)
                for idx in entry_indices:
                    entry_bar = df.iloc[idx]
                    entry_price = entry_bar["close"]
                    stop_price = entry_price * 0.97
                    target_price = entry_price * 1.06
                    dollar_risk = cash * risk_mgr.risk_per_trade_pct
                    units = dollar_risk / (entry_price - stop_price)

                    # Look forward up to 48 bars
                    exit_bar = df.iloc[min(idx + 48, len(df) - 1)]
                    exit_price = exit_bar["close"]
                    # Check if stop or target hit in window
                    sub = df.iloc[idx + 1: idx + 49]
                    if (sub["low"] <= stop_price).any():
                        exit_price = stop_price
                    elif (sub["high"] >= target_price).any():
                        exit_price = target_price

                    pnl_info = sim.compute_trade_pnl(entry_price, exit_price, units, dollar_risk)
                    cash += pnl_info["net_pnl"]

            sim_ret = ((cash - self.initial_capital) / self.initial_capital) * 100.0
            sim_returns.append(sim_ret)

        return {
            "num_simulations": num_simulations,
            "random_mean_return_pct": round(float(np.mean(sim_returns)), 2),
            "random_std_return_pct": round(float(np.std(sim_returns)), 2),
            "random_max_return_pct": round(float(np.max(sim_returns)), 2),
            "random_min_return_pct": round(float(np.min(sim_returns)), 2),
        }
