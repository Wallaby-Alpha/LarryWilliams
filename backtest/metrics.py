"""
Institutional Performance Metrics Module.

Calculates:
- Total Return, CAGR, Annualized Volatility, Max Drawdown
- Sharpe, Sortino, Calmar, Profit Factor
- Win Rate, Average Win, Average Loss, Win/Loss Ratio
- Average R Multiple, Median R, Expectancy per trade
- Consecutive Wins / Losses, Exposure %, Turnover
- Total Fees & Slippage paid
- MAE (Maximum Adverse Excursion) & MFE (Maximum Favorable Excursion)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


def calculate_comprehensive_metrics(
    equity_curve: pd.Series,
    trade_log: pd.DataFrame,
    initial_capital: float = 100000.0,
    annual_periods: int = 365 * 24 # for 1-hour candles
) -> Dict[str, Any]:
    """
    Computes complete performance metrics from equity curve and trade log.
    """
    metrics: Dict[str, Any] = {}

    if equity_curve.empty or len(equity_curve) < 2:
        return {"total_trades": 0, "net_return_pct": 0.0}

    # 1. Equity & Return metrics
    ending_capital = float(equity_curve.iloc[-1])
    total_net_pnl = ending_capital - initial_capital
    net_return_pct = (total_net_pnl / initial_capital) * 100.0

    # Hourly returns
    returns = equity_curve.pct_change().dropna()
    num_bars = len(equity_curve)
    years = max(num_bars / annual_periods, 0.05)

    cagr = ((ending_capital / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if ending_capital > 0 else -100.0
    ann_vol = float(returns.std() * np.sqrt(annual_periods) * 100.0) if len(returns) > 1 else 0.0

    # Drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_pct = float(drawdown.min() * 100.0)

    # Risk-adjusted ratios
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(annual_periods)) if returns.std() > 0 else 0.0
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(annual_periods)
    sortino = float(returns.mean() * np.sqrt(annual_periods) / downside_std) if downside_std > 0 else 0.0
    calmar = float(cagr / abs(max_drawdown_pct)) if abs(max_drawdown_pct) > 0 else 0.0

    # 2. Trade Log Metrics
    num_trades = len(trade_log)
    if num_trades > 0:
        net_pnls = trade_log["net_pnl"]
        r_multiples = trade_log["r_multiple"] if "r_multiple" in trade_log.columns else pd.Series([0.0] * num_trades)

        winning_trades = trade_log[net_pnls > 0]
        losing_trades = trade_log[net_pnls < 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / num_trades) * 100.0

        gross_profits = winning_trades["net_pnl"].sum()
        gross_losses = abs(losing_trades["net_pnl"].sum())
        profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (99.0 if gross_profits > 0 else 0.0)

        avg_win = float(winning_trades["net_pnl"].mean()) if win_count > 0 else 0.0
        avg_loss = float(abs(losing_trades["net_pnl"].mean())) if loss_count > 0 else 0.0
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        expectancy_dollar = float(net_pnls.mean())
        expectancy_r = float(r_multiples.mean())
        median_r = float(r_multiples.median())

        largest_winner = float(net_pnls.max())
        largest_loser = float(net_pnls.min())

        avg_holding_hours = float(trade_log["holding_hours"].mean()) if "holding_hours" in trade_log.columns else 0.0

        # Streaks
        is_win = (net_pnls > 0).astype(int)
        win_streaks = (is_win != is_win.shift()).cumsum()
        max_consecutive_wins = int(is_win.groupby(win_streaks).sum().max()) if win_count > 0 else 0
        
        is_loss = (net_pnls < 0).astype(int)
        loss_streaks = (is_loss != is_loss.shift()).cumsum()
        max_consecutive_losses = int(is_loss.groupby(loss_streaks).sum().max()) if loss_count > 0 else 0

        # Fees & Slippage
        total_fees = float(trade_log["fees"].sum()) if "fees" in trade_log.columns else 0.0
        total_slippage = float(trade_log["slippage"].sum()) if "slippage" in trade_log.columns else 0.0
        gross_pnl = float(trade_log["gross_pnl"].sum()) if "gross_pnl" in trade_log.columns else total_net_pnl

        # MAE and MFE
        avg_mae_r = float(trade_log["mae_r"].mean()) if "mae_r" in trade_log.columns else 0.0
        avg_mfe_r = float(trade_log["mfe_r"].mean()) if "mfe_r" in trade_log.columns else 0.0

    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = avg_loss = win_loss_ratio = 0.0
        expectancy_dollar = expectancy_r = median_r = 0.0
        largest_winner = largest_loser = avg_holding_hours = 0.0
        max_consecutive_wins = max_consecutive_losses = 0
        total_fees = total_slippage = gross_pnl = 0.0
        avg_mae_r = avg_mfe_r = 0.0

    metrics.update({
        "initial_capital": initial_capital,
        "ending_capital": round(ending_capital, 2),
        "total_net_pnl": round(total_net_pnl, 2),
        "net_return_pct": round(net_return_pct, 2),
        "cagr_pct": round(cagr, 2),
        "annualized_volatility_pct": round(ann_vol, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "profit_factor": round(profit_factor, 2),
        "win_rate_pct": round(win_rate, 2),
        "total_trades": num_trades,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "win_loss_ratio": round(win_loss_ratio, 2),
        "expectancy_dollar": round(expectancy_dollar, 2),
        "expectancy_r": round(expectancy_r, 2),
        "median_r": round(median_r, 2),
        "largest_winner": round(largest_winner, 2),
        "largest_loser": round(largest_loser, 2),
        "avg_holding_hours": round(avg_holding_hours, 1),
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "gross_pnl": round(gross_pnl, 2),
        "total_fees": round(total_fees, 2),
        "total_slippage": round(total_slippage, 2),
        "avg_mae_r": round(avg_mae_r, 2),
        "avg_mfe_r": round(avg_mfe_r, 2),
    })

    return metrics
