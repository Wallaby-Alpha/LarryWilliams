"""
Event-Driven Multi-Asset Portfolio Backtesting Engine (High Performance).

Simulates candle-by-candle execution with zero look-ahead bias:
- Multi-asset portfolio tracking with realistic cash & equity accounting
- Vector-accelerated signal detection + event-driven portfolio management
- Modular exit models: 1R, 1.5R, 2R, 3R, trailing stop, %R reversal, RS loss, partial 2R+trail
- Time-based exits (12h, 24h, 36h, 48h, 72h)
- BTC regime exit behavior (exit_all, stop_new_entries, do_nothing)
- High-fidelity intra-candle MAE / MFE calculation
- Full trade log recording all entry features for feature analysis.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from strategies.williams_signal_engine import WilliamsSignalEngine
from execution.risk_manager import RiskManager
from execution.simulator import ExecutionSimulator
from .metrics import calculate_comprehensive_metrics

logger = logging.getLogger("BacktestEngine")


class BacktestEngine:
    """
    Chronological portfolio backtesting engine.
    """

    def __init__(self, config: Dict[str, Any], initial_capital: float = 100000.0):
        self.config = config
        self.initial_capital = initial_capital
        self.signal_engine = WilliamsSignalEngine(config)
        self.risk_manager = RiskManager(config)
        self.simulator = ExecutionSimulator(config)

        self.exit_cfg = config.get("exit_strategy", {})
        self.time_stop_cfg = config.get("time_stop", {})
        self.btc_exit_cfg = config.get("btc_regime_exit", {})

    def run_backtest(
        self,
        btc_daily_df: pd.DataFrame,
        btc_1h_df: pd.DataFrame,
        universe_1h_dfs: Dict[str, pd.DataFrame],
        universe_4h_dfs: Dict[str, pd.DataFrame],
        universe_daily_dfs: Dict[str, pd.DataFrame],
        relative_strength_tables: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Executes fast chronological backtest across the universe.
        """
        # 1. Precompute BTC Daily Regime
        btc_regime_df = self.signal_engine.evaluate_btc_regime(btc_daily_df)
        btc_regime_df["timestamp"] = pd.to_datetime(btc_regime_df["timestamp"])
        btc_regime_map = dict(zip(btc_regime_df["timestamp"].dt.floor("1D"), btc_regime_df["btc_long_regime"]))

        # 2. Build Multi-Timeframe Feature Tables and vectorize signal readiness
        coin_feature_tables: Dict[str, pd.DataFrame] = {}
        coin_bar_dict: Dict[str, Dict[Any, Dict[str, Any]]] = {}

        min_7d = float(self.config.get("relative_strength", {}).get("rs_7d_min_percentile", 80.0))
        min_30d = float(self.config.get("relative_strength", {}).get("rs_30d_min_percentile", 80.0))
        max_ext = float(self.config.get("extension_filter", {}).get("max_extension_pct", 0.15))
        min_pb_atr = float(self.config.get("pullback", {}).get("min_pullback_atr", 1.0))
        min_pb_pct = float(self.config.get("pullback", {}).get("min_pullback_pct", 0.02))
        max_pb_pct = float(self.config.get("pullback", {}).get("max_pullback_pct", 0.08))
        pb_model = self.config.get("pullback", {}).get("model", "model_a")
        wr_thresh = float(self.config.get("williams_r", {}).get("oversold_threshold", -80.0))
        pa_model = self.config.get("price_action_confirmation", {}).get("model", "close_above_prev_high")

        for sym, df_1h in universe_1h_dfs.items():
            df_4h = universe_4h_dfs.get(sym, pd.DataFrame())
            df_daily = universe_daily_dfs.get(sym, pd.DataFrame())
            if df_1h.empty or df_daily.empty:
                continue

            rs_metrics = {
                "rs_7d_pctile": relative_strength_tables.get("pctile_7d", pd.DataFrame()).get(sym, pd.Series()),
                "rs_30d_pctile": relative_strength_tables.get("pctile_30d", pd.DataFrame()).get(sym, pd.Series()),
                "rs_90d_pctile": relative_strength_tables.get("pctile_90d", pd.DataFrame()).get(sym, pd.Series()),
                "leadership_score": relative_strength_tables.get("leadership_score", pd.DataFrame()).get(sym, pd.Series()),
            }

            feat_df = self.signal_engine.build_coin_feature_table(df_1h, df_4h, df_daily, rs_metrics)
            feat_df = feat_df.sort_values("timestamp").reset_index(drop=True)

            # --- Vectorized Condition Flags ---
            # RS leadership
            c_leader = (feat_df["rs_7d_pctile"] >= min_7d) & (feat_df["rs_30d_pctile"] >= min_30d)
            # Trend
            c_trend = feat_df["daily_uptrend"] & feat_df["four_hour_uptrend"]
            # Extension
            c_ext = feat_df["ext_from_ema20"] <= max_ext
            # Pullback
            if pb_model == "model_a":
                c_pb = feat_df["pullback_depth_atr"] >= min_pb_atr
            elif pb_model == "model_b":
                c_pb = (feat_df["pullback_depth_pct"] >= min_pb_pct) & (feat_df["pullback_depth_pct"] <= max_pb_pct)
            elif pb_model == "model_d":
                c_pb = (feat_df["low"] < feat_df["ema20"]) & (feat_df["low"] >= feat_df["sma50"])
            else:
                c_pb = feat_df["pullback_depth_atr"] >= min_pb_atr
            
            # Williams %R cross
            c_wr = (feat_df["williams_r"].shift(1) <= wr_thresh) & (feat_df["williams_r"] > wr_thresh)

            # Price Action Confirmation
            if pa_model == "close_above_prev_high":
                c_pa = feat_df["close"] > feat_df["high"].shift(1)
            elif pa_model == "break_swing_high":
                c_pa = feat_df["close"] > feat_df["swing_high"]
            elif pa_model == "close_above_ema20":
                c_pa = feat_df["close"] > feat_df["ema20"]
            elif pa_model == "bullish_engulfing":
                c_pa = (feat_df["close"] > feat_df["open"].shift(1)) & (feat_df["open"] <= feat_df["close"].shift(1))
            else: # williams_cross_only
                c_pa = pd.Series([True] * len(feat_df), index=feat_df.index)

            feat_df["signal_ready_base"] = c_leader & c_trend & c_ext & c_pb & c_wr & c_pa

            coin_feature_tables[sym] = feat_df
            # Create fast dict lookup: {ts: row_dict}
            coin_bar_dict[sym] = feat_df.set_index("timestamp").to_dict(orient="index")

        # 3. Align Unified Timestamps
        all_timestamps = sorted(list(set.union(*[set(d.keys()) for d in coin_bar_dict.values()])))
        if len(all_timestamps) < 50:
            return {"metrics": {"total_trades": 0}, "trade_log": pd.DataFrame(), "equity_curve": pd.Series()}

        # 4. State Tracking
        cash = self.initial_capital
        open_positions: Dict[str, Dict[str, Any]] = {}
        pending_orders: List[Dict[str, Any]] = []
        completed_trades: List[Dict[str, Any]] = []
        equity_records: List[Dict[str, Any]] = []

        exit_model = self.exit_cfg.get("model", "2R")
        trailing_atr_mult = float(self.exit_cfg.get("trailing_atr_mult", 1.5))
        time_stop_enabled = self.time_stop_cfg.get("enabled", True)
        max_holding_hours = int(self.time_stop_cfg.get("max_holding_hours", 48))
        btc_exit_action = self.btc_exit_cfg.get("action", "stop_new_entries")

        # 5. Chronological Bar-by-Bar Loop (Ultra-fast dict lookups)
        for step_idx, ts in enumerate(all_timestamps):
            day_key = ts.floor("1D")
            btc_regime_active = btc_regime_map.get(day_key, True)

            # --- A. EXECUTE PENDING ORDERS ON CURRENT BAR OPEN ---
            for order in pending_orders:
                sym = order["symbol"]
                sym_dict = coin_bar_dict.get(sym)
                if sym_dict and ts in sym_dict:
                    bar = sym_dict[ts]
                    candle_open = bar["open"]
                    fill_price, slip_pct, fee_pct = self.simulator.simulate_entry_fill(candle_open)

                    stop_price = order["planned_stop"]
                    atr_val = order["signal_info"]["atr"]
                    
                    if self.risk_manager.stop_loss_model == "atr":
                        stop_price = self.risk_manager.calculate_stop_loss(fill_price, order["planned_low"], atr_val, "atr")

                    r_dist = fill_price - stop_price
                    if r_dist <= 0:
                        continue

                    curr_equity = cash + sum(pos["units"] * pos["current_price"] for pos in open_positions.values())
                    units, dollar_pos, dollar_risk = self.risk_manager.calculate_position_size(curr_equity, fill_price, stop_price)

                    total_cost = (units * fill_price) * (1.0 + fee_pct)
                    if total_cost <= cash and units > 0:
                        cash -= total_cost
                        
                        if exit_model == "1R":
                            target_price = fill_price + r_dist * 1.0
                        elif exit_model == "1.5R":
                            target_price = fill_price + r_dist * 1.5
                        elif exit_model == "2R" or exit_model == "partial_2r_trail":
                            target_price = fill_price + r_dist * 2.0
                        elif exit_model == "3R":
                            target_price = fill_price + r_dist * 3.0
                        else:
                            target_price = fill_price + r_dist * 2.0

                        open_positions[sym] = {
                            "symbol": sym,
                            "entry_time": ts,
                            "entry_price": fill_price,
                            "stop_price": stop_price,
                            "target_price": target_price,
                            "units": units,
                            "remaining_units": units,
                            "initial_units": units,
                            "dollar_risk": dollar_risk,
                            "r_distance": r_dist,
                            "current_price": fill_price,
                            "highest_high": bar["high"],
                            "lowest_low": bar["low"],
                            "holding_hours": 0,
                            "partial_taken": False,
                            "signal_info": order["signal_info"],
                            "mae_r": 0.0,
                            "mfe_r": 0.0,
                        }
            pending_orders = []

            # --- B. UPDATE & MANAGE EXISTING POSITIONS ---
            closed_syms = []
            for sym, pos in open_positions.items():
                sym_dict = coin_bar_dict.get(sym)
                if not sym_dict or ts not in sym_dict:
                    continue

                bar = sym_dict[ts]
                b_open, b_high, b_low, b_close = bar["open"], bar["high"], bar["low"], bar["close"]
                pos["holding_hours"] += 1
                pos["current_price"] = b_close

                # Track MAE & MFE
                pos["highest_high"] = max(pos["highest_high"], b_high)
                pos["lowest_low"] = min(pos["lowest_low"], b_low)
                adverse_dist = pos["entry_price"] - pos["lowest_low"]
                favorable_dist = pos["highest_high"] - pos["entry_price"]
                pos["mae_r"] = max(pos["mae_r"], adverse_dist / pos["r_distance"])
                pos["mfe_r"] = max(pos["mfe_r"], favorable_dist / pos["r_distance"])

                exit_triggered = False
                exit_price = b_close
                exit_reason = "NONE"
                is_stop = False

                if not btc_regime_active and btc_exit_action == "exit_all":
                    exit_triggered = True
                    exit_price = b_open
                    exit_reason = "BTC_REGIME_VIOLATION"

                if not exit_triggered:
                    stop_hit, target_hit, _ = self.simulator.resolve_candle_collisions(
                        b_open, b_high, b_low, b_close, pos["stop_price"], pos["target_price"]
                    )
                    if stop_hit:
                        exit_triggered = True
                        exit_price = pos["stop_price"]
                        exit_reason = "STOP_LOSS"
                        is_stop = True
                    elif target_hit:
                        if exit_model == "partial_2r_trail" and not pos["partial_taken"]:
                            partial_units = pos["remaining_units"] * 0.5
                            pos["remaining_units"] -= partial_units
                            pos["partial_taken"] = True
                            pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])
                            p_fill, _, _ = self.simulator.simulate_exit_fill(pos["target_price"], is_stop=False)
                            pnl_part = self.simulator.compute_trade_pnl(pos["entry_price"], p_fill, partial_units, pos["dollar_risk"] * 0.5)
                            cash += (partial_units * p_fill) - pnl_part["fees"] - pnl_part["slippage"]
                        else:
                            exit_triggered = True
                            exit_price = pos["target_price"]
                            exit_reason = f"TARGET_{exit_model}"

                if not exit_triggered and (exit_model == "trailing_stop" or pos["partial_taken"]):
                    atr_cur = bar.get("atr", pos["r_distance"])
                    trail_level = pos["highest_high"] - (atr_cur * trailing_atr_mult)
                    if trail_level > pos["stop_price"]:
                        pos["stop_price"] = trail_level
                    if b_low <= pos["stop_price"]:
                        exit_triggered = True
                        exit_price = pos["stop_price"]
                        exit_reason = "TRAILING_STOP"
                        is_stop = True

                if not exit_triggered and exit_model == "williams_overbought_reversal":
                    wr_cur = bar.get("williams_r", -50.0)
                    if wr_cur > -20.0 and b_close < bar.get("ema20", b_close):
                        exit_triggered = True
                        exit_price = b_close
                        exit_reason = "WILLIAMS_OVERBOUGHT_REVERSAL"

                if not exit_triggered and exit_model == "rs_loss":
                    rs_7d = bar.get("rs_7d_pctile", 80.0)
                    if rs_7d < 50.0:
                        exit_triggered = True
                        exit_price = b_close
                        exit_reason = "RS_LEADERSHIP_LOST"

                if not exit_triggered and time_stop_enabled:
                    if pos["holding_hours"] >= max_holding_hours:
                        cur_r = (b_close - pos["entry_price"]) / pos["r_distance"]
                        min_req_r = float(self.time_stop_cfg.get("min_profit_threshold_r", 0.5))
                        if cur_r < min_req_r:
                            exit_triggered = True
                            exit_price = b_close
                            exit_reason = f"TIME_STOP_{max_holding_hours}H"

                if exit_triggered:
                    exec_exit_price, slip_pct, fee_pct = self.simulator.simulate_exit_fill(exit_price, is_stop=is_stop)
                    pnl_info = self.simulator.compute_trade_pnl(
                        pos["entry_price"], exec_exit_price, pos["remaining_units"], pos["dollar_risk"]
                    )
                    cash += (pos["remaining_units"] * exec_exit_price) - pnl_info["fees"] - pnl_info["slippage"]

                    sig = pos["signal_info"]
                    trade_record = {
                        "timestamp_entry": pos["entry_time"],
                        "timestamp_exit": ts,
                        "symbol": sym,
                        "btc_regime": btc_regime_active,
                        "rs_7d": sig.get("rs_7d", 0.0),
                        "rs_30d": sig.get("rs_30d", 0.0),
                        "rs_90d": sig.get("rs_90d", 0.0),
                        "leadership_score": sig.get("leadership_score", 0.0),
                        "daily_trend": True,
                        "four_h_trend": True,
                        "williams_r_entry": sig.get("wr_curr", 0.0),
                        "williams_r_min": sig.get("wr_min", 0.0),
                        "momentum_failure_flag": sig.get("momentum_failure", False),
                        "pullback_depth_atr": sig.get("pullback_atr", 0.0),
                        "volume_ratio": sig.get("volume_ratio", 1.0),
                        "entry_price": pos["entry_price"],
                        "stop_price": pos["stop_price"],
                        "exit_price": exec_exit_price,
                        "units": pos["initial_units"],
                        "gross_pnl": pnl_info["gross_pnl"],
                        "fees": pnl_info["fees"],
                        "slippage": pnl_info["slippage"],
                        "net_pnl": pnl_info["net_pnl"],
                        "r_multiple": pnl_info["r_multiple"],
                        "holding_hours": pos["holding_hours"],
                        "reason_for_exit": exit_reason,
                        "mae_r": round(pos["mae_r"], 2),
                        "mfe_r": round(pos["mfe_r"], 2),
                    }
                    completed_trades.append(trade_record)
                    closed_syms.append(sym)

            for sym in closed_syms:
                del open_positions[sym]

            # --- C. SCAN FOR NEW SIGNALS ON CURRENT BAR ---
            if btc_regime_active or btc_exit_action == "do_nothing":
                curr_open_count = len(open_positions) + len(pending_orders)
                curr_cum_risk = sum(self.risk_manager.risk_per_trade_pct for _ in range(curr_open_count))

                if curr_open_count < self.risk_manager.max_open_positions:
                    for sym, sym_dict in coin_bar_dict.items():
                        if sym in open_positions or any(o["symbol"] == sym for o in pending_orders):
                            continue

                        bar = sym_dict.get(ts)
                        if bar and bar.get("signal_ready_base", False):
                            can_open, _ = self.risk_manager.can_open_new_position(curr_open_count, curr_cum_risk)
                            if can_open:
                                planned_stop = self.risk_manager.calculate_stop_loss(
                                    bar["close"], bar.get("low", bar["close"] * 0.98), bar.get("atr", bar["close"] * 0.02)
                                )
                                sig_info = {
                                    "rs_7d": bar.get("rs_7d_pctile", 80.0),
                                    "rs_30d": bar.get("rs_30d_pctile", 80.0),
                                    "rs_90d": bar.get("rs_90d_pctile", 50.0),
                                    "leadership_score": bar.get("leadership_score", 70.0),
                                    "wr_curr": bar.get("williams_r", -75.0),
                                    "wr_min": bar.get("wr_min_recent", -85.0),
                                    "momentum_failure": bool(bar.get("momentum_failure_flag", False)),
                                    "pullback_atr": bar.get("pullback_depth_atr", 1.0),
                                    "volume_ratio": bar.get("volume_ratio", 1.0),
                                    "atr": bar.get("atr", bar["close"] * 0.02),
                                }
                                pending_orders.append({
                                    "symbol": sym,
                                    "signal_bar_time": ts,
                                    "signal_info": sig_info,
                                    "planned_stop": planned_stop,
                                    "planned_low": bar.get("low", bar["close"] * 0.98)
                                })
                                curr_open_count += 1
                                curr_cum_risk += self.risk_manager.risk_per_trade_pct
                                if curr_open_count >= self.risk_manager.max_open_positions:
                                    break

            # --- D. RECORD PORTFOLIO EQUITY ---
            curr_equity = cash + sum(pos["remaining_units"] * pos["current_price"] for pos in open_positions.values())
            equity_records.append({"timestamp": ts, "equity": curr_equity, "cash": cash, "open_positions": len(open_positions)})

        equity_df = pd.DataFrame(equity_records).set_index("timestamp")["equity"]
        trade_log_df = pd.DataFrame(completed_trades)

        metrics = calculate_comprehensive_metrics(
            equity_curve=equity_df,
            trade_log=trade_log_df,
            initial_capital=self.initial_capital
        )

        return {
            "metrics": metrics,
            "trade_log": trade_log_df,
            "equity_curve": equity_df,
            "portfolio_history": pd.DataFrame(equity_records)
        }
