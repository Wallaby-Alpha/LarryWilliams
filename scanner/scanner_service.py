"""
Live Scanner Service for MEXC Crypto Markets.

Runs scanning workflows using the UNIFIED WilliamsSignalEngine:
1. Fetches current market data via MEXC REST API
2. Evaluates BTC market regime
3. Filters liquid universe (>= $5M 24h volume)
4. Computes cross-sectional relative strength percentiles and leadership score
5. Classifies state for each coin: NO SETUP, WATCH, SETUP DEVELOPING, READY, TRIGGERED
6. Formats clean table and rich narrative explanation for traders.
"""

import time
import logging
from typing import Dict, List, Any, Optional
import pandas as pd

from data.mexc_client import MEXCClient
from data.universe import UniverseManager
from data.data_manager import DataManager
from indicators.relative_strength import RelativeStrengthEngine
from strategies.williams_signal_engine import WilliamsSignalEngine
from execution.risk_manager import RiskManager
from alerts.telegram_notifier import TelegramNotifier

logger = logging.getLogger("ScannerService")


class ScannerService:
    """
    Live scanning service for crypto relative strength & Williams %R setups.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = MEXCClient()
        self.universe_mgr = UniverseManager(
            quote_asset=config.get("universe", {}).get("quote_asset", "USDT"),
            min_24h_quote_volume_usd=float(config.get("universe", {}).get("min_24h_quote_volume_usd", 5000000.0)),
            exclude_coins=config.get("universe", {}).get("exclude_coins", ["BTCUSDT"])
        )
        self.rs_engine = RelativeStrengthEngine(config.get("relative_strength", {}).get("composite_weights"))
        self.signal_engine = WilliamsSignalEngine(config)
        self.risk_manager = RiskManager(config)
        self.notifier = TelegramNotifier()

    def scan_once(self, max_candidates: int = 40) -> Dict[str, Any]:
        """
        Executes a single scan iteration across live MEXC market.
        Falls back to local/cached data gracefully if network is unavailable.
        """
        scan_time = pd.Timestamp.utcnow()
        logger.info(f"Starting MEXC scan at {scan_time}...")

        # 1. Fetch BTC Daily and 1H data
        btc_daily = self.client.get_historical_klines_paginated("BTCUSDT", interval="1d", total_bars=100)
        btc_1h = self.client.get_klines("BTCUSDT", interval="1h", limit=100)

        # Fallback to synthetic if empty (e.g. offline/testing environment)
        using_synthetic = False
        if btc_daily.empty:
            logger.warning("Live MEXC API returned empty data; utilizing synthetic universe for scan.")
            using_synthetic = True
            b_1h_syn, u_syn = DataManager.generate_synthetic_universe(num_coins=25, num_days=120)
            btc_1h = b_1h_syn.tail(100)
            btc_daily = DataManager.resample_ohlcv(b_1h_syn, target_rule="1d")
            candidates = list(u_syn.keys())
            coin_1h_map = {k: v.tail(100) for k, v in u_syn.items()}
        else:
            # 2. Get active exchange symbols and tickers
            exchange_symbols = self.client.get_exchange_info()
            tickers_24h = self.client.get_24hr_tickers()
            candidates = self.universe_mgr.filter_active_symbols(exchange_symbols, tickers_24h)
            candidates = candidates[:max_candidates]
            coin_1h_map = {}
            coin_daily_map = {}
            coin_4h_map = {}
            coin_daily_closes = {}

            for sym in candidates:
                # 1d klines for RS and daily trend (100 bars)
                kd = self.client.get_klines(sym, interval="1d", limit=100)
                if not kd.empty and len(kd) >= 14:
                    coin_daily_map[sym] = kd
                    coin_daily_closes[sym] = kd.set_index("timestamp")["close"]
                
                # 4h klines for 50-period 4H SMA and slope (needs >= 50 bars)
                k4h = self.client.get_klines(sym, interval="4h", limit=100)
                if not k4h.empty and len(k4h) >= 50:
                    coin_4h_map[sym] = k4h
                else:
                    coin_4h_map[sym] = DataManager.resample_ohlcv(k1h, target_rule="4h") if "k1h" in locals() and not k1h.empty else pd.DataFrame()

                # 1h klines for setups & pullbacks
                k1h = self.client.get_klines(sym, interval="1h", limit=150)
                if not k1h.empty and len(k1h) >= 30:
                    coin_1h_map[sym] = k1h

        # 3. Evaluate BTC Regime
        btc_regime_df = self.signal_engine.evaluate_btc_regime(btc_daily)
        btc_status_row = btc_regime_df.iloc[-1]
        btc_long_regime = bool(btc_status_row["btc_long_regime"])
        btc_regime_info = {
            "price": float(btc_status_row["close"]),
            "sma_50": float(btc_status_row["btc_sma_slow"]),
            "sma_slope": float(btc_status_row["btc_sma_slope"]),
            "return_30d_pct": float(btc_status_row["btc_n_day_ret"] * 100.0),
            "regime_active": btc_long_regime,
        }

        btc_daily_close = btc_daily.set_index("timestamp")["close"]
        rs_results = self.rs_engine.compute_universe_metrics(coin_daily_closes, btc_daily_close)

        # 5. Evaluate Signals for Each Coin
        signal_results = []
        leaders_list = []

        pctile_7d_df = rs_results["pctile_7d"]
        pctile_30d_df = rs_results["pctile_30d"]
        pctile_90d_df = rs_results["pctile_90d"]
        leadership_df = rs_results["leadership_score"]

        for sym in coin_1h_map.keys():
            df_1h = coin_1h_map[sym]
            df_4h = coin_4h_map.get(sym, pd.DataFrame())
            df_daily = coin_daily_map.get(sym, pd.DataFrame())

            # Current RS metrics
            r7 = float(pctile_7d_df[sym].iloc[-1]) if sym in pctile_7d_df else 50.0
            r30 = float(pctile_30d_df[sym].iloc[-1]) if sym in pctile_30d_df else 50.0
            r90 = float(pctile_90d_df[sym].iloc[-1]) if sym in pctile_90d_df else 50.0
            lead_score = float(leadership_df[sym].iloc[-1]) if sym in leadership_df else 50.0

            cur_close = df_1h["close"].iloc[-1]
            leaders_list.append({
                "symbol": sym,
                "leadership_score": lead_score,
                "rs_7d": round(r7, 1),
                "rs_30d": round(r30, 1),
                "rs_90d": round(r90, 1),
                "price": cur_close
            })

            # Multi-timeframe feature merge
            rs_metrics = {
                "rs_7d_pctile": r7,
                "rs_30d_pctile": r30,
                "rs_90d_pctile": r90,
                "leadership_score": lead_score
            }

            feat_df = self.signal_engine.build_coin_feature_table(df_1h, df_4h, df_daily, rs_metrics)
            if len(feat_df) < 2:
                continue

            curr_row = feat_df.iloc[-1]
            prev_row = feat_df.iloc[-2]

            sig = self.signal_engine.evaluate_bar(curr_row, prev_row, btc_regime=btc_long_regime)

            # Planned stop & risk calculation
            planned_stop = self.risk_manager.calculate_stop_loss(
                curr_row["close"], sig.get("swing_low", curr_row["low"]), sig.get("atr", curr_row["close"] * 0.02)
            )
            r_dist = curr_row["close"] - planned_stop
            target_2r = curr_row["close"] + (r_dist * 2.0)
            rr_ratio = round(abs(target_2r - curr_row["close"]) / abs(curr_row["close"] - planned_stop), 2) if r_dist > 0 else 0.0

            sig_state = sig.get("state", "NO_SETUP")

            # Fire Telegram Notifications (handles anti-spam automatically)
            candle_ts = curr_row.get("timestamp", scan_time)
            if sig_state == "READY":
                self.notifier.send_trade_entry_alert(
                    symbol=sym,
                    price=float(curr_row["close"]),
                    stop_price=planned_stop,
                    candle_timestamp=candle_ts,
                    signal_info={
                        "leadership_score": lead_score,
                        "rs_7d": r7,
                        "rs_30d": r30,
                        "wr_curr": round(float(curr_row.get("williams_r", -50.0)), 1),
                        "pullback_atr": round(float(curr_row.get("pullback_depth_atr", 0.0)), 2),
                        "explanation": sig.get("reason", "")
                    }
                )
            elif sig_state == "SETUP_DEVELOPING":
                self.notifier.send_watchlist_alert(
                    symbol=sym,
                    price=float(curr_row["close"]),
                    candle_timestamp=candle_ts,
                    signal_info={
                        "leadership_score": lead_score,
                        "rs_7d": r7,
                        "rs_30d": r30,
                        "pullback_atr": round(float(curr_row.get("pullback_depth_atr", 0.0)), 2),
                        "williams_r": round(float(curr_row.get("williams_r", -50.0)), 1)
                    }
                )

            signal_results.append({
                "symbol": sym,
                "state": sig_state,
                "is_ready": sig.get("is_ready", False),
                "leadership_score": lead_score,
                "rs_7d": round(r7, 1),
                "rs_30d": round(r30, 1),
                "rs_90d": round(r90, 1),
                "daily_trend": bool(curr_row.get("daily_uptrend", False)),
                "four_h_trend": bool(curr_row.get("four_hour_uptrend", False)),
                "ext_from_ema20": float(curr_row.get("ext_from_ema20", 0.0)),
                "williams_r": round(float(curr_row.get("williams_r", -50.0)), 1),
                "pullback_atr": round(float(curr_row.get("pullback_depth_atr", 0.0)), 2),
                "volume_ratio": round(float(curr_row.get("volume_ratio", 1.0)), 2),
                "price": round(curr_row["close"], 4),
                "stop": planned_stop,
                "target_2r": round(target_2r, 4),
                "rr_ratio": rr_ratio,
                "explanation": sig.get("reason", "")
            })

        # Sort leaders
        leaders_df = pd.DataFrame(leaders_list).sort_values("leadership_score", ascending=False).reset_index(drop=True)
        signals_df = pd.DataFrame(signal_results).sort_values("leadership_score", ascending=False).reset_index(drop=True)

        # Diagnose Almost-Tradeable / Pipeline Candidates
        almost_tradeable = []
        for s in signal_results:
            if s["is_ready"]:
                continue
            max_ext = float(self.config.get("extension_filter", {}).get("max_extension_pct", 0.20))
            if s["leadership_score"] >= 60 or s["rs_7d"] >= 60 or s["rs_30d"] >= 70:
                missing = ""
                curr_status = ""
                if not s["daily_trend"] or not s["four_h_trend"]:
                    missing = "Daily/4H Moving Average Trend Alignment"
                    curr_status = "High RS but moving average structure not yet bullish"
                elif s.get("ext_from_ema20", 0.0) > max_ext:
                    missing = f"Extended (+{s.get('ext_from_ema20', 0.0)*100:.1f}% > {int(max_ext*100)}% max from 20D EMA)"
                    curr_status = "Parabolic move; waiting for consolidation"
                elif s["pullback_atr"] < 1.0:
                    missing = f"Pullback Depth ({s['pullback_atr']:.2f} ATR / 1.0 ATR required)"
                    curr_status = "Trending up strongly; no pullback yet"
                elif s["williams_r"] <= -80:
                    missing = f"Williams %R recovery ({s['williams_r']:.1f} <= -80)"
                    curr_status = "Oversold dip active; waiting for cross back above -80"
                elif s["williams_r"] > -80:
                    missing = f"Williams %R dip ({s['williams_r']:.1f} > -80)"
                    curr_status = "Pullback started but momentum not yet oversold"
                else:
                    missing = "Price Action confirmation candle"
                    curr_status = "Waiting for candle close above previous high"

                almost_tradeable.append({
                    "symbol": s["symbol"],
                    "leadership_score": s["leadership_score"],
                    "rs_7d": s["rs_7d"],
                    "price": s["price"],
                    "missing_condition": missing,
                    "current_status": curr_status
                })

        almost_tradeable = sorted(almost_tradeable, key=lambda x: x["leadership_score"], reverse=True)

        return {
            "timestamp": str(scan_time),
            "btc_regime": btc_regime_info,
            "using_synthetic_data": using_synthetic,
            "leaders": leaders_df.head(25),
            "signals": signals_df,
            "ready_signals": signals_df[signals_df["state"] == "READY"],
            "active_setups": signals_df[signals_df["state"].isin(["READY", "SETUP_DEVELOPING"])],
            "almost_tradeable": almost_tradeable,
        }
