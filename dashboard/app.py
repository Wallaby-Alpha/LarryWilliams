"""
Streamlit Web Dashboard for MEXC Larry Williams System.

Sections:
1. Market Regime (BTC Price, 50D SMA, Slope, 30D Return, Status)
2. Leaders (Top 25 Coins ranked by Composite Leadership Score)
3. Active Setups & Ready Signals (State Machine & narrative rationale)
4. Interactive Backtest Explorer (Parameters, Equity Curves, Benchmarks)
5. Feature Analysis (Statistical breakdown by RS, %R depth, exits)
6. Multi-Timeframe Visualizer (Interactive Candlestick + %R + Volume subplots)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Internal modules
from data.data_manager import DataManager
from data.universe import UniverseManager
from indicators.relative_strength import RelativeStrengthEngine
from strategies.williams_signal_engine import WilliamsSignalEngine
from backtest.engine import BacktestEngine
from backtest.benchmarks import BenchmarkSuite
from reports.feature_analysis import FeatureAnalyzer
from scanner.scanner_service import ScannerService

st.set_page_config(page_title="MEXC Williams %R Scanner & Backtester", layout="wide", page_icon="📈")


@st.cache_data
def load_base_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


@st.cache_data
def get_sample_or_cached_data():
    b_1h, u_1h = DataManager.generate_synthetic_universe(num_coins=25, num_days=150, seed=42)
    b_daily = DataManager.resample_ohlcv(b_1h, target_rule="1d")
    u_4h = {s: DataManager.resample_ohlcv(df, target_rule="4h") for s, df in u_1h.items()}
    u_daily = {s: DataManager.resample_ohlcv(df, target_rule="1d") for s, df in u_1h.items()}

    # Compute Relative Strength tables
    closes = {s: df.set_index("timestamp")["close"] for s, df in u_daily.items()}
    b_close = b_daily.set_index("timestamp")["close"]
    rs_engine = RelativeStrengthEngine()
    rs_tables = rs_engine.compute_universe_metrics(closes, b_close)

    return b_daily, b_1h, u_1h, u_4h, u_daily, rs_tables


def main():
    st.title("⚡ MEXC Larry Williams Crypto Scanner & Backtester")
    st.caption("Institutional Relative Strength • Multi-Timeframe Trend • Williams %R Momentum Reversals • Risk-Based Sizing")

    config = load_base_config()
    b_daily, b_1h, u_1h, u_4h, u_daily, rs_tables = get_sample_or_cached_data()

    # Sidebar Navigation
    st.sidebar.header("Navigation")
    section = st.sidebar.radio(
        "Go to Section",
        ["Market Regime & Leaders", "Live Scanner & Signals", "Interactive Backtester", "Feature & Edge Analysis", "Multi-Timeframe Chart Visualizer"]
    )

    # 1. Market Regime & Leaders
    if section == "Market Regime & Leaders":
        st.subheader("1. BTC Market Regime")
        engine = WilliamsSignalEngine(config)
        btc_regime_df = engine.evaluate_btc_regime(b_daily)
        cur_btc = btc_regime_df.iloc[-1]
        is_bull = bool(cur_btc["btc_long_regime"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BTC Price", f"${cur_btc['close']:,.2f}")
        c2.metric("BTC 50-Day SMA", f"${cur_btc['btc_sma_slow']:,.2f}", f"{cur_btc['btc_sma_slope']*100:.2f}% slope")
        c3.metric("BTC 30-Day Return", f"{cur_btc['btc_n_day_ret']*100:.2f}%")
        c4.metric("Regime Filter", "BULLISH / ON" if is_bull else "OFF", delta="Trading Allowed" if is_bull else "Trading Blocked")

        st.markdown("---")
        st.subheader("2. Top 25 Relative Strength Leaders (Composite Score)")
        pctile_7d = rs_tables["pctile_7d"].iloc[-1]
        pctile_30d = rs_tables["pctile_30d"].iloc[-1]
        pctile_90d = rs_tables["pctile_90d"].iloc[-1]
        lead_scores = rs_tables["leadership_score"].iloc[-1]

        leaders_df = pd.DataFrame({
            "Symbol": lead_scores.index,
            "Leadership Score": lead_scores.values.round(1),
            "7D RS Percentile": pctile_7d.values.round(1),
            "30D RS Percentile": pctile_30d.values.round(1),
            "90D RS Percentile": pctile_90d.values.round(1),
        }).sort_values("Leadership Score", ascending=False).reset_index(drop=True)

        st.dataframe(leaders_df.head(25), use_container_width=True)

    # 2. Live Scanner & Signals
    elif section == "Live Scanner & Signals":
        st.subheader("Live Scanner & Setup States")
        st.info("Scanner continuously scans MEXC markets and categorizes coins into: NO SETUP, WATCH, SETUP DEVELOPING, and READY.")
        
        scanner = ScannerService(config)
        scan_res = scanner.scan_once()

        ready = scan_res["ready_signals"]
        st.markdown(f"### 🎯 Ready Signals ({len(ready)})")
        if not ready.empty:
            st.dataframe(ready[["symbol", "price", "stop", "target_2r", "rr_ratio", "williams_r", "pullback_atr", "explanation"]], use_container_width=True)
        else:
            st.warning("No coin currently satisfies 100% of entry confirmation conditions.")

        st.markdown("### 🔍 Active & Developing Setups")
        active = scan_res["active_setups"]
        if not active.empty:
            st.dataframe(active[["symbol", "state", "leadership_score", "williams_r", "pullback_atr", "explanation"]], use_container_width=True)
        else:
            st.write("No developing setups currently in progress.")

    # 3. Interactive Backtester
    elif section == "Interactive Backtester":
        st.subheader("Interactive Strategy Backtester")
        
        # Sidebar Strategy Controls
        st.sidebar.markdown("### Strategy Parameters")
        rs_thresh = st.sidebar.slider("RS 7D & 30D Percentile Threshold", 50, 95, 80, step=5)
        wr_thresh = st.sidebar.slider("Williams %R Oversold Threshold", -95, -70, -80, step=5)
        exit_model = st.sidebar.selectbox("Exit Model", ["1R", "1.5R", "2R", "3R", "trailing_stop", "partial_2r_trail", "williams_overbought_reversal"])
        risk_pct = st.sidebar.select_slider("Risk Per Trade", options=[0.0025, 0.005, 0.0075, 0.01, 0.015], value=0.005, format_func=lambda x: f"{x*100:.2f}%")
        max_pos = st.sidebar.slider("Max Open Positions", 1, 15, 5)

        run_btn = st.sidebar.button("Run Backtest", type="primary")

        if run_btn:
            bt_cfg = copy.deepcopy(config)
            bt_cfg["relative_strength"]["rs_7d_min_percentile"] = float(rs_thresh)
            bt_cfg["relative_strength"]["rs_30d_min_percentile"] = float(rs_thresh)
            bt_cfg["williams_r"]["oversold_threshold"] = float(wr_thresh)
            bt_cfg["exit_strategy"]["model"] = exit_model
            bt_cfg["risk_and_sizing"]["risk_per_trade_pct"] = float(risk_pct)
            bt_cfg["portfolio_constraints"]["max_open_positions"] = int(max_pos)

            with st.spinner("Running event-driven multi-asset backtest..."):
                engine = BacktestEngine(bt_cfg)
                res = engine.run_backtest(b_daily, b_1h, u_1h, u_4h, u_daily, rs_tables)

            metrics = res["metrics"]
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Net Return", f"{metrics.get('net_return_pct', 0):.2f}%")
            m2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            m3.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
            m4.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.1f}%")
            m5.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
            m6.metric("Total Trades", f"{metrics.get('total_trades', 0)}")

            # Equity Curve Chart
            eq_curve = res["equity_curve"]
            bench = BenchmarkSuite()
            btc_eq = bench.buy_and_hold_asset(b_1h, "BTC")["equity_BTC"]
            alt_eq = bench.equal_weight_altcoin_basket(u_1h)["equity_altcoin_basket"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq_curve.index, y=eq_curve.values, mode="lines", name="Williams System (Net)", line=dict(color="#00FFAA", width=2.5)))
            fig.add_trace(go.Scatter(x=btc_eq.index, y=btc_eq.values, mode="lines", name="Buy & Hold BTC", line=dict(color="#FF9900", width=1.5, dash="dot")))
            fig.add_trace(go.Scatter(x=alt_eq.index, y=alt_eq.values, mode="lines", name="Equal-Weight Alt Basket", line=dict(color="#8888FF", width=1.5, dash="dash")))

            fig.update_layout(title="Portfolio Equity Curve vs Benchmarks", template="plotly_dark", height=450, yaxis_title="Portfolio Equity ($)")
            st.plotly_chart(fig, use_container_width=True)

            # Trade Log
            st.subheader("Trade Log")
            st.dataframe(res["trade_log"], use_container_width=True)

    # 4. Feature & Edge Analysis
    elif section == "Feature & Edge Analysis":
        st.subheader("Strategy Feature & Performance Attribution")
        st.write("Evaluating which specific rules contribute genuine incremental edge:")

        engine = BacktestEngine(config)
        res = engine.run_backtest(b_daily, b_1h, u_1h, u_4h, u_daily, rs_tables)
        analyzer = FeatureAnalyzer(res["trade_log"])
        feat_res = analyzer.generate_full_analysis()

        if feat_res.get("status") == "SUCCESS":
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Performance by 7D Relative Strength Decile")
                st.table(pd.DataFrame(feat_res["rs_7d_performance"]))

                st.markdown("#### Performance by Williams %R Depth")
                st.table(pd.DataFrame(feat_res["williams_depth_performance"]))

            with c2:
                st.markdown("#### Performance by Pullback ATR Depth")
                st.table(pd.DataFrame(feat_res["pullback_atr_performance"]))

                st.markdown("#### Performance by Exit Reason")
                st.table(pd.DataFrame(feat_res["exit_reason_performance"]))
        else:
            st.info("Run a backtest with trades to populate feature statistics.")

    # 5. Multi-Timeframe Visualizer
    elif section == "Multi-Timeframe Chart Visualizer":
        st.subheader("Multi-Timeframe Candle & Williams %R Inspector")
        symbol = st.selectbox("Select Coin", list(u_1h.keys()))
        df1 = u_1h[symbol].tail(120).copy()

        # Compute Williams %R and MAs
        from indicators.technical import williams_r, ema, sma
        df1["williams_r"] = williams_r(df1["high"], df1["low"], df1["close"], period=14)
        df1["ema20"] = ema(df1["close"], 20)
        df1["sma50"] = sma(df1["close"], 50)

        # Candlestick chart + %R subplot + Volume subplot
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            row_heights=[0.55, 0.25, 0.20],
            subplot_titles=[f"{symbol} 1H Candles & MAs", "Williams %R(14)", "Volume"]
        )

        fig.add_trace(go.Candlestick(
            x=df1["timestamp"], open=df1["open"], high=df1["high"], low=df1["low"], close=df1["close"], name="1H OHLC"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=df1["ema20"], line=dict(color="#FFCC00", width=1.2), name="1H EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=df1["sma50"], line=dict(color="#3399FF", width=1.2), name="1H SMA 50"), row=1, col=1)

        # %R trace + oversold lines
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=df1["williams_r"], line=dict(color="#FF00FF", width=1.5), name="Williams %R"), row=2, col=1)
        fig.add_hline(y=-80, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-20, line_dash="dash", line_color="green", row=2, col=1)

        # Volume
        fig.add_trace(go.Bar(x=df1["timestamp"], y=df1["volume"], marker_color="#555555", name="Volume"), row=3, col=1)

        fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    import copy
    main()
