# MEXC Larry Williams Trend & Relative Strength Pullback System

An institutional-grade Python crypto trading scanner and historical backtesting system using the MEXC Spot API and multi-timeframe OHLCV data.

The system adapts Larry Williams' momentum & pullback principles to crypto markets:
$$\text{Market Regime} \longrightarrow \text{Leading Altcoins} \longrightarrow \text{Confirm Trend} \longrightarrow \text{Wait for Pullback} \longrightarrow \text{Williams \%R Reversal} \longrightarrow \text{Price/Volume Confirmation} \longrightarrow \text{Risk-Based Sizing} \longrightarrow \text{Mechanical Exits}$$

---

## 1. System Architecture

```
/mexc-williams-system
├── config/
│   └── default_config.yaml         # Fully configurable parameter schema
├── data/
│   ├── mexc_client.py              # MEXC Spot REST API v3 client with rate-limiting & pagination
│   ├── universe.py                 # Dynamic universe selection (USDT pairs, volume filters, survivorship bias)
│   ├── validator.py                # Data quality validation layer (gaps, spikes, duplicates)
│   └── data_manager.py             # Parquet/CSV caching, multi-timeframe resampling, synthetic data generator
├── indicators/
│   ├── technical.py                # Exact Williams %R, SMA, EMA, ATR, Slopes, Extension
│   ├── relative_strength.py        # 7D, 30D, 90D excess return & cross-sectional percentile rankings
│   └── pullback.py                 # Pullback depth (ATR/%), swing high, pullback relative strength
├── strategies/
│   └── williams_signal_engine.py   # UNIFIED signal engine shared by backtester and live scanner
├── execution/
│   ├── risk_manager.py             # Position sizing (risk % / stop distance) & portfolio constraints
│   └── simulator.py                # Realistic execution, fees, slippage, collision resolution
├── backtest/
│   ├── engine.py                   # Event-driven chronological multi-asset portfolio backtester
│   ├── metrics.py                  # Institutional metrics (Sharpe, Sortino, Calmar, MAE/MFE, Expectancy)
│   ├── walk_forward.py             # In-sample vs Out-of-sample and rolling walk-forward optimizer
│   ├── overfitting.py              # Parameter cliff and fragility detector
│   └── benchmarks.py               # Buy & Hold BTC, Altcoin Basket, and Monte Carlo Randomized Entry
├── reports/
│   └── feature_analysis.py         # Feature attribution (RS deciles, %R depth, exit reason breakdown)
├── scanner/
│   ├── scanner_service.py          # Live MEXC scanner with multi-state classification
│   └── run_scanner.py              # CLI scanner runner (one-shot or continuous hourly loop)
├── dashboard/
│   └── app.py                      # Interactive Streamlit multi-page web dashboard
├── tests/                          # Comprehensive unit tests with synthetic datasets
│   ├── test_indicators.py
│   ├── test_strategy_signals.py
│   ├── test_execution_risk.py
│   └── test_lookahead_bias.py
└── run_research_study.py           # Master research script running the full quantitative study
```

---

## 2. Core Strategy Logic

### 1. BTC Market Regime Filter
- **Condition:** $\text{BTC Close} > \text{BTC 50D SMA} \quad \text{AND} \quad \text{SMA 50 is rising} \quad \text{AND} \quad \text{BTC 30D Return} > 0\%$
- Configurable parameters: `fast_sma` (20), `slow_sma` (50, 100, 200), `slope_lookback`, `min_return_pct`.

### 2. Relative Strength Leadership
- Calculates 7D, 30D, and 90D coin returns vs BTC return:
  $$\text{Excess Return} = \text{Coin Return} - \text{BTC Return}$$
- Cross-sectional percentile ranking across the eligible liquid universe:
  - $\text{RS\_7D\_PERCENTILE} \ge 80.0$
  - $\text{RS\_30D\_PERCENTILE} \ge 80.0$
- Composite Leadership Score:
  $$\text{LEADERSHIP\_SCORE} = 0.40 \times \text{RS}_{30D} + 0.35 \times \text{RS}_{7D} + 0.25 \times \text{RS}_{90D}$$

### 3. Individual Coin Trend Filter
- **Daily:** Price > 50 SMA, 20 SMA > 50 SMA, 50 SMA is rising.
- **4-Hour:** Price > 50 SMA, 50 SMA is rising.
- **Extension Filter:** Avoid buying parabolic coins; reject if Price > 20D EMA by more than 15%.

### 4. Pullback Detection (1-Hour)
- **Model A (Default):** Price falls $\ge 1.0\text{ ATR}$ from recent 24-bar swing high.
- **Model B:** Price retraces 2% to 8%.
- **Model C:** Price retraces between 0.5 and 1.8 ATR.
- **Model D:** Price falls below 1H EMA 20 but holds above 1H SMA 50.

### 5. Williams %R Reversal
- 14-period Williams %R on 1H:
  $$\%R = \frac{\text{Highest High}_{14} - \text{Close}}{\text{Highest High}_{14} - \text{Lowest Low}_{14}} \times -100$$
- Condition: %R previously reached oversold ($< -80$), then crosses back **ABOVE** $-80$.
- Optional Momentum Failure divergence detection: 1st pullback reaches $\le -90$, 2nd pullback higher low in %R while price tests low.

### 6. Price Action Entry Confirmation
- **Default:** Current 1H candle closes above previous candle's high ($\text{Close}_t > \text{High}_{t-1}$).
- Alternatives: Break above pullback swing high, Close above 1H EMA 20, Bullish engulfing.

### 7. Risk Management & Position Sizing
- **Risk Per Trade:** Default 0.5% of account equity.
- **Stop Loss:** Pullback swing low, or Entry $- 1.5 \times \text{ATR}$, or Structural $+$ ATR buffer.
- **Position Sizing:**
  $$\text{Units} = \frac{\text{Equity} \times \text{Risk \%}}{\text{Entry} - \text{Stop}}$$
- **Portfolio Constraints:** Max 5 open positions, Max 2.5% total portfolio risk, Max 20% equity cap per coin.

### 8. Modular Exit Models
- Fixed 1R, 1.5R, 2R, 3R
- Trailing Stop ($1.5 \times \text{ATR}$ from highest high)
- Williams %R Overbought Reversal (%R reaches $> -20$ then crosses back below)
- Partial Exit: 50% closed at 2R, remainder trailed with breakeven stop
- Time Stop: Exit if trade has not reached $+0.5R$ within 48 hours.

---

## 3. Telegram Dual-Tier Alert System

The scanner sends two distinct tiers of alerts directly to your phone via Telegram:

1. **Watchlist / Developing Setup (Heads-Up Alert):**
   - Triggered when a top-decile leader (`RS >= 80th percentile`) is in a daily/4H uptrend and begins a $\ge 1.0\text{ ATR}$ pullback with Williams %R dropping into oversold ($< -80$).
   - Gives you early notice of coins setting up *before* the entry triggers.

2. **Trade Entry Signal (Execution Alert):**
   - Triggered immediately when the 1H candle closes with Williams %R crossing back above $-80$ and price action confirming upward momentum.
   - Includes **Exact Entry Price**, **Stop Loss**, **Take Profit Targets (1.5R, 2.0R, 3.0R)**, **Risk/Reward ratio**, and **Position Sizing** tailored to your account size.

3. **BTC Regime Alerts:**
   - Immediate notification when BTC crosses its 50-day SMA or slope turns negative, flagging whether altcoin long entries are active or paused.

### How to Configure Telegram:
1. Open Telegram and search for `@BotFather`. Send `/newbot`, choose a name and username, and copy the API Token.
2. Search for `@userinfobot` on Telegram, click `/start`, and copy your numeric ID.
3. Copy `.env.example` to `.env` and fill in:
   ```env
   TELEGRAM_BOT_TOKEN=your_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ACCOUNT_EQUITY=10000.0
   ```
4. Test notifications instantly:
   ```bash
   python alerts/test_telegram.py
   ```

---

## 4. DigitalOcean Droplet 24/7 Deployment

### Option A: 1-Command Automated Script
1. Create an Ubuntu 22.04 / 24.04 Droplet on DigitalOcean ($4 to $6/mo plan).
2. SSH into your droplet:
   ```bash
   ssh root@<DROPLET_IP>
   ```
3. Clone your GitHub repository and run the deploy script:
   ```bash
   git clone https://github.com/Wallaby-Alpha/LarryWilliams.git
   cd LarryWilliams
   chmod +x deploy.sh
   ./deploy.sh
   ```
4. Put your Telegram bot token in `.env`:
   ```bash
   nano .env
   sudo systemctl restart mexc-scanner.service
   ```
5. Check live logs:
   ```bash
   sudo journalctl -u mexc-scanner.service -f
   ```

### Option B: Docker Compose
```bash
docker compose up -d
docker compose logs -f mexc-scanner
```

---

## 5. Local Execution & Testing

### Run Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Run Comprehensive Research Study
```bash
python run_research_study.py
```

### Run Live MEXC Scanner (One-Shot or Hourly Loop)
```bash
# One-shot scan
python scanner/run_scanner.py --once

# Continuous 24/7 hourly loop
python scanner/run_scanner.py --loop --interval 3600
```

### Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
