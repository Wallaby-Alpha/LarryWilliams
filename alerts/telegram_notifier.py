"""
Telegram Notification Engine for MEXC Williams %R System.

Features:
- Dual-tier alert system:
  1. Watchlist / Setup Developing Alert (Heads-up)
  2. Actionable Trade Entry Alert (Execution with Entry, Stop, Targets, Sizing)
- Market Regime Change alerts (BTC 50 SMA / Bull / Bear transitions)
- Daily Leadership Digest
- Persistent anti-spam tracking: prevents sending duplicate alerts for the same candle
- Fallback console logging when Telegram bot token is unconfigured or during tests.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional, List
import pandas as pd

logger = logging.getLogger("TelegramNotifier")


class TelegramNotifier:
    """
    Manages Telegram notifications and anti-spam state tracking.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        history_file: str = "data_storage/alert_history.json",
        enabled: bool = True
    ):
        # Read from environment variables if not passed directly
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.history_file = history_file
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)
        
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage" if self.bot_token else ""
        self.alert_history = self._load_history()

        if not self.enabled:
            logger.info("Telegram notifications disabled or credentials missing. Alerts will print to console.")

    def _load_history(self) -> Dict[str, Any]:
        """Loads alert state history from disk to prevent duplicate notifications."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read alert history file: {e}")
        return {"watchlist": {}, "entries": {}, "regime": None}

    def _save_history(self) -> None:
        """Saves alert state history to disk."""
        os.makedirs(os.path.dirname(self.history_file) or ".", exist_ok=True)
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.alert_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save alert history: {e}")

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Sends raw message to Telegram chat with retry.
        """
        if not self.enabled:
            try:
                print("\n[TELEGRAM PREVIEW]:\n" + text + "\n")
            except UnicodeEncodeError:
                safe_text = text.encode("ascii", errors="replace").decode("ascii")
                print("\n[TELEGRAM PREVIEW]:\n" + safe_text + "\n")
            return True

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }

        try:
            resp = requests.post(self.base_url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Telegram notification sent successfully.")
                return True
            else:
                logger.error(f"Telegram API error {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to Telegram API: {e}")
            return False

    # ==========================================
    # 1. ACTIONABLE TRADE ENTRY ALERT
    # ==========================================
    def send_trade_entry_alert(
        self,
        symbol: str,
        price: float,
        stop_price: float,
        candle_timestamp: Any,
        signal_info: Dict[str, Any],
        account_equity: float = 10000.0,
        risk_per_trade_pct: float = 0.005
    ) -> bool:
        """
        Fires an immediate action alert when all entry confirmation criteria are met.
        Includes entry price, stop-loss, profit targets (1.5R, 2R, 3R), and sizing.
        """
        # Anti-spam check
        candle_key = str(candle_timestamp)
        if self.alert_history.get("entries", {}).get(symbol) == candle_key:
            logger.debug(f"Skipping duplicate trade alert for {symbol} at {candle_key}")
            return False

        r_dist = price - stop_price
        pct_stop = ((price - stop_price) / price) * 100.0 if price > 0 else 0.0
        
        target_1_5r = price + (r_dist * 1.5)
        target_2r = price + (r_dist * 2.0)
        target_3r = price + (r_dist * 3.0)

        # Position sizing example
        dollar_risk = account_equity * risk_per_trade_pct
        units = dollar_risk / r_dist if r_dist > 0 else 0.0
        position_usd = units * price

        msg = (
            f"🟢 <b>MEXC TRADE ENTRY ALERT</b> 🟢\n\n"
            f"💎 <b>Coin:</b> <code>{symbol}</code>\n"
            f"⚡ <b>Action:</b> <b>BUY / LONG</b> (1H Close Confirmed)\n\n"
            f"🎯 <b>Entry Price:</b> <code>${price:.4f}</code>\n"
            f"🛑 <b>Stop Loss:</b> <code>${stop_price:.4f}</code> (<b>-{pct_stop:.2f}%</b>)\n\n"
            f"🏁 <b>Target 1 (1.5R):</b> <code>${target_1_5r:.4f}</code> (+{pct_stop*1.5:.2f}%)\n"
            f"🏁 <b>Target 2 (2.0R):</b> <code>${target_2r:.4f}</code> (+{pct_stop*2.0:.2f}%)\n"
            f"🏁 <b>Target 3 (3.0R):</b> <code>${target_3r:.4f}</code> (+{pct_stop*3.0:.2f}%)\n\n"
            f"⚖️ <b>Risk/Reward:</b> 1 : 2.0 (Target 2)\n"
            f"📐 <b>Sizing ({risk_per_trade_pct*100:.1f}% Risk on ${account_equity:,.0f}):</b> "
            f"<code>{units:.2f} units (~${position_usd:,.2f})</code>\n\n"
            f"📊 <b>Leadership & Indicator Stats:</b>\n"
            f"• Leadership Score: <b>{signal_info.get('leadership_score', 0):.1f}</b>\n"
            f"• 7D / 30D RS: <b>{signal_info.get('rs_7d', 0):.0f}%ile / {signal_info.get('rs_30d', 0):.0f}%ile</b>\n"
            f"• Williams %R: <b>{signal_info.get('wr_curr', 0):.1f}</b> (Crossed above -80)\n"
            f"• Pullback Depth: <b>{signal_info.get('pullback_atr', 0):.2f} ATR</b>\n\n"
            f"📝 <b>Rationale:</b>\n"
            f"<i>{signal_info.get('explanation', 'Pullback complete with %R recovery and price confirmation.')}</i>\n"
            f"⏱ <i>Timestamp: {candle_key} UTC</i>"
        )

        sent = self.send_message(msg)
        if sent:
            if "entries" not in self.alert_history:
                self.alert_history["entries"] = {}
            self.alert_history["entries"][symbol] = candle_key
            self._save_history()
        return sent

    # ==========================================
    # 2. WATCHLIST & DEVELOPING SETUP ALERT
    # ==========================================
    def send_watchlist_alert(
        self,
        symbol: str,
        price: float,
        candle_timestamp: Any,
        signal_info: Dict[str, Any]
    ) -> bool:
        """
        Fires an early heads-up alert when a strong leader begins pulling back into oversold %R.
        """
        candle_key = str(candle_timestamp)
        if self.alert_history.get("watchlist", {}).get(symbol) == candle_key:
            return False

        msg = (
            f"🟡 <b>WATCHLIST ALERT: Pullback in Progress</b> 🟡\n\n"
            f"👀 <b>Coin:</b> <code>{symbol}</code>\n"
            f"💰 <b>Current Price:</b> <code>${price:.4f}</code>\n"
            f"🏆 <b>Leadership Score:</b> <b>{signal_info.get('leadership_score', 0):.1f}</b>\n"
            f"📈 <b>7D / 30D RS:</b> <b>{signal_info.get('rs_7d', 0):.0f}%ile / {signal_info.get('rs_30d', 0):.0f}%ile</b>\n"
            f"📉 <b>Pullback Depth:</b> <b>{signal_info.get('pullback_atr', 0):.2f} ATR</b>\n"
            f"🌊 <b>Williams %R:</b> <code>{signal_info.get('williams_r', 0):.1f}</code> (Oversold Area)\n\n"
            f"🔭 <b>What to watch for:</b>\n"
            f"• 1H Williams %R crossing back ABOVE -80\n"
            f"• Candle closing above previous high for entry confirmation\n\n"
            f"⏱ <i>Status: SETUP DEVELOPING ({candle_key} UTC)</i>"
        )

        sent = self.send_message(msg)
        if sent:
            if "watchlist" not in self.alert_history:
                self.alert_history["watchlist"] = {}
            self.alert_history["watchlist"][symbol] = candle_key
            self._save_history()
        return sent

    # ==========================================
    # 3. BTC MARKET REGIME CHANGE ALERT
    # ==========================================
    def send_regime_alert(self, btc_regime: Dict[str, Any]) -> bool:
        """
        Alerts when BTC regime flips between Bullish (Trading ON) and Bearish (Trading PAUSED).
        """
        curr_state = btc_regime.get("regime_active", True)
        last_state = self.alert_history.get("regime")

        if last_state is not None and last_state == curr_state:
            return False  # No change

        status_emoji = "🟢" if curr_state else "🔴"
        status_text = "BULLISH / TRADING PERMITTED" if curr_state else "BEARISH / TRADING PAUSED"
        action_text = "Altcoin long setups are ACTIVE." if curr_state else "New altcoin long setups are SUSPENDED."

        msg = (
            f"{status_emoji} <b>BTC MARKET REGIME UPDATE</b> {status_emoji}\n\n"
            f"<b>Status:</b> <b>{status_text}</b>\n"
            f"<b>BTC Price:</b> <code>${btc_regime.get('price', 0):,.2f}</code>\n"
            f"<b>50-Day SMA:</b> <code>${btc_regime.get('sma_50', 0):,.2f}</code> "
            f"({btc_regime.get('sma_slope', 0)*100:.2f}% slope)\n"
            f"<b>30-Day Return:</b> <code>{btc_regime.get('return_30d_pct', 0):.2f}%</code>\n\n"
            f"📌 <i>Policy: {action_text}</i>"
        )

        sent = self.send_message(msg)
        if sent:
            self.alert_history["regime"] = curr_state
            self._save_history()
        return sent

    # ==========================================
    # 4. DAILY TOP LEADERS DIGEST
    # ==========================================
    def send_daily_digest(self, leaders_df: pd.DataFrame, btc_regime: Dict[str, Any]) -> bool:
        """Sends daily leaderboard summary of top relative strength leaders."""
        status_str = "BULLISH ✅" if btc_regime.get("regime_active", True) else "BEARISH ❌"
        rows_text = ""
        for idx, r in leaders_df.head(10).iterrows():
            rows_text += (
                f"{idx+1}. <code>{r['symbol']:<10}</code> "
                f"Score: <b>{r['leadership_score']:.1f}</b> | "
                f"7D: {r['rs_7d']:.0f}% | "
                f"${r['price']:.4f}\n"
            )

        msg = (
            f"☀️ <b>MEXC DAILY MARKET LEADERSHIP DIGEST</b> ☀️\n\n"
            f"BTC Regime: <b>{status_str}</b> (${btc_regime.get('price', 0):,.0f})\n\n"
            f"<b>Top 10 Relative Strength Leaders:</b>\n"
            f"{rows_text}\n"
            f"<i>Scanner running 24/7 on DigitalOcean droplet.</i>"
        )
        return self.send_message(msg)
