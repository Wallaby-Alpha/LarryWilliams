"""
Test Utility for Telegram Alert Engine.

Usage:
    # Using environment variables TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    python alerts/test_telegram.py

    # Or passing token and chat_id via CLI:
    python alerts/test_telegram.py --token <BOT_TOKEN> --chat-id <CHAT_ID>
"""

import os
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import pandas as pd

try:
    from dotenv import load_dotenv
    for p in [".env", "../.env", "/opt/mexc-williams-system/.env", os.path.expanduser("~/LarryWilliams/.env")]:
        if os.path.exists(p):
            load_dotenv(p)
            break
except ImportError:
    pass

from alerts.telegram_notifier import TelegramNotifier


def main():
    parser = argparse.ArgumentParser(description="Test Telegram Notifier for MEXC Williams Scanner")
    parser.add_argument("--token", help="Telegram Bot Token")
    parser.add_argument("--chat-id", help="Telegram Chat ID")
    args = parser.parse_args()

    token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = args.chat_id or os.getenv("TELEGRAM_CHAT_ID")

    notifier = TelegramNotifier(bot_token=token, chat_id=chat_id)

    print("=" * 70)
    print(" TESTING TELEGRAM NOTIFICATIONS ")
    print(f" Token configured: {'YES' if bool(notifier.bot_token) else 'NO (Console Preview Mode)'}")
    print(f" Chat ID configured: {'YES' if bool(notifier.chat_id) else 'NO'}")
    print("=" * 70)

    # 1. Send Test Watchlist Alert
    print("\n1. Sending Sample Watchlist (Tier 1) Alert...")
    notifier.send_watchlist_alert(
        symbol="NEARUSDT",
        price=4.6250,
        candle_timestamp=pd.Timestamp.utcnow(),
        signal_info={
            "leadership_score": 94.4,
            "rs_7d": 100.0,
            "rs_30d": 100.0,
            "pullback_atr": 1.45,
            "williams_r": -84.2
        }
    )

    # 2. Send Test Trade Entry Alert
    print("\n2. Sending Sample Trade Entry (Tier 2) Alert...")
    notifier.send_trade_entry_alert(
        symbol="NEARUSDT",
        price=4.6200,
        stop_price=4.4100,
        candle_timestamp=pd.Timestamp.utcnow(),
        signal_info={
            "leadership_score": 94.4,
            "rs_7d": 100.0,
            "rs_30d": 100.0,
            "wr_curr": -74.1,
            "pullback_atr": 1.45,
            "explanation": "Top-decile 7D/30D RS leader. Daily & 4H uptrends active. 1H pullback of 1.45 ATR. Williams %R crossed back above -80. Reversal candle closed above previous candle high."
        },
        account_equity=10000.0,
        risk_per_trade_pct=0.005
    )

    # 3. Send Test Regime Alert
    print("\n3. Sending Sample BTC Regime Alert...")
    notifier.send_regime_alert({
        "regime_active": True,
        "price": 84420.0,
        "sma_50": 74950.0,
        "sma_slope": 0.0042,
        "return_30d_pct": 7.51
    })

    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
