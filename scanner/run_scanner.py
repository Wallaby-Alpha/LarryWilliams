"""
Command-Line Runner for Live MEXC Scanner.

Usage:
    python run_scanner.py --once
    python run_scanner.py --loop --interval 3600
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import yaml
import argparse
import logging
from scanner.scanner_service import ScannerService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RunScanner")


def load_config(path: str = "config/default_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="MEXC Larry Williams Trend/Pullback Scanner")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to YAML config")
    parser.add_argument("--once", action="store_true", help="Run scan once and exit")
    parser.add_argument("--loop", action="store_true", help="Run continuously every interval")
    parser.add_argument("--interval", type=int, default=3600, help="Loop interval in seconds (default: 3600)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    scanner = ScannerService(cfg)

    while True:
        try:
            res = scanner.scan_once()
            regime = res["btc_regime"]
            status_str = "BULLISH / ON" if regime["regime_active"] else "BEARISH / OFF"
            print("\n" + "=" * 80)
            print(f" MEXC SCAN REPORT — {res['timestamp']}")
            print(f" BTC Regime: {status_str} | Price: ${regime['price']:,.2f} | 50 SMA: ${regime['sma_50']:,.2f} | 30D Ret: {regime['return_30d_pct']:.2f}%")
            print("=" * 80)

            print("\n[TOP LEADERS]")
            print(res["leaders"][["symbol", "leadership_score", "rs_7d", "rs_30d", "price"]].head(10).to_string(index=False))

            ready = res["ready_signals"]
            print(f"\n[READY SIGNALS FOR IMMEDIATE EXECUTION: {len(ready)}]")
            if not ready.empty:
                for _, r in ready.iterrows():
                    print(f"  -> {r['symbol']}: Price={r['price']} | Stop={r['stop']} | Target 2R={r['target_2r']} | RR={r['rr_ratio']}")
                    print(f"     Explanation: {r['explanation']}\n")
            else:
                print("  No symbols currently satisfy all entry conditions.")

            setups = res["active_setups"]
            print(f"\n[ACTIVE DEVELOPING SETUPS: {len(setups)}]")
            if not setups.empty:
                print(setups[["symbol", "state", "leadership_score", "williams_r", "pullback_atr", "explanation"]].head(10).to_string(index=False))

            if args.once or not args.loop:
                break

            print(f"\nSleeping for {args.interval}s until next hourly scan...")
            time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\nScanner stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error during scan: {e}", exc_info=True)
            if args.once:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
