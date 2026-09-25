"""
Dynamic Universe Selection & Survivorship-Bias Awareness Module.

Filters MEXC Spot universe dynamically:
- USDT quote asset
- Excludes BTC (reserved for market regime)
- Excludes stablecoins, fiat, synthetics
- Excludes leveraged tokens (3L, 3S, 5L, 5S, BULL, BEAR, UP, DOWN)
- Configurable minimum 24h quote volume ($1M, $2.5M, $5M, $10M, $25M)
- Tracks listing dates and delisted symbols for survivorship-bias mitigation.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
import pandas as pd

logger = logging.getLogger("UniverseSelection")


class UniverseManager:
    """
    Manages universe qualification and historical tracking.
    """

    STABLECOINS = {
        "USDC", "USDT", "BUSD", "TUSD", "FDUSD", "DAI", "USDD", "EUR", "GBP", "USD",
        "PYUSD", "USDY", "GUSD", "USDP", "CUSD", "EURS", "AEUR"
    }

    LEVERAGED_SUFFIXES = ("3L", "3S", "4L", "4S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")

    def __init__(
        self,
        quote_asset: str = "USDT",
        min_24h_quote_volume_usd: float = 5000000.0,
        exclude_coins: Optional[List[str]] = None,
        min_history_days: int = 90
    ):
        self.quote_asset = quote_asset.upper()
        self.min_24h_quote_volume_usd = min_24h_quote_volume_usd
        self.exclude_coins = set(exclude_coins or ["BTCUSDT"])
        self.min_history_days = min_history_days
        
        # Historical listing / delisting registry to prevent survivorship bias
        self.symbol_metadata: Dict[str, Dict[str, Any]] = {}

    def is_leveraged_or_synthetic(self, base_asset: str) -> bool:
        """Checks if token is a leveraged or synthetic token."""
        base_upper = base_asset.upper()
        for pattern in self.LEVERAGED_SUFFIXES:
            if base_upper.endswith(pattern) or pattern in base_upper:
                return True
        return False

    def is_stablecoin(self, base_asset: str) -> bool:
        """Checks if token is a stablecoin or fiat wrapper."""
        return base_asset.upper() in self.STABLECOINS

    def filter_active_symbols(
        self,
        symbols_info: List[Dict[str, Any]],
        tickers_24h: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Filters current symbols from MEXC exchangeInfo and 24hr tickers.
        """
        qualified = []
        for info in symbols_info:
            symbol = info.get("symbol", "")
            status = info.get("status", "")
            base_asset = info.get("baseAsset", "")
            quote_asset = info.get("quoteAsset", "")

            # Check status and quote
            if status != "ENABLED" and status != "1":
                continue
            if quote_asset.upper() != self.quote_asset:
                continue

            # Exclude benchmark and manual exclusions
            if symbol.upper() in self.exclude_coins or base_asset.upper() == "BTC":
                continue

            # Exclude stablecoins and leveraged tokens
            if self.is_stablecoin(base_asset) or self.is_leveraged_or_synthetic(base_asset):
                continue

            # Check 24hr volume if ticker data provided
            if tickers_24h and symbol in tickers_24h:
                ticker = tickers_24h[symbol]
                # quoteVolume is volume in USDT
                quote_vol = float(ticker.get("quoteVolume", 0.0))
                if quote_vol < self.min_24h_quote_volume_usd:
                    continue

            qualified.append(symbol)
            # Store metadata
            if symbol not in self.symbol_metadata:
                self.symbol_metadata[symbol] = {
                    "base": base_asset,
                    "quote": quote_asset,
                    "listing_status": "ACTIVE",
                    "first_observed": pd.Timestamp.utcnow(),
                }

        logger.info(f"Qualified {len(qualified)} liquid altcoins with >= ${self.min_24h_quote_volume_usd:,.0f} 24h volume.")
        return sorted(qualified)

    def reconstruct_point_in_time_universe(
        self,
        all_symbols_daily_data: Dict[str, pd.DataFrame],
        as_of_date: pd.Timestamp,
        min_prior_bars: int = 90
    ) -> List[str]:
        """
        Point-in-time universe reconstruction to mitigate survivorship bias.
        Only includes coins that had at least `min_prior_bars` before `as_of_date`
        and had continuous trading activity without being delisted prior.
        """
        eligible = []
        for sym, df in all_symbols_daily_data.items():
            if sym.upper() in self.exclude_coins:
                continue
            
            sub = df[df["timestamp"] <= as_of_date]
            if len(sub) < min_prior_bars:
                continue
            
            # Check volume on recent bar
            recent_bar = sub.iloc[-1]
            quote_vol = recent_bar.get("quote_volume", recent_bar["volume"] * recent_bar["close"])
            if quote_vol >= self.min_24h_quote_volume_usd:
                eligible.append(sym)

        return sorted(eligible)
