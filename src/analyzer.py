"""Technical Analysis Module."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


class MarketAnalyzer:
    """Technical analysis calculator."""

    def analyze(self, bars: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run full technical analysis on bar data."""
        df = self._bars_to_dataframe(bars)
        
        # Calculate indicators
        df = self._calculate_sma(df, [20, 50, 200])
        df = self._calculate_ema(df, [12, 26])
        df = self._calculate_rsi(df)
        df = self._calculate_macd(df)
        df = self._calculate_bollinger_bands(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Generate signals
        signals = self._generate_signals(df)
        
        return {
            "price": latest["close"],
            "sma_20": latest.get("sma_20"),
            "sma_50": latest.get("sma_50"),
            "sma_200": latest.get("sma_200"),
            "rsi": latest.get("rsi"),
            "macd": latest.get("macd"),
            "macd_signal": latest.get("macd_signal"),
            "bb_upper": latest.get("bb_upper"),
            "bb_lower": latest.get("bb_lower"),
            "signals": signals,
            "trend": self._determine_trend(df),
        }

    def _bars_to_dataframe(self, bars: List[Dict]) -> pd.DataFrame:
        """Convert bar data to DataFrame."""
        df = pd.DataFrame(bars)
        df["timestamp"] = pd.to_datetime(df["t"])
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df.set_index("timestamp").sort_index()
        return df

    def _calculate_sma(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Simple Moving Averages."""
        for period in periods:
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
        return df

    def _calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Exponential Moving Averages."""
        for period in periods:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD."""
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df["bb_middle"] = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()
        df["bb_upper"] = df["bb_middle"] + (std * 2)
        df["bb_lower"] = df["bb_middle"] - (std * 2)
        return df

    def _generate_signals(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate trading signals from indicators."""
        latest = df.iloc[-1]
        signals = {}
        
        # RSI Signal
        rsi = latest.get("rsi", 50)
        if rsi < 30:
            signals["rsi"] = "OVERSOLD"
        elif rsi > 70:
            signals["rsi"] = "OVERBOUGHT"
        else:
            signals["rsi"] = "NEUTRAL"
        
        # MACD Signal
        if latest.get("macd", 0) > latest.get("macd_signal", 0):
            signals["macd"] = "BULLISH"
        else:
            signals["macd"] = "BEARISH"
        
        # Bollinger Bands Signal
        price = latest["close"]
        if price < latest.get("bb_lower", price):
            signals["bollinger"] = "OVERSOLD"
        elif price > latest.get("bb_upper", price):
            signals["overbought"] = "OVERBOUGHT"
        else:
            signals["bollinger"] = "NEUTRAL"
        
        # Moving Average Signal
        if latest.get("sma_20", 0) > latest.get("sma_50", 0):
            signals["ma_cross"] = "BULLISH"
        else:
            signals["ma_cross"] = "BEARISH"
        
        return signals

    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine overall trend."""
        latest = df.iloc[-1]
        price = latest["close"]
        
        above_sma_20 = price > latest.get("sma_20", price)
        above_sma_50 = price > latest.get("sma_50", price)
        above_sma_200 = price > latest.get("sma_200", price)
        
        if above_sma_20 and above_sma_50 and above_sma_200:
            return "STRONG_UPTREND"
        elif above_sma_20 and above_sma_50:
            return "UPTREND"
        elif not above_sma_20 and not above_sma_50 and not above_sma_200:
            return "STRONG_DOWNTREND"
        elif not above_sma_20 and not above_sma_50:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
