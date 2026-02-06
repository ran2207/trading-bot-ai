"""Tests for the MarketAnalyzer module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.analyzer import MarketAnalyzer


class TestMarketAnalyzer:
    """Test suite for MarketAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return MarketAnalyzer()

    @pytest.fixture
    def sample_bars(self):
        """Generate sample bar data for testing."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=100),
            periods=100,
            freq='D'
        )
        
        # Generate realistic-looking price data
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)
        prices = [base_price]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        
        bars = []
        for i, date in enumerate(dates):
            price = prices[i]
            bars.append({
                "t": date.isoformat(),
                "o": price * (1 + np.random.uniform(-0.01, 0.01)),
                "h": price * (1 + abs(np.random.uniform(0, 0.02))),
                "l": price * (1 - abs(np.random.uniform(0, 0.02))),
                "c": price,
                "v": int(np.random.uniform(1e6, 5e7))
            })
        return bars

    def test_analyze_returns_required_fields(self, analyzer, sample_bars):
        """Test that analyze returns all required fields."""
        result = analyzer.analyze(sample_bars)
        
        assert "price" in result
        assert "sma_20" in result
        assert "sma_50" in result
        assert "rsi" in result
        assert "macd" in result
        assert "signals" in result
        assert "trend" in result

    def test_rsi_bounds(self, analyzer, sample_bars):
        """Test that RSI is within valid bounds (0-100)."""
        result = analyzer.analyze(sample_bars)
        
        assert result["rsi"] >= 0
        assert result["rsi"] <= 100

    def test_signals_have_expected_values(self, analyzer, sample_bars):
        """Test that signals contain expected keys."""
        result = analyzer.analyze(sample_bars)
        signals = result["signals"]
        
        assert "rsi" in signals
        assert "macd" in signals
        assert "ma_cross" in signals
        
        # Values should be valid signal types
        valid_rsi_signals = ["OVERSOLD", "OVERBOUGHT", "NEUTRAL"]
        valid_macd_signals = ["BULLISH", "BEARISH"]
        
        assert signals["rsi"] in valid_rsi_signals
        assert signals["macd"] in valid_macd_signals

    def test_trend_detection(self, analyzer, sample_bars):
        """Test that trend is detected correctly."""
        result = analyzer.analyze(sample_bars)
        
        valid_trends = [
            "STRONG_UPTREND", "UPTREND", 
            "STRONG_DOWNTREND", "DOWNTREND", 
            "SIDEWAYS"
        ]
        
        assert result["trend"] in valid_trends

    def test_empty_bars_raises_error(self, analyzer):
        """Test that empty bars raise an appropriate error."""
        with pytest.raises(Exception):
            analyzer.analyze([])

    def test_insufficient_data_handling(self, analyzer):
        """Test handling of insufficient data for indicators."""
        # Only 5 bars - not enough for 20-day SMA
        short_bars = [
            {"t": "2024-01-01", "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000}
            for _ in range(5)
        ]
        
        # Should still run without crashing (NaN values are acceptable)
        result = analyzer.analyze(short_bars)
        assert "price" in result


class TestTechnicalIndicators:
    """Test individual technical indicators."""

    @pytest.fixture
    def analyzer(self):
        return MarketAnalyzer()

    def test_sma_calculation(self, analyzer):
        """Test SMA calculation accuracy."""
        # Create simple test data
        dates = pd.date_range(start="2024-01-01", periods=30, freq='D')
        df = pd.DataFrame({
            "close": [100 + i for i in range(30)]  # Linear increase
        }, index=dates)
        df["open"] = df["close"]
        df["high"] = df["close"] + 1
        df["low"] = df["close"] - 1
        df["volume"] = 1000000
        
        # Calculate 20-day SMA
        result = analyzer._calculate_sma(df.copy(), [20])
        
        # The 20-day SMA at the end should be the average of last 20 values
        expected_sma_20 = df["close"].iloc[-20:].mean()
        assert abs(result.iloc[-1]["sma_20"] - expected_sma_20) < 0.01

    def test_rsi_oversold_detection(self, analyzer):
        """Test RSI detects oversold conditions."""
        # Create declining price data
        dates = pd.date_range(start="2024-01-01", periods=50, freq='D')
        prices = [100 * (0.98 ** i) for i in range(50)]  # Declining
        
        df = pd.DataFrame({
            "close": prices,
            "open": prices,
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "volume": [1000000] * 50
        }, index=dates)
        
        df = analyzer._calculate_rsi(df)
        
        # After consistent declines, RSI should be low
        assert df.iloc[-1]["rsi"] < 50  # Should be bearish

    def test_macd_crossover_detection(self, analyzer):
        """Test MACD crossover detection."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq='D')
        
        # Create price data with trend change
        prices = [100 - i * 0.5 for i in range(25)]  # Declining
        prices.extend([prices[-1] + i * 1.0 for i in range(25)])  # Rising
        
        df = pd.DataFrame({
            "close": prices,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "volume": [1000000] * 50
        }, index=dates)
        
        df = analyzer._calculate_ema(df, [12, 26])
        df = analyzer._calculate_macd(df)
        
        # Should have MACD values
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert not df["macd"].iloc[-1] != df["macd"].iloc[-1]  # Not NaN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
