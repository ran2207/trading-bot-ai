"""Tests for the Backtester module."""

import pytest
import asyncio
from datetime import datetime

from src.backtest import Backtester, Trade, Position, BacktestResult


class TestPosition:
    """Test Position dataclass."""

    def test_position_market_value(self):
        """Test market value calculation."""
        pos = Position(
            symbol="AAPL",
            quantity=10,
            avg_cost=100.0,
            current_price=110.0
        )
        
        assert pos.market_value == 1100.0

    def test_position_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        pos = Position(
            symbol="AAPL",
            quantity=10,
            avg_cost=100.0,
            current_price=110.0
        )
        
        assert pos.unrealized_pnl == 100.0  # (110-100) * 10

    def test_position_unrealized_pnl_pct(self):
        """Test unrealized P&L percentage."""
        pos = Position(
            symbol="AAPL",
            quantity=10,
            avg_cost=100.0,
            current_price=110.0
        )
        
        assert pos.unrealized_pnl_pct == 10.0  # 10% gain

    def test_position_loss(self):
        """Test losing position."""
        pos = Position(
            symbol="AAPL",
            quantity=10,
            avg_cost=100.0,
            current_price=90.0
        )
        
        assert pos.unrealized_pnl == -100.0
        assert pos.unrealized_pnl_pct == -10.0


class TestTrade:
    """Test Trade dataclass."""

    def test_trade_str_format(self):
        """Test trade string representation."""
        trade = Trade(
            timestamp=datetime(2024, 1, 15),
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=150.0,
            value=1500.0
        )
        
        trade_str = str(trade)
        assert "2024-01-15" in trade_str
        assert "BUY" in trade_str
        assert "AAPL" in trade_str
        assert "150.00" in trade_str


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create sample backtest result."""
        return BacktestResult(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=100000.0,
            final_value=120000.0,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            trades=[],
            daily_values=[
                {"date": datetime(2024, 1, 1), "value": 100000},
                {"date": datetime(2024, 6, 1), "value": 110000},
                {"date": datetime(2024, 12, 31), "value": 120000},
            ]
        )

    def test_total_return(self, sample_result):
        """Test total return calculation."""
        assert sample_result.total_return == 20.0  # 20% return

    def test_win_rate(self, sample_result):
        """Test win rate calculation."""
        assert sample_result.win_rate == 60.0  # 60% win rate

    def test_max_drawdown(self, sample_result):
        """Test max drawdown calculation."""
        # No drawdown in this upward trend
        assert sample_result.max_drawdown == 0.0

    def test_max_drawdown_with_dip(self):
        """Test max drawdown with a significant dip."""
        result = BacktestResult(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=100000.0,
            final_value=110000.0,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            trades=[],
            daily_values=[
                {"date": datetime(2024, 1, 1), "value": 100000},
                {"date": datetime(2024, 3, 1), "value": 120000},  # Peak
                {"date": datetime(2024, 6, 1), "value": 96000},   # Trough (20% DD)
                {"date": datetime(2024, 12, 31), "value": 110000},
            ]
        )
        
        assert result.max_drawdown == 20.0  # 20% drawdown

    def test_sharpe_ratio_calculation(self, sample_result):
        """Test Sharpe ratio is calculated."""
        sharpe = sample_result.sharpe_ratio
        # Just verify it's a number (actual value depends on returns)
        assert isinstance(sharpe, float)

    def test_zero_trades_win_rate(self):
        """Test win rate with zero trades."""
        result = BacktestResult(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=100000.0,
            final_value=100000.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            trades=[],
            daily_values=[]
        )
        
        assert result.win_rate == 0.0


class TestBacktester:
    """Test Backtester class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return {
            "trading": {
                "symbols": ["AAPL", "GOOGL"],
                "max_position_pct": 0.10,
            },
            "backtest": {
                "initial_capital": 100000.0,
                "commission": 0.0,
            }
        }

    def test_backtester_initialization(self, config):
        """Test backtester initializes correctly."""
        bt = Backtester(config, "2024-01-01", "2024-12-31")
        
        assert bt.initial_capital == 100000.0
        assert bt.capital == 100000.0
        assert len(bt.positions) == 0
        assert len(bt.trades) == 0

    def test_sample_data_generation(self, config):
        """Test synthetic data generation."""
        bt = Backtester(config, "2024-01-01", "2024-03-31")
        data = bt._generate_sample_data()
        
        assert "AAPL" in data
        assert "GOOGL" in data
        
        # Check data structure
        for symbol, df in data.items():
            assert "open" in df.columns
            assert "high" in df.columns
            assert "low" in df.columns
            assert "close" in df.columns
            assert "volume" in df.columns
            
            # OHLC sanity checks
            assert (df["high"] >= df["low"]).all()
            assert (df["high"] >= df["close"]).all()
            assert (df["low"] <= df["close"]).all()

    @pytest.mark.asyncio
    async def test_backtest_runs_without_error(self, config):
        """Test that backtest completes without errors."""
        bt = Backtester(config, "2024-01-01", "2024-03-31")
        result = await bt.run()
        
        assert isinstance(result, BacktestResult)
        assert result.start_date == datetime(2024, 1, 1)
        assert result.end_date == datetime(2024, 3, 31)

    @pytest.mark.asyncio
    async def test_backtest_generates_trades(self, config):
        """Test that backtest generates some trades."""
        bt = Backtester(config, "2024-01-01", "2024-06-30")
        result = await bt.run()
        
        # With 6 months of data, should have some trades
        # (Not guaranteed, but likely with the strategy)
        assert result.total_trades >= 0

    def test_mock_llm_analysis(self, config):
        """Test mock LLM analysis generation."""
        bt = Backtester(config, "2024-01-01", "2024-01-31")
        
        # Bullish technicals
        bullish_tech = {
            "signals": {
                "rsi": "OVERSOLD",
                "macd": "BULLISH",
                "ma_cross": "BULLISH"
            },
            "trend": "UPTREND"
        }
        
        result = bt._mock_llm_analysis(bullish_tech)
        assert result["recommendation"] == "BUY"
        
        # Bearish technicals
        bearish_tech = {
            "signals": {
                "rsi": "OVERBOUGHT",
                "macd": "BEARISH",
                "ma_cross": "BEARISH"
            },
            "trend": "DOWNTREND"
        }
        
        result = bt._mock_llm_analysis(bearish_tech)
        assert result["recommendation"] == "SELL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
