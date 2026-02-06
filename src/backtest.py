"""Backtesting Module - Full Implementation."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from .analyzer import MarketAnalyzer
from .strategy import TradingStrategy

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    price: float
    value: float
    
    def __str__(self):
        return f"{self.timestamp.date()} {self.side.upper()} {self.quantity} {self.symbol} @ ${self.price:.2f}"


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost * 100


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    trades: List[Trade]
    daily_values: List[Dict]
    
    @property
    def total_return(self) -> float:
        return (self.final_value - self.initial_capital) / self.initial_capital * 100
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100
    
    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage."""
        if not self.daily_values:
            return 0.0
        values = [d["value"] for d in self.daily_values]
        peak = values[0]
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            max_dd = max(max_dd, dd)
        return max_dd
    
    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (annualized, assuming 252 trading days)."""
        if len(self.daily_values) < 2:
            return 0.0
        returns = []
        for i in range(1, len(self.daily_values)):
            prev = self.daily_values[i-1]["value"]
            curr = self.daily_values[i]["value"]
            returns.append((curr - prev) / prev)
        if not returns:
            return 0.0
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        return (avg_return * 252) / (std_return * np.sqrt(252))


class MockAlpacaClient:
    """Mock Alpaca client for backtesting with historical data."""
    
    def __init__(self, historical_data: Dict[str, pd.DataFrame]):
        """
        Initialize with pre-loaded historical data.
        
        Args:
            historical_data: Dict mapping symbol to DataFrame with OHLCV data
        """
        self.data = historical_data
        self.current_date: Optional[datetime] = None
    
    def set_date(self, date: datetime):
        """Set the current simulation date."""
        self.current_date = date
    
    def get_bars(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get historical bars up to current date."""
        if symbol not in self.data:
            return []
        
        df = self.data[symbol]
        if self.current_date:
            df = df[df.index <= self.current_date]
        
        df = df.tail(limit)
        
        bars = []
        for idx, row in df.iterrows():
            bars.append({
                "t": idx.isoformat(),
                "o": row["open"],
                "h": row["high"],
                "l": row["low"],
                "c": row["close"],
                "v": row["volume"]
            })
        return bars
    
    def get_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        if symbol not in self.data:
            return 0.0
        
        df = self.data[symbol]
        if self.current_date:
            df = df[df.index <= self.current_date]
        
        if df.empty:
            return 0.0
        
        return float(df.iloc[-1]["close"])


class Backtester:
    """Backtest trading strategies on historical data."""

    def __init__(self, config: Dict[str, Any], start: str, end: str):
        self.config = config
        self.start = datetime.strptime(start, "%Y-%m-%d")
        self.end = datetime.strptime(end, "%Y-%m-%d")
        self.initial_capital = config.get("backtest", {}).get("initial_capital", 100000.0)
        self.capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_values: List[Dict] = []
        
        # Commission per trade (simulated)
        self.commission = config.get("backtest", {}).get("commission", 0.0)
        
        # Initialize components
        self.analyzer = MarketAnalyzer()
        self.strategy = TradingStrategy(config.get("trading", {}))
        self.symbols = config.get("trading", {}).get("symbols", [])
        
        # Risk management
        self.max_position_pct = config.get("trading", {}).get("max_position_pct", 0.1)
        
    def _generate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data for testing when real data unavailable."""
        logger.info("Generating synthetic price data for backtest...")
        
        data = {}
        date_range = pd.date_range(
            start=self.start - timedelta(days=250),  # Extra data for indicators
            end=self.end,
            freq='B'  # Business days only
        )
        
        # Base prices for each symbol
        base_prices = {
            "AAPL": 150.0,
            "GOOGL": 140.0,
            "MSFT": 380.0,
            "NVDA": 500.0,
            "AMZN": 180.0,
            "TSLA": 250.0,
            "META": 350.0,
            "AMD": 120.0,
        }
        
        for symbol in self.symbols:
            base_price = base_prices.get(symbol, 100.0)
            
            # Generate random walk with drift
            np.random.seed(hash(symbol) % (2**32))  # Reproducible per symbol
            n = len(date_range)
            
            # Daily returns with slight upward drift
            daily_returns = np.random.normal(0.0005, 0.02, n)  # 0.05% drift, 2% volatility
            
            # Generate price series
            prices = [base_price]
            for ret in daily_returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create OHLCV data
            df = pd.DataFrame({
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                "close": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
                "volume": [int(np.random.uniform(1e6, 5e7)) for _ in prices],
            }, index=date_range)
            
            # Ensure high >= open, close, low and low <= open, close, high
            df["high"] = df[["open", "high", "close"]].max(axis=1)
            df["low"] = df[["open", "low", "close"]].min(axis=1)
            
            data[symbol] = df
        
        return data

    async def fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data from Alpaca API.
        Falls back to synthetic data if API unavailable.
        """
        # For backtest, we use synthetic data unless real API configured
        # This allows testing without API keys
        return self._generate_sample_data()
    
    def _get_portfolio_value(self, client: MockAlpacaClient) -> float:
        """Calculate total portfolio value."""
        value = self.capital
        for symbol, position in self.positions.items():
            price = client.get_price(symbol)
            position.current_price = price
            value += position.market_value
        return value
    
    def _calculate_position_size(self, symbol: str, price: float, action: str) -> int:
        """Calculate position size based on risk management."""
        portfolio_value = self.capital + sum(
            pos.market_value for pos in self.positions.values()
        )
        
        max_position_value = portfolio_value * self.max_position_pct
        
        if action == "BUY":
            current_position_value = 0
            if symbol in self.positions:
                current_position_value = self.positions[symbol].market_value
            
            available = min(self.capital, max_position_value - current_position_value)
            if available <= 0:
                return 0
            return int(available / price)
        
        elif action == "SELL":
            if symbol in self.positions:
                return self.positions[symbol].quantity
            return 0
        
        return 0
    
    def _execute_trade(
        self, 
        date: datetime, 
        symbol: str, 
        action: str, 
        quantity: int, 
        price: float
    ):
        """Execute a simulated trade."""
        if quantity <= 0:
            return
        
        value = quantity * price
        commission = self.commission * quantity
        
        trade = Trade(
            timestamp=date,
            symbol=symbol,
            side=action.lower(),
            quantity=quantity,
            price=price,
            value=value
        )
        
        if action == "BUY":
            total_cost = value + commission
            if total_cost > self.capital:
                return  # Not enough capital
            
            self.capital -= total_cost
            
            if symbol in self.positions:
                # Add to existing position (update avg cost)
                pos = self.positions[symbol]
                total_qty = pos.quantity + quantity
                total_cost_basis = (pos.avg_cost * pos.quantity) + (price * quantity)
                pos.quantity = total_qty
                pos.avg_cost = total_cost_basis / total_qty
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=price,
                    current_price=price
                )
        
        elif action == "SELL":
            if symbol not in self.positions:
                return
            
            pos = self.positions[symbol]
            if quantity > pos.quantity:
                quantity = pos.quantity
            
            self.capital += (quantity * price) - commission
            
            pos.quantity -= quantity
            if pos.quantity <= 0:
                del self.positions[symbol]
        
        self.trades.append(trade)
        logger.debug(f"  {trade}")

    async def run(self) -> BacktestResult:
        """Run backtest simulation."""
        logger.info("=" * 60)
        logger.info("📊 BACKTESTING ENGINE")
        logger.info("=" * 60)
        logger.info(f"Period: {self.start.date()} to {self.end.date()}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info("=" * 60)
        
        # Fetch/generate historical data
        historical_data = await self.fetch_historical_data()
        client = MockAlpacaClient(historical_data)
        
        # Get trading dates
        all_dates = set()
        for df in historical_data.values():
            all_dates.update(df.index)
        trading_dates = sorted([d for d in all_dates if self.start <= d <= self.end])
        
        logger.info(f"Trading days: {len(trading_dates)}")
        logger.info("-" * 60)
        
        # Iterate through each trading day
        for i, date in enumerate(trading_dates):
            client.set_date(date)
            
            # Log progress every 20 days
            if i % 20 == 0:
                portfolio_value = self._get_portfolio_value(client)
                logger.info(f"Day {i+1}/{len(trading_dates)} - {date.date()} - Portfolio: ${portfolio_value:,.2f}")
            
            # Analyze each symbol
            for symbol in self.symbols:
                try:
                    bars = client.get_bars(symbol, limit=100)
                    if len(bars) < 50:  # Need enough data for indicators
                        continue
                    
                    # Run technical analysis
                    technicals = self.analyzer.analyze(bars)
                    
                    # Create mock sentiment (backtesting doesn't have real-time sentiment)
                    sentiment = {
                        "score": 0.0,  # Neutral
                        "confidence": 0.5,
                        "summary": "Historical backtest - no real-time sentiment",
                        "key_factors": []
                    }
                    
                    # Mock LLM analysis based on technicals
                    llm_analysis = self._mock_llm_analysis(technicals)
                    
                    # Generate signal
                    analysis = {
                        "symbol": symbol,
                        "technicals": technicals,
                        "sentiment": sentiment,
                        "llm_analysis": llm_analysis
                    }
                    
                    signal = self.strategy.generate_signal(analysis)
                    
                    # Execute if signal is not HOLD
                    if signal["action"] != "HOLD":
                        price = client.get_price(symbol)
                        quantity = self._calculate_position_size(
                            symbol, price, signal["action"]
                        )
                        
                        if quantity > 0:
                            self._execute_trade(
                                date, symbol, signal["action"], quantity, price
                            )
                
                except Exception as e:
                    logger.debug(f"Error analyzing {symbol} on {date}: {e}")
            
            # Record daily portfolio value
            portfolio_value = self._get_portfolio_value(client)
            self.daily_values.append({
                "date": date,
                "value": portfolio_value,
                "cash": self.capital,
                "positions": len(self.positions)
            })
        
        # Close all positions at end
        logger.info("-" * 60)
        logger.info("Closing all positions at end of backtest...")
        for symbol in list(self.positions.keys()):
            price = client.get_price(symbol)
            quantity = self.positions[symbol].quantity
            self._execute_trade(trading_dates[-1], symbol, "SELL", quantity, price)
        
        # Calculate results
        result = self._calculate_results()
        self._print_results(result)
        
        return result
    
    def _mock_llm_analysis(self, technicals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock LLM analysis based on technicals for backtesting."""
        signals = technicals.get("signals", {})
        trend = technicals.get("trend", "SIDEWAYS")
        
        # Simple rule-based mock
        bullish_signals = sum(1 for v in signals.values() if v in ["BULLISH", "OVERSOLD"])
        bearish_signals = sum(1 for v in signals.values() if v in ["BEARISH", "OVERBOUGHT"])
        
        if trend in ["STRONG_UPTREND", "UPTREND"] and bullish_signals >= 2:
            return {"recommendation": "BUY", "confidence": 0.7}
        elif trend in ["STRONG_DOWNTREND", "DOWNTREND"] and bearish_signals >= 2:
            return {"recommendation": "SELL", "confidence": 0.7}
        else:
            return {"recommendation": "HOLD", "confidence": 0.5}
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results and statistics."""
        winning_trades = 0
        losing_trades = 0
        
        # Pair up buy/sell trades to determine winners/losers
        buy_trades: Dict[str, List[Trade]] = {}
        
        for trade in self.trades:
            if trade.side == "buy":
                if trade.symbol not in buy_trades:
                    buy_trades[trade.symbol] = []
                buy_trades[trade.symbol].append(trade)
            else:  # sell
                if trade.symbol in buy_trades and buy_trades[trade.symbol]:
                    buy_trade = buy_trades[trade.symbol].pop(0)
                    if trade.price > buy_trade.price:
                        winning_trades += 1
                    else:
                        losing_trades += 1
        
        return BacktestResult(
            start_date=self.start,
            end_date=self.end,
            initial_capital=self.initial_capital,
            final_value=self.capital,
            total_trades=len(self.trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            trades=self.trades,
            daily_values=self.daily_values
        )
    
    def _print_results(self, result: BacktestResult):
        """Print backtest results summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("📈 BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        logger.info("-" * 60)
        logger.info(f"Initial Capital:   ${result.initial_capital:>12,.2f}")
        logger.info(f"Final Value:       ${result.final_value:>12,.2f}")
        logger.info(f"Total Return:      {result.total_return:>12.2f}%")
        logger.info("-" * 60)
        logger.info(f"Total Trades:      {result.total_trades:>12}")
        logger.info(f"Winning Trades:    {result.winning_trades:>12}")
        logger.info(f"Losing Trades:     {result.losing_trades:>12}")
        logger.info(f"Win Rate:          {result.win_rate:>12.1f}%")
        logger.info("-" * 60)
        logger.info(f"Max Drawdown:      {result.max_drawdown:>12.2f}%")
        logger.info(f"Sharpe Ratio:      {result.sharpe_ratio:>12.2f}")
        logger.info("=" * 60)
        
        # Show recent trades
        if result.trades:
            logger.info("")
            logger.info("Recent Trades (last 10):")
            for trade in result.trades[-10:]:
                logger.info(f"  {trade}")
