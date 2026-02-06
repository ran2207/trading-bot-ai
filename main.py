#!/usr/bin/env python3
"""
Trading Bot AI - Main Entry Point
AI-powered trading bot with market analysis and automated execution.

Features:
- Alpaca API integration for market data and trading
- LLM-powered market sentiment analysis
- Technical analysis with indicators (RSI, MACD, moving averages)
- Risk management rules
- Natural language interface for querying positions
- Backtesting framework

Author: Built with AI assistance (Claude/GPT)
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml

from src.alpaca_client import AlpacaClient
from src.analyzer import MarketAnalyzer
from src.sentiment import SentimentAnalyzer
from src.strategy import TradingStrategy
from src.portfolio import PortfolioManager
from src.llm import LLMAdvisor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = Path(path)
    
    if not config_path.exists():
        # Try to load from environment variables
        logger.info("Config file not found, loading from environment variables...")
        return {
            "alpaca": {
                "api_key": os.environ.get("ALPACA_API_KEY", ""),
                "secret_key": os.environ.get("ALPACA_SECRET_KEY", ""),
            },
            "openai": {
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
            },
            "trading": {
                "symbols": ["AAPL", "GOOGL", "MSFT", "NVDA", "AMZN"],
                "max_position_pct": 0.10,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05,
            },
            "backtest": {
                "initial_capital": 100000.0,
                "commission": 0.0,
            }
        }
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config: dict, mode: str = "paper"):
        self.config = config
        self.mode = mode
        self.trade_history = []
        
        # Validate API keys
        if not config.get("alpaca", {}).get("api_key"):
            raise ValueError("Alpaca API key is required. Set ALPACA_API_KEY environment variable or config.yaml")
        if not config.get("openai", {}).get("api_key"):
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or config.yaml")
        
        # Initialize components
        base_url = (
            "https://paper-api.alpaca.markets" 
            if mode == "paper" 
            else "https://api.alpaca.markets"
        )
        
        self.client = AlpacaClient(
            api_key=config["alpaca"]["api_key"],
            secret_key=config["alpaca"]["secret_key"],
            base_url=base_url
        )
        
        self.analyzer = MarketAnalyzer()
        self.sentiment = SentimentAnalyzer(config["openai"]["api_key"])
        self.llm = LLMAdvisor(config["openai"]["api_key"])
        self.strategy = TradingStrategy(config.get("trading", {}))
        self.portfolio = PortfolioManager(self.client, config.get("trading", {}))
        
        self.symbols = config.get("trading", {}).get("symbols", [])
        
        logger.info(f"Trading bot initialized in {mode.upper()} mode")
        logger.info(f"Trading symbols: {', '.join(self.symbols)}")

    async def analyze_symbol(self, symbol: str) -> dict:
        """Run full analysis on a symbol."""
        logger.info(f"📊 Analyzing {symbol}...")
        
        # Get market data
        bars = await self.client.get_bars(symbol, timeframe="1D", limit=100)
        
        if not bars:
            logger.warning(f"No bar data available for {symbol}")
            return {"symbol": symbol, "error": "No data available"}
        
        # Technical analysis
        technicals = self.analyzer.analyze(bars)
        logger.info(f"  Technical signals: {technicals['signals']}")
        logger.info(f"  Trend: {technicals['trend']}")
        
        # Sentiment analysis
        sentiment = await self.sentiment.analyze(symbol)
        logger.info(f"  Sentiment score: {sentiment['score']:.2f}")
        
        # LLM analysis
        llm_analysis = await self.llm.analyze_market(
            symbol=symbol,
            technicals=technicals,
            sentiment=sentiment
        )
        logger.info(f"  LLM recommendation: {llm_analysis['recommendation']}")
        
        return {
            "symbol": symbol,
            "technicals": technicals,
            "sentiment": sentiment,
            "llm_analysis": llm_analysis,
            "timestamp": datetime.now().isoformat()
        }

    async def execute_strategy(self, analysis: dict) -> dict | None:
        """Execute trading strategy based on analysis."""
        signal = self.strategy.generate_signal(analysis)
        
        logger.info(f"  Signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
        
        if signal["action"] == "HOLD":
            logger.info(f"  ⏸️  HOLD - No action for {analysis['symbol']}")
            return None
        
        # Get current price for position sizing
        try:
            quote = await self.client.get_latest_quote(analysis["symbol"])
            current_price = float(quote.get("quote", {}).get("ap", 0) or 
                                 analysis["technicals"].get("price", 0))
        except Exception:
            current_price = analysis["technicals"].get("price", 0)
        
        if current_price <= 0:
            logger.warning(f"  ⚠️  Cannot get current price for {analysis['symbol']}")
            return None
        
        # Calculate position size
        position_size = self.portfolio.calculate_position_size(
            symbol=analysis["symbol"],
            signal=signal,
            current_price=current_price
        )
        
        if position_size == 0:
            logger.info(f"  ⚠️  Position size is 0, skipping")
            return None
        
        # Execute order
        if self.mode in ["paper", "live"]:
            try:
                order = await self.client.submit_order(
                    symbol=analysis["symbol"],
                    qty=position_size,
                    side=signal["action"].lower(),
                    type="market"
                )
                logger.info(f"  ✅ {signal['action']} order submitted: {position_size} shares of {analysis['symbol']}")
                
                # Record in history
                self.trade_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": analysis["symbol"],
                    "action": signal["action"],
                    "quantity": position_size,
                    "order_id": order.get("id"),
                })
                
                return order
            except Exception as e:
                logger.error(f"  ❌ Order failed: {e}")
                return None
        
        # Simulation mode
        return {
            "simulated": True, 
            "action": signal["action"], 
            "qty": position_size,
            "symbol": analysis["symbol"]
        }

    async def run_cycle(self) -> list:
        """Run one analysis and trading cycle."""
        logger.info("=" * 60)
        logger.info(f"🤖 Starting trading cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Mode: {self.mode.upper()}")
        logger.info("=" * 60)
        
        # Get account info
        try:
            account = await self.client.get_account()
            equity = float(account.get('equity', 0))
            cash = float(account.get('cash', 0))
            logger.info(f"💰 Account equity: ${equity:,.2f} | Cash: ${cash:,.2f}")
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return []
        
        results = []
        
        for symbol in self.symbols:
            logger.info("-" * 40)
            try:
                # Analyze
                analysis = await self.analyze_symbol(symbol)
                
                if "error" in analysis:
                    continue
                
                # Execute
                order = await self.execute_strategy(analysis)
                
                results.append({
                    "symbol": symbol,
                    "analysis": analysis,
                    "order": order
                })
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        logger.info("=" * 60)
        logger.info("🏁 Trading cycle complete")
        logger.info(f"   Symbols analyzed: {len(results)}")
        logger.info(f"   Orders placed: {sum(1 for r in results if r.get('order'))}")
        logger.info("=" * 60)
        
        return results

    async def run(self):
        """Main run loop."""
        logger.info("🚀 Starting Trading Bot AI")
        logger.info(f"   Mode: {self.mode.upper()}")
        
        while True:
            try:
                # Check if market is open
                clock = await self.client.get_clock()
                
                if clock.get("is_open"):
                    await self.run_cycle()
                else:
                    next_open = clock.get("next_open", "unknown")
                    logger.info(f"📴 Market is closed. Next open: {next_open}")
                
                # Wait 15 minutes between cycles
                logger.info("💤 Sleeping for 15 minutes...")
                await asyncio.sleep(900)
                
            except KeyboardInterrupt:
                logger.info("👋 Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)


async def run_chat_interface(config: dict, mode: str):
    """Run the natural language chat interface."""
    from src.chat import TradingChatInterface, ChatCLI
    
    # Initialize bot for chat
    bot = TradingBot(config, mode=mode)
    
    # Create chat interface with bot access
    chat_interface = TradingChatInterface(
        api_key=config["openai"]["api_key"],
        trading_bot=bot
    )
    
    # Run CLI
    cli = ChatCLI(chat_interface)
    await cli.run()


async def run_analysis_only(config: dict, symbols: list = None):
    """Run analysis without trading (read-only mode)."""
    bot = TradingBot(config, mode="paper")
    
    symbols = symbols or config.get("trading", {}).get("symbols", [])
    
    logger.info("=" * 60)
    logger.info("📊 MARKET ANALYSIS REPORT")
    logger.info(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    for symbol in symbols:
        logger.info("")
        try:
            analysis = await bot.analyze_symbol(symbol)
            
            if "error" in analysis:
                continue
            
            # Print summary
            tech = analysis["technicals"]
            sent = analysis["sentiment"]
            llm = analysis["llm_analysis"]
            
            logger.info(f"{'─' * 40}")
            logger.info(f"📈 {symbol}")
            logger.info(f"   Price: ${tech['price']:.2f}")
            logger.info(f"   Trend: {tech['trend']}")
            logger.info(f"   RSI: {tech['rsi']:.1f}")
            logger.info(f"   MACD: {tech['macd']:.4f}")
            logger.info(f"   Sentiment: {sent['score']:.2f}")
            logger.info(f"   Recommendation: {llm['recommendation']}")
            
            signal = bot.strategy.generate_signal(analysis)
            logger.info(f"   Signal: {signal['action']} ({signal['confidence']:.0%} confidence)")
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    logger.info("")
    logger.info("=" * 60)


async def main():
    parser = argparse.ArgumentParser(
        description="Trading Bot AI - AI-powered stock trading assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode paper              # Run in paper trading mode
  python main.py --mode backtest --start 2024-01-01 --end 2024-12-31
  python main.py --mode chat               # Interactive chat interface
  python main.py --mode analyze            # Analysis only (no trading)
  python main.py --mode analyze --symbol AAPL TSLA NVDA
        """
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "backtest", "chat", "analyze"],
        default="paper",
        help="Operating mode (default: paper)"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--start", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--symbol", nargs="+", help="Symbols to analyze (for analyze mode)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    if args.mode == "backtest":
        if not args.start or not args.end:
            parser.error("Backtest mode requires --start and --end dates")
        
        from src.backtest import Backtester
        backtester = Backtester(config, args.start, args.end)
        await backtester.run()
    
    elif args.mode == "chat":
        await run_chat_interface(config, mode="paper")
    
    elif args.mode == "analyze":
        await run_analysis_only(config, symbols=args.symbol)
    
    else:
        # Paper or live trading
        if args.mode == "live":
            logger.warning("⚠️  LIVE TRADING MODE - Real money at risk!")
            response = input("Type 'I UNDERSTAND' to continue: ")
            if response != "I UNDERSTAND":
                logger.info("Cancelled.")
                return
        
        bot = TradingBot(config, mode=args.mode)
        await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
