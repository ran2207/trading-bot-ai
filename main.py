#!/usr/bin/env python3
"""
Trading Bot AI - Main Entry Point
AI-powered trading bot with market analysis and automated execution
"""

import argparse
import asyncio
import logging
from datetime import datetime

import yaml

from src.alpaca_client import AlpacaClient
from src.analyzer import MarketAnalyzer
from src.sentiment import SentimentAnalyzer
from src.strategy import TradingStrategy
from src.portfolio import PortfolioManager
from src.llm import LLMAdvisor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config: dict, mode: str = "paper"):
        self.config = config
        self.mode = mode
        
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
        self.strategy = TradingStrategy(config["trading"])
        self.portfolio = PortfolioManager(self.client, config["trading"])
        
        self.symbols = config["trading"]["symbols"]

    async def analyze_symbol(self, symbol: str) -> dict:
        """Run full analysis on a symbol."""
        logger.info(f"📊 Analyzing {symbol}...")
        
        # Get market data
        bars = await self.client.get_bars(symbol, timeframe="1D", limit=100)
        
        # Technical analysis
        technicals = self.analyzer.analyze(bars)
        logger.info(f"  Technical signals: {technicals['signals']}")
        
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
        
        if signal["action"] == "HOLD":
            logger.info(f"  ⏸️  HOLD - No action for {analysis['symbol']}")
            return None
        
        # Calculate position size
        position_size = self.portfolio.calculate_position_size(
            symbol=analysis["symbol"],
            signal=signal
        )
        
        if position_size == 0:
            logger.info(f"  ⚠️  Position size is 0, skipping")
            return None
        
        # Execute order
        if self.mode == "paper" or self.mode == "live":
            order = await self.client.submit_order(
                symbol=analysis["symbol"],
                qty=position_size,
                side=signal["action"].lower(),
                type="market"
            )
            logger.info(f"  ✅ {signal['action']} order submitted: {position_size} shares")
            return order
        
        return {"simulated": True, "action": signal["action"], "qty": position_size}

    async def run_cycle(self):
        """Run one analysis and trading cycle."""
        logger.info("=" * 50)
        logger.info(f"🤖 Starting trading cycle - {datetime.now()}")
        logger.info(f"   Mode: {self.mode.upper()}")
        logger.info("=" * 50)
        
        # Get account info
        account = await self.client.get_account()
        logger.info(f"💰 Account equity: ${float(account['equity']):,.2f}")
        
        results = []
        
        for symbol in self.symbols:
            try:
                # Analyze
                analysis = await self.analyze_symbol(symbol)
                
                # Execute
                order = await self.execute_strategy(analysis)
                
                results.append({
                    "symbol": symbol,
                    "analysis": analysis,
                    "order": order
                })
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        logger.info("=" * 50)
        logger.info("🏁 Trading cycle complete")
        logger.info("=" * 50)
        
        return results

    async def run(self):
        """Main run loop."""
        logger.info("🚀 Starting Trading Bot AI")
        
        while True:
            try:
                # Check if market is open
                clock = await self.client.get_clock()
                
                if clock["is_open"]:
                    await self.run_cycle()
                else:
                    logger.info("📴 Market is closed. Waiting...")
                
                # Wait 15 minutes between cycles
                await asyncio.sleep(900)
                
            except KeyboardInterrupt:
                logger.info("👋 Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)


async def main():
    parser = argparse.ArgumentParser(description="Trading Bot AI")
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "backtest"],
        default="paper",
        help="Trading mode"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--start", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Backtest end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.mode == "backtest":
        from src.backtest import Backtester
        backtester = Backtester(config, args.start, args.end)
        await backtester.run()
    else:
        bot = TradingBot(config, mode=args.mode)
        await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
