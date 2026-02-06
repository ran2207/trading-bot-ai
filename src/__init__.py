"""
Trading Bot AI - Source Package

AI-powered algorithmic trading with technical analysis,
sentiment analysis, and LLM-driven insights.
"""

from .alpaca_client import AlpacaClient
from .analyzer import MarketAnalyzer
from .sentiment import SentimentAnalyzer
from .llm import LLMAdvisor
from .strategy import TradingStrategy
from .portfolio import PortfolioManager, RiskManager
from .backtest import Backtester, BacktestResult, Trade, Position
from .chat import TradingChatInterface, ChatCLI

__all__ = [
    "AlpacaClient",
    "MarketAnalyzer",
    "SentimentAnalyzer",
    "LLMAdvisor",
    "TradingStrategy",
    "PortfolioManager",
    "RiskManager",
    "Backtester",
    "BacktestResult",
    "Trade",
    "Position",
    "TradingChatInterface",
    "ChatCLI",
]

__version__ = "1.0.0"
