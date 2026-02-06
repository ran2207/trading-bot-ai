# Trading Bot AI 🤖📈

An AI-powered algorithmic trading bot that combines technical analysis, sentiment analysis, and LLM-driven insights to make trading decisions.

> ⚠️ **DISCLAIMER**: This is for educational purposes only. Algorithmic trading involves significant risk. Never trade with money you can't afford to lose. Always use paper trading first.

## Features

### 🔌 Alpaca API Integration
- Real-time and historical market data
- Paper trading and live trading support
- Position management and order execution
- Account balance and portfolio tracking

### 🧠 LLM-Powered Analysis
- GPT-4 market sentiment analysis
- Natural language market insights
- AI-driven trading recommendations
- Confidence scoring for decisions

### 📊 Technical Analysis
- **Moving Averages**: SMA (20, 50, 200), EMA (12, 26)
- **RSI**: Relative Strength Index with overbought/oversold detection
- **MACD**: Moving Average Convergence Divergence with signal line
- **Bollinger Bands**: Volatility-based price channels
- **Trend Detection**: Multi-timeframe trend analysis

### ⚖️ Risk Management
- Position sizing based on portfolio percentage
- Stop-loss and take-profit calculations
- Maximum position limits
- Portfolio risk assessment
- Kelly Criterion-inspired sizing

### 💬 Natural Language Interface
- Chat with your trading bot
- Query positions in plain English
- Get market analysis conversationally
- Execute trades via natural language commands

### 📈 Backtesting Framework
- Historical data simulation
- Synthetic data generation for testing
- Performance metrics (Sharpe ratio, max drawdown, win rate)
- Trade-by-trade analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/ran2207/trading-bot-ai.git
cd trading-bot-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy the example config:
```bash
cp config.example.yaml config.yaml
```

2. Edit `config.yaml` with your API keys:
```yaml
alpaca:
  api_key: "your-alpaca-api-key"
  secret_key: "your-alpaca-secret-key"

openai:
  api_key: "sk-your-openai-api-key"

trading:
  symbols:
    - "AAPL"
    - "GOOGL"
    - "MSFT"
    - "NVDA"
    - "AMZN"
  max_position_pct: 0.10
  stop_loss_pct: 0.02
  take_profit_pct: 0.05
```

Or use environment variables:
```bash
export ALPACA_API_KEY="your-key"
export ALPACA_SECRET_KEY="your-secret"
export OPENAI_API_KEY="sk-your-key"
```

### Getting API Keys

**Alpaca** (for trading):
1. Sign up at [alpaca.markets](https://alpaca.markets)
2. Go to Paper Trading → API Keys
3. Generate a new API key pair

**OpenAI** (for AI analysis):
1. Sign up at [platform.openai.com](https://platform.openai.com)
2. Go to API Keys
3. Create a new secret key

## Usage

### Paper Trading Mode (Recommended for Testing)
```bash
python main.py --mode paper
```

### Analysis Only (No Trading)
```bash
python main.py --mode analyze
python main.py --mode analyze --symbol AAPL TSLA NVDA
```

### Interactive Chat Interface
```bash
python main.py --mode chat
```

Example chat commands:
- "Show my positions"
- "What's my account balance?"
- "Analyze AAPL for me"
- "What's the current price of TSLA?"
- "Buy 10 shares of MSFT"

### Backtesting
```bash
python main.py --mode backtest --start 2024-01-01 --end 2024-12-31
```

### Live Trading (Use with Extreme Caution!)
```bash
python main.py --mode live
```

## Project Structure

```
trading-bot-ai/
├── main.py                 # Entry point and bot orchestration
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── src/
│   ├── __init__.py
│   ├── alpaca_client.py   # Alpaca API client
│   ├── analyzer.py        # Technical analysis engine
│   ├── sentiment.py       # LLM sentiment analysis
│   ├── llm.py            # LLM advisory module
│   ├── strategy.py        # Trading strategy logic
│   ├── portfolio.py       # Portfolio & risk management
│   ├── chat.py           # Natural language interface
│   └── backtest.py       # Backtesting framework
└── tests/
    ├── conftest.py
    ├── test_analyzer.py
    ├── test_strategy.py
    └── test_backtest.py
```

## How It Works

### Trading Cycle

1. **Data Collection**: Fetches historical bars from Alpaca
2. **Technical Analysis**: Calculates indicators (RSI, MACD, MAs, BBands)
3. **Sentiment Analysis**: LLM analyzes market sentiment for the symbol
4. **LLM Advisory**: GPT-4 provides trading recommendation
5. **Signal Generation**: Combines all inputs into a trading signal
6. **Risk Check**: Validates against position limits and risk rules
7. **Execution**: Places order if signal is actionable

### Signal Scoring

The strategy combines multiple signals with weights:
- Technical indicators: ~40%
- Sentiment analysis: ~30%
- LLM recommendation: ~30%

A total score > 0.5 generates BUY, < -0.5 generates SELL.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_analyzer.py -v
```

## Development

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

---

## 🤖 AI-Assisted Development

This project was built with significant assistance from AI tools, demonstrating modern AI-augmented software development.

### Tools Used

| Tool | Usage |
|------|-------|
| **Claude (Anthropic)** | Architecture design, code implementation, documentation |
| **GPT-4 (OpenAI)** | Integrated for sentiment analysis and trading advice |

### What AI Helped With

#### 1. **Architecture & Design**
- Modular project structure
- Separation of concerns (analysis, strategy, execution)
- Async-first design for API interactions

#### 2. **Code Implementation**
- Technical indicator calculations (RSI, MACD, Bollinger Bands)
- Alpaca API client with proper error handling
- Backtesting framework with synthetic data generation
- Natural language chat interface

#### 3. **Risk Management Logic**
- Position sizing algorithms
- Kelly Criterion implementation
- Stop-loss/take-profit calculations
- Portfolio risk assessment

#### 4. **Testing**
- Comprehensive test suites
- Edge case handling
- Mock data generation

#### 5. **Documentation**
- This README
- Inline code documentation
- Usage examples

### Human Contributions

- **Project vision and requirements definition**
- **API key management and security decisions**
- **Final code review and validation**
- **Trading strategy parameters and risk tolerance**
- **Deployment and operational decisions**

### Lessons Learned

1. **AI excels at boilerplate**: Standard patterns like API clients and data processing are quick wins for AI assistance.

2. **Domain knowledge matters**: The human needs to understand trading concepts to guide the AI effectively.

3. **Verification is essential**: AI-generated financial code must be thoroughly tested before any real use.

4. **Iterative refinement**: Best results came from iterating on AI output with specific feedback.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Alpaca Markets](https://alpaca.markets) for their excellent trading API
- [OpenAI](https://openai.com) for GPT-4
- [Anthropic](https://anthropic.com) for Claude AI assistance

---

**Remember**: Always paper trade first. Past performance does not guarantee future results. Trade responsibly.
