"""Pytest configuration and fixtures."""

import pytest
import asyncio
import sys

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return {
        "alpaca": {
            "api_key": "test_api_key",
            "secret_key": "test_secret_key",
        },
        "openai": {
            "api_key": "test_openai_key",
        },
        "trading": {
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "max_position_pct": 0.10,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
        },
        "backtest": {
            "initial_capital": 100000.0,
            "commission": 0.0,
        }
    }
