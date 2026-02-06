"""Portfolio Management Module."""

from typing import Dict, Any


class PortfolioManager:
    """Manage portfolio and position sizing."""

    def __init__(self, client, config: Dict[str, Any]):
        self.client = client
        self.config = config
        self.max_position_pct = config.get("max_position_pct", 0.1)
        self.stop_loss_pct = config.get("stop_loss_pct", 0.02)
        self.take_profit_pct = config.get("take_profit_pct", 0.05)

    async def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        account = await self.client.get_account()
        return float(account.get("equity", 0))

    async def get_available_cash(self) -> float:
        """Get available cash for trading."""
        account = await self.client.get_account()
        return float(account.get("cash", 0))

    async def get_position_value(self, symbol: str) -> float:
        """Get current position value for a symbol."""
        position = await self.client.get_position(symbol)
        if position:
            return float(position.get("market_value", 0))
        return 0.0

    def calculate_position_size(
        self, symbol: str, signal: Dict[str, Any], current_price: float = None
    ) -> int:
        """Calculate position size based on risk management rules."""
        import asyncio
        
        # Get account value synchronously (simplified)
        loop = asyncio.get_event_loop()
        
        async def _get_values():
            portfolio_value = await self.get_portfolio_value()
            position_value = await self.get_position_value(symbol)
            cash = await self.get_available_cash()
            return portfolio_value, position_value, cash
        
        portfolio_value, position_value, cash = loop.run_until_complete(_get_values())
        
        # Max position value
        max_position = portfolio_value * self.max_position_pct
        
        # Current position
        current_position = position_value
        
        if signal["action"] == "BUY":
            # How much more can we buy?
            available = min(cash, max_position - current_position)
            if available <= 0:
                return 0
            
            if current_price:
                return int(available / current_price)
            return 0
        
        elif signal["action"] == "SELL":
            # Can only sell what we have
            if current_price and current_position > 0:
                return int(current_position / current_price)
            return 0
        
        return 0

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price."""
        if side == "buy":
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price."""
        if side == "buy":
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
